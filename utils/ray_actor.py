import inspect
import os
import time

import ray
import torch
from accelerate import dispatch_model, infer_auto_device_map
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .gac_gen_utils import *


def get_remote_model_generator_class(num_gpus):
    # Dynamically register the ModelGenerator class as a Ray remote class, specifying the required number of GPUs
    return ray.remote(num_gpus=num_gpus)(ModelGenerator)


class ModelGenerator:
    def __init__(
        self,
        model_path,
        model_name,
        max_memory={0: "80GiB"},
        model_ensemble_weight=1,
        use_cache=True,
        quantization="none",
    ):

        quantization_options = {
            "8bit": BitsAndBytesConfig(load_in_8bit=True),
            "4bit": BitsAndBytesConfig(load_in_4bit=True),
            "none": None,
        }

        # Retrieve the appropriate quantization_config
        quantization_config = quantization_options.get(quantization)

        # Raise an error if an invalid quantization option is provided
        if quantization_config is None and quantization != "none":
            raise ValueError(
                f"Invalid quantization value '{quantization}'. Allowed values are: 'none', '8bit', '4bit'."
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )

        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=model._get_no_split_modules("auto"),
        )

        # https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/modeling_utils.py#L3773
        device_map_kwargs = {"device_map": device_map}
        if "skip_keys" in inspect.signature(dispatch_model).parameters:
            device_map_kwargs["skip_keys"] = model._skip_keys_device_placement

        self.model_name = model_name
        self.model_ensemble_weight = model_ensemble_weight
        self.use_cache = use_cache

        # Load model to GPU
        self.model = dispatch_model(model, **device_map_kwargs)
        if self.model_name in ["Yi-34B-Chat", "Yi-6B-Chat"]:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, padding_side="left", use_fast=False, trust_remote_code=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, padding_side="left", trust_remote_code=True
            )

        # Make sure use greedy search
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = 1.0
        self.model.generation_config.top_p = 1.0

        if (
            isinstance(self.model.generation_config.eos_token_id, list)
            and len(self.model.generation_config.eos_token_id) > 1
        ):
            logger.warning(
                f"For model {self.model_name}, the eos_token_id in generation_config more than one, we only take first one."
            )
            self.model.generation_config.eos_token_id = self.model.generation_config.eos_token_id[
                0
            ]

        if self.model.generation_config.eos_token_id and (
            self.model.generation_config.eos_token_id != self.tokenizer.eos_token_id
        ):
            logger.warning(
                f"For model {self.model_name}, the eos_token_id is inconsistent between the generation config and the tokenizer ({self.model.generation_config.eos_token_id} and {self.tokenizer.eos_token_id}). We will forcefully set the tokenizer to be consistent with the generation config ({self.model.generation_config.eos_token_id})."
            )
            self.tokenizer.eos_token_id = self.model.generation_config.eos_token_id

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        if (
            self.model_name == "Starling-LM-7B-alpha"
            and len(self.tokenizer) > self.model.vocab_size
        ):
            logger.warning(
                f"Model {self.model_name} used! You need remove sep_token from tokenizer_config.json because it cause vocab size +1!"
            )

    def get_vocab_size(self):
        if len(self.tokenizer.get_vocab()) != self.model.config.vocab_size:
            logger.warning(
                f"For model {self.model_name}, the vocab_size of the tokenizer and model config are not equal! We will create the mapping matrix base on the model config."
            )
        return self.model.config.vocab_size

    def get_ensemble_weight(self):
        return self.model_ensemble_weight

    def get_input_ids(self):
        return self.state["input_ids"]

    def check_if_stop(self):
        if self.state["unfinished_sequences"].max() == 0:
            self.state["this_peer_finished"] = True

        # stop if we exceed the maximum length
        if torch.all(
            self.state["stopping_criteria"](
                self.state["input_ids"], self.state["scores"]
            )
        ):
            self.state["this_peer_finished"] = True

        return self.state["this_peer_finished"]

    def update_unfinished_sequences(self, unfinished_sequences):
        self.state["unfinished_sequences"] = unfinished_sequences.to(
            self.state["unfinished_sequences"].device
        )

    def get_unfinished_sequences(self):
        return self.state["unfinished_sequences"]

    def update_input_ids_and_model_kwargs(self, next_tokens_list):
        self.state["next_tokens_list"] = next_tokens_list
        (
            self.state["input_ids"],
            self.state["modefl_kwargs"],
            self.state["unfinished_sequences"],
        ) = update_input_ids_and_model_kwargs(self.model, self.state)

    def get_one_token(self):

        st = time.time()
        self.state["next_tokens_scores"], self.state["outputs"] = get_one_token(
            self.model, self.state
        )
        time_used = time.time() - st

        return self.model_ensemble_weight * self.state["next_tokens_scores"], time_used

    def generate_prepare(self, *args, **kwargs):
        self.state = generate_prepare(model=self.model, **self.inputs, **kwargs)
        self.state["model_kwargs"]["use_cache"] = self.use_cache

    def get_max_position_embeddings(self):
        return self.model.config.max_position_embeddings

    def get_model_name(self):
        return self.model_name

    def get_tokenizer(self):
        return self.tokenizer

    def prepare_inputs_for_model(
        self, chat_list, min_max_position_embeddings=4096, apply_chat_template=False
    ):
        # Calculate the truncation length as 75% of the minimum max_position_embeddings
        truncation_length = int(min_max_position_embeddings * 0.75)
        input_texts = []

        # Apply the chat template and collect the processed text
        for chat in chat_list:
            if apply_chat_template:
                # Assume the tokenizer has an apply_chat_template method
                processed_text = self.tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            else:
                processed_text = chat[0]["content"]
            input_texts.append(processed_text)

        self.inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            max_length=truncation_length,
            truncation=True,
        ).to(next(self.model.parameters()).device)

        return self.inputs.input_ids
