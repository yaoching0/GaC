from typing import Any, Dict, List, Set, Tuple

import ray
import yaml
import torch
from .ray_actor import get_remote_model_generator_class
from .gac_gen_utils import *

def setup_model_actors_and_data(config: List[Dict], norm_type: str, threshold: float) -> Tuple[List[Any], List[Any], Set[str], List[Dict[int, int]], Dict[int, str], Dict[Any, str], List[Dict[str, int]], int]:
    """
    Sets up model actors based on configurations and preprocesses necessary data for text generation.

    Args:
        config (List[Dict]): Configuration list where each element is a dictionary specifying
                             model path and memory specifications.
        norm_type (str): The type of normalization to apply ('average' or 'score').
        threshold (float)

    Returns:
        Tuple containing:
        - model_actors_list (List[ActorHandle]): List of Ray actor handles for the model generators.
        - tokenizers (List[Tokenizer]): List of tokenizer instances fetched from each model actor.
        - vocab_union (Set[str]): Unified set of all tokens across the tokenizers' vocabularies.
        - mapping_matrices (List[torch.sparse_coo_tensor]): A list of sparse COO tensors, each representing
          a mapping matrix from a model's tokenizer token IDs to the token IDs in the unified vocabulary.
          Each matrix corresponds to a tokenizer and maps its original token IDs to new token IDs in the
          unified vocabulary. The shape of each matrix is [model_vocab_size, len(vocab_union)], where
          model_vocab_size is the size of the tokenizer's vocabulary.
        - index_to_vocab (Dict[int, str]): Mapping from unique indices to tokens in the unified vocabulary.
        - special_prefix_tokens_dict (Dict[Tokenizer, str]): Mapping of each tokenizer to its special prefix token.
        - byte_mappings_list (List[Dict[str, int]]): List of byte value mappings for '<0x00>' to '<0xFF>'
          for each tokenizer.
        - min_max_position_embeddings (int): The minimum of the maximum position embeddings across all model actors.
        - model_name_list (List[str]): list of model name in model_actors_list
        - primary_index (int)
        - threshold (float)
    """
    update_scores(config, norm_type)
    config = normalize_scores(config)
    logger.info(f"Model ensemble weights: {[(c['name'], round(c['score'],4)) for c in config]}")

    # find primary model
    primary_index = check_priorities(config)
    if primary_index != -1:
        real_threshold = threshold*config[primary_index]["score"]
        logger.info(f"Gate model is {config[primary_index]['name']} with threshold {threshold}, and other ensembled models KV cache will be disabled!\nPlease note that for threshold ensemble, we currently only support batch size = 1.")

    else:
        real_threshold = threshold = 1
        logger.info(f"Every token will be ensembled, meaning the threshold will not be ignored!")

    config = validate_and_update_quantization(config)

    # Initialize model actors based on configuration and GPU requirements
    model_actors_list = [
        get_remote_model_generator_class(model_config["num_gpus"]).remote(
            model_path=model_config["weight"], max_memory=model_config["max_memory"], model_name=model_config["name"], model_ensemble_weight=model_config["score"], use_cache=(primary_index == -1) or (i == primary_index), quantization=model_config["quantization"]
        )
        for i,model_config in enumerate(config)
    ]

    # Fetch tokenizer for each model
    tokenizers = [
        ray.get(model_actor.get_tokenizer.remote()) for model_actor in model_actors_list
    ]

    model_name_list = [
        ray.get(model_actor.get_model_name.remote()) for model_actor in model_actors_list
    ]

    # Determine special prefix tokens for all tokenizers
    special_prefix_tokens_dict = get_special_prefix_tokens_for_all(tokenizers)

    # Create a unified vocabulary and mappings for tokenizers
    vocab_union, tokenizers_mapping, index_to_vocab, byte_mappings_list = get_vocab_union_and_mapping(
        tokenizers
    )

    model_vocab_size_list = [
        ray.get(model_actor.get_vocab_size.remote()) for model_actor in model_actors_list
    ]

    mapping_matrices = [
        create_mapping_matrix(mapping, len(vocab_union), vocab_size)
        for mapping, tokenizer, vocab_size in zip(tokenizers_mapping, tokenizers, model_vocab_size_list)
    ]

    # Find the minimum max position embeddings across all models
    min_max_position_embeddings = min(
        ray.get(model_actor.get_max_position_embeddings.remote())
        for model_actor in model_actors_list
    )

    return (
        model_actors_list,
        tokenizers,
        vocab_union,
        mapping_matrices,
        index_to_vocab,
        special_prefix_tokens_dict,
        byte_mappings_list,
        min_max_position_embeddings,
        model_name_list,
        primary_index,
        real_threshold,
    )
    
def validate_and_update_quantization(model_config: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Validates the 'quantization' field in each dictionary of a list of model configurations,
    and adds the 'quantization' field with a default value of 'none' if it's missing.

    Args:
        model_config (List[Dict[str, str]]): 
            A list of dictionaries, where each dictionary represents a model configuration.
            Each dictionary should contain a 'quantization' key, which must have one of the 
            following values: 'none', '8bit', or '4bit'. If the 'quantization' key is missing,
            it will be added with a default value of 'none'.

    Raises:
        ValueError: If any 'quantization' value is not one of 'none', '8bit', or '4bit'.

    Returns:
        List[Dict[str, str]]: The updated list of model configurations with valid 'quantization' values.
    """
    
    # Define the valid quantization options
    valid_quantization_values = {'none', '8bit', '4bit'}
    
    # Loop through each configuration in the input list
    for idx, config in enumerate(model_config):
        # Check if 'quantization' key exists, if not, set it to 'none'
        if 'quantization' not in config:
            config['quantization'] = 'none'
        
        # Get the 'quantization' value
        quantization_value = config['quantization']
        
        # Check if the value is valid, otherwise raise an error with details
        if quantization_value not in valid_quantization_values:
            raise ValueError(
                f"Invalid quantization value '{quantization_value}' in config at index {idx}. "
                f"Allowed values are: {valid_quantization_values}"
            )
    
    # Return the updated list of configurations
    return model_config

def check_priorities(dict_list):
    """
    Check the list of dictionaries to ensure that there is exactly one "primary" priority and all priorities are valid.

    Args:
    dict_list (list of dict): A list where each item is a dictionary with a key "priority" whose value should be either "supportive" or "primary".

    Returns:
    int: Index of the first dictionary with "primary" as priority if there is exactly one, otherwise returns -1.
    """
    allowed_priorities = ["supportive", "primary"]
    primary_index = -1
    primary_count = 0

    for index, d in enumerate(dict_list):
        priority = d.get("priority")

        # Check if the priority is within the allowed values
        if priority not in allowed_priorities:
            raise ValueError(f"'priority' value '{priority}' at index {index} is not allowed!")

        # Check for primary priority and count them
        if priority == "primary":
            primary_count += 1
            if primary_count == 1:
                primary_index = index

    # Warn if there is more than one primary priority
    if primary_count > 1:
        raise ValueError("More than one 'primary' found!")

    return primary_index


def normalize_scores(config, n=1):
    """
    Normalizes the scores of each configuration in the list of dictionaries by multiplying each score by n,
    and then normalizing these scores to a 0 to 1 range such that their sum is 1.
    
    Parameters:
        config (list of dict): A list of dictionaries, each representing a configuration with a 'score' key.
        n (int, optional): The factor to multiply each score by before normalization. Defaults to 1.
    
    Returns:
        list of dict: The input list of dictionaries with normalized 'score' values.
    """
    
    # Extract scores and multiply by n
    scores = np.array([configuration['score'] for configuration in config]) ** n
    
    # Normalize scores to sum to 1
    normalized_scores = scores / np.sum(scores)
    
    # Update the scores in the original list of dictionaries
    for configuration, new_score in zip(config, normalized_scores):
        configuration['score'] = new_score
    
    return config

def extract_generated_texts(tokenizer, input_ids_0: torch.Tensor, output: torch.Tensor) -> List[str]:
    """
    Extract generated text from the model's output, excluding the input portion and any left-side padding.

    :param tokenizer: The tokenizer used, which must have a pad_token_id attribute.
    :param input_ids_0: Token IDs input to the model, shaped (batch_size, sequence_length).
                        Input may contain left-side padding.
    :param output: Model output token IDs, shaped (batch_size, output_sequence_length).
                Output sequence contains both the input sequence and the generated response.
    :return: A list of strings, where each string is the generated text for the corresponding batch.

    Function logic:
    - For each sample, find the non-pad portion in input_ids_0.
    - Search for a matching sequence in the output that corresponds to the non-pad portion.
    - Extract from the end of the matched sequence in the output to the end of the output as the response.
    - Decode the token IDs of the response into text using the tokenizer.
    """
    pad_token_id = tokenizer.pad_token_id
    generated_texts = []

    for i in range(output.shape[0]):
        # Find the index of the first non-pad token in input_ids_0
        non_pad_indices = (input_ids_0[i] != pad_token_id).nonzero().squeeze()
        
        if non_pad_indices.dim() == 0:
            non_pad_indices = non_pad_indices.unsqueeze(0)

        first_non_pad_index = non_pad_indices[0].item() if non_pad_indices.numel() > 0 else -1

        if first_non_pad_index == -1:
            raise ValueError("No non-pad tokens found in the input for batch index {}".format(i))

        # Construct the input_ids tensor of the non-pad portion for the current sample
        input_ids_non_pad = input_ids_0[i, first_non_pad_index:]

        found_match = False
        for pos in range(output.shape[1]):
            if pos + input_ids_non_pad.shape[0] <= output.shape[1]:
                if torch.equal(output[i, pos:pos+input_ids_non_pad.shape[0]], input_ids_non_pad):
                    found_match = True
                    response_start_index = pos + input_ids_non_pad.shape[0]
                    break

        if not found_match:
            raise ValueError(f"No matching sequence found in the output for batch index {i}")

        response_ids = output[i, response_start_index:]

        decoded_text = tokenizer.decode(response_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        generated_texts.append(decoded_text)

    return generated_texts

def update_scores(config, norm_type):
    """
    This function updates each dictionary in a list by different strategies based on the norm_type value.
    - 'average': Sets all scores to 1.
    - 'score': Leaves the "score" values unchanged.
    
    If the norm_type is not one of the specified values, an error is raised.

    Parameters:
    - config (list of dict): A list of dictionaries, each containing the fields "score" and "ece".
    - norm_type (str): The type of normalization to apply ('average' or 'score').

    Returns:
    - The updated list of dictionaries according to the specified normalization type.
    
    Raises:
    - ValueError: If norm_type is not one of the specified values.
    """
    if norm_type == 'average':
        for item in config:
            item["score"] = 1
    elif norm_type == 'score':
        pass
    else:
        raise ValueError(f"Invalid norm_type: {norm_type}. Expected 'average' or 'score'.")

    return config

def load_yaml_config(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    try:
        config_api_server = config['CONFIG_API_SERVER']
        norm_type_api_server = config['NORM_TYPE_API_SERVER']
        threshold_api_server = config['THRESHOLD_API_SERVER']
    except KeyError as e:
        raise ValueError(f"Missing required configuration key: {e}")

    return config_api_server, norm_type_api_server, threshold_api_server

# init RAY
ray.init()
