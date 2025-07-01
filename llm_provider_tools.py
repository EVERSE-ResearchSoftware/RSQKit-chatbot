from vision.ocr_by_provider import analyse_image_provider as provider_ocr
from settings import RESOURCES, PROVIDER_TO_RESOURCE_KEY, PROVIDER_ID_TO_NAME
from openai import OpenAI
from typing import List, Union, Optional, Dict
import requests
from dotenv import load_dotenv
import json
import os
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Custom exception for LLM client errors"""

    pass


def resolve_vision_function(provider: str):
    """Resolve chat function based on provider"""
    provider_key = _get_resource_key(provider)

    if provider_key in PROVIDER_ID_TO_NAME:
        return provider_ocr
    else:
        raise LLMClientError(f"Unsupported provider: {provider}")


def _get_resource_key(provider: str) -> str:
    """Convert provider name to resource key"""

    if provider not in PROVIDER_TO_RESOURCE_KEY:
        raise LLMClientError(f"Unknown provider: {provider}")

    return PROVIDER_TO_RESOURCE_KEY[provider]


def _get_provider_config(provider: str) -> dict:
    """Get configuration for a specific provider"""
    resource_key = _get_resource_key(provider)

    if resource_key not in RESOURCES:
        raise LLMClientError(f"Configuration missing for provider: {provider}")

    config = RESOURCES[resource_key]
    required_keys = ["base_url", "default_embedding"]

    for key in required_keys:
        if key not in config:
            raise LLMClientError(f"Missing '{key}' in {provider} configuration")

    return config


def _create_client(provider: str) -> OpenAI:
    """Create OpenAI client for embeddings"""
    config = _get_provider_config(provider)
    resource_key = _get_resource_key(provider)

    api_key = os.environ.get(config["api_env_var"], "required-but-needed-for-ollama")

    if not api_key and resource_key != "ollama":
        raise LLMClientError(f"API_KEY environment variable required for {provider}")

    return OpenAI(base_url=config["base_url"], api_key=api_key)


def get_embedding(
    input: Union[str, List[str]], provider: str, model_name: Optional[str] = None
) -> Union[List[float], List[List[float]]]:
    """
    Get embeddings from a specified provider.

    Args:
        input: Text string or list of strings to embed
        provider: Provider name ('Ollama' or 'Albert API')
        model_name: Optional model name, uses default if not provided

    Returns:
        Single embedding list if input is string, list of embeddings if input is list

    Raises:
        LLMClientError: If provider is unsupported or configuration is invalid
    """

    try:

        # Fetch provider configuration
        config = _get_provider_config(provider)
        model = model_name or config.get("default_embedding")

        # Check if the provider supports embedding
        supports_embedding = config["supports_embedding"]
        # Fallback to referenced fall_back_provider if the provider doesn't support embedding
        if not supports_embedding:
            # get embedding from fallback_provider
            fall_back_provider = config["fall_back_provider"]
            return get_embedding(
                input=input, provider=fall_back_provider, model_name=model
            )

        # Create client based on the provider
        # Normalize input to a list for uniform processing
        is_string = isinstance(input, str)
        if is_string:
            input = [input]
        client = _create_client(provider)

        # Generate embeddings
        response = client.embeddings.create(
            model=model, input=input, encoding_format="float"
        )

        # Extract embeddings from response
        embeddings = [d.embedding for d in response.data]

        # Return single embedding if input was a string
        return embeddings[0] if is_string else embeddings

    except KeyError as e:
        logger.error(f"Missing configuration key: {e}")
        raise LLMClientError(f"Missing configuration key: {e}") from e

    except Exception as e:
        logger.error(f"Failed to get embeddings from {provider}: {str(e)}")
        raise LLMClientError(f"Embedding request failed: {str(e)}") from e


def rerank_docs(
    prompt: str, input: List[str], provider: str, model_name: Optional[str] = None
) -> Dict:

    config = _get_provider_config(provider)
    supports_reranker = config["supports_reranker"]
    if not supports_reranker:  # No reranker available
        reranked_docs_data = {
            "indices": [_ for _ in range(len(input))],  # same order as the input
            "scores": [len(input) - idx for idx in range(len(input))],  # made up scores
        }
        return reranked_docs_data

    required_keys = ["rerank_url", "default_reranker"]

    for key in required_keys:
        if key not in config:
            raise LLMClientError(f"Missing '{key}' in {provider} configuration")
    api_key = os.environ.get(
        config["api_env_var"], "required-but-not-needed-for-ollama"
    )
    model = model_name or config.get("default_reranker")
    url = config.get("rerank_url")
    headers = {"authorization": f"Bearer {api_key}", "content-type": "application/json"}
    data = {"prompt": prompt, "input": input, "model": model}
    if url is not None:
        response = requests.post(url=url, headers=headers, json=data)
        if response.status_code == 200:
            response_dict = json.loads(response.content.decode())
            reranked_docs_data = {
                "indices": [d.get("index") for d in response_dict["data"]],
                "scores": [d.get("score") for d in response_dict["data"]],
            }
            return reranked_docs_data
        else:
            raise ValueError(f"This error occured {response.status_code}")
    else:
        raise ValueError(f"url for reranker not available for provider: {provider}")


def rerank_results(results, provider, query, top_rerank=3):
    """
    Rerank search results using a specified provider.

    Args:
        results: Query results containing documents, ids, and metadatas
        provider: The provider to use for reranking
        query: The search query/prompt to use for reranking
        top_rerank: Number of top results to return (default: 3)

    Returns:
        dict: Dictionary with keys "reranked_docs" and "reranked_metadata"
    """
    relevant_docs = [doc for doc in results["documents"][0]]
    metadatas = [item for item in results["metadatas"][0]]
    reranked_docs_data = rerank_docs(
        prompt=query, input=relevant_docs, provider=provider
    )
    reranked_indices = reranked_docs_data["indices"]
    scores = reranked_docs_data["scores"]
    reranked_docs = [relevant_docs[i] for i in reranked_indices[:top_rerank]]
    reranked_metadatas = [metadatas[i] for i in reranked_indices[:top_rerank]]
    reranked_scores = [scores[i] for i in reranked_indices[:top_rerank]]

    return {
        "reranked_docs": reranked_docs,
        "reranked_metadatas": reranked_metadatas,
        "reranked_scores": reranked_scores,
    }


def get_default_llm(selected_provider: str) -> str:

    resource_key = PROVIDER_TO_RESOURCE_KEY.get(selected_provider)

    if not resource_key:
        raise ValueError(f"Unknown provider: '{selected_provider}'")

    resource = RESOURCES.get(resource_key)
    if not resource:
        raise KeyError(f"No resource found for key: '{resource_key}'")

    llm_model = resource.get("default_llm")
    if not llm_model:
        raise KeyError(f"'default_llm' not defined for provider '{resource_key}'")

    return llm_model


def get_default_vison_model(selected_provider: str) -> str:

    resource_key = PROVIDER_TO_RESOURCE_KEY.get(selected_provider)

    if not resource_key:
        raise ValueError(f"Unknown provider: '{selected_provider}'")

    resource = RESOURCES.get(resource_key)
    if not resource:
        raise KeyError(f"No resource found for key: '{resource_key}'")

    vision_model = resource.get("default_vision")
    if not vision_model:
        raise KeyError(f"'default_llm' not defined for provider '{resource_key}'")

    return vision_model


def _get_ai_resources(provider):
    provider_resource_key = _get_resource_key(provider=provider)
    return RESOURCES.get(provider_resource_key)
