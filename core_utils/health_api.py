# from settings import (
#     RESOURCES,
#     ConfigKeys,
#     PROVIDER_TO_RESOURCE_KEY,
#     PROVIDER_ID_TO_NAME,
# )
# import os
# from dotenv import load_dotenv

# OLLAMA_DISPLAY_NAME = PROVIDER_ID_TO_NAME.get("ollama", "Ollama")

# load_dotenv()


# # Function to check if api_key is present
# def check_api_key(provider: str):
#     # Ollama doesn't need an api_key but a dummy api_key is required by the OpenAI client
#     is_ollama = provider == OLLAMA_DISPLAY_NAME
#     if is_ollama:
#         return True

#     provider_id = PROVIDER_TO_RESOURCE_KEY.get(provider, "")
#     provider_resource = RESOURCES.get(provider_id, {})
#     is_api_key_present = (
#         os.environ.get(provider_resource.get(ConfigKeys.API_ENV_VAR, "")) is not None
#     )  # boolean
#     return is_api_key_present

from settings import (
    RESOURCES,
    ConfigKeys,
    PROVIDER_TO_RESOURCE_KEY,
    StatusAPI,
    ModelTypeKeys,
    ModelTypes,
)
from typing import Dict, List, TypedDict, Optional
import logging

from llms.openai_interface import _get_client
import os
from dotenv import load_dotenv

load_dotenv()


# Function to check if api_key is present
def check_api_key(provider: str):
    provider_id = PROVIDER_TO_RESOURCE_KEY.get(provider, "")
    provider_resources = RESOURCES.get(provider_id, {})
    is_api_key_present = (
        os.environ.get(provider_resources.get(ConfigKeys.API_ENV_VAR, "")) is not None
    )  # boolean
    return is_api_key_present


def check_health_api(provider: str) -> str:
    client = _get_client(provider=provider)
    try:
        available_models = [model.id for model in client.models.list()]
        return StatusAPI.UP if len(available_models) > 0 else StatusAPI.DOWN
    except Exception as e:
        print("API health check failed:", e)
        return StatusAPI.DOWN


from typing import List, TypedDict
import logging


def get_available_models(provider: str) -> ModelTypes:
    """
    Fetches and categorizes available models by type from the given provider.

    Args:
        provider (str): The name of the provider to use for the API client.

    Returns:
        ModelTypes: A dictionary with keys 'embeddings', 'llms', 'vlms', and 'rerankers',
                   each mapping to a list of model IDs that match the respective type.

    Raises:
        ValueError: If the provider is not valid or client creation fails.
    """
    logger = logging.getLogger(__name__)

    if not provider or not isinstance(provider, str):
        logger.error("Provider must be a non-empty string")
        return _get_empty_model_types()

    try:
        client = _get_client(provider=provider)
        models = client.models.list()

        model_type_mapping = {
            ModelTypeKeys.EMBEDDINGS_API_TYPE: [],
            ModelTypeKeys.LLMS_API_TYPE: [],
            ModelTypeKeys.VLMS_API_TYPE: [],
            ModelTypeKeys.RERANKERS_API_TYPE: [],
            ModelTypeKeys.TRANSCRIPTION_API_TYPE: [],
        }

        for model in models:
            if model.type in model_type_mapping:
                model_type_mapping[model.type].append(model.id)
            else:
                logger.warning(f"Unknown model type: {model.type}")

        return {
            ModelTypeKeys.EMBEDDINGS: model_type_mapping[
                ModelTypeKeys.EMBEDDINGS_API_TYPE
            ],
            ModelTypeKeys.LLMS: model_type_mapping[ModelTypeKeys.LLMS_API_TYPE],
            ModelTypeKeys.VLMS: model_type_mapping[ModelTypeKeys.VLMS_API_TYPE],
            ModelTypeKeys.RERANKERS: model_type_mapping[
                ModelTypeKeys.RERANKERS_API_TYPE
            ],
            ModelTypeKeys.TRANSCRIPTION: model_type_mapping[
                ModelTypeKeys.TRANSCRIPTION_API_TYPE
            ],
        }

    except Exception as e:
        logger.error(f"Failed to fetch models for provider '{provider}': {e}")
        return _get_empty_model_types()


def _get_empty_model_types() -> ModelTypes:
    """Helper function to return an empty model types dictionary."""
    return {
        ModelTypeKeys.EMBEDDINGS: [],
        ModelTypeKeys.LLMS: [],
        ModelTypeKeys.VLMS: [],
        ModelTypeKeys.RERANKERS: [],
        ModelTypeKeys.TRANSCRIPTION: [],
    }
