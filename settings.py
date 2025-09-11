from pathlib import Path
import os
import yaml
from typing import List, TypedDict
from dataclasses import dataclass


# Define a class for configuration keys
class ConfigKeys:
    """Constants for YAML configuration keys."""

    PROVIDERS = "providers"
    DIRECTORIES = "directories"
    NAME = "name"
    MODELS = "models"
    API_ENV_VAR_VISION = "api_env_var_vision"
    BASE_URL = "base_url"
    BASE_URL_VISION = "base_url_vision"
    DEFAULT_LLM = "default_llm"
    DEFAULT_EMBEDDING = "default_embedding"
    DEFAULT_RERANKER = "default_reranker"
    DEFAULT_VISION = "default_vision"
    API_ENV_VAR = "api_env_var"
    FALL_BACK_PROVIDER = "fall_back_provider"
    SUPPORTS_EMBEDDING = "supports_embedding"
    SUPPORTS_RERANKER = "supports_reranker"
    RERANK_URL = "rerank_url"
    DOCUMENTS_DIR = "documents_dir"
    CHROMA_PERSIST_DIR = "chroma_persist_dir"


class StatusAPI:
    UP = "up"
    DOWN = "down"


class ModelTypeKeys:
    EMBEDDINGS = "embeddings"
    LLMS = "llms"
    VLMS = "vlms"
    RERANKERS = "rerankers"
    TRANSCRIPTION = "transcriptions"

    EMBEDDINGS_API_TYPE = "text-embeddings-inference"
    LLMS_API_TYPE = "text-generation"
    VLMS_API_TYPE = "image-text-to-text"
    RERANKERS_API_TYPE = "text-classification"
    TRANSCRIPTION_API_TYPE = "automatic-speech-recognition"


class ModelTypes(TypedDict):
    embeddings: List[str]
    llms: List[str]
    vlms: List[str]
    rerankers: List[str]
    audio_texts: List[str]


class StreamlitKeys:
    SELECTED_LLM_PROVIDER = "selected_llm_provider"


# Define paths to YAML configuration files
YAML_CONFIG_PATH = Path(__file__).resolve().parent / "provider_config.yaml"
YAML_DIRECTORIES_PATH = Path(__file__).resolve().parent / "directories.yaml"
EMAIL_SETTINGS_FILE = Path(__file__).resolve().parent / "email_settings.yaml"


def load_yaml(file_path):
    """Load a YAML file and return its content."""
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️ Warning: YAML file not found at {file_path}.")
        return {}
    except yaml.YAMLError as e:
        print(f"⚠️ Warning: YAML parsing error in {file_path}: {e}")
        return {}


# Load YAML configuration files
YAML_CONFIG_PROVIDERS = load_yaml(YAML_CONFIG_PATH)
YAML_CONFIG_DIRECTORIES = load_yaml(YAML_DIRECTORIES_PATH)


# Extract configuration from YAML
DIRECTORIES = YAML_CONFIG_DIRECTORIES.get(ConfigKeys.DIRECTORIES, {})
CONFIG_PROVIDERS = YAML_CONFIG_PROVIDERS.get(ConfigKeys.PROVIDERS, {})

# Create mappings for provider IDs and names
PROVIDER_ID_TO_NAME = {
    provider_id: provider.get(ConfigKeys.NAME)
    for provider_id, provider in CONFIG_PROVIDERS.items()
}
PROVIDER_TO_RESOURCE_KEY = {value: key for key, value in PROVIDER_ID_TO_NAME.items()}

# Define constants
COLLECTIONS_SESSION = "COLLECTIONS_SESSION"
PERMANENT_CHROMA_COLLECTION = "permanent_collection"
# Set up directories
DOCUMENTS_DIR = str(
    Path(__file__).resolve().parent / DIRECTORIES.get(ConfigKeys.DOCUMENTS_DIR, "")
)
CHROMA_PERSIST_DIR = str(
    Path(__file__).resolve().parent / DIRECTORIES.get(ConfigKeys.CHROMA_PERSIST_DIR, "")
)

# Create document directory if it doesn't exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)


# Build RESOURCES dictionary from YAML
def build_resources(config_providers):
    """Build the RESOURCES dictionary from the provider configuration."""
    resources = {}
    for provider_id, provider_config in config_providers.items():
        models = provider_config.get(ConfigKeys.MODELS, {})
        api_env_var = provider_config.get(ConfigKeys.API_ENV_VAR)
        api_env_var_vision = provider_config.get(
            ConfigKeys.API_ENV_VAR_VISION, api_env_var
        )
        resources[provider_id] = {
            ConfigKeys.BASE_URL: provider_config.get(ConfigKeys.BASE_URL),
            ConfigKeys.BASE_URL_VISION: provider_config.get(ConfigKeys.BASE_URL_VISION),
            ConfigKeys.DEFAULT_LLM: models.get(ConfigKeys.DEFAULT_LLM),
            ConfigKeys.DEFAULT_EMBEDDING: models.get(ConfigKeys.DEFAULT_EMBEDDING),
            ConfigKeys.DEFAULT_RERANKER: models.get(ConfigKeys.DEFAULT_RERANKER),
            ConfigKeys.DEFAULT_VISION: models.get(ConfigKeys.DEFAULT_VISION),
            ConfigKeys.RERANK_URL: provider_config.get(ConfigKeys.RERANK_URL),
            ConfigKeys.API_ENV_VAR: api_env_var,
            ConfigKeys.API_ENV_VAR_VISION: api_env_var_vision,
            ConfigKeys.FALL_BACK_PROVIDER: provider_config.get(
                ConfigKeys.FALL_BACK_PROVIDER
            ),
            ConfigKeys.SUPPORTS_EMBEDDING: provider_config.get(
                ConfigKeys.SUPPORTS_EMBEDDING, False
            ),
            ConfigKeys.SUPPORTS_RERANKER: provider_config.get(
                ConfigKeys.SUPPORTS_RERANKER, False
            ),
        }
    return resources


RESOURCES = build_resources(CONFIG_PROVIDERS)
