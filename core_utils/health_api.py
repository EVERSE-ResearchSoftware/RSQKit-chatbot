from settings import (
    RESOURCES,
    ConfigKeys,
    PROVIDER_TO_RESOURCE_KEY,
    PROVIDER_ID_TO_NAME,
)
import os
from dotenv import load_dotenv

OLLAMA_DISPLAY_NAME = PROVIDER_ID_TO_NAME.get("ollama", "Ollama")

load_dotenv()


# Function to check if api_key is present
def check_api_key(provider: str):
    # Ollama doesn't need an api_key but a dummy api_key is required by the OpenAI client
    is_ollama = provider == OLLAMA_DISPLAY_NAME
    if is_ollama:
        return True

    provider_id = PROVIDER_TO_RESOURCE_KEY.get(provider, "")
    provider_resource = RESOURCES.get(provider_id, {})
    is_api_key_present = (
        os.environ.get(provider_resource.get(ConfigKeys.API_ENV_VAR, "")) is not None
    )  # boolean
    return is_api_key_present
