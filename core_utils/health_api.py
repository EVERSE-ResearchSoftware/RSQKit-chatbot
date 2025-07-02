from settings import RESOURCES, ConfigKeys, PROVIDER_TO_RESOURCE_KEY
import os
from dotenv import load_dotenv

load_dotenv()


# Function to check if api_key is present
def check_api_key(provider: str):
    provider_id = PROVIDER_TO_RESOURCE_KEY.get(provider, "")
    provider_resource = RESOURCES.get(provider_id, {})
    is_api_key_present = (
        os.environ.get(provider_resource.get(ConfigKeys.API_ENV_VAR, "")) is not None
    )  # boolean
    return is_api_key_present
