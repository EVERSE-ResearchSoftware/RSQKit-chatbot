from settings import PROVIDER_TO_RESOURCE_KEY, RESOURCES
from openai import OpenAI
import os


def create_client(provider):
    try:
        resource_key = PROVIDER_TO_RESOURCE_KEY[provider]
        base_url = RESOURCES[resource_key].get("base_url")
        client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get(RESOURCES[resource_key].get("api_env_var")),
        )
        return client

    except Exception as e:
        raise ValueError(
            f"The following error while creating OpenAI client for the provider {provider}: {e}"
        )
