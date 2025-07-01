# llm_client.py
import os
from typing import List, Dict, Generator, Optional
from openai import OpenAI
from settings import RESOURCES, PROVIDER_TO_RESOURCE_KEY


# Cache clients at module level
_client_cache = {}


def _get_client(provider: str) -> OpenAI:
    """Get or create cached client for provider"""

    if provider not in _client_cache:
        provider_key = PROVIDER_TO_RESOURCE_KEY[provider]
        if provider_key not in RESOURCES:
            raise ValueError(f"Unknown provider: {provider}")

        config = RESOURCES[provider_key]

        # Get API key just-in-time
        api_key = ""
        if api_env_var := config.get("api_env_var"):
            api_key = os.getenv(api_env_var, "")

        # Create and cache client
        _client_cache[provider] = OpenAI(base_url=config["base_url"], api_key=api_key)

    return _client_cache[provider]


def get_chat_response_stream(
    provider: str, messages: List[Dict], model_name: Optional[str] = None, **kwargs
) -> Generator[str, None, None]:
    """Universal chat streaming function with client caching"""
    provider_key = PROVIDER_TO_RESOURCE_KEY[provider]
    if provider_key not in RESOURCES:
        raise ValueError(f"Unknown provider: {provider}")

    config = RESOURCES[provider_key]
    client = _get_client(provider)  # Get cached client
    model = model_name or config["default_llm"]

    response_stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=kwargs.get("temperature", 0.0),
        **{k: v for k, v in kwargs.items() if k != "temperature"},
    )

    for token_data in response_stream:
        if token_data.choices and len(token_data.choices) > 0:
            delta = token_data.choices[0].delta
            content = getattr(delta, "reasoning_content", None) or getattr(
                delta, "content", None
            )
            if content:
                yield content


def get_chat_response(
    provider: str, messages: List[Dict], model_name: Optional[str] = None, **kwargs
) -> str:
    """Universal chat function with client caching"""
    provider_key = PROVIDER_TO_RESOURCE_KEY[provider]
    if provider_key not in RESOURCES:
        raise ValueError(f"Unknown provider: {provider}")

    config = RESOURCES[provider_key]
    client = _get_client(provider)  # Get cached client
    model = model_name or config["default_llm"]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        temperature=kwargs.get("temperature", 0.0),
        **{k: v for k, v in kwargs.items() if k != "temperature"},
    )

    # Handle both reasoning_content and regular content like the stream version
    message = response.choices[0].message
    content = getattr(message, "reasoning_content", None) or getattr(
        message, "content", None
    )

    return content


# Optional: Clear cache function for testing/development
def clear_client_cache():
    """Clear client cache (useful for config reloads)"""
    global _client_cache
    _client_cache.clear()
