from settings import (
    RESOURCES,
    ConfigKeys,
    PROVIDER_TO_RESOURCE_KEY,
    StatusAPI,
    ModelTypeKeys,
    ModelTypes,
)

from pathlib import Path
import logging

from llms.openai_interface import _get_client
import os
from dotenv import load_dotenv

load_dotenv()

import json
from datetime import datetime, timedelta

# Path to the JSON file that stores the last scan timestamps and cached data
last_api_scan_providers = Path(__file__).parent / "last_api_scan_providers.json"


# Helper: load JSON data (create file if it doesn't exist)
def _load_last_scan_info():
    if not last_api_scan_providers.exists():
        last_api_scan_providers.write_text("{}")
    try:
        with last_api_scan_providers.open("r") as fp:
            return json.load(fp) or {}
    except json.JSONDecodeError:
        return {}


# Helper: save JSON data
def _save_last_scan_info(data):
    with last_api_scan_providers.open("w") as fp:
        json.dump(data, fp, indent=2)


# Cache in memory for quick access
_last_scan_info = _load_last_scan_info()


def _update_provider_entry(provider, *, health_status=None, models=None):
    """
    Update the JSON entry for *provider* with the current timestamp and
    optionally the health status and available models.
    """
    global _last_scan_info
    entry = _last_scan_info.get(provider, {})
    entry.update(
        {
            "last_scan": datetime.now().isoformat(),
        }
    )
    if health_status is not None:
        entry["health_status"] = health_status
    if models is not None:
        entry["models"] = models
    _last_scan_info[provider] = entry
    _save_last_scan_info(_last_scan_info)


def check_api_key(provider: str):
    provider_id = PROVIDER_TO_RESOURCE_KEY.get(provider, "")
    provider_resources = RESOURCES.get(provider_id, {})
    is_api_key_present = (
        os.environ.get(provider_resources.get(ConfigKeys.API_ENV_VAR, "")) is not None
    )
    return is_api_key_present


def check_health_api(provider: str) -> str:
    """Return the health status of a provider with 2‑hour caching."""
    provider_data = _last_scan_info.get(provider)

    now = datetime.now()
    last_scan = None
    if provider_data:
        try:
            last_scan = datetime.fromisoformat(provider_data.get("last_scan", ""))
        except Exception:
            last_scan = None

    if not provider_data or not last_scan or now - last_scan > timedelta(hours=2):
        # Run the health check
        client = _get_client(provider=provider)
        try:
            available_models = [model.id for model in client.models.list()]
            status = StatusAPI.UP if len(available_models) > 0 else StatusAPI.DOWN
        except Exception as e:
            print("API health check failed:", e)
            status = StatusAPI.DOWN

        # Fetch models for caching (this will also update the JSON entry)
        models = get_available_models(provider)

        # Update JSON entry with status and models
        _update_provider_entry(provider, health_status=status, models=models)
    else:
        # Return cached status
        status = provider_data.get("health_status", StatusAPI.DOWN)

    return status


def get_available_models(provider: str) -> ModelTypes:
    """Fetch (or cache‑return) available models by type with 2‑hour caching."""
    provider_data = _last_scan_info.get(provider)

    now = datetime.now()
    last_scan = None
    if provider_data:
        try:
            last_scan = datetime.fromisoformat(provider_data.get("last_scan", ""))
        except Exception:
            last_scan = None

    if not provider_data or not last_scan or now - last_scan > timedelta(hours=2):
        logger = logging.getLogger(__name__)
        if not provider or not isinstance(provider, str):
            logger.error("Provider must be a non-empty string")
            result = _get_empty_model_types()
        else:
            try:
                client = _get_client(provider=provider)
                models = client.models.list()
                model_type_mapping = {
                    ModelTypeKeys.EMBEDDINGS_API_TYPE: [],
                    ModelTypeKeys.LLMS_API_TYPE: [],
                    ModelTypeKeys.VLMS_API_TYPE: [],
                    ModelTypeKeys.RERANKERS_API_TYPE: [],
                    ModelTypeKeys.TRANSCRIPTION_API_TYPE: [],
                    ModelTypeKeys.ALL_MODELS: [],
                }
                for model in models:
                    if hasattr(model, "type"):
                        if model.type in model_type_mapping:
                            model_type_mapping[model.type].append(model.id)
                        else:
                            logger.warning(f"Unknown model type: {model.type}")
                    else:
                        logger.warning(
                            f"Provider {provider} doesn't sort model by type"
                        )
                    model_type_mapping[ModelTypeKeys.ALL_MODELS].append(model.id)

                result = {
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
                    ModelTypeKeys.ALL_MODELS: model_type_mapping[
                        ModelTypeKeys.ALL_MODELS
                    ],
                }
            except Exception as e:
                logger.error(f"Failed to fetch models for provider '{provider}': {e}")
                result = _get_empty_model_types()

        # Cache the result (this also updates last_scan timestamp)
        _update_provider_entry(provider, models=result)
    else:
        # Return cached model info
        result = provider_data.get("models", _get_empty_model_types())

    return result


def _get_empty_model_types() -> ModelTypes:
    """Helper function to return an empty model types dictionary."""
    return {
        ModelTypeKeys.EMBEDDINGS: [],
        ModelTypeKeys.LLMS: [],
        ModelTypeKeys.VLMS: [],
        ModelTypeKeys.RERANKERS: [],
        ModelTypeKeys.TRANSCRIPTION: [],
        ModelTypeKeys.ALL_MODELS: [],
    }
