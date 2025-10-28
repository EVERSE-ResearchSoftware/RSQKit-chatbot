import streamlit as st
from settings import (
    COLLECTIONS_SESSION,
    PROVIDER_ID_TO_NAME,
    StatusAPI,
)

from typing import Dict, Any
from functools import lru_cache
from llm_provider_tools import get_default_llm
from provider_persistence import ProviderPersistence
from core_utils.health_api import check_api_key, check_health_api, get_available_models
from app_config import (
    get_selected_llm_key,
    get_page_icon_mapping,
    ICONS,
    display_navigation_links,
    ensure_session_state_key,
    init_page_session_state,
    init_global_session_state,
)
from ui.ui_llm_settings import display_llm_settings


# Pure functions - use @lru_cache for better performance
@lru_cache(maxsize=128)
def get_provider_icon_mapping():
    """Cache the provider icon mapping - pure function"""
    return {
        key: ICONS["local"] if key == "ollama" else ICONS["remote"]
        for key in PROVIDER_ID_TO_NAME.keys()
    }


@lru_cache(maxsize=128)
def get_provider_options():
    """Cache provider options with icons - pure function"""
    icon_mapping = get_provider_icon_mapping()
    return {
        display_name: f"{icon_mapping[key]} {display_name}"
        for key, display_name in PROVIDER_ID_TO_NAME.items()
    }


@lru_cache(maxsize=128)
def get_icon(provider_key: str) -> str:
    """Get icon for provider with caching - pure function"""
    return ICONS["local"] if provider_key == "ollama" else ICONS["remote"]


# External state functions - keep @st.cache_data
@st.cache_data(ttl=60)
def get_provider_status(provider: str) -> Dict[str, Any]:
    """Get provider API key and health status with caching"""
    return {
        "has_api_key": check_api_key(provider=provider),
        "is_healthy": check_health_api(provider=provider) != StatusAPI.DOWN,
    }


@st.cache_data(ttl=300)
def get_cached_models(provider: str):
    """Get available models with caching"""
    return get_available_models(provider=provider)


def get_sidebar_config_with_persistence(page_key: str) -> Dict[str, Any]:
    """Enhanced sidebar config with provider persistence"""

    # Ensure session state is initialized
    init_page_session_state(page_key)

    # Get provider with persistence
    provider = ProviderPersistence.get_provider(page_key)

    # If no provider found, set empty but don't persist empty values
    if not provider:
        provider = ""

    config = {
        "page_icons": get_page_icon_mapping(),
        "provider_options": get_provider_options(),
        "provider": provider,
        "provider_page_key": f"provider_{page_key}",
        "needs_provider_selection": page_key != COLLECTIONS_SESSION,
        "needs_model_selection": page_key != COLLECTIONS_SESSION,
    }

    # Only fetch expensive data if provider is selected AND not empty
    if provider and provider.strip():
        try:
            config["default_llm"] = get_default_llm(selected_provider=provider)
            config["provider_status"] = get_provider_status(provider)
            config["available_models"] = get_cached_models(provider)
        except Exception as e:
            st.error(f"Error loading provider data for '{provider}': {e}")
            config["provider_status"] = {"has_api_key": False, "is_healthy": False}
            config["available_models"] = {"llms": [], "vlms": []}
    else:
        # Set default empty values when no provider is selected
        config["provider_status"] = {"has_api_key": False, "is_healthy": False}
        config["available_models"] = {"llms": [], "vlms": []}

    return config


def render_provider_selection_with_persistence(config: Dict[str, Any], page_key: str):
    """Enhanced provider selection with persistence"""

    current_provider = config["provider"]

    with st.expander(f"{ICONS['provider']} AI Provider", expanded=True):
        provider_options = config["provider_options"]

        # Get current index safely
        current_index = 0
        if current_provider and (current_provider in provider_options):
            current_index = list(provider_options.keys()).index(current_provider)
        elif len(provider_options) > 0:
            current_index = 0

        provider = st.radio(
            label="Provider",
            options=list(provider_options.keys()),
            format_func=lambda x: provider_options[x],
            index=current_index,
            label_visibility="hidden",
            key=f"provider_radio_{page_key}",
        )

        # Update persistence if provider changed
        if provider != current_provider:
            ProviderPersistence.set_provider(page_key, provider)
            # Clear cache to force refresh
            get_sidebar_config_with_persistence(page_key=page_key)
            st.rerun()

        # Show status if available in config
        if provider and "provider_status" in config:
            status = config["provider_status"]

            if not status["has_api_key"]:
                st.error(f"Missing API_KEY for {provider}")

            if not status["is_healthy"]:
                st.write(f"API is down {ICONS['api_down']}")


def reset_chat(page_key: str):
    """Reset chat messages for a given page"""
    ensure_session_state_key(page_key, {"messages": []})
    if page_key in st.session_state and isinstance(st.session_state[page_key], dict):
        st.session_state[page_key]["messages"] = []
    else:
        st.session_state[page_key] = {"messages": []}


def render_model_selection(page_key: str, config: Dict[str, Any]):
    """Render model selection using pre-loaded config"""
    provider = config["provider"]

    if not provider:
        st.warning("Please select a provider first.")
        return

    # Only render if models are available in config
    if "available_models" not in config:
        st.warning("Loading models...")
        return

    available_models = config["available_models"]
    llm_key = get_selected_llm_key(page_key=page_key, provider=provider)

    # Combine LLM and VLM models
    llms_vlms_models = (
        available_models.get("llms", [])
        + available_models.get("vlms", [])
        + available_models.get("all-models", [])
    )

    if llms_vlms_models:
        # Ensure the LLM key exists in session state
        _default_llm = config.get("default_llm", "")
        index_default_llm = llms_vlms_models.index(_default_llm) if _default_llm else 0
        default_model = llms_vlms_models[-1] if llms_vlms_models else ""
        ensure_session_state_key(llm_key, _default_llm)

        st.selectbox(
            label="LLMs",
            options=llms_vlms_models,
            index=index_default_llm,
            label_visibility="hidden",
            key=llm_key,
        )
        display_llm_settings()
    else:
        st.warning("No models available for this provider.")


def render_navigation_links():
    """Render navigation links with cached icons"""
    display_navigation_links()


# Main sidebar rendering function - optimized with defensive checks
def sidebar(page_key: str):
    """Main sidebar function with optimized rendering and defensive session state"""
    try:
        # Initialize session state first
        init_page_session_state(page_key)

        # Load all config at once
        # config = get_sidebar_config(page_key)
        config = get_sidebar_config_with_persistence(page_key)

        # Display navigation links
        render_navigation_links()

        with st.sidebar:
            # Divider
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # New chat button (conditional rendering)
            if config["needs_provider_selection"]:
                if st.button(
                    f"{ICONS['new_chat']} New Chat",
                    help="Start a new chat",
                    use_container_width=True,
                    key=f"new_chat_{page_key}",  # Explicit key
                ):
                    reset_chat(page_key)
                    st.success("✅ New chat started!")

            # Provider selection (conditional rendering)
            if config["needs_provider_selection"]:
                # render_provider_selection(config)
                render_provider_selection_with_persistence(config, page_key=page_key)

            # Model selection (conditional rendering)
            if config["needs_model_selection"]:
                with st.expander("Select LLM", expanded=False):
                    init_global_session_state()
                    render_model_selection(page_key, config)

    except Exception as e:
        st.error(f"Error rendering sidebar: {e}")
        st.write("Please refresh the page or check the console for more details.")


# Debug function to check session state (optional)
def debug_session_state():
    """Debug function to inspect session state - remove in production"""
    if st.checkbox("Show Session State Debug"):
        st.write("Session State Keys:")
        for key, value in st.session_state.items():
            st.write(f"- {key}: {type(value)} = {value}")
