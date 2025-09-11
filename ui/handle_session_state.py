import streamlit as st
from settings import StreamlitKeys
from settings import PROVIDER_TO_RESOURCE_KEY


def get_selected_llm_key(page_key, provider):
    provider_id = PROVIDER_TO_RESOURCE_KEY.get(provider, "")
    return StreamlitKeys.SELECTED_LLM_PROVIDER + f"_{page_key}_{provider_id}"


def get_selected_llm(page_key: str, provider):

    selected_llm_provider_key = get_selected_llm_key(
        page_key=page_key, provider=provider
    )
    return st.session_state[selected_llm_provider_key]


# Initialize session state for global settings
def init_global_session_state():
    """Initialize session state variables that persist across pages"""
    # RAG Parameters
    if "retrieval_k" not in st.session_state:
        st.session_state.retrieval_k = 5
    if "top_rerank" not in st.session_state:
        st.session_state.top_rerank = 3
    # LLM creativity
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.0
