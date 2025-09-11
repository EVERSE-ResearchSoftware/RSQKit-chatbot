import streamlit as st
from app_config import ICONS
# from ui.handle_session_state import init_global_session_state


@st.cache_data
def get_strategy_options():
    """Cache strategy options to avoid recreating list on every run"""
    return ["hybrid", "semantic", "keyword", "multi_step"]


def display_rag_settings():
    # Only initialize if not already done
    if not hasattr(st.session_state, "_rag_initialized"):
        # init_global_session_state()
        st.session_state._rag_initialized = True

    with st.sidebar:
        with st.expander(f"{ICONS['rag_settings']} RAG Settings", expanded=False):

            # Core RAG sliders - use session state keys directly
            st.slider(
                "Top K Documents",
                min_value=1,
                max_value=7,
                value=st.session_state.retrieval_k,
                key="retrieval_k",
                help="Number of documents to retrieve for RAG context",
            )

            st.slider(
                "Top Rerank Documents",
                min_value=1,
                max_value=7,
                value=st.session_state.top_rerank,
                key="top_rerank",
                help="Number of documents to rerank after initial retrieval",
            )

            # Multi-Retrieval Settings
            st.subheader("Multi-Retrieval Settings")
            st.checkbox(
                "Enable Multi-Retrieval",
                value=st.session_state.get("enable_multi_retrieval", True),
                key="enable_multi_retrieval",
            )

            st.checkbox(
                "Show Retrieval Details",
                value=st.session_state.get("show_retrieval_details", False),
                key="show_retrieval_details",
            )

            st.checkbox(
                "Show Query Decomposition",
                value=st.session_state.get("show_decomposition", True),
                key="show_decomposition",
            )

            # Retrieval Strategy Settings
            st.subheader("Retrieval Strategy Settings")
            strategy_options = get_strategy_options()
            default_index = strategy_options.index(
                st.session_state.get("default_strategy", "hybrid")
            )

            st.selectbox(
                "Default Strategy",
                options=strategy_options,
                index=default_index,
                key="default_strategy",
            )

            st.slider(
                "Max Subqueries",
                min_value=1,
                max_value=10,
                value=st.session_state.get("max_subqueries", 5),
                key="max_subqueries",
                help="Maximum number of subqueries for multi-step retrieval",
            )


def get_rag_settings():
    """
    Separate function to get current RAG settings without UI rendering.
    Use this when you only need the values without displaying the UI.
    """
    return {
        "enable_multi_retrieval": st.session_state.get("enable_multi_retrieval", True),
        "show_retrieval_details": st.session_state.get("show_retrieval_details", False),
        "show_decomposition": st.session_state.get("show_decomposition", True),
        "default_strategy": st.session_state.get("default_strategy", "hybrid"),
        "max_subqueries": st.session_state.get("max_subqueries", 5),
        "retrieval_k": st.session_state.get("retrieval_k", 5),
        "top_rerank": st.session_state.get("top_rerank", 3),
    }


# Enhanced session state initialization
def init_rag_session_state():
    """Initialize all RAG-related session state variables"""
    defaults = {
        "retrieval_k": 5,
        "top_rerank": 3,
        "enable_multi_retrieval": True,
        "show_retrieval_details": False,
        "show_decomposition": True,
        "default_strategy": "hybrid",
        "max_subqueries": 5,
        "temperature": 0.0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
