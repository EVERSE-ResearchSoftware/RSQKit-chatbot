import streamlit as st
from app_config import ICONS


@st.cache_data
def get_strategy_options():
    """Cache strategy options to avoid recreating list on every run"""
    return ["hybrid", "semantic", "keyword", "multi_step"]


def display_rag_settings():
    # Only initialize if not already done
    if not hasattr(st.session_state, "_rag_initialized"):
        st.session_state._rag_initialized = True

    with st.sidebar:
        with st.expander(f"{ICONS['rag_settings']} RAG Settings", expanded=False):

            # Core RAG sliders - use session state keys directly
            st.slider(
                "Top K Documents",
                min_value=1,
                max_value=10,
                value=st.session_state.retrieval_k,
                key="retrieval_k",
                help="Number of documents to retrieve for RAG context",
            )

            st.slider(
                "Top Rerank Documents",
                min_value=1,
                max_value=10,
                value=st.session_state.top_rerank,
                key="top_rerank",
                help="Number of documents to rerank after initial retrieval",
            )

            st.slider(
                "Chunk Size",
                min_value=400,
                max_value=1000,
                value=st.session_state.get("chunk_size", 1000),
                key="chunk_size",
                help="Length of each chunk (max 1000)",
            )

            # Chunk overlap slider
            st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=200,
                value=st.session_state.get("chunk_overlap", 0),
                key="chunk_overlap",
                help="Length of overlapping text between chunks (max 200)",
            )
            # Multi-Retrieval Settings
            st.subheader("Multi-Retrieval Settings")
            st.checkbox(
                "Enable Multi-Retrieval",
                value=st.session_state.get("enable_multi_retrieval", False),
                key="enable_multi_retrieval",
            )
            st.checkbox(
                "Enable History",
                value=st.session_state.get("enable_history", False),
                key="enable_history",
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
            st.checkbox(
                "Augment Chunk",
                value=st.session_state.get("augment_chunk", True),
                key="augment_chunk",
            )
            st.checkbox(
                "Answer Per Chunk",
                value=st.session_state.get("answer_per_chunk", False),
                key="answer_per_chunk",
            )
            # Retrieval Strategy Settings
            st.subheader("Retrieval Strategy Settings")
            strategy_options = get_strategy_options()
            default_index = strategy_options.index(
                st.session_state.get("default_strategy", "hybrid")
            )
            # Chunk overlap slider
            st.slider(
                "Semantic Weight",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("alpha", 0.5),
                key="alpha",
                help="Semantic weight in hybrid search. 0 means pure keyword seach.",
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
                max_value=5,
                value=st.session_state.get("max_subqueries", 5),
                key="max_subqueries",
                help="Maximum number of subqueries for multi-step retrieval",
            )


def display_chunking_settings():
    """Display only the chunking-related settings in the sidebar."""
    with st.sidebar:
        with st.expander(f"{ICONS['rag_settings']} Chunking Settings", expanded=False):
            # Chunk Size slider
            st.slider(
                "Chunk Size",
                min_value=400,
                max_value=1000,
                value=st.session_state.get("chunk_size", 1000),
                key="chunk_size",
                help="Length of each chunk (max 1000)",
            )

            # Chunk overlap slider
            st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=200,
                value=st.session_state.get("chunk_overlap", 0),
                key="chunk_overlap",
                help="Length of overlapping text between chunks (max 200)",
            )


def get_rag_settings():
    """
    Separate function to get current RAG settings without UI rendering.
    Use this when you only need the values without displaying the UI.
    """
    return {
        "enable_history": st.session_state.get("enable_history", False),
        "enable_multi_retrieval": st.session_state.get("enable_multi_retrieval", True),
        "show_retrieval_details": st.session_state.get("show_retrieval_details", False),
        "show_decomposition": st.session_state.get("show_decomposition", True),
        "default_strategy": st.session_state.get("default_strategy", "hybrid"),
        "max_subqueries": st.session_state.get("max_subqueries", 10),
        "retrieval_k": st.session_state.get("retrieval_k", 10),
        "top_rerank": st.session_state.get("top_rerank", 2),
        "chunk_size": st.session_state.get("chunk_size", 1000),
        "chunk_overlap": st.session_state.get("chunk_overlap", 0),
        "augment_chunk": st.session_state.get("augment_chunk", True),
        "answer_per_chunk": st.session_state.get("answer_per_chunk", True),
        "alpha": st.session_state.get("alpha", 0.5),
    }


def get_chunking_settings():
    """
    Separate function to get current RAG settings without UI rendering.
    Use this when you only need the values without displaying the UI.
    """
    return {
        "chunk_size": st.session_state.get("chunk_size", 1000),
        "chunk_overlap": st.session_state.get("chunk_overlap", 0),
    }
