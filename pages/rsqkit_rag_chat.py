# import streamlit as st
# import chromadb
# from chroma_data_ingestor import get_embedding
# from ui.header import sidebar
# from app_config import ICONS
# from ui.custom_display import view_sources
# from ui.custom_styles import CSS_CONTENT, inject_page_styles
# from ui.ui_rag_settings import get_rag_settings, display_rag_settings
# from settings import CHROMA_PERSIST_DIR
# from llm_provider_tools import (
#     get_embedding,
#     rerank_results,
#     get_default_llm,
# )
# from prompt_templates.prompts import RAG_SYSTEM_PROMPT
# from prompt_templates.prompt_builder import build_rag_context
# from dotenv import load_dotenv
# import os
# from llms.openai_interface import get_chat_response_stream
# from core_utils.hybrid_search import HybridSearch
# from core_utils.retrieval_utils import (
#     create_multi_retrieval_engine,
#     agentic_query_processing,
# )
# from ui.handle_session_state import get_selected_llm


# load_dotenv()





# st.markdown(CSS_CONTENT, unsafe_allow_html=True)

# # Sidebar and model setup
# sidebar(page_key=current_page_key)

# selected_provider = st.session_state[f"provider_{current_page_key}"]
# # update llm to use the one on sidebar
# llm_model = get_selected_llm(
#     page_key=current_page_key, provider=selected_provider
# ) or get_default_llm(selected_provider=selected_provider)


# chat_function = get_chat_response_stream

# inject_page_styles()
# st.markdown(
#     f'<h1 class="main-title">{ICONS["rsqkit_chat"]} RSQKit Chat</h1>',
#     unsafe_allow_html=True,
# )

# # Initialize ChromaDB
# db_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
# collection = db_client.get_or_create_collection(RSQ_KIT_CHROMA_COLLECTION)


# # @st.cache_resource
# def initialize_components():
#     """Initialize hybrid search and multi-retrieval engine"""
#     hybrid_searcher = HybridSearch(collection, alpha=0.7)
#     # Use the factory function to create the engine
#     multi_retrieval_engine = create_multi_retrieval_engine(
#         collection=collection,
#         selected_provider=selected_provider,
#         chat_function=chat_function,
#         model_name=llm_model,
#     )
#     return hybrid_searcher, multi_retrieval_engine


# def init_rag_bot_messages():
#     if current_page_key not in st.session_state:
#         st.session_state[current_page_key] = {
#             "messages": [{"role": "system", "content": RAG_SYSTEM_PROMPT}],
#             "retrieval_history": [],
#         }


# def display_chat_history():
#     for message in st.session_state[current_page_key]["messages"]:
#         if message["role"] != "system":
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])


# def main():
#     init_rag_bot_messages()

#     # Check if documents exist
#     all_results = collection.get()
#     documents = all_results["documents"]

#     if not documents:
#         if st.chat_input("Ask a question..."):
#             st.info(
#                 f"There are no documents in the collection '{RSQ_KIT_CHROMA_COLLECTION}'."
#             )
#         return

#     # Initialize components
#     hybrid_searcher, multi_retrieval_engine = initialize_components()

#     # Display chat history
#     display_chat_history()

#     # Multi-retrieval controls in sidebar
#     display_rag_settings()
#     settings_for_rag = get_rag_settings()
#     query = st.chat_input("Ask a question (you can ask multiple questions at once)...")

#     if query:
#         # Show user input
#         with st.chat_message("user"):
#             st.markdown(query)

#         agentic_query_processing(
#             query=query,
#             selected_provider=selected_provider,
#             rerank_results=rerank_results,
#             chat_function=chat_function,
#             get_embedding=get_embedding,
#             llm_model=llm_model,
#             multi_retrieval_engine=multi_retrieval_engine,
#             hybrid_searcher=hybrid_searcher,
#             view_sources=view_sources,
#             build_rag_context=build_rag_context,
#             session_key=current_page_key,
#             **settings_for_rag,
#         )


# if __name__ == "__main__":
#     main()


import streamlit as st
import chromadb
from ui.header import sidebar
from app_config import ICONS
from ui.custom_styles import CSS_CONTENT, inject_page_styles
from ui.ui_rag_settings import get_rag_settings, display_rag_settings
from ui.chat_history import display_chat_history
from settings import CHROMA_PERSIST_DIR
from llm_provider_tools import get_default_llm
from prompt_templates.prompts import RAG_SYSTEM_PROMPT
from prompt_templates.prompt_builder import build_rag_context
from dotenv import load_dotenv
from llms.openai_interface import get_chat_response_stream
from core_utils.rag_hybrid_utils import initialize_components
from core_utils.retrieval_utils import agentic_query_processing
from app_config import get_selected_llm

load_dotenv()


RSQ_KIT_CHROMA_COLLECTION = "rsqkit"
current_page_key = "RSQ_KIT_RAG"
# ===== OPTIMIZATION 1: Cache expensive operations =====
@st.cache_resource
def get_chromadb_client():
    """Cache the ChromaDB client - persists across reruns"""
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


@st.cache_resource
def get_collection(_client):
    """Cache the collection - persists across reruns
    Note: _ prefix in _client tells Streamlit not to hash this parameter"""
    return _client.get_or_create_collection(RSQ_KIT_CHROMA_COLLECTION)


@st.cache_data(ttl=60)  # Cache for 60 seconds
def check_documents_exist(_collection):
    """Cache document existence check with TTL to allow for updates"""
    all_results = _collection.get()
    documents = all_results.get("documents", [])
    return len(documents) > 0


@st.cache_resource
def get_cached_components(
    _collection, _chat_function, llm_model, selected_provider, alpha
):
    """Cache the RAG components initialization
    Note: _ prefix for unhashable parameters"""
    _, multi_retrieval_engine = initialize_components(
        collection=_collection,
        chat_function=_chat_function,
        llm_model=llm_model,
        selected_provider=selected_provider,
        alpha=alpha,
    )
    return multi_retrieval_engine


# ===== OPTIMIZATION 2: Initialize session state efficiently =====
def init_session_state():
    """Initialize all session state at once"""
    if current_page_key not in st.session_state:
        st.session_state[current_page_key] = {
            "messages": [{"role": "system", "content": RAG_SYSTEM_PROMPT}],
            "retrieval_history": [],
        }

    # Initialize RAG settings defaults if not present
    rag_defaults = {
        #"retrieval_k": 10,
        "top_rerank": 4,
        #"chunk_size": 1000, # already set in app_config
        #"chunk_overlap": 0, # already initialized in app_config
        "enable_multi_retrieval": False,
        "enable_history": False,
        "show_retrieval_details": False,
        "show_decomposition": True,
        "augment_chunk": False,
        "answer_per_chunk": False,
        "alpha": 0.5,
        "default_strategy": "hybrid",
        "max_subqueries": 5,
    }

    for key, default_value in rag_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# ===== OPTIMIZATION 3: Cache LLM model selection =====
@st.cache_data
def get_llm_model(page_key, provider):
    """Cache LLM model selection"""
    return get_selected_llm(page_key=page_key, provider=provider) or get_default_llm(
        selected_provider=provider
    )


# ===== STEP 1: Initialize session state FIRST =====
init_session_state()

# ===== STEP 2: Inject CSS BEFORE any content =====
inject_page_styles()
st.markdown(
    f'<h1 class="main-title">{ICONS["rsqkit_chat"]} RSQKit Chat</h1>',
    unsafe_allow_html=True,
)
st.markdown(CSS_CONTENT, unsafe_allow_html=True)

# ===== STEP 3: Sidebar and model setup =====
sidebar(page_key=current_page_key)

# OPTIMIZATION: Cache provider and model lookups
selected_provider = st.session_state.get(f"provider_{current_page_key}", "OpenAI")
llm_model = get_llm_model(current_page_key, selected_provider)
chat_function = get_chat_response_stream

# ===== STEP 4: Get cached ChromaDB client and collection =====
db_client = get_chromadb_client()
collection = get_collection(db_client)

# ===== STEP 5: Display RAG settings in sidebar =====
display_rag_settings()

# OPTIMIZATION: Only get settings when needed (moved to main)


def main():
    """Main chat interface"""

    # OPTIMIZATION: Use cached document check
    has_documents = check_documents_exist(collection)

    if not has_documents:
        if st.chat_input("Ask a question..."):
            st.info(
                f"There are no documents in the collection '{RSQ_KIT_CHROMA_COLLECTION}'."
            )
        return

    # OPTIMIZATION: Get settings only when needed for processing
    settings_for_rag = get_rag_settings()

    # OPTIMIZATION: Use cached components with current alpha value
    alpha = st.session_state.get("alpha", 0.5)

    # Only reinitialize if alpha changed (use a key to track)
    cache_key = f"{llm_model}_{selected_provider}_{alpha}"
    if (
        "last_cache_key" not in st.session_state
        or st.session_state.last_cache_key != cache_key
    ):
        st.session_state.last_cache_key = cache_key
        # Clear the cache for this specific configuration
        get_cached_components.clear()

    multi_retrieval_engine = get_cached_components(
        collection, chat_function, llm_model, selected_provider, alpha
    )

    # Display chat history
    display_chat_history(current_page_key)

    # Chat input
    query = st.chat_input(
        "Ask a question (you can ask multiple questions at once)... I can make mistake. Please verify my answers"
    )

    if query:
        # Show user input
        with st.chat_message("user"):
            st.markdown(query)

        # OPTIMIZATION: Only process when there's actual input
        agentic_query_processing(
            query=query,
            selected_provider=selected_provider,
            chat_function=chat_function,
            llm_model=llm_model,
            multi_retrieval_engine=multi_retrieval_engine,
            build_rag_context=build_rag_context,
            session_key=current_page_key,
            **settings_for_rag,
        )


# OPTIMIZATION: Use Streamlit's built-in fragment for the main chat area
# This prevents the entire page from rerunning when only the chat updates
if __name__ == "__main__":
    main()
