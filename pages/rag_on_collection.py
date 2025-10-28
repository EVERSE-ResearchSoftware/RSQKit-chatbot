import streamlit as st
import chromadb
from core_utils.retrieval_utils import agentic_query_processing
from core_utils.rag_hybrid_utils import initialize_components
from ui.ui_rag_settings import display_rag_settings, get_rag_settings
from ui.chat_history import display_chat_history
from dotenv import load_dotenv
from ui.header import sidebar
from app_config import ICONS, DOCUMENT_CHAT_COLLECTION
from ui.custom_styles import CSS_CONTENT, inject_page_styles
from settings import CHROMA_PERSIST_DIR
from prompt_templates.prompt_builder import build_rag_context
from llms.openai_interface import get_chat_response_stream
from llm_provider_tools import get_default_llm
from app_config import get_selected_llm

load_dotenv()

current_page_key = "rag_collections"


# ===== OPTIMIZATION 1: Cache expensive operations =====
@st.cache_resource
def get_chromadb_client():
    """Cache the ChromaDB client - persists across reruns"""
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


@st.cache_data(ttl=30)  # Refresh every 30 seconds to catch new collections
def get_available_collections(_client):
    """Cache available collections list with TTL for updates
    Note: _ prefix tells Streamlit not to hash this parameter"""
    collections = [
        col.name
        for col in _client.list_collections()
        if col.name != DOCUMENT_CHAT_COLLECTION
    ]
    return collections


@st.cache_resource
def get_collection_instance(_client, collection_name):
    """Cache collection instances by name"""
    return _client.get_or_create_collection(collection_name)


@st.cache_data(ttl=60)  # Cache for 60 seconds
def check_collection_documents(_collection_name):
    """Cache document existence check per collection"""
    client = get_chromadb_client()
    collection = get_collection_instance(client, _collection_name)
    all_results = collection.get(limit=1)  # Just check if any exist
    documents = all_results.get("documents", [])
    return len(documents) > 0


@st.cache_data(ttl=60)
def get_document_count(_collection_name):
    """Cache document count for display"""
    client = get_chromadb_client()
    collection = get_collection_instance(client, _collection_name)
    all_results = collection.get()
    return len(all_results.get("documents", []))


@st.cache_resource
def get_cached_components(
    _collection, _chat_function, llm_model, selected_provider, alpha=0.5
):
    """Cache the RAG components initialization per collection
    Note: _ prefix for unhashable parameters"""
    _, multi_retrieval_engine = initialize_components(
        collection=_collection,
        chat_function=_chat_function,
        llm_model=llm_model,
        alpha=alpha,
        selected_provider=selected_provider,
    )
    return multi_retrieval_engine


# ===== OPTIMIZATION 2: Cache LLM model selection =====
@st.cache_data
def get_llm_model(page_key, provider):
    """Cache LLM model selection"""
    return get_selected_llm(page_key=page_key, provider=provider) or get_default_llm(
        selected_provider=provider
    )


# ===== OPTIMIZATION 3: Efficient session state initialization =====
def init_session_state_for_collection(collection_key):
    """Initialize session state for a specific collection"""
    if collection_key not in st.session_state:
        st.session_state[collection_key] = {
            "messages": [],
            "retrieval_history": [],
        }

    # Initialize RAG settings defaults if not present
    rag_defaults = {
        "retrieval_k": 5,
        "top_rerank": 3,
        "chunk_size": 1000,
        "chunk_overlap": 0,
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


# ===== STEP 1: Inject CSS and display title =====
st.markdown(CSS_CONTENT, unsafe_allow_html=True)
inject_page_styles()
st.markdown(
    f'<h1 class="main-title">{ICONS["rag_collections"]} RAG on a Collection</h1>',
    unsafe_allow_html=True,
)

# ===== STEP 2: Get cached ChromaDB client and collections =====
chroma_client = get_chromadb_client()
collections = get_available_collections(chroma_client)

if not collections:
    st.info("No collections available. Please create a collection first.")
    st.stop()  # Stop execution here if no collections

# ===== STEP 3: Collection selection with persistence =====
# Store selected collection in session state for persistence across reruns
if "selected_collection_index" not in st.session_state:
    st.session_state.selected_collection_index = 0


# Use a callback to update the index when selection changes
def on_collection_change():
    # Clear component cache when collection changes
    get_cached_components.clear()
    # Update the index
    st.session_state.selected_collection_index = collections.index(
        st.session_state.collection_selector
    )


selected_collection = st.selectbox(
    "Select a collection",
    collections,
    index=st.session_state.selected_collection_index,
    key="collection_selector",
    on_change=on_collection_change,
)

# Display document count for the selected collection
doc_count = get_document_count(selected_collection)
st.caption(f"📄 {doc_count} documents in collection")

# ===== STEP 4: Set up tracking keys and get collection =====
collection_tracking_key = f"{current_page_key}_{selected_collection}"
collection = get_collection_instance(chroma_client, selected_collection)

# ===== STEP 5: Initialize session state for this collection =====
init_session_state_for_collection(collection_tracking_key)

# ===== STEP 6: Sidebar and provider setup =====
sidebar(page_key=collection_tracking_key)

# Get provider with fallback
provider_key = f"provider_{current_page_key}_{selected_collection}"
selected_provider = st.session_state.get(provider_key, "OpenAI")

# ===== STEP 7: Get cached LLM model =====
llm_model = get_llm_model(collection_tracking_key, selected_provider)
chat_function = get_chat_response_stream

# ===== STEP 8: Display RAG settings =====
display_rag_settings()


def main():
    """Main chat interface"""

    # OPTIMIZATION: Use cached document check
    has_documents = check_collection_documents(selected_collection)

    if not has_documents:
        st.error("No documents found in the collection.")
        if st.chat_input("Ask a question..."):
            st.info("Please add documents to the collection first.")
        return

    # Display chat history for this collection
    display_chat_history(current_page_key=collection_tracking_key)

    # OPTIMIZATION: Get settings only when processing a query
    settings_for_rag = get_rag_settings()

    # OPTIMIZATION: Use cached components with current settings
    alpha = st.session_state.get("alpha", 0.5)

    # Create a cache key for component initialization
    cache_key = f"{selected_collection}_{llm_model}_{selected_provider}_{alpha}"

    # Track if we need to reinitialize components
    if "last_component_cache_key" not in st.session_state:
        st.session_state.last_component_cache_key = None

    if st.session_state.last_component_cache_key != cache_key:
        st.session_state.last_component_cache_key = cache_key
        # Clear cache for this specific configuration if settings changed
        get_cached_components.clear()

    # Get cached components with a spinner for better UX
    with st.spinner("Initializing RAG engine..."):
        multi_retrieval_engine = get_cached_components(
            collection, chat_function, llm_model, selected_provider, alpha
        )

    # Chat input
    query = st.chat_input("Ask a question...")

    if query:
        # Show user input
        with st.chat_message("user"):
            st.markdown(query)

        # Process the query
        agentic_query_processing(
            query=query,
            selected_provider=selected_provider,
            chat_function=chat_function,
            llm_model=llm_model,
            multi_retrieval_engine=multi_retrieval_engine,
            build_rag_context=build_rag_context,
            session_key=collection_tracking_key,
            **settings_for_rag,
        )



if __name__ == "__main__":
    main()
