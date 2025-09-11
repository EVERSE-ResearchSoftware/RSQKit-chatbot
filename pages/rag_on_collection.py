import streamlit as st
import chromadb
from core_utils.hybrid_search import HybridSearch
from core_utils.retrieval_utils import agentic_query_processing
from ui.ui_rag_settings import display_rag_settings, get_rag_settings
from dotenv import load_dotenv
from ui.header import sidebar
from app_config import ICONS, DOCUMENT_CHAT_COLLECTION
from ui.custom_styles import CSS_CONTENT, inject_page_styles
from settings import CHROMA_PERSIST_DIR
from ui.custom_display import view_sources
from prompt_templates.prompt_builder import build_rag_context
from llms.openai_interface import get_chat_response_stream
from llm_provider_tools import (
    get_embedding,
    rerank_results,
    get_default_llm,
)
from core_utils.retrieval_utils import (
    create_multi_retrieval_engine,
)
from ui.handle_session_state import get_selected_llm

load_dotenv()

# Initialize persistent ChromaDB clien
# ----


current_page_key = "rag_collections"


st.markdown(CSS_CONTENT, unsafe_allow_html=True)

# ---- end css


inject_page_styles()
st.markdown(
    f'<h1 class="main-title">{ICONS["rag_collections"]} RAG on a Collection</h1>',
    unsafe_allow_html=True,
)


# Initialize ChromaDB client and collection (persistent directory)
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collections = [
    col.name
    for col in chroma_client.list_collections()
    if col.name != DOCUMENT_CHAT_COLLECTION
]
if not collections:
    st.info("No collections available. Please create a collection first.")

# Select a collection
selected_collection = st.selectbox(
    "Select a collection", collections, index=0 if collections else None
)

# tracking_key
collection_tracking_key = rf"{current_page_key}_{selected_collection}"
collection = chroma_client.get_or_create_collection(selected_collection)

sidebar(page_key=collection_tracking_key)

selected_provider = st.session_state[
    f"provider_{current_page_key}_{selected_collection}"
]  # to track chat history per collection
llm_model = get_selected_llm(
    page_key=collection_tracking_key, provider=selected_provider
) or get_default_llm(selected_provider=selected_provider)
chat_function = get_chat_response_stream


# Chat history
def init_rag_bot_messages():
    if collection_tracking_key not in st.session_state:
        st.session_state[collection_tracking_key] = {
            "messages": [],
            "retrieval_history": [],
        }


def initialize_components():
    """Initialize hybrid search and multi-retrieval engine"""
    hybrid_searcher = HybridSearch(collection, alpha=0.7)
    # Use the factory function to create the engine
    multi_retrieval_engine = create_multi_retrieval_engine(
        collection=collection,
        selected_provider=selected_provider,
        chat_function=chat_function,
        model_name=llm_model,
    )
    return hybrid_searcher, multi_retrieval_engine


# Display chat history
def display_chat_history():
    for message in st.session_state[collection_tracking_key]["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main():
    init_rag_bot_messages()
    display_chat_history()

    # Check if documents exist
    all_results = collection.get()
    documents = all_results["documents"]

    if not documents:
        st.error("No documents found in the collection.")
        if st.chat_input("Ask a question..."):
            st.info("Please add documents to the collection first.")
        return

    # Initialize components
    hybrid_searcher, multi_retrieval_engine = initialize_components()

    if hybrid_searcher is None:
        st.error("Unable to load RAG engine.")
        return

    display_rag_settings()
    settings_for_rag = get_rag_settings()

    # Chat input
    query = st.chat_input("Ask a question...")

    if query:
        # Show user input
        with st.chat_message("user"):
            st.markdown(query)

        agentic_query_processing(
            query=query,
            selected_provider=selected_provider,
            rerank_results=rerank_results,
            chat_function=chat_function,
            get_embedding=get_embedding,
            llm_model=llm_model,
            multi_retrieval_engine=multi_retrieval_engine,
            hybrid_searcher=hybrid_searcher,
            view_sources=view_sources,
            build_rag_context=build_rag_context,
            session_key=collection_tracking_key,
            **settings_for_rag,
        )


if __name__ == "__main__":
    main()

