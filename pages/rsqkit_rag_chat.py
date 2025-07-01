import streamlit as st
import chromadb
from chroma_data_ingestor import get_embedding
from ui.header import sidebar, ICONS
from ui.custom_display import view_sources
from ui.custom_styles import CSS_CONTENT, inject_page_styles
from ui.rag_settings_ui import display_rag_settings
from settings import CHROMA_PERSIST_DIR
from llm_provider_tools import (
    get_embedding,
    rerank_results,
    get_default_llm,
)
from prompt_templates.prompts import RAG_SYSTEM_PROMPT
from prompt_templates.prompt_builder import build_rag_context
from dotenv import load_dotenv
import os
from llms.openai_interface import get_chat_response_stream
from core_utils.hybrid_search import HybridSearch
from core_utils.retrieval_utils import (
    create_multi_retrieval_engine,
    agentic_query_processing,
)


load_dotenv()
RSQ_KIT_RAG_SESSION_KEY = "RSQ_KIT_RAG"
RSQ_KIT_CHROMA_COLLECTION = "rsqkit"

# Streamlit App Configuration
st.set_page_config(
    page_title="RSQKit Chatbot",
    page_icon=ICONS["rsqkit_chat"],
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CSS_CONTENT, unsafe_allow_html=True)

# Sidebar and model setup
sidebar(page_key=RSQ_KIT_RAG_SESSION_KEY)
selected_provider = st.session_state[f"provider_{RSQ_KIT_RAG_SESSION_KEY}"]
llm_model = get_default_llm(selected_provider=selected_provider)
chat_function = get_chat_response_stream

inject_page_styles()
st.markdown(
    f'<h1 class="main-title">{ICONS["rsqkit_chat"]} RSQKit Chat</h1>',
    unsafe_allow_html=True,
)

# Initialize ChromaDB
db_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = db_client.get_or_create_collection(RSQ_KIT_CHROMA_COLLECTION)


# @st.cache_resource
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


def init_rag_bot_messages():
    if RSQ_KIT_RAG_SESSION_KEY not in st.session_state:
        st.session_state[RSQ_KIT_RAG_SESSION_KEY] = {
            "messages": [{"role": "system", "content": RAG_SYSTEM_PROMPT}],
            "retrieval_history": [],
        }


def display_chat_history():
    for message in st.session_state[RSQ_KIT_RAG_SESSION_KEY]["messages"]:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def main():
    init_rag_bot_messages()

    # Check if documents exist
    all_results = collection.get()
    documents = all_results["documents"]

    if not documents:
        if st.chat_input("Ask a question..."):
            st.info("There are no documents in the collection 'documents'.")
        return

    # Initialize components
    hybrid_searcher, multi_retrieval_engine = initialize_components()

    # Display chat history
    display_chat_history()

    # Multi-retrieval controls in sidebar

    settings_for_rag = display_rag_settings()
    enable_multi_retrieval = settings_for_rag["enable_multi_retrieval"]
    show_retrieval_details = settings_for_rag["show_retrieval_details"]
    show_decomposition = settings_for_rag["show_decomposition"]
    max_subqueries = settings_for_rag["max_subqueries"]
    default_strategy = settings_for_rag["default_strategy"]
    # Chat input
    query = st.chat_input("Ask a question (you can ask multiple questions at once)...")

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
            show_decomposition=show_decomposition,
            default_strategy=default_strategy,
            show_retrieval_details=show_retrieval_details,
            max_subqueries=max_subqueries,
            view_sources=view_sources,
            enable_multi_retrieval=enable_multi_retrieval,
            build_rag_context=build_rag_context,
            session_key=RSQ_KIT_RAG_SESSION_KEY,
        )


if __name__ == "__main__":
    main()
