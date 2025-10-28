import streamlit as st
import os
import chromadb
from dotenv import load_dotenv

from core_utils.rag_hybrid_utils import initialize_components
from core_utils.retrieval_utils import agentic_query_processing
from ui.ui_rag_settings import display_rag_settings, get_rag_settings

from chroma_data_ingestor import (
    node_pipeline,
    create_batch_embeddings,
    add_nodes_to_chroma,
)

from ui.header import sidebar
from app_config import ICONS, DOCUMENT_CHAT_COLLECTION, get_selected_llm
from ui.custom_styles import CSS_CONTENT, inject_page_styles
from ui.chat_history import display_chat_history
from settings import CHROMA_PERSIST_DIR, DOCUMENTS_DIR
from prompt_templates.prompt_builder import build_rag_context
from llm_provider_tools import get_default_llm
from llms.openai_interface import get_chat_response_stream

# ========= BASE SETUP =========
load_dotenv()
current_page_key = "document_chat"


# ========= OPTIMIZATION 1: Cache expensive resources =========
@st.cache_resource
def get_chromadb_client():
    """Persistent ChromaDB client across reruns."""
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


@st.cache_resource
def get_collection_instance(_client, collection_name):
    """Cache collection instance for the temporary RAG."""
    return _client.get_or_create_collection(collection_name)


@st.cache_data(ttl=30)
def check_collection_documents(_collection_name):
    """Fast existence check to drive UI paths (LLM-only vs RAG)."""
    client = get_chromadb_client()
    collection = get_collection_instance(client, _collection_name)
    res = collection.get(limit=1)
    docs = res.get("documents", [])
    return len(docs) > 0


@st.cache_data(ttl=30)
def get_document_count(_collection_name):
    """Lightweight count for caption/UI."""
    client = get_chromadb_client()
    collection = get_collection_instance(client, _collection_name)
    res = collection.get()
    return len(res.get("documents", []))


@st.cache_resource
def get_cached_components(
    _collection, _chat_function, llm_model, selected_provider, alpha=0.5
):
    """Cache RAG components by configuration."""
    hybrid_searcher, multi_retrieval_engine = initialize_components(
        collection=_collection,
        chat_function=_chat_function,
        llm_model=llm_model,
        alpha=alpha,
        selected_provider=selected_provider,
    )
    return hybrid_searcher, multi_retrieval_engine


# ========= OPTIMIZATION 2: Cache LLM model selection =========
@st.cache_data
def get_llm_model(page_key, provider):
    """Cache LLM model lookup with fallback."""
    return get_selected_llm(page_key=page_key, provider=provider) or get_default_llm(
        selected_provider=provider
    )


# ========= OPTIMIZATION 3: Efficient session state init =========
def init_session_state_defaults():
    """One-time init for page-scoped state & defaults."""
    # Chat buckets for this page
    if current_page_key not in st.session_state:
        st.session_state[current_page_key] = {
            "messages": [],
            "retrieval_history": [],
        }

    # Provider default (robust to missing key)
    provider_key = f"provider_{current_page_key}"
    if provider_key not in st.session_state:
        st.session_state[provider_key] = "OpenAI"

    # RAG defaults (align with your other page)
    rag_defaults = {
        "retrieval_k": 10,
        "top_rerank": 4,
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
    for k, v in rag_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ========= UI BOOTSTRAP =========
st.markdown(CSS_CONTENT, unsafe_allow_html=True)
inject_page_styles()
st.markdown(
    f'<h1 class="main-title">{ICONS["document_chat"]} Document Chat</h1>',
    unsafe_allow_html=True,
)

# Sidebar (first so provider is available)
sidebar(page_key=current_page_key)

# Init session defaults now that sidebar may have set provider
init_session_state_defaults()

# ========= CORE OBJECTS (cached) =========
chroma_client = get_chromadb_client()
temporary_collection_name = DOCUMENT_CHAT_COLLECTION
collection = get_collection_instance(chroma_client, temporary_collection_name)

# Provider / LLM
provider_key = f"provider_{current_page_key}"
selected_provider = st.session_state.get(provider_key, "OpenAI")
chat_function = get_chat_response_stream
llm_model = get_llm_model(current_page_key, selected_provider)

# ========= HELPER: file handling (kept same logic) =========
docs_path = DOCUMENTS_DIR


def save_uploaded_file(uploaded_file):
    """Persist uploaded file to DOCUMENTS_DIR."""
    os.makedirs(docs_path, exist_ok=True)
    file_path = os.path.join(docs_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def handle_uploaded_files(uploaded_files, current_collection):
    """Process -> chunk -> embed -> add to chroma."""
    file_paths, all_chunks = [], []

    for uploaded_file in uploaded_files:
        file_path = save_uploaded_file(uploaded_file)
        file_paths.append(file_path)

        chunks = node_pipeline(file_path=file_path, chunk_overlap=100, chunk_size=1000)
        if chunks:
            all_chunks.extend(chunks)
            st.success(
                f"Fichier '{uploaded_file.name}' traité avec succès ({len(chunks)} chunks extraits)."
            )
        else:
            st.error(f"Erreur lors du traitement de '{uploaded_file.name}'.")

    if all_chunks:
        # Compute embeddings in batch, then add to collection
        create_batch_embeddings(nodes=all_chunks, provider=selected_provider)
        add_nodes_to_chroma(collection=current_collection, nodes=all_chunks)
        st.success(
            f"Base de connaissances mise à jour avec {len(all_chunks)} nouveaux chunks."
        )

        # Track in session for UX
        st.session_state["file_paths"] = file_paths
        st.session_state["processed"] = True

        # Invalidate cached document checks to reflect new state
        check_collection_documents.clear()
        get_document_count.clear()


# ========= WARNER =========
def warn_user(_collection_name):
    has_docs = check_collection_documents(_collection_name)
    if not has_docs:
        st.warning("⚠️ No document provided. Using LLM only.")


# ========= CONTROLS & INPUTS =========
uploaded_files = st.file_uploader(
    label="Upload and Chat",
    type=None,
    label_visibility="hidden",
    accept_multiple_files=True,
)

display_rag_settings()  # settings panel (values read only when needed)

spacer1, col1, col2, spacer2 = st.columns([1, 2, 2, 1])
with col1:
    process_btn = st.button("Create Temporary RAG")
with col2:
    clear_btn = st.button("Clear Temporary RAG", type="secondary")

query = st.chat_input("Ask a question...")

with st.sidebar:
    # Show a quick document count & warning
    doc_count = get_document_count(temporary_collection_name)
    st.caption(f"📄 {doc_count} chunks in temporary collection")
    warn_user(temporary_collection_name)


# ========= MAIN FLOW =========
def main():
    # Display chat history for this page
    display_chat_history(current_page_key=current_page_key)

    # Handle create/clear operations
    if process_btn:
        if uploaded_files:
            with st.spinner("Traitement des documents en cours..."):
                handle_uploaded_files(
                    uploaded_files=uploaded_files, current_collection=collection
                )
                # RAG components depend on collection contents -> clear cached components
                get_cached_components.clear()
        else:
            st.info("Please upload files first.")

    elif clear_btn:
        # Delete all docs in the temp collection
        all_ids = collection.get().get("ids", [])
        if all_ids:
            collection.delete(ids=all_ids)
            st.warning("Temporary RAG cleared.")
            # Invalidate caches reflecting collection state
            check_collection_documents.clear()
            get_document_count.clear()
            get_cached_components.clear()

    # Decide mode based on presence of any docs (fast cached check)
    has_documents = check_collection_documents(temporary_collection_name)

    # If user asked something:
    if query:
        # Always echo user message
        with st.chat_message("user"):
            st.markdown(query)

        if not has_documents:
            # ===== LLM-only path (unchanged logic, just tidied) =====
            st.session_state[current_page_key]["messages"].append(
                {"role": "user", "content": query}
            )
            with st.chat_message("assistant"):
                full_resp = ""
                placeholder = st.empty()
                for token in chat_function(
                    model_name=llm_model,
                    messages=st.session_state[current_page_key]["messages"],
                    provider=selected_provider,
                ):
                    full_resp += token
                    placeholder.write(full_resp)
            st.session_state[current_page_key]["messages"].append(
                {"role": "assistant", "content": full_resp}
            )
            return

        # ===== RAG path =====
        # Pull settings only when needed
        settings_for_rag = get_rag_settings()
        alpha = st.session_state.get("alpha", 0.5)

        # Recompute cache key and clear component cache if config changed
        cache_key = (
            f"{temporary_collection_name}_{llm_model}_{selected_provider}_{alpha}"
        )
        if "last_component_cache_key" not in st.session_state:
            st.session_state.last_component_cache_key = None
        if st.session_state.last_component_cache_key != cache_key:
            st.session_state.last_component_cache_key = cache_key
            get_cached_components.clear()

        # Get (cached) RAG engines
        with st.spinner("Initializing RAG engine..."):
            _, multi_retrieval_engine = get_cached_components(
                collection, chat_function, llm_model, selected_provider, alpha
            )

        # Process query with agentic RAG
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


if __name__ == "__main__":
    main()
