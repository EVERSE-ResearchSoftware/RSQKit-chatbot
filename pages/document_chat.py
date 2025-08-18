import streamlit as st
import os
import chromadb
from core_utils.hybrid_search import HybridSearch
from core_utils.retrieval_utils import (
    agentic_query_processing,
    create_multi_retrieval_engine,
)
from ui.ui_rag_settings import display_rag_settings, get_rag_settings

from dotenv import load_dotenv
from chroma_data_ingestor import (
    node_pipeline,
    create_batch_embeddings,
    add_nodes_to_chroma,
)
from ui.header import sidebar
from app_config import ICONS, DOCUMENT_CHAT_COLLECTION
from ui.custom_styles import CSS_CONTENT, inject_page_styles
from settings import CHROMA_PERSIST_DIR, DOCUMENTS_DIR
from prompt_templates.prompt_builder import build_rag_context
from llm_provider_tools import (
    get_embedding,
    rerank_results,
    get_default_llm,
)
from llms.openai_interface import get_chat_response_stream
from ui.custom_display import view_sources
from ui.handle_session_state import get_selected_llm


load_dotenv()

current_page_key = "document_chat"

# Initialize persistent ChromaDB client
db_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
temporary_collection = DOCUMENT_CHAT_COLLECTION
collection = db_client.get_or_create_collection(temporary_collection)

# Path to your text documents
docs_path = DOCUMENTS_DIR


st.markdown(CSS_CONTENT, unsafe_allow_html=True)


sidebar(page_key=current_page_key)

selected_provider = st.session_state[f"provider_{current_page_key}"]
chat_function = get_chat_response_stream
llm_model = get_selected_llm(
    page_key=current_page_key, provider=selected_provider
) or get_default_llm(selected_provider=selected_provider)

inject_page_styles()
st.markdown(
    f'<h1 class="main-title">{ICONS["document_chat"]} Document Chat</h1>',
    unsafe_allow_html=True,
)


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


# Chat history
def init_rag_bot_messages():
    if current_page_key not in st.session_state:
        st.session_state[current_page_key] = {"messages": [], "retrieval_history": []}


# Display chat history
def display_chat_history():
    for message in st.session_state[current_page_key]["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


uploaded_files = st.file_uploader(
    label="Upload and Chat",
    type=None,  # ["pdf", "png", "jpg", "jpeg"],
    label_visibility="hidden",
    accept_multiple_files=True,
)
# Chat input


# Fonction pour sauvegarder les fichiers téléchargés
def save_uploaded_file(uploaded_file):
    # Créer le répertoire de données s'il n'existe pas
    if not os.path.exists(docs_path):
        os.makedirs(docs_path, exist_ok=True)

    # Sauvegarder le fichier
    file_path = os.path.join(docs_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def handle_uploaded_files(uploaded_files, current_collection):
    file_paths = []
    all_chunks = []

    for uploaded_file in uploaded_files:
        file_path = save_uploaded_file(uploaded_file)
        file_paths.append(file_path)

        # Traiter le document
        chunks = node_pipeline(file_path=file_path, chunk_overlap=100, chunk_size=1000)
        if chunks:
            all_chunks.extend(chunks)
            st.success(
                f"Fichier '{uploaded_file.name}' traité avec succès ({len(chunks)} chunks extraits)."
            )
        else:
            st.error(f"Erreur lors du traitement de '{uploaded_file.name}'.")

    # Ajouter les chunks à la base vectorielle
    # compute all embeddings
    if all_chunks:
        create_batch_embeddings(nodes=all_chunks, provider=selected_provider)
        add_nodes_to_chroma(collection=current_collection, nodes=all_chunks)
        added_ids = [node.id for node in all_chunks]
        st.success(
            f"Base de connaissances mise à jour avec {len(added_ids)} nouveaux chunks."
        )

        # Stocker les chemins des fichiers dans la session
        st.session_state["file_paths"] = file_paths
        st.session_state["processed"] = True


def warn_user():
    all_results = collection.get()
    documents = all_results["documents"]
    if not documents:
        st.warning("⚠️ No document provided. Using LLM only.")


def main():
    init_rag_bot_messages()
    spacer1, col1, col2, spacer2 = st.columns([1, 2, 2, 1])
    with col1:
        process_btn = st.button("Create Temporary RAG")
    with col2:
        clear_btn = st.button("Clear Temporary RAG", type="secondary")
    query = st.chat_input("Ask a question...")
    with st.sidebar:
        warn_user()
    # Handle document processing when button is clicked
    if process_btn:
        if uploaded_files:
            with st.spinner("Traitement des documents en cours..."):
                handle_uploaded_files(
                    uploaded_files=uploaded_files, current_collection=collection
                )
        else:
            st.info("Please upload files first.")
    elif clear_btn:
        # Fetch all document IDs in the collection
        all_ids = collection.get()["ids"]

        # Delete all documents by ID
        if all_ids:
            collection.delete(ids=all_ids)
            st.warning("Temporary RAG cleared.")
    display_chat_history()

    # Always attempt to initialize hybrid searcher
    all_results = collection.get()
    documents = all_results["documents"]

    # Handle case when no documents are available
    if not documents:

        if query:
            # st.info("No document provided. Using LLM only.")
            # LLM-only conversation mode
            with st.chat_message("user"):
                st.markdown(query)

            # Add user message to session state
            st.session_state[current_page_key]["messages"].append(
                {"role": "user", "content": query}
            )

            # Stream LLM response without RAG context
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

            # Store assistant response
            st.session_state[current_page_key]["messages"].append(
                {"role": "assistant", "content": full_resp}
            )
        return

    # Initialize components
    hybrid_searcher, multi_retrieval_engine = initialize_components()

    if hybrid_searcher is None:
        st.error("Failed to initialize RAG engine.")
        return

    # Set hybrid searcher weight
    display_rag_settings()
    settings_for_rag = get_rag_settings()

    if query:
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
            session_key=current_page_key,
            **settings_for_rag,
        )


if __name__ == "__main__":
    main()
