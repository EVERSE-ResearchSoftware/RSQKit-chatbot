import streamlit as st
import os
import chromadb
from dotenv import load_dotenv
from pathlib import Path
from chroma_data_ingestor import (
    node_pipeline,
    create_batch_embeddings,
    add_nodes_to_chroma,
)
from ui.header import sidebar, ICONS
from settings import CHROMA_PERSIST_DIR, COLLECTIONS_SESSION, DOCUMENTS_DIR
from collections import defaultdict
from ui.custom_styles import CSS_CONTENT, inject_page_styles


load_dotenv()


# Initialize persistent ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

# Path to your text documents
docs_path = DOCUMENTS_DIR


st.set_page_config(
    page_title="Collections Management",
    page_icon=ICONS["collections"],
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS for styling
st.markdown(CSS_CONTENT, unsafe_allow_html=True)

sidebar(page_key=COLLECTIONS_SESSION)
selected_provider = st.session_state[f"provider_{COLLECTIONS_SESSION}"]

inject_page_styles()
st.markdown(
    f'<h1 class="main-title">{ICONS["collections"]} Collections Management</h1>',
    unsafe_allow_html=True,
)


# Initialize ChromaDB client and collection (persistent directory)
collections = [col.name for col in client.list_collections()]
if not collections:
    st.info("No collections available. Please create a collection first.")


def save_uploaded_file(uploaded_file):
    if not os.path.exists(docs_path):
        os.makedirs(docs_path, exist_ok=True)

    file_path = os.path.join(docs_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def add_files(
    uploaded_files,
    collection_name: str,
    provider: str = selected_provider,
    purpose: str = "update",
):
    do_create = purpose == "create"

    if not collection_name in collections and not do_create:
        st.error(f"Collection '{collection_name}' does not exist.")
    elif collection_name in collections and do_create:
        st.error(f"Collection '{collection_name}' already exists.")
    elif not uploaded_files:
        st.error("Please upload at least one document.")
    else:
        with st.spinner("Processing documents..."):
            all_chunks = []
            file_paths = []

            for uploaded_file in uploaded_files:
                file_path = save_uploaded_file(uploaded_file)
                file_paths.append(file_path)

                chunks = node_pipeline(
                    file_path=file_path, chunk_overlap=120, chunk_size=1000
                )
                if chunks:
                    all_chunks.extend(chunks)
                else:
                    st.error(f"Erreur lors du traitement de '{uploaded_file.name}'.")

            if all_chunks:
                create_batch_embeddings(nodes=all_chunks, provider=provider)
                collection = client.get_or_create_collection(collection_name)
                add_nodes_to_chroma(collection=collection, nodes=all_chunks)
                added_ids = [node.id for node in all_chunks]

            if purpose == "update":
                st.success(
                    f"Collection '{collection_name}' has been successfully updated with {len(all_chunks)} new chunks."
                )
            elif purpose == "create":
                st.success(
                    f"Collection '{collection_name}' created successfully with {len(all_chunks)} chunks."
                )


def get_collection_documents(collection):
    """Get and group documents from a collection"""
    try:
        result = collection.get(include=["metadatas"])
        if not result["ids"]:
            return {}

        documents = defaultdict(
            lambda: {
                "chunks": [],
                "chunk_count": 0,
                "metadata": {},
                "filename": "Unknown",
            }
        )

        for chunk_id, metadata in zip(result["ids"], result["metadatas"]):
            # Try to identify document using various metadata fields
            doc_key = None
            filename = "Unknown"

            if metadata and isinstance(metadata, dict):
                # Priority order for document identification

                if metadata.get("file-path"):
                    doc_key = metadata["file-path"]
                    filename = metadata.get(
                        "title", metadata.get("chunkSource", "Unknown").split("/")[-1]
                    )

                elif metadata.get("document_uuid"):
                    doc_key = metadata["document_uuid"]
                    filename = metadata.get(
                        "filename", metadata.get("source", "Unknown")
                    )
                elif metadata.get("file_hash") and metadata.get("filename"):
                    doc_key = f"{metadata['filename']}_{metadata['file_hash']}"
                    filename = metadata["filename"]
                elif metadata.get("source"):
                    # Extract filename from source path
                    source_path = metadata["source"]
                    filename = Path(source_path).name if source_path else "Unknown"
                    doc_key = source_path
                elif metadata.get("filename"):
                    filename = metadata["filename"]
                    doc_key = filename
                else:
                    # Look for any field that might contain a filename
                    for key, value in metadata.items():
                        if isinstance(value, str) and any(
                            ext in value.lower()
                            for ext in [".pdf", ".txt", ".docx", ".md"]
                        ):
                            filename = Path(value).name
                            doc_key = value
                            break

            # Fallback if no document key found
            if not doc_key:
                if "_chunk_" in chunk_id:
                    doc_key = chunk_id.split("_chunk_")[0]
                    filename = doc_key
                else:
                    doc_key = f"unknown_document_{hash(chunk_id) % 1000}"
                    filename = doc_key

            documents[doc_key]["chunks"].append(chunk_id)
            documents[doc_key]["chunk_count"] += 1
            documents[doc_key]["filename"] = filename

            # Store representative metadata (from first chunk)
            if not documents[doc_key]["metadata"]:
                documents[doc_key]["metadata"] = metadata or {}

        return dict(documents)
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return {}


def create_delete_filter(doc_key, doc_info):
    """Create a safe filter for deleting a specific document"""
    metadata = doc_info["metadata"]

    # Build the most specific filter possible
    where_conditions = []

    if metadata.get("document_uuid"):
        return {"document_uuid": metadata["document_uuid"]}

    if metadata.get("file_hash"):
        where_conditions.append({"file_hash": metadata["file_hash"]})

    if metadata.get("source"):
        where_conditions.append({"source": metadata["source"]})

    if metadata.get("filename"):
        where_conditions.append({"filename": metadata["filename"]})
    if metadata.get("file-path"):
        where_conditions.append({"file-path": metadata["file-path"]})

    # Combine conditions with AND
    if len(where_conditions) > 1:
        return {"$and": where_conditions}
    elif len(where_conditions) == 1:
        return where_conditions[0]
    else:
        # Fallback: delete by specific chunk IDs
        return None


def delete_document_safely(collection, doc_key, doc_info):
    """Safely delete a document from the collection"""
    try:
        # First try metadata-based deletion
        where_filter = create_delete_filter(doc_key, doc_info)

        if where_filter:
            # Preview what will be deleted
            preview = collection.get(where=where_filter, include=["metadatas"])
            expected_chunks = set(doc_info["chunks"])
            actual_chunks = set(preview["ids"])

            # Safety check: ensure we're deleting exactly what we expect
            if expected_chunks == actual_chunks:
                collection.delete(where=where_filter)
                return True, f"Successfully deleted {len(actual_chunks)} chunks"
            else:
                # Fallback to ID-based deletion for safety
                collection.delete(ids=doc_info["chunks"])
                return (
                    True,
                    f"Deleted {len(doc_info['chunks'])} chunks by ID (fallback method)",
                )
        else:
            # Delete by specific chunk IDs
            collection.delete(ids=doc_info["chunks"])
            return True, f"Deleted {len(doc_info['chunks'])} chunks by ID"

    except Exception as e:
        return False, f"Error deleting document: {str(e)}"


# Display current collections
st.subheader("Current Collections")
if collections:
    for collection in collections:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(collection)
        with col2:
            if st.button("Delete", key=f"delete_{collection}"):
                client.delete_collection(collection)
                st.success(f"Collection '{collection}' deleted successfully.")
                st.rerun()
else:
    st.info("No collections available.")

# Create new collection
st.subheader("Create New Collection")
with st.form("create_collection_form"):
    collection_name = st.text_input("Collection Name")
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True)
    create_collection = st.form_submit_button("Create Collection")
    if create_collection and collection_name:
        add_files(
            uploaded_files=uploaded_files,
            collection_name=collection_name,
            purpose="create",
        )

# Update collection section with document management
st.markdown("---")
st.subheader("Update Collection")

if collections:
    selected_collection = st.selectbox("Select a collection", collections, index=0)

    if selected_collection:
        # Get collection info
        collection = client.get_collection(selected_collection)
        documents = get_collection_documents(collection)

        # Collection stats
        total_chunks = sum(doc["chunk_count"] for doc in documents.values())
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", len(documents))
        with col2:
            st.metric("Total Chunks", total_chunks)
        with col3:
            if documents:
                avg_chunks = total_chunks / len(documents)
                st.metric("Avg Chunks/Doc", f"{avg_chunks:.1f}")

        # Document management tabs
        tab1, tab2 = st.tabs(["📄 Manage Documents", "➕ Add Documents"])

        with tab1:
            if documents:
                st.write("**Documents in this collection:**")

                for idx, (doc_key, doc_info) in enumerate(documents.items()):
                    with st.container():
                        st.markdown(
                            '<div class="document-item">', unsafe_allow_html=True
                        )

                        col1, col2, col3, col4 = st.columns([4, 2, 1, 1])

                        with col1:
                            st.write(f"**📄 {doc_info['filename']}**")
                            if doc_info["metadata"].get("source"):
                                st.caption(f"Source: {doc_info['metadata']['source']}")

                        with col2:
                            st.write(f"📊 {doc_info['chunk_count']} chunks")

                        with col3:
                            if st.button(
                                "ℹ️",
                                key=f"info_{selected_collection}_{idx}",
                                help="View details",
                            ):
                                detail_key = f"show_details_{selected_collection}_{idx}"
                                st.session_state[detail_key] = not st.session_state.get(
                                    detail_key, False
                                )

                        with col4:
                            delete_key = f"delete_{selected_collection}_{idx}"
                            confirm_key = f"confirm_delete_{selected_collection}_{idx}"

                            # Check if we're in confirmation mode
                            if st.session_state.get(confirm_key, False):
                                col_yes, col_no = st.columns(2)
                                with col_yes:
                                    if st.button(
                                        "✅",
                                        key=f"yes_{delete_key}",
                                        help="Confirm delete",
                                    ):
                                        success, message = delete_document_safely(
                                            collection, doc_key, doc_info
                                        )
                                        if success:
                                            st.success(message)
                                            # Reset confirmation state
                                            st.session_state[confirm_key] = False
                                            st.rerun()
                                        else:
                                            st.error(message)
                                with col_no:
                                    if st.button(
                                        "❌", key=f"no_{delete_key}", help="Cancel"
                                    ):
                                        st.session_state[confirm_key] = False
                                        st.rerun()
                            else:
                                if st.button(
                                    "🗑️", key=delete_key, help="Delete document"
                                ):
                                    st.session_state[confirm_key] = True
                                    st.rerun()

                        # Show details if requested
                        if st.session_state.get(
                            f"show_details_{selected_collection}_{idx}", False
                        ):
                            st.write("**Document Details:**")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write("**Metadata:**")
                                if doc_info["metadata"]:
                                    for key, value in doc_info["metadata"].items():
                                        st.write(f"- **{key}:** {value}")
                                else:
                                    st.write("No metadata available")
                            with col_b:
                                st.write("**Sample Chunk IDs:**")
                                for chunk_id in doc_info["chunks"][:5]:  # Show first 5
                                    st.code(chunk_id, language=None)
                                if len(doc_info["chunks"]) > 5:
                                    st.write(
                                        f"... and {len(doc_info['chunks']) - 5} more chunks"
                                    )

                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("---")

                        # Show confirmation message outside the button area
                        if st.session_state.get(confirm_key, False):
                            st.warning(
                                f"⚠️ Are you sure you want to delete '{doc_info['filename']}'? This will remove {doc_info['chunk_count']} chunks."
                            )
            else:
                st.info("No documents found in this collection.")

        with tab2:
            st.write("**Add new documents to the collection:**")
            new_uploaded_files = st.file_uploader(
                "Select files to upload", accept_multiple_files=True, key="update_files"
            )

            if st.button("Update Collection", key="update_collection_btn"):
                if new_uploaded_files:
                    add_files(
                        uploaded_files=new_uploaded_files,
                        collection_name=selected_collection,
                    )
                    st.rerun()  # Refresh to show new documents
                else:
                    st.warning("Please select files to upload.")
else:
    st.info("No collections available to update.")
