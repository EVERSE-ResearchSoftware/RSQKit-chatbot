import chromadb
import os
import argparse
from core_utils.document_reader.reader import FileReader
from core_utils.text_chunking.text_chunker import TextChunker
from llm_provider_tools import get_embedding
from settings import DOCUMENTS_DIR, PROVIDER_ID_TO_NAME
from dotenv import load_dotenv

load_dotenv()

# Initialize persistent ChromaDB client
client = chromadb.PersistentClient(path="./chromadb_store")
collection = client.get_or_create_collection("rsqkit")

# Path to your text documents
input_directory = DOCUMENTS_DIR


# Embedding function using api


# Load and store embeddings
def ingest_data(
    provider: str,
    directory: str = input_directory,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    collection_name: str = "rsqkit",
) -> None:
    """
    Ingest data from a directory of documents and store them in a vector store.

    Args:
        provider (str): The provider for generating embeddings.
        directory (str): The directory containing the documents to ingest. Defaults to input_directory.
        chunk_size (int): The size of each chunk. Defaults to 1000.
        chunk_overlap (int): The overlap between chunks. Defaults to 100.
        collection_name (str): The name of the collection in the vector store. Defaults to "documents".

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        IOError: If there is an error reading the files.
    """
    all_nodes = []
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")

    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            nodes_of_file = node_pipeline(
                file_path=file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            all_nodes.extend(nodes_of_file)
    except IOError as e:
        raise IOError(f"Error reading files in {directory}: {e}")

    # compute all embeddings
    create_batch_embeddings(nodes=all_nodes, provider=provider)

    # add to chromadb
    collection_in_db = client.get_or_create_collection(collection_name)
    add_nodes_to_chroma(collection=collection_in_db, nodes=all_nodes, batch_size=32)


def create_batch_embeddings(nodes, provider: str, batch_size=10):
    """
    Create embeddings for a batch of nodes.

    This function processes nodes in batches to generate embeddings using a specified provider.
    Each node's content is used to generate an embedding, which is then assigned back to the node.

    Parameters:
    nodes (list): A list of nodes to process.
    provider (str): The provider to use for generating embeddings.
    batch_size (int, optional): The size of each batch. Defaults to 10.

    Returns:
    None
    """
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i : i + batch_size]
        texts = [node.content for node in batch]

        embeddings = get_embedding(input=texts, provider=provider)
        for node, emb in zip(batch, embeddings):
            node.embedding = emb


def node_pipeline(file_path, chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    Processes a file by reading it, chunking the text, and returning the nodes.

    Args:
        file_path (str): The path to the file to be processed.
        chunk_size (int, optional): The size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 100.

    Returns:
        list: A list of nodes created from the file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        Exception: For any other errors that occur during processing.
    """
    try:
        reader = FileReader()
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        pages = reader.get_pages(file=file_path)
        nodes = chunker.create_chunks_with_source_page(file_path=file_path, pages=pages)
        return nodes
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def add_nodes_to_chroma(collection, nodes, batch_size=10):
    """
    Adds nodes to a Chroma collection in batches.

    Args:
        collection: The Chroma collection to which nodes will be added.
        nodes (list): A list of nodes to be added to the collection.
        batch_size (int, optional): The number of nodes to add per batch. Defaults to 10.

    Raises:
        TypeError: If nodes is not a list.
        ValueError: If batch_size is not a positive integer.
    """
    if not isinstance(nodes, list):
        raise TypeError("nodes must be a list")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    for i in range(0, len(nodes), batch_size):
        batch = nodes[i : i + batch_size]
        ids = [node.id for node in batch]
        contents = [node.content for node in batch]
        embeddings = [node.embedding for node in batch]
        metadata = [node.metadata for node in batch]
        collection.add(
            ids=ids, documents=contents, embeddings=embeddings, metadatas=metadata
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument(
        "--input_directory",
        default=DOCUMENTS_DIR,
        help="Path to the documents directory (default: DOCUMENTS_DIR)",
    )
    parser.add_argument(
        "--provider",
        default=PROVIDER_ID_TO_NAME["ollama"],
        help="Embedding provider (default: OLLAMA)",
    )
    parser.add_argument(
        "--collection", default="documents", help="Name of the ChromaDB collection"
    )
    args = parser.parse_args()

    ingest_data(
        provider=args.provider,
        directory=args.input_directory,
        collection_name=args.collection,
    )
    print("ChromaDB initialized with embeddings.")

    # Usage with default parameters
# python chroma_data_ingestor.py

# # Usage with input params
# python chroma_data_ingestor.py --input_directory /path/to/directory --provider OLLAMA --collection rsqkit
