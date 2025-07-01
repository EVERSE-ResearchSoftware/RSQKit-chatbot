import os
import pytest
from unittest.mock import Mock, patch, mock_open
from chroma_data_ingestor import ingest_data


@patch("chroma_data_ingestor.add_nodes_to_chroma")
@patch("chroma_data_ingestor.client.get_or_create_collection")
@patch("chroma_data_ingestor.create_batch_embeddings")
@patch("chroma_data_ingestor.node_pipeline")
@patch("os.path.exists")
@patch("os.listdir")
def test_ingest_data_success(
    mock_listdir,
    mock_path_exists,
    mock_node_pipeline,
    mock_create_batch_embeddings,
    mock_get_or_create_collection,
    mock_add_nodes_to_chroma,
):
    # Setup mocks
    mock_path_exists.return_value = True
    mock_listdir.return_value = ["file1.txt", "file2.txt"]
    mock_node_pipeline.side_effect = lambda file_path, chunk_size, chunk_overlap: [
        f"node_{file_path}"
    ]
    mock_get_or_create_collection.return_value = Mock()
    mock_create_batch_embeddings.return_value = None
    mock_add_nodes_to_chroma.return_value = None

    # Call the function
    ingest_data(provider="provider_name", directory="test_directory")

    # Assertions
    mock_listdir.assert_called_once_with("test_directory")
    mock_node_pipeline.assert_any_call(
        file_path="test_directory/file1.txt", chunk_size=1000, chunk_overlap=100
    )
    mock_node_pipeline.assert_any_call(
        file_path="test_directory/file2.txt", chunk_size=1000, chunk_overlap=100
    )
    mock_create_batch_embeddings.assert_called_once_with(
        nodes=["node_test_directory/file1.txt", "node_test_directory/file2.txt"],
        provider="provider_name",
    )
    mock_get_or_create_collection.assert_called_once_with("rsqkit")
    mock_add_nodes_to_chroma.assert_called_once_with(
        collection=mock_get_or_create_collection.return_value,
        nodes=["node_test_directory/file1.txt", "node_test_directory/file2.txt"],
        batch_size=32,
    )


@patch("os.path.exists")
def test_ingest_data_directory_not_found(mock_path_exists):
    # Setup mock
    mock_path_exists.return_value = False

    # Call the function and assert it raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        ingest_data(provider="provider_name", directory="nonexistent_directory")
