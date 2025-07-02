import pytest
import os
import tempfile
from unittest.mock import Mock, patch
from chroma_data_ingestor import (
    ingest_data,
    create_batch_embeddings,
    node_pipeline,
    add_nodes_to_chroma,
)


class MockNode:
    """Mock node class for testing."""

    def __init__(self, id, content, metadata=None, embedding=None):
        self.id = id
        self.content = content
        self.metadata = metadata or {}
        self.embedding = embedding


@pytest.fixture
def mock_nodes():
    """Fixture providing mock nodes for testing."""
    return [
        MockNode(id="1", content="Test content 1", metadata={"source": "file1.txt"}),
        MockNode(id="2", content="Test content 2", metadata={"source": "file2.txt"}),
        MockNode(id="3", content="Test content 3", metadata={"source": "file3.txt"}),
    ]


@pytest.fixture
def mock_embeddings():
    """Fixture providing mock embeddings."""
    return [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]


@pytest.fixture
def temp_directory():
    """Fixture providing a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test files
        test_files = ["test1.txt", "test2.txt", "test3.txt"]
        for filename in test_files:
            file_path = os.path.join(tmp_dir, filename)
            with open(file_path, "w") as f:
                f.write(f"Test content for {filename}")
        yield tmp_dir


class TestIngestData:
    """Test cases for the ingest_data function."""

    @patch("chroma_data_ingestor.get_embedding")
    @patch("chroma_data_ingestor.node_pipeline")
    @patch("chroma_data_ingestor.client")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_ingest_data_success(
        self,
        mock_listdir,
        mock_exists,
        mock_client,
        mock_node_pipeline,
        mock_get_embedding,
        mock_nodes,
    ):
        """Test successful data ingestion."""
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["test1.txt", "test2.txt"]
        mock_node_pipeline.side_effect = [[mock_nodes[0]], [mock_nodes[1]]]
        mock_get_embedding.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection

        # Execute
        ingest_data(provider="test_provider", directory="/test/dir")

        # Verify
        mock_exists.assert_called_once_with("/test/dir")
        mock_listdir.assert_called_once_with("/test/dir")
        assert mock_node_pipeline.call_count == 2
        mock_get_embedding.assert_called()
        mock_client.get_or_create_collection.assert_called_once_with("rsqkit")

    @patch("os.path.exists")
    def test_ingest_data_directory_not_exists(self, mock_exists):
        """Test FileNotFoundError when directory doesn't exist."""
        mock_exists.return_value = False

        with pytest.raises(
            FileNotFoundError, match="The directory /nonexistent does not exist"
        ):
            ingest_data(provider="test_provider", directory="/nonexistent")

    @patch("chroma_data_ingestor.get_embedding")
    @patch("chroma_data_ingestor.node_pipeline")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_ingest_data_io_error(
        self, mock_listdir, mock_exists, mock_node_pipeline, mock_get_embedding
    ):
        """Test IOError handling during file processing."""
        mock_exists.return_value = True
        mock_listdir.side_effect = IOError("Permission denied")

        with pytest.raises(IOError, match="Error reading files"):
            ingest_data(provider="test_provider", directory="/test/dir")

    @patch("chroma_data_ingestor.get_embedding")
    @patch("chroma_data_ingestor.node_pipeline")
    @patch("chroma_data_ingestor.client")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_ingest_data_custom_parameters(
        self,
        mock_listdir,
        mock_exists,
        mock_client,
        mock_node_pipeline,
        mock_get_embedding,
        mock_nodes,
    ):
        """Test ingest_data with custom parameters."""
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["test.txt"]
        mock_node_pipeline.return_value = [mock_nodes[0]]
        mock_get_embedding.return_value = [[0.1, 0.2]]
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection

        # Execute with custom parameters
        ingest_data(
            provider="custom_provider",
            directory="/custom/dir",
            chunk_size=500,
            chunk_overlap=50,
            collection_name="custom_collection",
        )

        # Verify custom parameters were used
        mock_node_pipeline.assert_called_with(
            file_path="/custom/dir/test.txt", chunk_size=500, chunk_overlap=50
        )
        mock_client.get_or_create_collection.assert_called_once_with(
            "custom_collection"
        )


class TestCreateBatchEmbeddings:
    """Test cases for the create_batch_embeddings function."""

    @patch("chroma_data_ingestor.get_embedding")
    def test_create_batch_embeddings_success(self, mock_get_embedding, mock_nodes):
        """Test successful batch embedding creation."""
        mock_get_embedding.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        create_batch_embeddings(mock_nodes, provider="test_provider", batch_size=3)

        # Verify embeddings were assigned (use approximate equality for floats)
        import numpy as np

        np.testing.assert_allclose(mock_nodes[0].embedding, [0.1, 0.2], rtol=1e-10)
        np.testing.assert_allclose(mock_nodes[1].embedding, [0.3, 0.4], rtol=1e-10)
        np.testing.assert_allclose(mock_nodes[2].embedding, [0.5, 0.6], rtol=1e-10)

        # Verify API call
        mock_get_embedding.assert_called_once_with(
            input=["Test content 1", "Test content 2", "Test content 3"],
            provider="test_provider",
        )

    @patch("chroma_data_ingestor.get_embedding")
    def test_create_batch_embeddings_with_batching(
        self, mock_get_embedding, mock_nodes
    ):
        """Test batch embedding with smaller batch size."""
        mock_get_embedding.side_effect = [
            [[0.1, 0.2], [0.3, 0.4]],  # First batch
            [[0.5, 0.6]],  # Second batch
        ]

        create_batch_embeddings(mock_nodes, provider="test_provider", batch_size=2)

        # Verify embeddings were assigned correctly (use approximate equality)
        import numpy as np

        np.testing.assert_allclose(mock_nodes[0].embedding, [0.1, 0.2], rtol=1e-10)
        np.testing.assert_allclose(mock_nodes[1].embedding, [0.3, 0.4], rtol=1e-10)
        np.testing.assert_allclose(mock_nodes[2].embedding, [0.5, 0.6], rtol=1e-10)

        # Verify API was called twice
        assert mock_get_embedding.call_count == 2

    @patch("chroma_data_ingestor.get_embedding")
    def test_create_batch_embeddings_empty_nodes(self, mock_get_embedding):
        """Test batch embedding with empty nodes list."""
        create_batch_embeddings([], provider="test_provider")

        # Verify no API calls were made
        mock_get_embedding.assert_not_called()


class TestNodePipeline:
    """Test cases for the node_pipeline function."""

    @patch("chroma_data_ingestor.FileReader")
    @patch("chroma_data_ingestor.TextChunker")
    def test_node_pipeline_success(self, mock_chunker_class, mock_reader_class):
        """Test successful node pipeline processing."""
        # Setup mocks
        mock_reader = Mock()
        mock_chunker = Mock()
        mock_reader_class.return_value = mock_reader
        mock_chunker_class.return_value = mock_chunker

        mock_pages = ["page1", "page2"]
        mock_nodes = [MockNode("1", "content1"), MockNode("2", "content2")]

        mock_reader.get_pages.return_value = mock_pages
        mock_chunker.create_chunks_with_source_page.return_value = mock_nodes

        # Execute
        result = node_pipeline("/test/file.txt", chunk_size=500, chunk_overlap=50)

        # Verify
        mock_reader_class.assert_called_once()
        mock_chunker_class.assert_called_once_with(chunk_size=500, chunk_overlap=50)
        mock_reader.get_pages.assert_called_once_with(file="/test/file.txt")
        mock_chunker.create_chunks_with_source_page.assert_called_once_with(
            file_path="/test/file.txt", pages=mock_pages
        )
        assert result == mock_nodes

    @patch("chroma_data_ingestor.FileReader")
    def test_node_pipeline_file_not_found(self, mock_reader_class):
        """Test FileNotFoundError handling in node pipeline."""
        mock_reader = Mock()
        mock_reader_class.return_value = mock_reader
        mock_reader.get_pages.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            node_pipeline("/nonexistent/file.txt")

    @patch("chroma_data_ingestor.FileReader")
    def test_node_pipeline_generic_exception(self, mock_reader_class):
        """Test generic exception handling in node pipeline."""
        mock_reader = Mock()
        mock_reader_class.return_value = mock_reader
        mock_reader.get_pages.side_effect = Exception("Generic error")

        with pytest.raises(Exception, match="Generic error"):
            node_pipeline("/test/file.txt")


class TestAddNodesToChroma:
    """Test cases for the add_nodes_to_chroma function."""

    def test_add_nodes_to_chroma_success(self, mock_nodes):
        """Test successful addition of nodes to Chroma collection."""
        # Add embeddings to mock nodes
        for i, node in enumerate(mock_nodes):
            node.embedding = [0.1 * (i + 1), 0.2 * (i + 1)]

        mock_collection = Mock()

        add_nodes_to_chroma(mock_collection, mock_nodes, batch_size=2)

        # Verify collection.add was called correctly
        assert mock_collection.add.call_count == 2  # 3 nodes with batch_size=2

        # Check first batch call
        first_call = mock_collection.add.call_args_list[0]
        assert first_call[1]["ids"] == ["1", "2"]
        assert first_call[1]["documents"] == ["Test content 1", "Test content 2"]
        # Use approximate equality for floating point comparisons
        import numpy as np

        np.testing.assert_allclose(
            first_call[1]["embeddings"], [[0.1, 0.2], [0.2, 0.4]], rtol=1e-10
        )

        # Check second batch call
        second_call = mock_collection.add.call_args_list[1]
        assert second_call[1]["ids"] == ["3"]
        assert second_call[1]["documents"] == ["Test content 3"]
        # Use approximate equality for floating point comparisons
        import numpy as np

        np.testing.assert_allclose(
            second_call[1]["embeddings"], [[0.3, 0.6]], rtol=1e-10
        )

    def test_add_nodes_to_chroma_type_error(self):
        """Test TypeError when nodes is not a list."""
        mock_collection = Mock()

        with pytest.raises(TypeError, match="nodes must be a list"):
            add_nodes_to_chroma(mock_collection, "not_a_list")

    def test_add_nodes_to_chroma_batch_size_error(self, mock_nodes):
        """Test ValueError for invalid batch_size."""
        mock_collection = Mock()

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            add_nodes_to_chroma(mock_collection, mock_nodes, batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            add_nodes_to_chroma(mock_collection, mock_nodes, batch_size=-1)

    def test_add_nodes_to_chroma_empty_nodes(self):
        """Test adding empty nodes list."""
        mock_collection = Mock()

        add_nodes_to_chroma(mock_collection, [], batch_size=10)

        # Verify no calls were made
        mock_collection.add.assert_not_called()


class TestIntegration:
    """Integration tests combining multiple functions."""

    @patch("chroma_data_ingestor.get_embedding")
    @patch("chroma_data_ingestor.FileReader")
    @patch("chroma_data_ingestor.TextChunker")
    @patch("chroma_data_ingestor.client")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_full_integration_workflow(
        self,
        mock_listdir,
        mock_exists,
        mock_client,
        mock_chunker_class,
        mock_reader_class,
        mock_get_embedding,
    ):
        """Test the complete integration workflow."""
        # Setup comprehensive mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["test.txt"]

        # Mock file processing
        mock_reader = Mock()
        mock_chunker = Mock()
        mock_reader_class.return_value = mock_reader
        mock_chunker_class.return_value = mock_chunker

        mock_pages = ["Test page content"]
        mock_nodes = [MockNode("1", "Test content", {"source": "test.txt"})]

        mock_reader.get_pages.return_value = mock_pages
        mock_chunker.create_chunks_with_source_page.return_value = mock_nodes

        # Mock embedding generation
        mock_get_embedding.return_value = [[0.1, 0.2, 0.3]]

        # Mock ChromaDB
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection

        # Execute full workflow
        ingest_data(provider="test_provider", directory="/test/dir")

        # Verify the complete workflow
        assert mock_exists.called
        assert mock_listdir.called
        assert mock_reader.get_pages.called
        assert mock_chunker.create_chunks_with_source_page.called
        assert mock_get_embedding.called
        assert mock_collection.add.called

        # Verify the node has the embedding (use approximate equality)
        import numpy as np

        np.testing.assert_allclose(mock_nodes[0].embedding, [0.1, 0.2, 0.3], rtol=1e-10)

    @patch("chroma_data_ingestor.get_embedding")
    def test_embedding_error_handling(self, mock_get_embedding, mock_nodes):
        """Test error handling when embedding API fails."""
        mock_get_embedding.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            create_batch_embeddings(mock_nodes, provider="test_provider")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
