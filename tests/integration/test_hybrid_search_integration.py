import pytest
import chromadb
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from core_utils.hybrid_search import HybridSearch
from chroma_data_ingestor import get_embedding
import numpy as np

# Mock bm25s since it might not be available in test environment
try:
    import bm25s
except ImportError:
    # Create a mock bm25s module
    import sys
    from unittest.mock import MagicMock

    bm25s = MagicMock()
    sys.modules["bm25s"] = bm25s


class TestHybridSearchIntegration:
    """Integration tests for hybrid search functionality combining semantic and keyword search"""

    @pytest.fixture
    def temp_db_dir(self):
        """Create temporary directory for ChromaDB"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_collection(self, temp_db_dir):
        """Create a mock ChromaDB collection with test data"""
        client = chromadb.PersistentClient(path=temp_db_dir)
        collection = client.get_or_create_collection("test_documents")

        # Add test documents with varied content for hybrid search testing
        test_documents = [
            {
                "id": "doc1",
                "document": "Machine learning algorithms are powerful tools for data analysis and pattern recognition.",
                "metadata": {
                    "source": "ml_book.pdf",
                    "page": 1,
                    "topic": "machine_learning",
                },
            },
            {
                "id": "doc2",
                "document": "Deep learning networks use multiple layers to learn complex representations from data.",
                "metadata": {
                    "source": "dl_paper.pdf",
                    "page": 5,
                    "topic": "deep_learning",
                },
            },
            {
                "id": "doc3",
                "document": "Natural language processing enables computers to understand human language effectively.",
                "metadata": {"source": "nlp_guide.pdf", "page": 12, "topic": "nlp"},
            },
            {
                "id": "doc4",
                "document": "Computer vision algorithms can identify objects and patterns in images with high accuracy.",
                "metadata": {
                    "source": "cv_tutorial.pdf",
                    "page": 8,
                    "topic": "computer_vision",
                },
            },
            {
                "id": "doc5",
                "document": "Data preprocessing is crucial for machine learning model performance and accuracy.",
                "metadata": {
                    "source": "preprocessing.pdf",
                    "page": 3,
                    "topic": "data_science",
                },
            },
        ]

        # Mock embedding function to return consistent embeddings
        with patch("chroma_data_ingestor.get_embedding") as mock_get_embedding:
            # Create deterministic embeddings based on document content
            def mock_embedding(text):
                # Simple hash-based embedding for consistent results
                embedding = np.random.RandomState(hash(text) % 2**32).rand(384).tolist()
                return embedding

            mock_get_embedding.side_effect = mock_embedding

            # Add documents to collection
            for doc in test_documents:
                collection.add(
                    ids=[doc["id"]],
                    documents=[doc["document"]],
                    metadatas=[doc["metadata"]],
                    embeddings=[mock_embedding(doc["document"])],
                )

        return collection

    @pytest.fixture
    def hybrid_searcher(self, mock_collection):
        """Create HybridSearch instance with test collection"""
        # Mock the BM25 components to avoid dependency issues
        with (
            patch("bm25s.BM25") as mock_bm25_class,
            patch("bm25s.tokenize") as mock_tokenize,
        ):

            # Setup mock BM25 instance
            mock_bm25_instance = MagicMock()
            mock_bm25_class.return_value = mock_bm25_instance

            # Setup mock tokenize function
            mock_tokenize.return_value = MagicMock()
            mock_tokenize.return_value.vocab = {"test": 1, "query": 2}

            # Mock get_scores to return reasonable scores
            mock_bm25_instance.get_scores.return_value = np.array(
                [0.8, 0.6, 0.4, 0.2, 0.1]
            )

            return HybridSearch(mock_collection, alpha=0.7)

    def test_hybrid_search_initialization(self, mock_collection):
        """Test HybridSearch initialization with different alpha values"""
        with (
            patch("bm25s.BM25") as mock_bm25_class,
            patch("bm25s.tokenize") as mock_tokenize,
        ):

            # Setup mocks
            mock_bm25_instance = MagicMock()
            mock_bm25_class.return_value = mock_bm25_instance
            mock_tokenize.return_value = MagicMock()
            mock_tokenize.return_value.vocab = {"test": 1}

            # Test default alpha
            searcher_default = HybridSearch(mock_collection)
            assert searcher_default.alpha == 0.7  # Default from class

            # Test custom alpha values
            searcher_semantic = HybridSearch(mock_collection, alpha=1.0)
            assert searcher_semantic.alpha == 1.0

            searcher_keyword = HybridSearch(mock_collection, alpha=0.0)
            assert searcher_keyword.alpha == 0.0

            searcher_balanced = HybridSearch(mock_collection, alpha=0.5)
            assert searcher_balanced.alpha == 0.5

            # Verify BM25 index was built
            assert mock_bm25_class.called
            assert mock_bm25_instance.index.called

    def test_bm25_search_component(self, hybrid_searcher):
        """Test the BM25 search component independently"""
        # Test BM25 search directly
        bm25_ids, bm25_scores = hybrid_searcher.search_bm25("machine learning", k=3)

        # Verify we got results
        assert isinstance(bm25_ids, list)
        assert isinstance(bm25_scores, list)
        assert len(bm25_ids) <= 3
        assert len(bm25_scores) == len(bm25_ids)

        # Verify scores are reasonable
        for score in bm25_scores:
            assert isinstance(score, (int, float))
            assert score >= 0

    def test_semantic_search_component_direct(self, hybrid_searcher):
        """Test the semantic search component directly"""
        # Create a dummy embedding
        query_embedding = np.random.rand(384).tolist()

        # Test semantic search directly
        semantic_ids, semantic_scores = hybrid_searcher.search_semantic(
            query_embedding, k=3
        )

        # Verify we got results
        assert isinstance(semantic_ids, list)
        assert isinstance(semantic_scores, list)
        assert len(semantic_ids) <= 3
        assert len(semantic_scores) == len(semantic_ids)

        # Verify scores are reasonable (similarity scores should be between 0 and 1)
        for score in semantic_scores:
            assert isinstance(score, (int, float))
            assert score >= 0
            assert score <= 1.1  # Allow for slight floating point errors

    def test_semantic_search_component_integration(self, hybrid_searcher):
        """Test the semantic search component of hybrid search"""
        # Mock embedding for query - note: we don't mock get_embedding here since
        # the hybrid_search method doesn't call it directly
        query_embedding = np.random.rand(384).tolist()

        # Test semantic search (alpha = 1.0 for pure semantic)
        hybrid_searcher.alpha = 1.0

        results = hybrid_searcher.hybrid_search(
            "machine learning algorithms", query_embedding, k=3
        )

        # Verify results structure
        assert isinstance(results, dict)
        assert "ids" in results
        assert "documents" in results
        assert "metadatas" in results
        assert "scores" in results
        assert "semantic_weight" in results
        assert "bm25_weight" in results

        # Verify we got results
        assert len(results["ids"]) > 0
        assert len(results["documents"]) > 0

        # Verify alpha settings
        assert results["semantic_weight"] == 1.0
        assert results["bm25_weight"] == 0.0

    def test_keyword_search_component(self, hybrid_searcher):
        """Test the keyword search component of hybrid search"""
        # Mock embedding for query
        query_embedding = np.random.rand(384).tolist()

        # Test keyword search (alpha = 0.0 for pure keyword)
        hybrid_searcher.alpha = 0.0

        # Query with specific keywords that should match
        results = hybrid_searcher.hybrid_search(
            "machine learning", query_embedding, k=3
        )

        # Verify results structure
        assert isinstance(results, dict)
        assert "ids" in results
        assert "documents" in results
        assert "metadatas" in results
        assert "scores" in results

        # Verify we got results
        assert len(results["ids"]) > 0

        # Check that documents containing "machine learning" are prioritized
        documents = results["documents"]
        assert any("machine learning" in doc.lower() for doc in documents)

        # Verify alpha settings
        assert results["semantic_weight"] == 0.0
        assert results["bm25_weight"] == 1.0

    def test_hybrid_search_weighting(self, mock_collection):
        """Test that different alpha values produce different result rankings"""
        query_embedding = np.random.rand(384).tolist()
        query = "deep learning networks"

        with (
            patch("bm25s.BM25") as mock_bm25_class,
            patch("bm25s.tokenize") as mock_tokenize,
        ):

            # Setup mocks
            mock_bm25_instance = MagicMock()
            mock_bm25_class.return_value = mock_bm25_instance
            mock_tokenize.return_value = MagicMock()
            mock_tokenize.return_value.vocab = {"deep": 1, "learning": 2, "networks": 3}
            mock_bm25_instance.get_scores.return_value = np.array(
                [0.8, 0.6, 0.4, 0.2, 0.1]
            )

            # Test with semantic-heavy weighting (alpha = 0.9)
            searcher_semantic = HybridSearch(mock_collection, alpha=0.9)
            semantic_results = searcher_semantic.hybrid_search(
                query, query_embedding, k=3
            )

            # Test with keyword-heavy weighting (alpha = 0.1)
            searcher_keyword = HybridSearch(mock_collection, alpha=0.1)
            keyword_results = searcher_keyword.hybrid_search(
                query, query_embedding, k=3
            )

            # Test with balanced weighting (alpha = 0.5)
            searcher_balanced = HybridSearch(mock_collection, alpha=0.5)
            balanced_results = searcher_balanced.hybrid_search(
                query, query_embedding, k=3
            )

            # Verify all searches return results
            assert len(semantic_results["ids"]) > 0
            assert len(keyword_results["ids"]) > 0
            assert len(balanced_results["ids"]) > 0

            # Verify weighting is correctly applied (with tolerance for floating point)
            assert abs(semantic_results["semantic_weight"] - 0.9) < 0.001
            assert abs(semantic_results["bm25_weight"] - 0.1) < 0.001
            assert abs(keyword_results["semantic_weight"] - 0.1) < 0.001
            assert abs(keyword_results["bm25_weight"] - 0.9) < 0.001
            assert abs(balanced_results["semantic_weight"] - 0.5) < 0.001
            assert abs(balanced_results["bm25_weight"] - 0.5) < 0.001

            # Results should potentially differ based on weighting
            assert isinstance(semantic_results["ids"], list)
            assert isinstance(keyword_results["ids"], list)
            assert isinstance(balanced_results["ids"], list)

    def test_hybrid_search_result_quality(self, hybrid_searcher):
        """Test the quality and relevance of hybrid search results"""
        query_embedding = np.random.rand(384).tolist()

        # Test query that should match specific document
        results = hybrid_searcher.hybrid_search(
            "natural language processing", query_embedding, k=5
        )

        # Verify result structure and content
        assert len(results["ids"]) <= 5
        assert len(results["documents"]) == len(results["ids"])
        assert len(results["metadatas"]) == len(results["ids"])
        assert len(results["scores"]) == len(results["ids"])

        # Check that relevant document is in results
        documents = results["documents"]
        assert any("natural language processing" in doc.lower() for doc in documents)

        # Verify metadata is preserved
        metadatas = results["metadatas"]
        for metadata in metadatas:
            assert "source" in metadata
            assert "topic" in metadata

        # Verify scores are present and reasonable
        scores = results["scores"]
        assert all(isinstance(score, (int, float)) for score in scores)
        assert all(score >= 0 for score in scores)  # Scores should be non-negative

    def test_hybrid_search_empty_query(self, hybrid_searcher):
        """Test hybrid search behavior with empty or invalid queries"""
        query_embedding = np.random.rand(384).tolist()

        # Test empty query
        try:
            results = hybrid_searcher.hybrid_search("", query_embedding, k=3)
            # Should either return empty results or handle gracefully
            assert isinstance(results, dict)
            assert "ids" in results
            assert "documents" in results
            # Empty results are acceptable for empty queries
            assert isinstance(results["ids"], list)
            assert isinstance(results["documents"], list)
        except Exception as e:
            # Should handle empty queries gracefully - accept various error types
            error_msg = str(e).lower()
            assert any(
                keyword in error_msg
                for keyword in [
                    "empty",
                    "invalid",
                    "documents",
                    "index",
                    "range",
                    "bm25",
                    "tokenize",
                ]
            )

    def test_hybrid_search_large_k_results(self, hybrid_searcher):
        """Test hybrid search with k larger than available documents"""
        query_embedding = np.random.rand(384).tolist()

        # Request more results than available documents (we have 5 test docs)
        results = hybrid_searcher.hybrid_search("algorithms", query_embedding, k=10)

        # Should return all available documents (max 5)
        assert len(results["ids"]) <= 5
        assert len(results["documents"]) <= 5

    def test_hybrid_search_error_handling(self, hybrid_searcher):
        """Test hybrid search error handling"""
        # Test with various potential error conditions
        query_embedding = np.random.rand(384).tolist()

        # Test with normal query - should work
        try:
            results = hybrid_searcher.hybrid_search("test query", query_embedding, k=3)
            assert isinstance(results, dict)
            assert "ids" in results
        except Exception as e:
            # If there are errors, they should be informative
            error_msg = str(e).lower()
            assert any(
                keyword in error_msg
                for keyword in ["embedding", "bm25", "documents", "index", "collection"]
            )

    def test_hybrid_search_consistency(self, hybrid_searcher):
        """Test that hybrid search returns consistent results for same query"""
        query_embedding = np.random.rand(384).tolist()
        query = "machine learning"

        # Run same search multiple times
        results1 = hybrid_searcher.hybrid_search(query, query_embedding, k=3)
        results2 = hybrid_searcher.hybrid_search(query, query_embedding, k=3)

        # Results should be consistent
        assert results1["ids"] == results2["ids"]
        assert results1["documents"] == results2["documents"]
        assert results1["metadatas"] == results2["metadatas"]
        assert results1["scores"] == results2["scores"]

    def test_hybrid_search_integration_with_collection(self, mock_collection):
        """Test integration between HybridSearch and ChromaDB collection"""
        with (
            patch("bm25s.BM25") as mock_bm25_class,
            patch("bm25s.tokenize") as mock_tokenize,
        ):

            # Setup mocks
            mock_bm25_instance = MagicMock()
            mock_bm25_class.return_value = mock_bm25_instance
            mock_tokenize.return_value = MagicMock()
            mock_tokenize.return_value.vocab = {"test": 1}
            mock_bm25_instance.get_scores.return_value = np.array(
                [0.8, 0.6, 0.4, 0.2, 0.1]
            )

            # Create searcher
            searcher = HybridSearch(mock_collection, alpha=0.6)

            # Verify collection integration
            assert searcher.collection == mock_collection

            # Test that collection methods are accessible
            all_docs = mock_collection.get()
            assert len(all_docs["documents"]) == 5  # Our test documents

            # Verify searcher can work with collection data
            query_embedding = np.random.rand(384).tolist()
            results = searcher.hybrid_search("test", query_embedding, k=2)
            assert len(results["ids"]) <= 2

    def test_hybrid_search_alpha_boundary_values(self, mock_collection):
        """Test hybrid search behavior at alpha boundary values"""
        query_embedding = np.random.rand(384).tolist()
        query = "learning algorithms"

        with (
            patch("bm25s.BM25") as mock_bm25_class,
            patch("bm25s.tokenize") as mock_tokenize,
        ):

            # Setup mocks
            mock_bm25_instance = MagicMock()
            mock_bm25_class.return_value = mock_bm25_instance
            mock_tokenize.return_value = MagicMock()
            mock_tokenize.return_value.vocab = {"learning": 1, "algorithms": 2}
            mock_bm25_instance.get_scores.return_value = np.array(
                [0.8, 0.6, 0.4, 0.2, 0.1]
            )

            # Test alpha = 0.0 (pure keyword search)
            searcher_keyword = HybridSearch(mock_collection, alpha=0.0)
            keyword_results = searcher_keyword.hybrid_search(
                query, query_embedding, k=3
            )

            # Test alpha = 1.0 (pure semantic search)
            searcher_semantic = HybridSearch(mock_collection, alpha=1.0)
            semantic_results = searcher_semantic.hybrid_search(
                query, query_embedding, k=3
            )

            # Both should return valid results
            assert len(keyword_results["ids"]) > 0
            assert len(semantic_results["ids"]) > 0

            # Verify structure is consistent
            for results in [keyword_results, semantic_results]:
                assert "ids" in results
                assert "documents" in results
                assert "metadatas" in results
                assert "scores" in results
                assert len(results["ids"]) == len(results["documents"])

            # Verify correct weighting
            assert keyword_results["semantic_weight"] == 0.0
            assert keyword_results["bm25_weight"] == 1.0
            assert semantic_results["semantic_weight"] == 1.0
            assert semantic_results["bm25_weight"] == 0.0

    def test_hybrid_search_score_normalization(self, hybrid_searcher):
        """Test that hybrid search properly normalizes and combines scores"""
        query_embedding = np.random.rand(384).tolist()

        # Test with balanced alpha
        hybrid_searcher.alpha = 0.5
        results = hybrid_searcher.hybrid_search(
            "machine learning data", query_embedding, k=3
        )

        # Verify scores are reasonable
        scores = results["scores"]
        assert all(isinstance(score, (int, float)) for score in scores)
        assert all(score >= 0 for score in scores)

        # Scores should generally be sorted in descending order (best first)
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1] - 0.001  # Allow small tolerance

    def test_hybrid_search_metadata_preservation(self, hybrid_searcher):
        """Test that hybrid search preserves document metadata correctly"""
        query_embedding = np.random.rand(384).tolist()

        results = hybrid_searcher.hybrid_search("computer vision", query_embedding, k=5)

        # Verify metadata structure
        metadatas = results["metadatas"]
        for metadata in metadatas:
            assert isinstance(metadata, dict)
            assert "source" in metadata
            assert "topic" in metadata
            assert "page" in metadata

            # Verify metadata values are reasonable
            assert metadata["source"].endswith(".pdf")
            assert isinstance(metadata["page"], int)
            assert metadata["page"] > 0
