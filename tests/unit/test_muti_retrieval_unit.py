"""
Unit tests for multi-retrieval module components.
Tests individual classes and methods in isolation using mocks.
"""

import pytest
from unittest.mock import Mock, patch


# Import the classes we're testing
from core_utils.retrieval_utils import (
    SubQuery,
    RetrievalResult,
    SubQueryAnswer,
    MultiRetrievalEngine,
    rerank_hybrid_results,
)


def create_mock_streamlit():
    """Create a properly configured mock streamlit module"""
    mock_st = Mock()

    # Mock status component with proper context manager
    mock_status = Mock()
    mock_status.update = Mock()
    mock_st.status.return_value = mock_status
    mock_st.status.return_value.__enter__ = Mock(return_value=mock_status)
    mock_st.status.return_value.__exit__ = Mock(return_value=None)

    # Mock other streamlit components
    mock_st.write = Mock()
    mock_st.markdown = Mock()
    mock_st.subheader = Mock()
    mock_st.empty.return_value.markdown = Mock()
    mock_st.expander = Mock()
    mock_st.spinner = Mock()

    # Mock context managers for expander and spinner
    mock_st.expander.return_value.__enter__ = Mock()
    mock_st.expander.return_value.__exit__ = Mock()
    mock_st.spinner.return_value.__enter__ = Mock()
    mock_st.spinner.return_value.__exit__ = Mock()

    return mock_st


class TestDataClasses:
    """Test the dataclass structures"""

    def test_subquery_creation(self):
        """Test SubQuery dataclass initialization and default values"""
        # Create a basic SubQuery instance
        subquery = SubQuery(
            id="test_1",
            question="What is AI?",
            query_type="factual",
            priority=1,
            parent_query="Tell me about AI",
        )

        # Assert all fields are set correctly
        assert subquery.id == "test_1"
        assert subquery.question == "What is AI?"
        assert subquery.query_type == "factual"
        assert subquery.priority == 1
        assert subquery.parent_query == "Tell me about AI"
        # Test default values
        assert subquery.dependencies is None
        assert subquery.search_strategy == "hybrid"
        assert subquery.is_independent is True

    def test_retrieval_result_creation(self):
        """Test RetrievalResult dataclass with sample data"""
        # Create sample retrieval result
        result = RetrievalResult(
            subquery_id="test_1",
            question="What is AI?",
            documents=["AI is artificial intelligence", "Machine learning is a subset"],
            metadatas=[{"source": "doc1"}, {"source": "doc2"}],
            scores=[0.9, 0.8],
            strategy_used="hybrid",
        )

        # Verify all fields are properly assigned
        assert result.subquery_id == "test_1"
        assert len(result.documents) == 2
        assert len(result.metadatas) == 2
        assert len(result.scores) == 2
        assert result.strategy_used == "hybrid"

    def test_subquery_answer_creation(self):
        """Test SubQueryAnswer dataclass structure"""
        # Create a sample answer object
        answer = SubQueryAnswer(
            subquery_id="test_1",
            question="What is AI?",
            answer="AI stands for Artificial Intelligence",
            has_sufficient_context=True,
            sources_used=["source1", "source2"],
            metadatas_used=[{"doc": "1"}, {"doc": "2"}],
        )

        # Verify answer object properties
        assert answer.subquery_id == "test_1"
        assert answer.has_sufficient_context is True
        assert len(answer.sources_used) == 2
        assert len(answer.metadatas_used) == 2


class TestMultiRetrievalEngine:
    """Test the main MultiRetrievalEngine class methods"""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock MultiRetrievalEngine for testing"""
        # Mock the dependencies
        mock_chat_function = Mock()
        mock_hybrid_searcher = Mock()
        mock_hybrid_searcher.collection = Mock()

        # Create engine instance with mocked dependencies
        engine = MultiRetrievalEngine(
            chat_function=mock_chat_function,
            model_name="test-model",
            hybrid_searcher=mock_hybrid_searcher,
            selected_provider="test-provider",
        )
        return engine

    def test_engine_initialization(self, mock_engine):
        """Test that engine initializes with correct properties"""
        # Verify engine properties are set correctly
        assert mock_engine.model_name == "test-model"
        assert mock_engine.selected_provider == "test-provider"
        assert mock_engine.chat_function is not None
        assert mock_engine.hybrid_searcher is not None

    @patch("core_utils.retrieval_utils.st")  # Mock streamlit module properly
    def test_detect_multiple_questions_single_query(self, mock_st, mock_engine):
        """Test detection of single question queries"""
        # Setup mock streamlit status component
        mock_status = Mock()
        mock_status.update = Mock()
        mock_st.status.return_value = mock_status
        mock_st.status.return_value.__enter__ = Mock(return_value=mock_status)
        mock_st.status.return_value.__exit__ = Mock(return_value=None)
        mock_st.empty.return_value.markdown = Mock()

        # Mock the chat function to return a single query response
        mock_engine.chat_function.return_value = iter(
            [
                '{"is_multi_query": false, "question_count": 1, "complexity_score": 2, ',
                '"detected_questions": ["What is AI?"], "query_type": "single", ',
                '"requires_decomposition": false, "reasoning": "Single question detected"}',
            ]
        )

        # Test single question detection
        result = mock_engine.detect_multiple_questions("What is AI?")

        # Verify detection results
        assert result["is_multi_query"] is False
        assert result["question_count"] == 1
        assert result["query_type"] == "single"
        assert result["requires_decomposition"] is False

    @patch("core_utils.retrieval_utils.st")  # Mock streamlit module properly
    def test_detect_multiple_questions_complex_query(self, mock_st, mock_engine):
        """Test detection of complex multi-part queries"""
        # Setup mock streamlit status component
        mock_status = Mock()
        mock_status.update = Mock()
        mock_st.status.return_value = mock_status
        mock_st.status.return_value.__enter__ = Mock(return_value=mock_status)
        mock_st.status.return_value.__exit__ = Mock(return_value=None)
        mock_st.empty.return_value.markdown = Mock()

        # Mock response for complex query
        mock_engine.chat_function.return_value = iter(
            [
                '{"is_multi_query": true, "question_count": 2, "complexity_score": 6, ',
                '"detected_questions": ["What is AI?", "How does ML work?"], ',
                '"query_type": "multi_question", "requires_decomposition": true, ',
                '"reasoning": "Multiple questions detected with and conjunction"}',
            ]
        )

        # Test complex query detection
        result = mock_engine.detect_multiple_questions(
            "What is AI and how does ML work?"
        )

        # Verify multi-query detection
        assert result["is_multi_query"] is True
        assert result["question_count"] == 2
        assert result["requires_decomposition"] is True
        assert len(result["detected_questions"]) == 2

    @patch(
        "core_utils.retrieval_utils.st", None
    )  # Mock streamlit as None to avoid st.status() calls
    def test_detect_multiple_questions_fallback(self, mock_engine):
        """Test fallback detection when JSON parsing fails"""
        # Mock chat function to return invalid JSON
        mock_engine.chat_function.return_value = iter(["Invalid JSON response"])

        # Test fallback heuristic detection
        result = mock_engine.detect_multiple_questions(
            "What is AI? And how does it work?"
        )

        # Verify fallback detection works
        assert "is_multi_query" in result
        assert "reasoning" in result
        assert (
            result["reasoning"]
            == "Heuristic detection"
        )

    @patch("core_utils.retrieval_utils.st", create=True)
    def test_decompose_query_simple(self, mock_st, mock_engine):
        """Test query decomposition for simple queries"""
        # Mock detection info indicating no decomposition needed
        detection_info = {"requires_decomposition": False}

        # Test simple query decomposition
        result = mock_engine.decompose_query("What is AI?", detection_info)

        # Verify single subquery is created
        assert len(result) == 1
        assert result[0].id == "subq_1"
        assert result[0].question == "What is AI?"
        assert result[0].is_independent is True

    @patch("core_utils.retrieval_utils.st")
    def test_decompose_query_complex(self, mock_st, mock_engine):
        """Test query decomposition for complex queries"""
        # Setup mock streamlit status component
        mock_status = Mock()
        mock_status.update = Mock()
        mock_st.status.return_value = mock_status
        mock_st.status.return_value.__enter__ = Mock(return_value=mock_status)
        mock_st.status.return_value.__exit__ = Mock(return_value=None)
        mock_st.empty.return_value.markdown = Mock()

        # Mock chat function response for decomposition
        mock_engine.chat_function.return_value = iter(
            [
                '{"subqueries": [',
                '{"id": "subq_1", "question": "What is AI?", "query_type": "definitional", ',
                '"priority": 1, "dependencies": [], "search_strategy": "hybrid", "is_independent": true},',
                '{"id": "subq_2", "question": "How does ML work?", "query_type": "analytical", ',
                '"priority": 2, "dependencies": [], "search_strategy": "semantic", "is_independent": true}],',
                '"execution_order": ["subq_1", "subq_2"], "combination_strategy": "merge"}',
            ]
        )

        # Mock detection info requiring decomposition
        detection_info = {"requires_decomposition": True}

        # Test complex query decomposition
        result = mock_engine.decompose_query(
            "What is AI and how does ML work?", detection_info
        )

        # Verify multiple subqueries are created
        assert len(result) == 2
        assert result[0].id == "subq_1"
        assert result[1].id == "subq_2"
        assert all(sq.is_independent for sq in result)

    @patch("core_utils.retrieval_utils.st", None)
    def test_decompose_query_fallback(self, mock_engine):
        """Test decomposition fallback when JSON parsing fails"""
        # Mock chat function to return invalid JSON
        mock_engine.chat_function.return_value = iter(["Invalid JSON"])

        # Mock detection info requiring decomposition
        detection_info = {"requires_decomposition": True}

        # Test fallback decomposition
        result = mock_engine.decompose_query(
            "What is AI? How does it work?", detection_info
        )

        # Verify fallback creates subqueries based on question marks
        assert len(result) >= 1
        assert all(isinstance(sq, SubQuery) for sq in result)

    @patch("core_utils.retrieval_utils.get_embedding")
    def test_hybrid_retrieval(self, mock_get_embedding, mock_engine):
        """Test hybrid retrieval strategy execution"""
        # Mock embedding function
        mock_get_embedding.return_value = [0.1, 0.2, 0.3]

        # Mock hybrid searcher results
        mock_engine.hybrid_searcher.hybrid_search.return_value = {
            "documents": ["AI is artificial intelligence"],
            "metadatas": [{"source": "doc1"}],
            "scores": [0.9],
            "ids": ["id1"],
        }

        # Mock reranking results
        with patch("core_utils.retrieval_utils.rerank_hybrid_results") as mock_rerank:
            mock_rerank.return_value = {
                "reranked_docs": ["AI is artificial intelligence"],
                "reranked_metadatas": [{"source": "doc1"}],
                "reranked_scores": [0.95],
            }

            # Create test subquery
            subquery = SubQuery(
                id="test_1",
                question="What is AI?",
                query_type="factual",
                priority=1,
                parent_query="What is AI?",
            )

            # Execute hybrid retrieval
            result = mock_engine._hybrid_retrieval(subquery)

            # Verify retrieval result
            assert result.subquery_id == "test_1"
            assert len(result.documents) == 1
            assert result.strategy_used == "hybrid"
            assert result.documents[0] == "AI is artificial intelligence"

    def test_answer_subquery_traditional_rag_no_docs(self, mock_engine):
        """Test answering subquery when no documents are found"""
        # Create subquery
        subquery = SubQuery(
            id="test_1",
            question="What is AI?",
            query_type="factual",
            priority=1,
            parent_query="What is AI?",
        )

        # Create empty retrieval result
        retrieval_result = RetrievalResult(
            subquery_id="test_1",
            question="What is AI?",
            documents=[],
            metadatas=[],
            scores=[],
            strategy_used="hybrid",
        )

        # Mock build_rag_context function
        mock_build_context = Mock()

        # Test answering with no documents
        answer = mock_engine.answer_subquery_traditional_rag(
            subquery,
            retrieval_result,
            mock_build_context,
            "Template: {query} {context}",
        )

        # Verify no-context response
        assert answer.subquery_id == "test_1"
        assert answer.has_sufficient_context is False
        assert "cannot answer" in answer.answer.lower()
        assert len(answer.sources_used) == 0

    @patch("core_utils.retrieval_utils.st")
    def test_answer_subquery_traditional_rag_with_docs(self, mock_st, mock_engine):
        """Test answering subquery with available documents"""
        # Setup mock streamlit components
        mock_st.write = Mock()
        mock_st.empty.return_value.markdown = Mock()

        # Mock chat function response
        mock_engine.chat_function.return_value = iter(
            [
                "AI stands for Artificial Intelligence, which refers to computer systems ",
                "that can perform tasks typically requiring human intelligence.",
            ]
        )

        # Create subquery
        subquery = SubQuery(
            id="test_1",
            question="What is AI?",
            query_type="factual",
            priority=1,
            parent_query="What is AI?",
        )

        # Create retrieval result with documents
        retrieval_result = RetrievalResult(
            subquery_id="test_1",
            question="What is AI?",
            documents=["AI is artificial intelligence"],
            metadatas=[{"source": "doc1"}],
            scores=[0.9],
            strategy_used="hybrid",
        )

        # Mock build_rag_context function
        mock_build_context = Mock(return_value="Context: AI is artificial intelligence")

        # Test answering with documents
        answer = mock_engine.answer_subquery_traditional_rag(
            subquery,
            retrieval_result,
            mock_build_context,
            "Query: {query}\nContext: {context}",
            augment_chunk=False,  # chunk_augmentation,
            answer_per_chunk=False,
        )

        # Verify successful response
        assert answer.subquery_id == "test_1"
        assert answer.has_sufficient_context is True
        assert "artificial intelligence" in answer.answer.lower()
        assert len(answer.sources_used) == 1

    @patch("core_utils.retrieval_utils.st")
    def test_synthesize_final_answer_single(self, mock_st, mock_engine):
        """Test synthesis when only one subquery answer exists"""
        # Create single subquery answer
        answer = SubQueryAnswer(
            subquery_id="test_1",
            question="What is AI?",
            answer="AI is artificial intelligence",
            has_sufficient_context=True,
            sources_used=["source1"],
            metadatas_used=[{"doc": "1"}],
        )

        # Test synthesis with single answer
        # result = mock_engine.synthesize_final_answer("What is AI?", [answer])
        result = mock_engine.synthesize_final_answer("What is AI?", [answer])

        # Verify single answer is returned directly
        assert result == "AI is artificial intelligence"

    @patch("core_utils.retrieval_utils.st")
    def test_synthesize_final_answer_multiple(self, mock_st, mock_engine):
        """Test synthesis of multiple subquery answers"""
        # Setup mock streamlit components
        mock_st.subheader = Mock()
        mock_st.empty.return_value.markdown = Mock()

        # Mock chat function for synthesis
        mock_engine.chat_function.return_value = iter(
            [
                "Based on the sub-questions answered, AI stands for Artificial Intelligence ",
                "and machine learning is a subset of AI that enables computers to learn.",
            ]
        )

        # Create multiple subquery answers
        answers = [
            SubQueryAnswer(
                subquery_id="test_1",
                question="What is AI?",
                answer="AI is artificial intelligence",
                has_sufficient_context=True,
                sources_used=["source1"],
                metadatas_used=[{"doc": "1"}],
            ),
            SubQueryAnswer(
                subquery_id="test_2",
                question="What is ML?",
                answer="ML is machine learning",
                has_sufficient_context=True,
                sources_used=["source2"],
                metadatas_used=[{"doc": "2"}],
            ),
        ]

        # Test synthesis of multiple answers
        result = mock_engine.synthesize_final_answer("What is AI and ML?", answers)

        # Verify synthesis combines information
        assert "artificial intelligence" in result.lower()
        assert "machine learning" in result.lower()


class TestUtilityFunctions:
    """Test utility functions in the module"""

    @patch("llm_provider_tools.rerank_results")
    def test_rerank_hybrid_results(self, mock_rerank_func):
        """Test the rerank_hybrid_results utility function"""
        # Mock reranking function
        mock_rerank_func.return_value = {
            "reranked_docs": ["doc1", "doc2"],
            "reranked_metadatas": [{"id": 1}, {"id": 2}],
            "reranked_scores": [0.9, 0.8],
        }

        # Create sample hybrid results
        hybrid_results = {
            "documents": ["doc1", "doc2"],
            "ids": ["id1", "id2"],
            "metadatas": [{"id": 1}, {"id": 2}],
            "scores": [0.8, 0.7],
        }

        # Test reranking function
        result = rerank_hybrid_results(
            query="test query",
            hybrid_results=hybrid_results,
            selected_provider="test_provider",
            top_rerank=2,
            reranking_function=mock_rerank_func,
        )

        # Verify reranking was called correctly
        mock_rerank_func.assert_called_once()
        assert "reranked_docs" in result
        assert len(result["reranked_docs"]) == 2

    @patch("core_utils.hybrid_search.HybridSearch")
    @patch("llm_provider_tools.get_default_llm")
    @patch("llms.openai_interface.get_chat_response_stream")
    def test_create_multi_retrieval_engine(self, mock_chat, mock_llm, mock_hybrid):
        """Test factory function for creating MultiRetrievalEngine"""
        from core_utils.retrieval_utils import create_multi_retrieval_engine

        # Mock dependencies
        mock_collection = Mock()
        mock_llm.return_value = "test-model"
        mock_hybrid_instance = Mock()
        mock_hybrid.return_value = mock_hybrid_instance

        # Test engine creation
        engine = create_multi_retrieval_engine(
            collection=mock_collection, selected_provider="test_provider"
        )

        # Verify engine was created successfully
        assert isinstance(engine, MultiRetrievalEngine)
        assert engine.selected_provider == "test_provider"
        mock_hybrid.assert_called_once_with(mock_collection, alpha=0.7)


if __name__ == "__main__":
    # Run tests with pytest when script is executed directly
    pytest.main([__file__, "-v"])
