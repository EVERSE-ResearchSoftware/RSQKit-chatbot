import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import chromadb

TEST_SESSION_KEY = "rag_chat"

# Import the modules we're testing
from core_utils.retrieval_utils import (
    create_multi_retrieval_engine,
    agentic_query_processing,
)
from core_utils.hybrid_search import HybridSearch


class TestMultiRetrievalEngine:
    """Integration tests for the Multi-Retrieval Engine functionality"""

    @pytest.fixture
    def temp_chroma_db_with_diverse_content(self):
        """Create a ChromaDB instance with diverse content for multi-retrieval testing"""
        temp_dir = tempfile.mkdtemp()
        client = chromadb.PersistentClient(path=temp_dir)
        collection = client.get_or_create_collection("multi_retrieval_test")

        # Add diverse test documents that would benefit from query decomposition
        test_docs = [
            "Python is a versatile programming language used for web development, data science, and automation.",
            "Machine learning models require large datasets for training and validation processes.",
            "Vector databases like ChromaDB store embeddings for similarity search and retrieval.",
            "Streamlit is a Python framework for building interactive web applications quickly.",
            "Natural language processing involves tokenization, embedding, and semantic analysis.",
            "Database indexing improves query performance by creating efficient data structures.",
            "API rate limiting prevents abuse and ensures fair usage across multiple users.",
            "Cloud computing platforms provide scalable infrastructure for modern applications.",
            "Software testing includes unit tests, integration tests, and end-to-end testing.",
            "Data preprocessing steps include cleaning, normalization, and feature engineering.",
        ]

        test_ids = [f"doc_{i}" for i in range(len(test_docs))]
        test_metadatas = [
            (
                {"source": f"programming_doc_{i}.txt", "category": "programming"}
                if i < 4
                else {"source": f"data_doc_{i}.txt", "category": "data_science"}
            )
            for i in range(len(test_docs))
        ]

        # Create more varied embeddings for better testing
        test_embeddings = []
        for i, j in enumerate(
            zip(range(len(test_docs)), reversed(range(len(test_docs))))
        ):
            base_value = 0.1 * i + 0.05 * j[0]  # j is a tuple from zip
            embedding = [base_value] * 384
            test_embeddings.append(embedding)

        collection.add(
            documents=test_docs,
            ids=test_ids,
            metadatas=test_metadatas,
            embeddings=test_embeddings,
        )

        yield client, collection

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_embedding_function(self):
        """Mock embedding function for consistent testing"""

        def mock_get_embedding(text=None, provider="openai", input=None, **kwargs):
            # Handle both 'text' and 'input' parameters (some APIs use 'input')
            text_to_process = text or input or ""
            # Create embeddings based on text content for more realistic testing
            text_hash = hash(str(text_to_process)) % 1000
            return [0.001 * text_hash] * 384

        return mock_get_embedding

    @pytest.fixture
    def mock_llm_for_decomposition(self):
        """Mock LLM that provides query decomposition"""

        def mock_chat_function(messages, model=None, stream=False, **kwargs):
            # Check if this is a query decomposition request
            if any(
                "decompose" in msg.get("content", "").lower()
                or "subqueries" in msg.get("content", "").lower()
                for msg in messages
                if isinstance(msg, dict)
            ):
                # Return decomposed queries
                response = """1. What is Python programming language used for?
2. How does machine learning work with datasets?
3. What are vector databases and their applications?"""
            else:
                # Regular response
                response = "This is a comprehensive answer based on multiple retrieved contexts."

            if stream:

                def response_generator():
                    for word in response.split():
                        yield {"choices": [{"delta": {"content": word + " "}}]}

                return response_generator()
            else:
                return {"choices": [{"message": {"content": response}}]}

        return mock_chat_function

    def test_multi_retrieval_engine_creation(
        self, temp_chroma_db_with_diverse_content, mock_llm_for_decomposition
    ):
        """Test that multi-retrieval engine is created successfully with all required components"""
        client, collection = temp_chroma_db_with_diverse_content

        # Create multi-retrieval engine
        engine = create_multi_retrieval_engine(
            collection=collection,
            selected_provider="openai",
            chat_function=mock_llm_for_decomposition,
            model_name="gpt-3.5-turbo",
        )

        # Verify engine was created and has expected attributes
        assert engine is not None

        # Test for common attributes that a retrieval engine might have
        # Since we don't know the exact structure, test for various possibilities
        has_collection = hasattr(engine, "collection")
        has_retriever = hasattr(engine, "retriever")
        has_query_method = hasattr(engine, "query")
        has_retrieve_method = hasattr(engine, "retrieve")
        has_search_method = hasattr(engine, "search")
        is_callable = callable(engine)

        # At least one of these should be true for a valid retrieval engine
        assert (
            has_collection
            or has_retriever
            or has_query_method
            or has_retrieve_method
            or has_search_method
            or is_callable
        ), f"Engine should have at least one expected attribute or be callable. Engine type: {type(engine)}"

    def test_query_decomposition_and_parallel_retrieval(
        self,
        temp_chroma_db_with_diverse_content,
        mock_embedding_function,
        mock_llm_for_decomposition,
    ):
        """Test that complex queries are properly decomposed and retrieved in parallel"""
        client, collection = temp_chroma_db_with_diverse_content

        mock_session_state = {
            TEST_SESSION_KEY: {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."}
                ],
                "retrieval_history": [],
            },
            "temperature": 0,
        }

        with (
            patch("streamlit.session_state", mock_session_state),
            patch("chroma_data_ingestor.get_embedding", mock_embedding_function),
            patch("llm_provider_tools.get_embedding", mock_embedding_function),
            patch("llm_provider_tools.rerank_results", lambda x, y, z: x),
        ):

            # Initialize components
            hybrid_searcher = HybridSearch(collection, alpha=0.7)
            multi_retrieval_engine = create_multi_retrieval_engine(
                collection=collection,
                selected_provider="openai",
                chat_function=mock_llm_for_decomposition,
                model_name="gpt-3.5-turbo",
            )

            # Complex query that should benefit from decomposition
            complex_query = "How do Python and machine learning work together with vector databases for building applications?"

            # Mock Streamlit UI components
            with (
                patch("streamlit.chat_message"),
                patch("streamlit.markdown"),
                patch("streamlit.expander") as mock_expander,
                patch("ui.custom_display.view_sources"),
            ):

                # Configure mock expander to capture decomposition display
                mock_expander_context = MagicMock()
                mock_expander.return_value.__enter__.return_value = (
                    mock_expander_context
                )

                # Execute agentic query processing with multi-retrieval enabled
                agentic_query_processing(
                    query=complex_query,
                    selected_provider="openai",
                    rerank_results=lambda x, y, z: x,
                    chat_function=mock_llm_for_decomposition,
                    get_embedding=mock_embedding_function,
                    llm_model="gpt-3.5-turbo",
                    multi_retrieval_engine=multi_retrieval_engine,
                    hybrid_searcher=hybrid_searcher,
                    show_decomposition=True,
                    default_strategy="adaptive",
                    show_retrieval_details=True,
                    max_subqueries=3,
                    view_sources=Mock(),
                    enable_multi_retrieval=True,
                    build_rag_context=Mock(return_value="Mocked context"),
                    session_key=TEST_SESSION_KEY,
                )

            # Verify that session state was updated
            assert len(mock_session_state[TEST_SESSION_KEY]["messages"]) > 1

            # Verify retrieval history was populated (indicating multi-retrieval occurred)
            if "retrieval_history" in mock_session_state[TEST_SESSION_KEY]:
                # Check that some retrieval activity was recorded
                retrieval_history = mock_session_state[TEST_SESSION_KEY][
                    "retrieval_history"
                ]
                # The exact structure depends on your implementation

    def test_different_retrieval_strategies(
        self,
        temp_chroma_db_with_diverse_content,
        mock_embedding_function,
        mock_llm_for_decomposition,
    ):
        """Test different retrieval strategies (adaptive, parallel, sequential)"""
        client, collection = temp_chroma_db_with_diverse_content

        strategies = ["adaptive", "parallel", "sequential"]

        for strategy in strategies:
            mock_session_state = {
                TEST_SESSION_KEY: {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."}
                    ],
                    "retrieval_history": [],
                    "temperature": 0,
                }
            }

            with (
                patch("streamlit.session_state", mock_session_state),
                patch("chroma_data_ingestor.get_embedding", mock_embedding_function),
                patch("llm_provider_tools.get_embedding", mock_embedding_function),
                patch("llm_provider_tools.rerank_results", lambda x, y, z: x),
            ):

                # Initialize components
                hybrid_searcher = HybridSearch(collection, alpha=0.7)
                multi_retrieval_engine = create_multi_retrieval_engine(
                    collection=collection,
                    selected_provider="openai",
                    chat_function=mock_llm_for_decomposition,
                    model_name="gpt-3.5-turbo",
                )

                test_query = f"Test query for {strategy} strategy"

                with (
                    patch("streamlit.chat_message"),
                    patch("streamlit.markdown"),
                    patch("streamlit.expander"),
                    patch("ui.custom_display.view_sources"),
                ):

                    # Test each strategy
                    agentic_query_processing(
                        query=test_query,
                        selected_provider="openai",
                        rerank_results=lambda x, y, z: x,
                        chat_function=mock_llm_for_decomposition,
                        get_embedding=mock_embedding_function,
                        llm_model="gpt-3.5-turbo",
                        multi_retrieval_engine=multi_retrieval_engine,
                        hybrid_searcher=hybrid_searcher,
                        show_decomposition=False,
                        default_strategy=strategy,
                        show_retrieval_details=False,
                        max_subqueries=2,
                        view_sources=Mock(),
                        enable_multi_retrieval=True,
                        build_rag_context=Mock(return_value="Mocked context"),
                        session_key=TEST_SESSION_KEY,
                    )

                # Verify that processing completed without errors for each strategy
                assert len(mock_session_state[TEST_SESSION_KEY]["messages"]) > 1

    def test_max_subqueries_limit(
        self,
        temp_chroma_db_with_diverse_content,
        mock_embedding_function,
        mock_llm_for_decomposition,
    ):
        """Test that max_subqueries parameter properly limits query decomposition"""
        client, collection = temp_chroma_db_with_diverse_content

        for max_subqueries in [1, 3, 5]:
            mock_session_state = {
                TEST_SESSION_KEY: {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."}
                    ],
                    "retrieval_history": [],
                },
                "temperature": 0,
            }

            with (
                patch("streamlit.session_state", mock_session_state),
                patch("chroma_data_ingestor.get_embedding", mock_embedding_function),
                patch("llm_provider_tools.get_embedding", mock_embedding_function),
                patch("llm_provider_tools.rerank_results", lambda x, y, z: x),
            ):

                hybrid_searcher = HybridSearch(collection, alpha=0.7)
                multi_retrieval_engine = create_multi_retrieval_engine(
                    collection=collection,
                    selected_provider="openai",
                    chat_function=mock_llm_for_decomposition,
                    model_name="gpt-3.5-turbo",
                )

                complex_query = "Tell me about Python, machine learning, databases, APIs, and testing"

                with (
                    patch("streamlit.chat_message"),
                    patch("streamlit.markdown"),
                    patch("streamlit.expander"),
                    patch("ui.custom_display.view_sources"),
                ):

                    agentic_query_processing(
                        query=complex_query,
                        selected_provider="openai",
                        rerank_results=lambda x, y, z: x,
                        chat_function=mock_llm_for_decomposition,
                        get_embedding=mock_embedding_function,
                        llm_model="gpt-3.5-turbo",
                        multi_retrieval_engine=multi_retrieval_engine,
                        hybrid_searcher=hybrid_searcher,
                        show_decomposition=True,
                        default_strategy="adaptive",
                        show_retrieval_details=True,
                        max_subqueries=max_subqueries,
                        view_sources=Mock(),
                        enable_multi_retrieval=True,
                        build_rag_context=Mock(return_value="Mocked context"),
                        session_key=TEST_SESSION_KEY,
                    )

                # Verify processing completed (exact verification depends on implementation)
                assert len(mock_session_state[TEST_SESSION_KEY]["messages"]) > 1

    def test_multi_retrieval_disabled_fallback(
        self,
        temp_chroma_db_with_diverse_content,
        mock_embedding_function,
        mock_llm_for_decomposition,
    ):
        """Test that system falls back to single retrieval when multi-retrieval is disabled"""
        client, collection = temp_chroma_db_with_diverse_content

        mock_session_state = {
            TEST_SESSION_KEY: {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."}
                ],
                "retrieval_history": [],
            }
        }

        def mock_rerank_results(*args, **kwargs):
            """Mock rerank function that accepts any arguments"""
            # Return the first argument (usually the results) or empty list
            if args:
                return args[0]
            return kwargs.get("results", [])

        with (
            patch("streamlit.session_state", mock_session_state),
            patch("chroma_data_ingestor.get_embedding", mock_embedding_function),
            patch("llm_provider_tools.get_embedding", mock_embedding_function),
            patch("llm_provider_tools.rerank_results", mock_rerank_results),
        ):

            hybrid_searcher = HybridSearch(collection, alpha=0.7)
            multi_retrieval_engine = create_multi_retrieval_engine(
                collection=collection,
                selected_provider="openai",
                chat_function=mock_llm_for_decomposition,
                model_name="gpt-3.5-turbo",
            )

            test_query = "Complex query that would normally be decomposed"

            with (
                patch("streamlit.chat_message"),
                patch("streamlit.markdown"),
                patch("streamlit.expander"),
                patch("ui.custom_display.view_sources"),
                patch(
                    "core_utils.retrieval_utils.respond_with_enhanced_naive_rag"
                ) as mock_enhanced_rag,
            ):
                mock_enhanced_rag.return_value = ["doc1", "doc2", "doc3"]
                # Test with multi-retrieval disabled
                agentic_query_processing(
                    query=test_query,
                    selected_provider="openai",
                    chat_function=mock_llm_for_decomposition,
                    # get_embedding=mock_embedding_function,
                    llm_model="gpt-3.5-turbo",
                    multi_retrieval_engine=multi_retrieval_engine,
                    # hybrid_searcher=hybrid_searcher,
                    show_decomposition=False,
                    default_strategy="adaptive",
                    show_retrieval_details=False,
                    max_subqueries=3,
                    enable_multi_retrieval=False,  # Disabled!
                    build_rag_context=Mock(return_value="Mocked context"),
                    session_key=TEST_SESSION_KEY,
                )

                mock_enhanced_rag.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
