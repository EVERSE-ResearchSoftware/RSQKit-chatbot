import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch
import chromadb

# Import the modules we're testing
from chroma_data_ingestor import get_embedding
from core_utils.hybrid_search import HybridSearch
from core_utils.retrieval_utils import (
    create_multi_retrieval_engine,
    agentic_query_processing,
)
from prompt_templates.prompt_builder import build_rag_context

TEST_SESSION_KEY = "rsqkit_chat"


class TestCoreRAGPipeline:
    """Integration tests for the core RAG pipeline functionality"""

    @pytest.fixture
    def temp_chroma_db(self):
        """Create a temporary ChromaDB instance for testing"""
        temp_dir = tempfile.mkdtemp()
        client = chromadb.PersistentClient(path=temp_dir)
        collection = client.get_or_create_collection("test_documents")

        # Add test documents
        test_docs = [
            "Python is a high-level programming language known for its simplicity.",
            "Machine learning algorithms can identify patterns in large datasets.",
            "Vector databases store high-dimensional embeddings for similarity search.",
            "Retrieval-Augmented Generation combines information retrieval with language models.",
        ]

        test_ids = [f"doc_{i}" for i in range(len(test_docs))]
        test_metadatas = [
            {"source": f"test_doc_{i}.txt"} for i in range(len(test_docs))
        ]

        # Mock embeddings (simplified for testing)
        test_embeddings = [[0.1 * i] * 384 for i in range(len(test_docs))]

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
    def mock_llm_response(self):
        """Mock LLM response for testing"""

        def mock_chat_function(messages, model=None, stream=False, **kwargs):
            # Simulate streaming response
            response_text = "This is a test response based on the retrieved context."
            if stream:
                # Return a generator that yields chunks
                def response_generator():
                    words = response_text.split()
                    for word in words:
                        yield {"choices": [{"delta": {"content": word + " "}}]}

                return response_generator()
            else:
                return {"choices": [{"message": {"content": response_text}}]}

        return mock_chat_function

    @pytest.fixture
    def mock_embedding_function(self):
        """Mock embedding function for testing"""

        def mock_get_embedding(text, provider="openai"):
            # Return a simple mock embedding based on text length
            return [0.1] * 384

        return mock_get_embedding

    def test_end_to_end_rag_pipeline(
        self, temp_chroma_db, mock_llm_response, mock_embedding_function
    ):
        """Test the complete RAG pipeline from query to response"""
        client, collection = temp_chroma_db

        # Mock session state
        mock_session_state = {
            TEST_SESSION_KEY: {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."}
                ],
                "retrieval_history": [],
            }
        }

        with (
            patch("streamlit.session_state", mock_session_state),
            patch("chroma_data_ingestor.get_embedding", mock_embedding_function),
            patch("llm_provider_tools.get_embedding", mock_embedding_function),
            patch("llm_provider_tools.rerank_results", lambda x, y, z: x),
        ):  # No reranking for simplicity

            # Initialize hybrid searcher
            hybrid_searcher = HybridSearch(collection, alpha=0.7)

            # Create multi-retrieval engine
            multi_retrieval_engine = create_multi_retrieval_engine(
                collection=collection,
                selected_provider="openai",
                chat_function=mock_llm_response,
                model_name="gpt-3.5-turbo",
            )

            # Test query
            test_query = "What is Python programming language?"

            # Execute the agentic query processing
            with (
                patch("streamlit.chat_message"),
                patch("streamlit.markdown"),
                patch("streamlit.expander"),
                patch("ui.custom_display.view_sources"),
            ):

                agentic_query_processing(
                    query=test_query,
                    selected_provider="openai",
                    rerank_results=lambda x, y, z: x,  # Mock rerank
                    chat_function=mock_llm_response,
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
                    build_rag_context=build_rag_context,
                    session_key=TEST_SESSION_KEY,
                )

            # Verify that the session state was updated with messages
            assert len(mock_session_state[TEST_SESSION_KEY]["messages"]) > 1

            # Verify that user message was added
            user_messages = [
                msg
                for msg in mock_session_state[TEST_SESSION_KEY]["messages"]
                if msg["role"] == "user"
            ]
            assert len(user_messages) > 0
            assert user_messages[0]["content"] == test_query

            # Verify that assistant response was added
            assistant_messages = [
                msg
                for msg in mock_session_state[TEST_SESSION_KEY]["messages"]
                if msg["role"] == "assistant"
            ]
            assert len(assistant_messages) > 0

    def test_context_building_with_retrieved_docs(
        self, temp_chroma_db, mock_embedding_function
    ):
        """Test that context is properly built from retrieved documents"""
        client, collection = temp_chroma_db

        with (
            patch("chroma_data_ingestor.get_embedding", mock_embedding_function),
            patch("llm_provider_tools.get_embedding", mock_embedding_function),
        ):

            # Initialize hybrid searcher
            hybrid_searcher = HybridSearch(collection, alpha=0.7)

            # Get query embedding
            query_text = "Python programming"
            query_embedding = mock_embedding_function(query_text)

            # Perform a search with proper arguments
            results = hybrid_searcher.hybrid_search(
                query=query_text, query_embedding=query_embedding, k=2
            )

            # Verify we got results
            assert len(results["documents"]) > 0

            # Build context from results
            context = build_rag_context(results["documents"])

            # Verify context contains expected information
            assert isinstance(context, str)
            assert len(context) > 0
            assert "Python" in context  # Should contain relevant content

    def test_empty_collection_handling(self):
        """Test behavior when no documents are available"""
        temp_dir = tempfile.mkdtemp()
        client = chromadb.PersistentClient(path=temp_dir)
        empty_collection = client.get_or_create_collection("empty_test")

        try:
            # Check that collection is empty
            all_results = empty_collection.get()
            assert len(all_results["documents"]) == 0

            # The main app should handle this gracefully
            # This test verifies the collection state check

        finally:
            shutil.rmtree(temp_dir)

    def test_hybrid_search_integration(self, temp_chroma_db, mock_embedding_function):
        """Test that hybrid search properly combines semantic and keyword search"""
        client, collection = temp_chroma_db

        with (
            patch("chroma_data_ingestor.get_embedding", mock_embedding_function),
            patch("llm_provider_tools.get_embedding", mock_embedding_function),
        ):

            query_text = "programming language"
            query_embedding = mock_embedding_function(query_text)

            # Test different alpha values for hybrid search
            for alpha in [0.0, 0.5, 1.0]:  # Pure keyword, balanced, pure semantic
                hybrid_searcher = HybridSearch(collection, alpha=alpha)
                results = hybrid_searcher.hybrid_search(
                    query=query_text, query_embedding=query_embedding, k=2
                )

                # Should return results regardless of alpha
                assert len(results["documents"]) > 0
                assert len(results["metadatas"]) == len(results["documents"])
                assert len(results["ids"]) == len(results["documents"])


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
