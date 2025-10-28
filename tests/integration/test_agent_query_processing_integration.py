import pytest
from unittest.mock import Mock, patch
import streamlit as st

from core_utils.retrieval_utils import agentic_query_processing, generate_sources_html


def test_agentic_query_processing_multi_retrieval():
    session_key = "test_session"
    query = "Test query"
    selected_provider = "test_provider"
    enable_multi_retrieval = True
    show_retrieval_details = True
    show_decomposition = True
    max_subqueries = 5
    default_strategy = "hybrid"
    llm_model = "test_model"

    # Mock multi_retrieval_engine
    multi_retrieval_engine = Mock()
    detection_info = {"requires_decomposition": True, "complexity_score": 7}
    multi_retrieval_engine.detect_multiple_questions.return_value = detection_info

    subqueries = [
        Mock(
            id="subq_1",
            question="Subquery 1",
            query_type="simple",
            priority=1,
            parent_query=query,
            search_strategy="hybrid",
            is_independent=True,
        ),
        Mock(
            id="subq_2",
            question="Subquery 2",
            query_type="complex",
            priority=2,
            parent_query=query,
            search_strategy="vector",
            is_independent=False,
        ),
    ]
    multi_retrieval_engine.decompose_query.return_value = subqueries

    subquery_answers = [
        Mock(
            answer="Answer 1",
            sources_used=["doc1", "doc2"],
            metadatas_used=[{"source": "source1"}, {"source": "source2"}],
        )
    ]
    retrieval_results = {"subq_1": Mock(documents=["doc1", "doc2"])}
    multi_retrieval_engine.execute_multi_retrieval_with_streaming.return_value = (
        subquery_answers,
        retrieval_results,
    )
    multi_retrieval_engine.synthesize_final_answer.return_value = "Final answer"

    # Mock session_state
    with patch("streamlit.session_state", new_callable=dict):
        st.session_state = {session_key: {"messages": [], "retrieval_history": []}}

        agentic_query_processing(
            query=query,
            selected_provider=selected_provider,
            rerank_results=Mock(),
            chat_function=Mock(),
            build_rag_context=Mock(return_value="context"),
            llm_model=llm_model,
            session_key=session_key,
            multi_retrieval_engine=multi_retrieval_engine,
            hybrid_searcher=Mock(),
            get_embedding=Mock(),
            enable_multi_retrieval=enable_multi_retrieval,
            show_retrieval_details=show_retrieval_details,
            show_decomposition=show_decomposition,
            max_subqueries=max_subqueries,
            default_strategy=default_strategy,
        )

        # Check session_state
        expected_messages = [
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "content": "Final answer",
                "sources_html": generate_sources_html(
                    relevant_docs=["doc1", "doc2"],
                    metadatas=[{"source": "source1"}, {"source": "source2"}],
                ),
            },
        ]
        assert st.session_state[session_key]["messages"] == expected_messages

        # Check retrieval_history
        retrieval_history = st.session_state[session_key]["retrieval_history"]
        assert len(retrieval_history) == 1
        entry = retrieval_history[0]
        assert entry["query"] == query
        assert entry["subqueries"] == subqueries
        assert entry["detection_info"] == detection_info
        assert entry["subquery_answers"] == subquery_answers
        assert (
            entry["results_count"] == 2
        )  # len(result.documents) for retrieval_results


def test_agentic_query_processing_traditional():
    session_key = "test_session"
    query = "Test query"
    selected_provider = "test_provider"
    enable_multi_retrieval = False
    show_retrieval_details = True
    llm_model = "test_model"

    # Mocks
    hybrid_searcher = Mock()
    hybrid_results = {
        "documents": ["doc1", "doc2", "doc3", "doc4", "doc5"],
        "ids": ["id1", "id2", "id3", "id4", "id5"],
        "metadatas": [
            {"source": "source1"},
            {"source": "source2"},
            {"source": "source3"},
            {"source": "source4"},
            {"source": "source5"},
        ],
        "scores": [0.9, 0.8, 0.7, 0.6, 0.5],
    }
    hybrid_searcher.hybrid_search.return_value = hybrid_results
    get_embedding = Mock()
    get_embedding.return_value = [0.1, 0.2, 0.3]
    rerank_results = Mock()
    reranked_data = {
        "reranked_docs": ["doc1", "doc2", "doc3"],
        "reranked_metadatas": [
            {"source": "source1"},
            {"source": "source2"},
            {"source": "source3"},
        ],
    }
    rerank_results.return_value = reranked_data
    build_rag_context = Mock(return_value="context")
    chat_function = Mock()
    chat_function.return_value = iter(
        ["F", "i", "n", "a", "l", " ", "a", "n", "s", "w", "e", "r"]
    )

    with patch(
        "core_utils.retrieval_utils.respond_with_enhanced_naive_rag"
    ) as mock_enhanced_rag:
        mock_enhanced_rag.return_value = ["doc1", "doc2", "doc3"]

        # Mock session_state
        with patch("streamlit.session_state", new_callable=dict):
            st.session_state = {session_key: {"messages": [], "retrieval_history": []}}

            agentic_query_processing(
                query=query,
                selected_provider=selected_provider,
                chat_function=chat_function,
                build_rag_context=build_rag_context,
                llm_model=llm_model,
                session_key=session_key,
                multi_retrieval_engine=Mock(),
                enable_multi_retrieval=enable_multi_retrieval,
                show_retrieval_details=show_retrieval_details,
            )

            mock_enhanced_rag.assert_called_once()
            # assert st.session_state[session_key]["messages"] == expected_messages

            # Check retrieval_history is empty
            retrieval_history = st.session_state[session_key]["retrieval_history"]
            assert retrieval_history == []

    # # Mock session_state
    # with patch("streamlit.session_state", new_callable=dict):
    #     st.session_state = {session_key: {"messages": [], "retrieval_history": []}}

    #     agentic_query_processing(
    #         query=query,
    #         selected_provider=selected_provider,
    #         chat_function=chat_function,
    #         build_rag_context=build_rag_context,
    #         llm_model=llm_model,
    #         session_key=session_key,
    #         multi_retrieval_engine=Mock(),
    #         enable_multi_retrieval=enable_multi_retrieval,
    #         show_retrieval_details=show_retrieval_details,
    #     )

    #     # Check session_state
    #     expected_messages = [
    #         {"role": "user", "content": query},
    #         {"role": "assistant", "content": "Final answer",
    #          "sources_html": generate_sources_html(relevant_docs=["doc1", "doc2", "doc3"],metadatas=[
    #             {"source": "source1"},
    #             {"source": "source2"},
    #             {"source": "source3"},
    #         ])},

    #     ]
    #     assert st.session_state[session_key]["messages"] == expected_messages

    #     # Check retrieval_history is empty
    #     retrieval_history = st.session_state[session_key]["retrieval_history"]
    #     assert retrieval_history == []
