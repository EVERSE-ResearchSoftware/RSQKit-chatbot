import json
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import streamlit as st
from llm_provider_tools import get_embedding, rerank_results
from prompt_templates.prompts import RAG_PROMPT_TEMPLATE
from llms.openai_interface import get_chat_response_stream


@dataclass
class SubQuery:
    id: str
    question: str
    query_type: str
    priority: int
    parent_query: str
    dependencies: List[str] = None
    search_strategy: str = "hybrid"
    is_independent: bool = True  # New field to track independence


@dataclass
class RetrievalResult:
    subquery_id: str
    question: str
    documents: List[str]
    metadatas: List[Dict]
    scores: List[float]
    strategy_used: str


@dataclass
class SubQueryAnswer:
    subquery_id: str
    question: str
    answer: str
    has_sufficient_context: bool
    sources_used: List[str]
    metadatas_used: List[Dict]


def rerank_hybrid_results(
    query,
    hybrid_results,
    selected_provider,
    top_rerank: int = 3,
    reranking_function=rerank_results,
):
    # Format results for reranker
    chroma_format_results = {
        "documents": [hybrid_results["documents"]],
        "ids": [hybrid_results["ids"]],
        "metadatas": [hybrid_results["metadatas"]],
        "distances": [[1 - score for score in hybrid_results["scores"]]],
    }

    reranked_data = reranking_function(
        results=chroma_format_results,
        provider=selected_provider,
        query=query,
        top_rerank=top_rerank,
    )
    return reranked_data


class MultiRetrievalEngine:
    def __init__(self, chat_function, model_name, hybrid_searcher, selected_provider):
        self.chat_function = chat_function
        self.model_name = model_name
        self.hybrid_searcher = hybrid_searcher
        self.selected_provider = selected_provider
        # Store the collection reference for creating new HybridSearch instances
        self.collection = hybrid_searcher.collection if hybrid_searcher else None

    def detect_multiple_questions(self, query: str) -> Dict[str, Any]:
        """Detect if the query contains multiple questions or complex requirements"""
        detection_prompt = f"""
        Analyze this query to detect if it contains multiple questions or complex information needs:
        
        Query: "{query}"
        
        Look for:
        1. Multiple question marks
        2. Words like "and", "also", "what about", "compare", "versus"
        3. Lists or enumerations
        4. Different topics or concepts
        5. Sequential requests
        
        Respond in JSON format:
        {{
            "is_multi_query": true/false,
            "question_count": number,
            "complexity_score": 0-10,
            "detected_questions": ["list of individual questions if multiple"],
            "query_type": "single|multi_question|comparative|sequential|complex",
            "requires_decomposition": true/false,
            "reasoning": "explanation of detection"
        }}
        """

        messages = [{"role": "user", "content": detection_prompt}]

        try:
            with st.status(
                "🤔 Thinking: Detecting query complexity...", expanded=True
            ) as status:
                thinking_placeholder = st.empty()
                thinking_buffer = ""

                for token in self.chat_function(
                    model_name=self.model_name,
                    messages=messages,
                    provider=self.selected_provider,
                ):
                    thinking_buffer += token
                    thinking_placeholder.markdown(f"```\n{thinking_buffer}\n```")

                response = thinking_buffer
            status.update(
                label="🤔 Thinking: Detecting query complexity...", expanded=False
            )
        except Exception:
            # Fallback: aggregate tokens without any st calls
            response = ""
            for token in self.chat_function(
                model_name=self.model_name,
                messages=messages,
                provider=self.selected_provider,
            ):
                response += token
        try:
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
        except:
            # Fallback detection using simple heuristics
            question_marks = query.count("?")
            multi_indicators = len(
                re.findall(
                    r"\b(and|also|what about|compare|versus|vs|additionally|furthermore)\b",
                    query.lower(),
                )
            )

            return {
                "is_multi_query": question_marks > 1 or multi_indicators > 0,
                "question_count": max(1, question_marks),
                "complexity_score": min(10, question_marks * 2 + multi_indicators),
                "detected_questions": [query],
                "query_type": "multi_question" if question_marks > 1 else "single",
                "requires_decomposition": question_marks > 1 or multi_indicators > 1,
                "reasoning": "Heuristic detection based on punctuation and keywords",
            }

    def decompose_query(self, query: str, detection_info: Dict) -> List[SubQuery]:
        """Break down complex query into subqueries"""
        if not detection_info.get("requires_decomposition", False):
            return [
                SubQuery(
                    id="subq_1",
                    question=query,
                    query_type="simple",
                    priority=1,
                    parent_query=query,
                    is_independent=True,
                )
            ]

        decomposition_prompt = f"""
        Break down this complex query into specific subqueries that can be searched independently:
        
        Original Query: "{query}"
        Detection Info: {json.dumps(detection_info, indent=2)}
        
        For each subquery, determine if it can be answered independently or if it depends on other subqueries.
        Independent subqueries should be answerable without needing information from other parts.
        
        Format as JSON:
        {{
            "subqueries": [
                {{
                    "id": "subq_1",
                    "question": "specific searchable question",
                    "query_type": "factual|analytical|comparative|definitional",
                    "priority": 1,
                    "dependencies": [],
                    "search_strategy": "hybrid|semantic|keyword|multi_step",
                    "is_independent": true
                }}
            ],
            "execution_order": ["subq_1", "subq_2", ...],
            "combination_strategy": "merge|compare|sequence|synthesize"
        }}
        """

        messages = [{"role": "user", "content": decomposition_prompt}]

        # Stream the thinking process
        try:
            with st.status(
                "🤔 Thinking: Decomposing query...", expanded=True
            ) as status:
                thinking_placeholder = st.empty()
                thinking_buffer = ""

                for token in self.chat_function(
                    model_name=self.model_name,
                    messages=messages,
                    provider=self.selected_provider,
                ):
                    thinking_buffer += token
                    thinking_placeholder.markdown(f"```\n{thinking_buffer}\n```")

                response = thinking_buffer
                status.update(label="🤔 Thinking: Decomposing query...", expanded=False)
        except Exception:
            response = ""
            for token in self.chat_function(
                model_name=self.model_name,
                messages=messages,
                provider=self.selected_provider,
            ):
                response += token

        try:
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            subqueries = []
            for sq in data.get("subqueries", []):
                subquery = SubQuery(
                    id=sq["id"],
                    question=sq["question"],
                    query_type=sq["query_type"],
                    priority=sq["priority"],
                    parent_query=query,
                    dependencies=sq.get("dependencies", []),
                    search_strategy=sq.get("search_strategy", "hybrid"),
                    is_independent=sq.get("is_independent", True),
                )
                subqueries.append(subquery)

            return subqueries

        except Exception as e:
            # Try to report the error via streamlit if available—silently ignore otherwise
            try:
                st.error(f"Error in query decomposition: {e}")
            except Exception:
                pass
            # Fallback: split by question marks or conjunctions
            parts = re.split(r"[\?]|(?:\band\s)|(?:\balso\s)", query)
            subqueries = []
            for i, part in enumerate(parts):
                if part.strip():
                    subquery = SubQuery(
                        id=f"subq_{i+1}",
                        question=part.strip()
                        + ("?" if not part.strip().endswith("?") else ""),
                        query_type="factual",
                        priority=i + 1,
                        parent_query=query,
                        is_independent=True,
                    )
                    subqueries.append(subquery)
            return subqueries

    def execute_subquery(self, subquery: SubQuery) -> RetrievalResult:
        """Execute a single subquery using appropriate retrieval strategy"""
        try:
            if subquery.search_strategy == "multi_step":
                # Multi-step retrieval for complex subqueries
                return self._multi_step_retrieval(subquery)
            elif subquery.search_strategy == "semantic":
                # Semantic-focused search
                return self._semantic_retrieval(subquery)
            elif subquery.search_strategy == "keyword":
                # Keyword-focused search
                return self._hybrid_retrieval(subquery)  # Using hybrid as fallback
            else:
                # Default hybrid search
                return self._hybrid_retrieval(subquery)

        except Exception as e:
            if "st" in globals():
                st.error(f"Error executing subquery {subquery.id}: {e}")
            return RetrievalResult(
                subquery_id=subquery.id,
                question=subquery.question,
                documents=[],
                metadatas=[],
                scores=[],
                strategy_used="error",
            )

    def _hybrid_retrieval(
        self, subquery: SubQuery, k: int = 5, top_rerank: int = 3
    ) -> RetrievalResult:
        """Standard hybrid retrieval"""
        query_embedding = get_embedding(
            input=subquery.question, provider=self.selected_provider
        )

        hybrid_results = self.hybrid_searcher.hybrid_search(
            query=subquery.question, query_embedding=query_embedding, k=k
        )
        # rerank first
        reranked_results = rerank_hybrid_results(
            query=subquery.question,
            hybrid_results=hybrid_results,
            selected_provider=self.selected_provider,
            top_rerank=top_rerank,
        )
        return RetrievalResult(
            subquery_id=subquery.id,
            question=subquery.question,
            documents=reranked_results["reranked_docs"],
            metadatas=reranked_results["reranked_metadatas"],
            scores=reranked_results["reranked_scores"],
            strategy_used="hybrid",
        )

    def _semantic_retrieval(
        self, subquery: SubQuery, k: int = 5, top_rerank: int = 3
    ) -> RetrievalResult:
        """Semantic-focused retrieval with higher alpha"""

        query_embedding = get_embedding(
            input=subquery.question, provider=self.selected_provider
        )

        results = self.hybrid_searcher.hybrid_search(
            query=subquery.question, query_embedding=query_embedding, k=k
        )
        reranked_results = rerank_hybrid_results(
            query=subquery.question,
            hybrid_results=results,
            selected_provider=self.selected_provider,
            top_rerank=top_rerank,
        )
        return RetrievalResult(
            subquery_id=subquery.id,
            question=subquery.question,
            documents=reranked_results["reranked_docs"],
            metadatas=reranked_results["reranked_metadatas"],
            scores=reranked_results["reranked_scores"],
            strategy_used="semantic",
        )

    def _multi_step_retrieval(
        self, subquery: SubQuery, k: int = 5, top_rerank: int = 3
    ) -> RetrievalResult:
        """Multi-step retrieval for complex queries"""
        # First, get initial results
        initial_result = self._hybrid_retrieval(subquery)

        if not initial_result.documents:
            return initial_result

        # Generate follow-up query based on initial results
        followup_prompt = f"""
        Based on these initial search results for the question "{subquery.question}",
        generate a refined search query to get more specific information:
        
        Initial Results Preview: {initial_result.documents[0][:200] if initial_result.documents else "No results"}
        
        Provide a refined search query that would get more targeted information:
        """

        messages = [{"role": "user", "content": followup_prompt}]
        refined_query = ""
        for token in self.chat_function(
            model_name=self.model_name,
            messages=messages,
            provider=self.selected_provider,
        ):
            refined_query += token

        # Execute refined search
        refined_embedding = get_embedding(
            input=refined_query.strip(), provider=self.selected_provider
        )
        refined_results = self.hybrid_searcher.hybrid_search(
            query=refined_query.strip(), query_embedding=refined_embedding, k=k
        )
        reranked_refined_results = rerank_hybrid_results(
            query=refined_query.strip(),
            hybrid_results=refined_results,
            top_rerank=top_rerank,
            selected_provider=self.selected_provider,
        )
        # Combine results
        combined_docs = (
            initial_result.documents + reranked_refined_results["reranked_docs"]
        )
        combined_metas = (
            initial_result.metadatas + reranked_refined_results["reranked_metadatas"]
        )
        combined_scores = (
            initial_result.scores + reranked_refined_results["reranked_scores"]
        )

        return RetrievalResult(
            subquery_id=subquery.id,
            question=subquery.question,
            documents=combined_docs[:8],  # Limit total results
            metadatas=combined_metas[:8],
            scores=combined_scores[:8],
            strategy_used="multi_step",
        )

    def answer_subquery_traditional_rag(
        self,
        subquery: SubQuery,
        retrieval_result: RetrievalResult,
        build_rag_context,
        rag_prompt_template: str,
    ) -> SubQueryAnswer:
        """Answer a single subquery using traditional RAG approach (more lenient)"""

        if not retrieval_result.documents:
            return SubQueryAnswer(
                subquery_id=subquery.id,
                question=subquery.question,
                answer="I cannot answer this question based on the available context as no relevant documents were found.",
                has_sufficient_context=False,
                sources_used=[],
                metadatas_used=[],
            )

        # Use traditional RAG approach - build context and answer directly
        context = build_rag_context(docs=retrieval_result.documents)
        prompt = rag_prompt_template.format(query=subquery.question, context=context)
        messages = [{"role": "user", "content": prompt}]

        answer_buffer = ""

        if "st" in globals():
            st.write(f"**Answering:** {subquery.question}")
            answer_placeholder = st.empty()

            for token in self.chat_function(
                model_name=self.model_name,
                messages=messages,
                provider=self.selected_provider,
            ):
                answer_buffer += token
                answer_placeholder.markdown(answer_buffer)
        else:
            for token in self.chat_function(
                model_name=self.model_name,
                messages=messages,
                provider=self.selected_provider,
            ):
                answer_buffer += token

        # Traditional RAG assumes the context is workable if documents exist
        has_sufficient_context = len(retrieval_result.documents) > 0

        return SubQueryAnswer(
            subquery_id=subquery.id,
            question=subquery.question,
            answer=answer_buffer,
            has_sufficient_context=has_sufficient_context,
            sources_used=retrieval_result.documents,
            metadatas_used=retrieval_result.metadatas,
        )

    def execute_multi_retrieval_with_streaming(
        self, subqueries: List[SubQuery], build_rag_context, rag_prompt_template: str
    ) -> Tuple[List[SubQueryAnswer], Dict[str, RetrievalResult]]:
        """Execute subqueries with traditional RAG for each independent question"""

        results = {}
        subquery_answers = []

        # Separate independent and dependent subqueries
        independent_subqueries = [sq for sq in subqueries if sq.is_independent]
        dependent_subqueries = [sq for sq in subqueries if not sq.is_independent]

        # Process independent subqueries using traditional RAG approach
        if "st" in globals():
            st.subheader("🔍 Processing Independent Questions with Traditional RAG")

        for subquery in sorted(independent_subqueries, key=lambda sq: sq.priority):
            if "st" in globals():
                with st.container():
                    st.write(f"**Question {subquery.id}:** {subquery.question}")

                    # Retrieve context
                    with st.spinner(f"Retrieving context for {subquery.id}..."):
                        result = self.execute_subquery(subquery)
                        results[subquery.id] = result

                    # Answer using traditional RAG (more lenient)
                    answer = self.answer_subquery_traditional_rag(
                        subquery, result, build_rag_context, rag_prompt_template
                    )
                    subquery_answers.append(answer)

                    # Show sources
                    if answer.sources_used:
                        with st.expander(f"📚 Sources for {subquery.id}"):
                            for i, (doc, meta) in enumerate(
                                zip(answer.sources_used[:3], answer.metadatas_used[:3])
                            ):
                                st.write(f"**Source {i+1}:** {doc}...")
                                if meta:
                                    st.write(f"*Metadata:* file: {meta['title']}")
                                    st.write(
                                        f"*Metadata:* page range: {meta['page-range']}"
                                    )

                    st.divider()
            else:
                result = self.execute_subquery(subquery)
                results[subquery.id] = result
                answer = self.answer_subquery_traditional_rag(
                    subquery, result, build_rag_context, rag_prompt_template
                )
                subquery_answers.append(answer)

        # Process dependent subqueries (if any)
        if dependent_subqueries:
            if "st" in globals():
                st.subheader("🔗 Processing Dependent Questions")

            for subquery in sorted(dependent_subqueries, key=lambda sq: sq.priority):
                # Check dependencies
                if subquery.dependencies:
                    for dep_id in subquery.dependencies:
                        if dep_id not in results:
                            if "st" in globals():
                                st.warning(
                                    f"Dependency {dep_id} not yet resolved for {subquery.id}"
                                )

                # Retrieve additional context if needed
                result = self.execute_subquery(subquery)
                results[subquery.id] = result

                # For dependent queries, combine context with related subqueries
                combined_docs = result.documents
                combined_metas = result.metadatas

                # Add context from dependencies
                for dep_id in subquery.dependencies:
                    if dep_id in results:
                        dep_result = results[dep_id]
                        combined_docs.extend(dep_result.documents)
                        combined_metas.extend(dep_result.metadatas)

                # Create a modified result with combined context
                combined_result = RetrievalResult(
                    subquery_id=result.subquery_id,
                    question=result.question,
                    documents=combined_docs[:8],  # Limit total
                    metadatas=combined_metas[:8],
                    scores=result.scores,
                    strategy_used=result.strategy_used + "_with_dependencies",
                )

                if "st" in globals():
                    st.write(f"**Question {subquery.id}:** {subquery.question}")

                # Use traditional RAG for dependent questions too
                answer = self.answer_subquery_traditional_rag(
                    subquery, combined_result, build_rag_context, rag_prompt_template
                )
                subquery_answers.append(answer)

                if "st" in globals() and answer.sources_used:
                    with st.expander(f"📚 Sources for {subquery.id}"):
                        for i, (doc, meta) in enumerate(
                            zip(answer.sources_used[:3], answer.metadatas_used[:3])
                        ):
                            st.write(f"**Source {i+1}:** {doc[:200]}...")
                            if meta:
                                st.write(f"*Metadata:* {meta}")
                    st.divider()

        return subquery_answers, results

    def synthesize_final_answer(
        self, original_query: str, subquery_answers: List[SubQueryAnswer]
    ) -> str:
        """Synthesize all subquery answers into a final comprehensive answer"""

        if len(subquery_answers) == 1:
            return subquery_answers[0].answer

        # Prepare synthesis prompt
        answers_text = []
        for answer in subquery_answers:
            status = "✅" if answer.has_sufficient_context else "⚠️"
            answers_text.append(
                f"{status} **Q: {answer.question}**\nA: {answer.answer}\n"
            )

        synthesis_prompt = f"""
        Original question: "{original_query}"
        
        I have answered the following sub-questions:
        
        {chr(10).join(answers_text)}
        
        Please provide a comprehensive, well-structured answer that synthesizes all the above information to fully address the original question. 
        
        If some sub-questions couldn't be answered due to insufficient context, acknowledge this in your final response.
        
        Structure your response clearly and make sure it directly addresses the original question.
        """

        messages = [{"role": "user", "content": synthesis_prompt}]

        final_answer_buffer = ""

        if "st" in globals():
            st.subheader("🎯 Final Comprehensive Answer")
            ##
            # Show thinking process
            thinking_placeholder = st.empty()
            thinking_buffer = ""
            for token in self.chat_function(
                model_name=self.model_name,
                messages=messages,
                provider=self.selected_provider,
            ):
                thinking_buffer += token
                final_answer_buffer += token
                thinking_placeholder.markdown(thinking_buffer)

        else:
            for token in self.chat_function(
                model_name=self.model_name,
                messages=messages,
                provider=self.selected_provider,
            ):
                final_answer_buffer += token

        return final_answer_buffer


def create_multi_retrieval_engine(
    collection, selected_provider, chat_function=None, model_name=None
):
    """
    Factory function to create a MultiRetrievalEngine instance.
    This helps avoid session state and caching issues across different pages.
    """
    try:
        from core_utils.hybrid_search import HybridSearch
        from llm_provider_tools import get_default_llm

        # Use provided parameters or get defaults
        if chat_function is None:
            chat_function = get_chat_response_stream
        if model_name is None:
            model_name = get_default_llm(selected_provider=selected_provider)

        # Create hybrid searcher
        hybrid_searcher = HybridSearch(collection, alpha=0.7)

        # Create and return the engine
        return MultiRetrievalEngine(
            chat_function=chat_function,
            model_name=model_name,
            hybrid_searcher=hybrid_searcher,
            selected_provider=selected_provider,
        )
    except ImportError as e:
        if "st" in globals():
            st.error(f"Failed to create MultiRetrievalEngine: {e}")
        raise


def display_multi_retrieval_results(
    subqueries: List[SubQuery],
    retrieval_results: Dict[str, RetrievalResult],
    detection_info: Dict[str, Any],
):
    """Display results from multi-retrieval execution"""
    with st.expander("🔍 Multi-Retrieval Process", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Query Detection")
            st.json(detection_info)

        with col2:
            st.subheader("Subquery Execution")
            for subquery in subqueries:
                if subquery.id in retrieval_results:
                    result = retrieval_results[subquery.id]
                    status_icon = "✅" if result.documents else "❌"
                    independence_icon = "🔗" if not subquery.is_independent else "🔄"
                    st.write(
                        f"{status_icon}{independence_icon} **{subquery.id}**: {subquery.question}"
                    )
                    st.write(
                        f"   Strategy: {result.strategy_used}, Results: {len(result.documents)}"
                    )
                else:
                    st.write(f"❌ **{subquery.id}**: {subquery.question}")
                    st.write("   Status: Failed to execute")

        # Show retrieval statistics
        st.subheader("Retrieval Statistics")
        total_docs = sum(len(result.documents) for result in retrieval_results.values())
        unique_strategies = set(
            result.strategy_used for result in retrieval_results.values()
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents Retrieved", total_docs)
        with col2:
            st.metric("Subqueries Executed", len(retrieval_results))
        with col3:
            st.metric("Strategies Used", len(unique_strategies))

        # Strategy breakdown
        strategy_counts = {}
        for result in retrieval_results.values():
            strategy_counts[result.strategy_used] = (
                strategy_counts.get(result.strategy_used, 0) + 1
            )

        st.write("**Strategy Distribution:**")
        for strategy, count in strategy_counts.items():
            st.write(f"- {strategy}: {count} subqueries")


def agentic_query_processing(
    query,
    selected_provider: str,
    rerank_results,
    chat_function,
    build_rag_context,
    llm_model,
    view_sources,
    session_key: str,
    multi_retrieval_engine,
    hybrid_searcher,
    get_embedding,
    enable_multi_retrieval=False,
    show_retrieval_details=False,
    show_decomposition=False,
    max_subqueries=5,
    default_strategy="hybrid",
    rag_prompt_template: str = RAG_PROMPT_TEMPLATE,
):
    subqueries = []
    detection_info = {}

    if enable_multi_retrieval:
        # Step 1: Detect multiple questions
        with st.spinner("Detecting multiple questions..."):
            detection_info = multi_retrieval_engine.detect_multiple_questions(query)

        if show_decomposition:
            with st.expander("📝 Query Detection Results"):
                st.json(detection_info)

        # Step 2: Decompose query if needed
        if detection_info.get("requires_decomposition", False):
            with st.spinner("Decomposing query into subqueries..."):
                subqueries = multi_retrieval_engine.decompose_query(
                    query, detection_info
                )
                # Limit subqueries
                subqueries = subqueries[:max_subqueries]

            if show_decomposition:
                with st.expander("🔧 Query Decomposition"):
                    st.write(f"**Original Query:** {query}")
                    st.write(f"**Decomposed into {len(subqueries)} subqueries:**")
                    for sq in subqueries:
                        independence_status = (
                            "Independent" if sq.is_independent else "Dependent"
                        )
                        st.write(f"- **{sq.id}**: {sq.question}")
                        st.write(
                            f"  Type: {sq.query_type}, Priority: {sq.priority}, Strategy: {sq.search_strategy}, Status: {independence_status}"
                        )
        else:
            # Single query - create one subquery
            subqueries = [
                SubQuery(
                    id="subq_1",
                    question=query,
                    query_type="simple",
                    priority=1,
                    parent_query=query,
                    search_strategy=default_strategy,
                    is_independent=True,
                )
            ]

        # Step 3: Execute multi-retrieval with streaming
        with st.chat_message("assistant"):
            st.write("Processing your multi-part question...")

            subquery_answers, retrieval_results = (
                multi_retrieval_engine.execute_multi_retrieval_with_streaming(
                    subqueries, build_rag_context, rag_prompt_template
                )
            )

            # Synthesize final answer if multiple subqueries
            if len(subqueries) > 1:
                final_answer = multi_retrieval_engine.synthesize_final_answer(
                    query, subquery_answers
                )
            else:
                final_answer = (
                    subquery_answers[0].answer
                    if subquery_answers
                    else "No answer could be generated."
                )

        # Display multi-retrieval results
        if show_retrieval_details and len(subqueries) > 1:
            display_multi_retrieval_results(
                subqueries, retrieval_results, detection_info
            )

        # Store final answer in session
        st.session_state[session_key]["messages"].append(
            {"role": "user", "content": query}
        )
        st.session_state[session_key]["messages"].append(
            {"role": "assistant", "content": final_answer}
        )

        # Store retrieval info in session
        st.session_state[session_key]["retrieval_history"].append(
            {
                "query": query,
                "subqueries": subqueries,
                "detection_info": detection_info,
                "subquery_answers": subquery_answers,
                "results_count": sum(
                    len(result.documents) for result in retrieval_results.values()
                ),
            }
        )

        # Display sources from all subqueries
        all_sources = []
        all_metadatas = []
        for answer in subquery_answers:
            if answer.sources_used:
                all_sources.extend(answer.sources_used[:2])  # Limit per subquery
                all_metadatas.extend(answer.metadatas_used[:2])

        if all_sources:
            with st.expander("🔍 All Sources Used"):
                view_sources(relevant_docs=all_sources, metadatas=all_metadatas)

    else:
        # Traditional RAG approach
        query_embedding = get_embedding(input=query, provider=selected_provider)
        hybrid_results = hybrid_searcher.hybrid_search(
            query=query, query_embedding=query_embedding, k=5
        )

        chroma_format_results = {
            "documents": [hybrid_results["documents"]],
            "ids": [hybrid_results["ids"]],
            "metadatas": [hybrid_results["metadatas"]],
            "distances": [[1 - score for score in hybrid_results["scores"]]],
        }

        reranked_data = rerank_results(
            results=chroma_format_results,
            provider=selected_provider,
            query=query,
            top_rerank=3,
        )

        reranked_docs = reranked_data.get(
            "reranked_docs", hybrid_results["documents"][:3]
        )
        reranked_metadatas = reranked_data.get(
            "reranked_metadatas", hybrid_results["metadatas"][:3]
        )

        # Generate response using traditional approach
        context = build_rag_context(docs=reranked_docs)
        prompt = rag_prompt_template.format(query=query, context=context)

        st.session_state[session_key]["messages"].append(
            {"role": "user", "content": prompt}
        )

        # Stream response
        with st.chat_message("assistant"):
            full_resp = ""
            placeholder = st.empty()

            for token in chat_function(
                model_name=llm_model,
                messages=st.session_state[session_key]["messages"],
                provider=selected_provider,
            ):
                full_resp += token
                placeholder.write(full_resp)

        # Store response
        st.session_state[session_key]["messages"].append(
            {"role": "assistant", "content": full_resp}
        )
        st.session_state[session_key]["messages"][-2] = {
            "role": "user",
            "content": query,
        }

        # Display sources
        with st.expander("🔍 View Sources"):
            view_sources(relevant_docs=reranked_docs, metadatas=reranked_metadatas)

    # Display multi-retrieval statistics
    if enable_multi_retrieval and show_retrieval_details:
        with st.expander("📊 Multi-Retrieval Statistics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                if enable_multi_retrieval:
                    total_docs_used = sum(
                        len(answer.sources_used) for answer in subquery_answers
                    )
                    st.metric("Total Documents Used", total_docs_used)
                else:
                    st.metric("Total Documents Used", len(reranked_docs))
            with col2:
                if enable_multi_retrieval:
                    answered_questions = sum(
                        1
                        for answer in subquery_answers
                        if answer.answer
                        and "cannot answer" not in answer.answer.lower()
                    )
                    st.metric(
                        "Questions Answered",
                        f"{answered_questions}/{len(subquery_answers)}",
                    )
                else:
                    st.metric("Query Type", "Single Query")
            with col3:
                complexity = detection_info.get("complexity_score", 0)
                st.metric("Complexity Score", f"{complexity}/10")

                if enable_multi_retrieval and len(subqueries) > 1:
                    independent_count = sum(1 for sq in subqueries if sq.is_independent)
                    st.write(
                        f"Independent: {independent_count}, Dependent: {len(subqueries) - independent_count}"
                    )
