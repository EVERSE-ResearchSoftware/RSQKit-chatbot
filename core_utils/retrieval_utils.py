import json
import re
from typing import List, Dict, Any, Tuple, Optional
import streamlit as st
from llm_provider_tools import get_embedding, rerank_results
from prompt_templates.prompts import RAG_PROMPT_TEMPLATE, RAG_SYSTEM_PROMPT
from prompt_templates.prompt_builder import synthesize_prompt
from llms.openai_interface import get_chat_response_stream
from core_utils.hybrid_search import HybridSearch
from core_utils.data_types.retrieval_type import (
    RetrievalResult,
    SubQuery,
    SubQueryAnswer,
)
from prompt_templates.prompt_builder import (
    build_decomposition_prompt,
    build_detection_prompt,
    build_followup_prompt,
    build_synthesis_prompt,
)
from ui.custom_display import generate_sources_html
SOURCES_HTML = "sources_html"

# ============================================================================
# CACHED UTILITIES - Reduce redundant computations
# ============================================================================

@st.cache_data(show_spinner=False, ttl=300, max_entries=100)
def _cached_context_build(docs_tuple: tuple, meta_str: str, builder_name: str) -> str:
    """Lightweight cache for context building - uses tuples for hashability"""
    # This is a placeholder - actual build_rag_context is passed dynamically
    # Cache key is based on docs + metadata signature
    return ""  # Will be overridden in actual call


def _stream_response_efficiently(chat_function, model_name, messages, provider, label: str = "Processing"):
    """Unified streaming function with single placeholder - Claude-like experience"""
    placeholder = st.empty()
    buffer = ""
    
    with placeholder.container():
        st.markdown(f"**{label}...**")
        response_area = st.empty()
        
        for token in chat_function(
            model_name=model_name,
            messages=messages,
            provider=provider,
        ):
            buffer += token
            response_area.markdown(buffer)
    
    return buffer


def rerank_hybrid_results(
    query,
    hybrid_results,
    selected_provider,
    top_rerank: int = 3,
    reranking_function=rerank_results,
):
    """Optimized reranking - no changes needed, already efficient"""
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
    def __init__(
        self,
        chat_function,
        model_name,
        hybrid_searcher: "HybridSearch",
        selected_provider,
    ):
        self.chat_function = chat_function
        self.model_name = model_name
        self.hybrid_searcher = hybrid_searcher
        self.selected_provider = selected_provider
        self.collection = hybrid_searcher.collection if hybrid_searcher else None

    def detect_multiple_questions(self, query: str) -> Dict[str, Any]:
        """Streamlined detection with minimal UI"""
        detection_prompt = build_detection_prompt(query)
        messages = [{"role": "user", "content": detection_prompt}]

        try:
            response = _stream_response_efficiently(
                self.chat_function,
                self.model_name,
                messages,
                self.selected_provider,
                label="🤔 Analyzing query"
            )
        except Exception:
            response = "".join(self.chat_function(
                model_name=self.model_name,
                messages=messages,
                provider=self.selected_provider,
            ))

        try:
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
        except:
            # Fast fallback heuristic
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
                "reasoning": "Heuristic detection",
            }

    def decompose_query(self, query: str, detection_info: Dict) -> List[SubQuery]:
        """Streamlined decomposition"""
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

        decomposition_prompt = build_decomposition_prompt(query, detection_info)
        messages = [{"role": "user", "content": decomposition_prompt}]

        try:
            response = _stream_response_efficiently(
                self.chat_function,
                self.model_name,
                messages,
                self.selected_provider,
                label="🔧 Breaking down query"
            )
        except Exception:
            response = "".join(self.chat_function(
                model_name=self.model_name,
                messages=messages,
                provider=self.selected_provider,
            ))

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

        except Exception:
            # Fast fallback
            parts = re.split(r"[\?]|(?:\band\s)|(?:\balso\s)", query)
            return [
                SubQuery(
                    id=f"subq_{i+1}",
                    question=part.strip() + ("?" if not part.strip().endswith("?") else ""),
                    query_type="factual",
                    priority=i + 1,
                    parent_query=query,
                    is_independent=True,
                )
                for i, part in enumerate(parts) if part.strip()
            ]

    def execute_subquery(self, subquery: SubQuery, **kwargs) -> RetrievalResult:
        """Simplified subquery execution"""
        retrieval_k = kwargs.get("retrieval_k", 5)
        top_rerank = kwargs.get("top_rerank", 3)

        try:
            if subquery.search_strategy == "multi_step":
                return self._multi_step_retrieval(subquery, top_rerank=top_rerank, retrieval_k=retrieval_k)
            elif subquery.search_strategy == "semantic":
                return self._semantic_retrieval(subquery, top_rerank=top_rerank, retrieval_k=retrieval_k)
            else:
                return self._hybrid_retrieval(subquery, top_rerank=top_rerank, retrieval_k=retrieval_k)
        except Exception as e:
            st.error(f"⚠️ Error retrieving for {subquery.id}: {str(e)[:100]}")
            return RetrievalResult(
                subquery_id=subquery.id,
                question=subquery.question,
                documents=[],
                metadatas=[],
                scores=[],
                strategy_used="error",
            )

    def _hybrid_retrieval(
        self, subquery: SubQuery, retrieval_k: int = 5, top_rerank: int = 3
    ) -> RetrievalResult:
        """Optimized hybrid retrieval"""
        query_embedding = get_embedding(
            input=subquery.question, provider=self.selected_provider
        )
        
        hybrid_results = self.hybrid_searcher.hybrid_search(
            query=subquery.question,
            query_embedding=query_embedding,
            retrieval_k=retrieval_k,
        )
        
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
        self, subquery: SubQuery, retrieval_k: int = 5, top_rerank: int = 3
    ) -> RetrievalResult:
        """Optimized semantic retrieval"""
        query_embedding = get_embedding(
            input=subquery.question, provider=self.selected_provider
        )

        results = self.hybrid_searcher.hybrid_search(
            query=subquery.question,
            query_embedding=query_embedding,
            retrieval_k=retrieval_k,
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
        self, subquery: SubQuery, retrieval_k: int = 5, top_rerank: int = 3
    ) -> RetrievalResult:
        """Optimized multi-step retrieval"""
        initial_result = self._hybrid_retrieval(subquery, top_rerank=top_rerank, retrieval_k=retrieval_k)

        if not initial_result.documents:
            return initial_result

        followup_prompt = build_followup_prompt(subquery.question, initial_result)
        messages = [{"role": "user", "content": followup_prompt}]
        
        refined_query = "".join(self.chat_function(
            model_name=self.model_name,
            messages=messages,
            provider=self.selected_provider,
        ))

        refined_embedding = get_embedding(
            input=refined_query.strip(), provider=self.selected_provider
        )
        
        refined_results = self.hybrid_searcher.hybrid_search(
            query=refined_query.strip(),
            query_embedding=refined_embedding,
            retrieval_k=retrieval_k,
        )
        
        reranked_refined_results = rerank_hybrid_results(
            query=refined_query.strip(),
            hybrid_results=refined_results,
            top_rerank=top_rerank,
            selected_provider=self.selected_provider,
        )

        combined_docs = initial_result.documents + reranked_refined_results["reranked_docs"]
        combined_metas = initial_result.metadatas + reranked_refined_results["reranked_metadatas"]
        combined_scores = initial_result.scores + reranked_refined_results["reranked_scores"]

        return RetrievalResult(
            subquery_id=subquery.id,
            question=subquery.question,
            documents=combined_docs,
            metadatas=combined_metas,
            scores=combined_scores,
            strategy_used="multi_step",
        )

    def answer_subquery_traditional_rag(
        self,
        subquery: SubQuery,
        retrieval_result: RetrievalResult,
        build_rag_context,
        rag_prompt_template: str,
        augment_chunk: bool = True,
        answer_per_chunk: bool = True,
        **kwargs,
    ) -> SubQueryAnswer:
        """Optimized RAG answering with streamlined UI"""
        display_context = kwargs.get("display_context", False)

        if not retrieval_result.documents:
            return SubQueryAnswer(
                subquery_id=subquery.id,
                question=subquery.question,
                answer="I cannot answer this question - no relevant documents found.",
                has_sufficient_context=False,
                sources_used=[],
                metadatas_used=[],
            )

        # Show context only if requested (collapsed)
        if display_context:
            pre_context = build_rag_context(
                documents=retrieval_result.documents, 
                metadatas=retrieval_result.metadatas
            )
            with st.expander("📄 View Retrieved Context", expanded=False):
                st.code(pre_context, language="text")

        # Route to appropriate processing method
        if augment_chunk:
            return self._process_per_chunk_and_synthesize(
                subquery=subquery,
                retrieval_result=retrieval_result,
                build_rag_context=build_rag_context,
                rag_prompt_template=rag_prompt_template,
                use_augmented=True,
                display_context=display_context,
            )
        elif answer_per_chunk:
            return self._process_per_chunk_and_synthesize(
                subquery=subquery,
                retrieval_result=retrieval_result,
                build_rag_context=build_rag_context,
                rag_prompt_template=rag_prompt_template,
                use_augmented=False,
                display_context=display_context,
            )
        else:
            # Single-shot answer - fastest path
            context = build_rag_context(
                documents=[d.get("augmented_text", "") for d in retrieval_result.metadatas],
            )
            
            if display_context:
                with st.expander("📝 Augmented Context", expanded=False):
                    st.code(context, language="text")

            prompt = rag_prompt_template.format(query=subquery.question, context=context)
            messages = [{"role": "user", "content": prompt}]
            
            answer_buffer = _stream_response_efficiently(
                self.chat_function,
                self.model_name,
                messages,
                self.selected_provider,
                label=f"💡 Answering: {subquery.question[:60]}..."
            )

            return SubQueryAnswer(
                subquery_id=subquery.id,
                question=subquery.question,
                answer=answer_buffer,
                has_sufficient_context=len(retrieval_result.documents) > 0,
                sources_used=retrieval_result.documents,
                metadatas_used=retrieval_result.metadatas,
            )

    def _process_per_chunk_and_synthesize(
        self,
        subquery: SubQuery,
        retrieval_result: RetrievalResult,
        build_rag_context,
        rag_prompt_template: str,
        use_augmented: bool,
        display_context: bool,
    ) -> SubQueryAnswer:
        """Optimized chunk processing with better UX feedback"""
        individual_answers = []

        # Determine texts to process
        if use_augmented:
            texts_to_process = [
                (i, metadata.get("augmented_text", ""))
                for i, metadata in enumerate(retrieval_result.metadatas)
            ]
            context_label = "Augmented Context"
        else:
            texts_to_process = [
                (i, doc) for i, doc in enumerate(retrieval_result.documents)
            ]
            context_label = "Document"

        # Single progress indicator
        progress_text = st.empty()
        progress_text.markdown(f"**Processing {len(texts_to_process)} {context_label.lower()}s...**")

        # Process each chunk efficiently
        for i, text in texts_to_process:
            if not text.strip():
                continue

            individual_context = build_rag_context(
                documents=[text], 
                metadatas=retrieval_result.metadatas
            )

            if display_context:
                with st.expander(f"{context_label} {i+1}", expanded=False):
                    st.code(individual_context, language="text")

            # Update progress
            progress_text.markdown(f"**Processing {context_label} {i+1}/{len(texts_to_process)}**")

            prompt = rag_prompt_template.format(
                query=subquery.question, 
                context=individual_context
            )
            messages = [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            individual_answer = _stream_response_efficiently(
                self.chat_function,
                self.model_name,
                messages,
                self.selected_provider,
                label=f"{context_label} {i+1}"
            )

            if individual_answer.strip():
                individual_answers.append(f"{context_label} {i+1}: {individual_answer.strip()}")

        # Clear progress
        progress_text.empty()

        if not individual_answers:
            return SubQueryAnswer(
                subquery_id=subquery.id,
                question=subquery.question,
                answer=f"No valid contexts found to answer the question.",
                has_sufficient_context=False,
                sources_used=retrieval_result.documents,
                metadatas_used=retrieval_result.metadatas,
            )

        # Synthesize efficiently
        synthesis_prompt = f"""Based on the following individual answers to "{subquery.question}", 
provide a comprehensive, synthesized response. Remove redundancy and create a coherent answer.

Individual Answers:
{chr(10).join(individual_answers)}

Question: {subquery.question}

Synthesized Answer:"""

        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": synthesis_prompt},
        ]

        final_answer = _stream_response_efficiently(
            self.chat_function,
            self.model_name,
            messages,
            self.selected_provider,
            label="📊 Synthesizing insights"
        )

        # Show final answer prominently
        st.markdown("**✅ Final Answer:**")
        st.markdown(final_answer)

        return SubQueryAnswer(
            subquery_id=subquery.id,
            question=subquery.question,
            answer=final_answer,
            has_sufficient_context=len(individual_answers) > 0,
            sources_used=retrieval_result.documents,
            metadatas_used=retrieval_result.metadatas,
        )

    def execute_multi_retrieval_with_streaming(
        self,
        subqueries: List[SubQuery],
        build_rag_context,
        rag_prompt_template: str,
        **kwargs,
    ) -> Tuple[List[SubQueryAnswer], Dict[str, RetrievalResult]]:
        """Optimized multi-retrieval execution"""
        retrieval_k = kwargs.get("retrieval_k", 5)
        top_rerank = kwargs.get("top_rerank", 3)
        results = {}
        subquery_answers = []

        independent_subqueries = [sq for sq in subqueries if sq.is_independent]
        dependent_subqueries = [sq for sq in subqueries if not sq.is_independent]

        if independent_subqueries:
            st.markdown("**🔍 Processing Questions**")

        for subquery in sorted(independent_subqueries, key=lambda sq: sq.priority):
            with st.container():
                st.markdown(f"**Q{subquery.id[-1]}:** {subquery.question}")

                # Retrieve with minimal UI
                with st.spinner("Retrieving..."):
                    result = self.execute_subquery(
                        subquery, retrieval_k=retrieval_k, top_rerank=top_rerank
                    )
                    results[subquery.id] = result

                # Answer
                context_answer_chunk_params = {
                    "augment_chunk": st.session_state.get("augment_chunk", True),
                    "answer_per_chunk": st.session_state.get("answer_per_chunk", True),
                }
                answer = self.answer_subquery_traditional_rag(
                    subquery,
                    result,
                    build_rag_context,
                    rag_prompt_template,
                    **context_answer_chunk_params,
                )
                subquery_answers.append(answer)

                # Collapsed sources
                if answer.sources_used:
                    with st.expander(f"📚 Sources ({len(answer.sources_used)})", expanded=False):
                        for i, (doc, meta) in enumerate(zip(answer.sources_used, answer.metadatas_used)):
                            st.text(f"Source {i+1}: {doc[:150]}...")
                            if meta:
                                st.caption(f"📄 {meta.get('title', 'Unknown')} | Pages: {meta.get('page-range', 'N/A')}")

                st.divider()

        # Process dependent subqueries (if any)
        if dependent_subqueries:
            st.markdown("**🔗 Processing Dependent Questions**")

            for subquery in sorted(dependent_subqueries, key=lambda sq: sq.priority):
                result = self.execute_subquery(subquery, top_rerank=top_rerank, retrieval_k=retrieval_k)
                results[subquery.id] = result

                # Combine with dependencies
                combined_docs = result.documents
                combined_metas = result.metadatas

                for dep_id in subquery.dependencies:
                    if dep_id in results:
                        dep_result = results[dep_id]
                        combined_docs.extend(dep_result.documents)
                        combined_metas.extend(dep_result.metadatas)

                combined_result = RetrievalResult(
                    subquery_id=result.subquery_id,
                    question=result.question,
                    documents=combined_docs,
                    metadatas=combined_metas,
                    scores=result.scores,
                    strategy_used=result.strategy_used + "_with_deps",
                )

                st.markdown(f"**Q{subquery.id[-1]}:** {subquery.question}")
                answer = self.answer_subquery_traditional_rag(
                    subquery, combined_result, build_rag_context, rag_prompt_template
                )
                subquery_answers.append(answer)

                if answer.sources_used:
                    with st.expander(f"📚 Sources ({len(answer.sources_used)})", expanded=False):
                        for i, (doc, meta) in enumerate(zip(answer.sources_used, answer.metadatas_used)):
                            st.text(f"{doc[:150]}...")
                st.divider()

        return subquery_answers, results

    def validate_rag_answer_with_chat_history(
        self,
        original_query: Optional[str] = None,
        synthesized_answer: Optional[str] = None,
        chat_history: Optional[List[Dict]] = None,
        **kwargs,
    ):
        """Optimized history validation"""
        recap_prompt = f"Given this query: {original_query}\nThe answer found is: {synthesized_answer}.\nMake a new answer that takes the past conversation into account."
        
        # remove the sources_html for vllm deployed model following openai protocol
        chat_history = [{key: value for key, value in d.items() if key != SOURCES_HTML} for d in chat_history]
        print(chat_history)
        chat_history.append({"role": "user", "content": recap_prompt})
        
        temperature = st.session_state.get("temperature", 0.7)
        
        updated_buffer = _stream_response_efficiently(
            self.chat_function,
            self.model_name,
            chat_history,
            self.selected_provider,
            label="🧠 Refining with chat history"
        )

        return updated_buffer

    def synthesize_final_answer(
        self, original_query: str, subquery_answers: List[SubQueryAnswer], **kwargs
    ) -> str:
        """Optimized final synthesis"""
        history = kwargs.get("conversation_history", []).copy()
        enable_history = kwargs.get("enable_history", False)

        if len(subquery_answers) == 1:
            if not enable_history or not history:
                return subquery_answers[0].answer
            else:
                st.markdown("**🧠 Checking chat history...**")
                return self.validate_rag_answer_with_chat_history(
                    original_query=original_query,
                    synthesized_answer=subquery_answers[0].answer,
                    chat_history=history,
                )

        # Multi-answer synthesis
        synthesis_prompt = build_synthesis_prompt(original_query, subquery_answers)
        temp_messages = [{"role": "user", "content": synthesis_prompt}]
        
        temperature = st.session_state.get("temperature", 0.7)

        st.markdown("**🎯 Final Comprehensive Answer**")
        
        final_answer_buffer = _stream_response_efficiently(
            self.chat_function,
            self.model_name,
            temp_messages,
            self.selected_provider,
            label="Synthesizing"
        )

        if enable_history and history:
            st.markdown("**🧠 Refining with history...**")
            final_answer_buffer = self.validate_rag_answer_with_chat_history(
                original_query=original_query,
                synthesized_answer=final_answer_buffer,
                chat_history=history,
            )

        return final_answer_buffer


# ============================================================================
# FACTORY & UTILITY FUNCTIONS
# ============================================================================

def create_multi_retrieval_engine(
    collection, selected_provider, chat_function=None, model_name=None, **kwargs
):
    """Optimized factory function"""
    from core_utils.hybrid_search import HybridSearch
    from llm_provider_tools import get_default_llm

    if chat_function is None:
        chat_function = get_chat_response_stream
    if model_name is None:
        model_name = get_default_llm(selected_provider=selected_provider)

    alpha = kwargs.get("alpha", 0.7)
    hybrid_searcher = HybridSearch(collection, alpha=alpha)

    return MultiRetrievalEngine(
        chat_function=chat_function,
        model_name=model_name,
        hybrid_searcher=hybrid_searcher,
        selected_provider=selected_provider,
    )


def display_multi_retrieval_results(
    subqueries: List[SubQuery],
    retrieval_results: Dict[str, RetrievalResult],
    detection_info: Dict[str, Any],
):
    """Streamlined results display"""
    with st.expander("📊 Retrieval Details", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        total_docs = sum(len(r.documents) for r in retrieval_results.values())
        unique_strategies = set(r.strategy_used for r in retrieval_results.values())
        
        with col1:
            st.metric("Documents", total_docs)
        with col2:
            st.metric("Subqueries", len(retrieval_results))
        with col3:
            st.metric("Strategies", len(unique_strategies))

        # Compact subquery list
        for subquery in subqueries:
            if subquery.id in retrieval_results:
                result = retrieval_results[subquery.id]
                status = "✅" if result.documents else "❌"
                st.text(f"{status} {subquery.id}: {subquery.question[:60]}... ({len(result.documents)} docs)")


def reply_with_chat_history(
    prompt: str,
    chat_function,
    llm_model: str,
    selected_provider: str,
    use_streamlit: bool = True,
    session_key: Optional[str] = None,
    chat_history: Optional[List[Dict]] = None,
    **kwargs,
):
    """Optimized chat history handling"""
    if use_streamlit:
        st.session_state[session_key]["messages"].append(
            {"role": "user", "content": prompt}
        )

        cleaned_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state[session_key]["messages"]
        ]

        with st.chat_message("assistant"):
            full_resp = _stream_response_efficiently(
                chat_function,
                llm_model,
                cleaned_messages,
                selected_provider,
                label="Responding"
            )

        st.session_state[session_key]["messages"].append(
            {"role": "assistant", "content": full_resp}
        )
    else:
        cleaned_messages = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in chat_history
        ]
        
        full_resp = "".join(chat_function(
            model_name=llm_model,
            messages=cleaned_messages,
            provider=selected_provider,
        ))
        return full_resp


def respond_with_multi_retrieval_rag(
    query: str,
    build_rag_context,
    rag_prompt_template: str,
    retrieval_k: int,
    top_rerank: int,
    multi_retrieval_engine,
    session_key: str,
    enable_history: bool,
    default_strategy: str,
    max_subqueries: int,
    show_retrieval_details: bool,
    show_decomposition: bool,
):
    """Optimized multi-retrieval RAG"""
    # Detect
    detection_info = multi_retrieval_engine.detect_multiple_questions(query)

    if show_decomposition:
        with st.expander("🔍 Query Analysis", expanded=False):
            st.json(detection_info)

    # Decompose
    if detection_info.get("requires_decomposition", False):
        subqueries = multi_retrieval_engine.decompose_query(query, detection_info)
        subqueries = subqueries[:max_subqueries]

        if show_decomposition:
            with st.expander("🔧 Query Breakdown", expanded=False):
                st.markdown(f"**Original:** {query}")
                st.markdown(f"**Split into {len(subqueries)} parts:**")
                for sq in subqueries:
                    st.text(f"• {sq.id}: {sq.question}")
    else:
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

    # Execute with streaming
    with st.chat_message("assistant"):
        subquery_answers, retrieval_results = (
            multi_retrieval_engine.execute_multi_retrieval_with_streaming(
                subqueries=subqueries,
                build_rag_context=build_rag_context,
                rag_prompt_template=rag_prompt_template,
                top_rerank=top_rerank,
                retrieval_k=retrieval_k,
            )
        )

        # Synthesize if needed
        if len(subqueries) >= 1:
            final_answer = multi_retrieval_engine.synthesize_final_answer(
                query,
                subquery_answers,
                conversation_history=st.session_state[session_key]["messages"],
                enable_history=enable_history,
                session_key=session_key,
            )
        else:
            final_answer = "No answer could be generated."

    # Show detailed stats if requested
    if show_retrieval_details and len(subqueries) > 1:
        display_multi_retrieval_results(subqueries, retrieval_results, detection_info)

    # Store in session
    st.session_state[session_key]["messages"].append({"role": "user", "content": query})
    st.session_state[session_key]["messages"].append(
        {"role": "assistant", "content": final_answer}
    )

    # Store retrieval history
    if "retrieval_history" not in st.session_state[session_key]:
        st.session_state[session_key]["retrieval_history"] = []
    
    st.session_state[session_key]["retrieval_history"].append(
        {
            "query": query,
            "subqueries": subqueries,
            "detection_info": detection_info,
            "subquery_answers": subquery_answers,
            "results_count": sum(len(r.documents) for r in retrieval_results.values()),
        }
    )

    # Collect and display sources efficiently
    all_sources = []
    all_metadatas = []
    for answer in subquery_answers:
        if answer.sources_used:
            all_sources.extend(answer.sources_used)
            all_metadatas.extend(answer.metadatas_used)

    if all_sources:
        from ui.chat_history import _wrap_sources_html
        
        sources_html = generate_sources_html(
            relevant_docs=all_sources, 
            metadatas=all_metadatas, 
            link_style="dual"
        )
        st.session_state[session_key]["messages"][-1][SOURCES_HTML] = sources_html

        # Display collapsed sources
        st.markdown(_wrap_sources_html(sources_html), unsafe_allow_html=True)

    return subqueries, subquery_answers


# ============================================================================
# ENHANCED NAIVE RAG (One-shot optimized)
# ============================================================================

from ui.chat_history import _wrap_sources_html

def respond_with_enhanced_naive_rag(
    query: str,
    build_rag_context,
    retrieval_k: int,
    top_rerank: int,
    multi_retrieval_engine: "MultiRetrievalEngine",
    session_key: str,
    enable_history: bool = False,
    answer_per_chunk: bool = False,
    augment_chunk: bool = False,
    rag_prompt_template: str = RAG_PROMPT_TEMPLATE,
    **kwargs,
):
    """Optimized single-shot RAG with minimal overhead"""
    
    formatted_query = SubQuery(
        id="main", 
        question=query, 
        query_type="single", 
        priority=1, 
        parent_query="None"
    )
    
    # Fast retrieval
    with st.spinner("🔍 Retrieving documents..."):
        retrieval_result = multi_retrieval_engine._hybrid_retrieval(
            subquery=formatted_query, 
            retrieval_k=retrieval_k, 
            top_rerank=top_rerank
        )
    
    # Answer with streaming
    with st.chat_message("assistant"):
        pre_answer: SubQueryAnswer = (
            multi_retrieval_engine.answer_subquery_traditional_rag(
                subquery=formatted_query,
                retrieval_result=retrieval_result,
                build_rag_context=build_rag_context,
                answer_per_chunk=answer_per_chunk,
                augment_chunk=augment_chunk,
                rag_prompt_template=rag_prompt_template,
                **kwargs,
            )
        )

        # Store user message first
        st.session_state[session_key]["messages"].append(
            {"role": "user", "content": query}
        )
        
        # Synthesize final answer (with optional history)
        final_answer = multi_retrieval_engine.synthesize_final_answer(
            original_query=query,
            subquery_answers=[pre_answer],
            enable_history=enable_history,
            conversation_history=st.session_state[session_key]["messages"],
        )
        
        # Store assistant message
        st.session_state[session_key]["messages"].append(
            {"role": "assistant", "content": final_answer}
        )

    # Generate and store sources
    sources_html = generate_sources_html(
        retrieval_result.documents, 
        retrieval_result.metadatas, 
        link_style="dual"
    )
    st.session_state[session_key]["messages"][-1][SOURCES_HTML] = sources_html

    # Display collapsed sources
    show_sources = kwargs.get("show_sources", True)
    if show_sources:
        st.markdown(_wrap_sources_html(sources_html), unsafe_allow_html=True)

    return retrieval_result.documents


def show_retrieval_statistics(
    enable_multi_retrieval: bool,
    show_retrieval_details: bool,
    subquery_answers: List,
    subqueries: List,
    reranked_docs: List,
    detection_info: Dict,
):
    """Streamlined statistics display"""
    if not (enable_multi_retrieval and show_retrieval_details):
        return

    with st.expander("📊 Retrieval Statistics", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if enable_multi_retrieval:
                total = sum(len(a.sources_used) for a in subquery_answers)
                st.metric("Documents Used", total)
            else:
                st.metric("Documents Used", len(reranked_docs))
        
        with col2:
            if enable_multi_retrieval:
                answered = sum(
                    1 for a in subquery_answers 
                    if a.answer and "cannot answer" not in a.answer.lower()
                )
                st.metric("Answered", f"{answered}/{len(subquery_answers)}")
            else:
                st.metric("Query Type", "Single")
        
        with col3:
            complexity = detection_info.get("complexity_score", 0)
            st.metric("Complexity", f"{complexity}/10")


def agentic_query_processing(
    query: str,
    selected_provider: str,
    chat_function,
    build_rag_context,
    llm_model: str,
    session_key: str,
    multi_retrieval_engine: 'MultiRetrievalEngine',
    enable_multi_retrieval: bool = False,
    show_retrieval_details: bool = False,
    show_decomposition: bool = False,
    max_subqueries: int = 5,
    default_strategy: str = "hybrid",
    rag_prompt_template: str = RAG_PROMPT_TEMPLATE,
    retrieval_k: int = 10,
    top_rerank: int = 2,
    **kwargs,
):
    """Unified query processing with optimized routing"""
    
    reranked_docs = []
    subqueries = []
    detection_info = {}
    subquery_answers = []
    enable_history = kwargs.get("enable_history", False)

    # Check for chat-only mode (@ prefix)
    if query.startswith("@"):
        reply_with_chat_history(
            prompt=query.lstrip("@").strip(),
            llm_model=llm_model,
            chat_function=chat_function,
            selected_provider=selected_provider,
            use_streamlit=True,
            session_key=session_key,
        )
        return

    # Route to appropriate RAG method
    if enable_multi_retrieval:
        subqueries, subquery_answers = respond_with_multi_retrieval_rag(
            query=query,
            build_rag_context=build_rag_context,
            rag_prompt_template=rag_prompt_template,
            retrieval_k=retrieval_k,
            top_rerank=top_rerank,
            multi_retrieval_engine=multi_retrieval_engine,
            session_key=session_key,
            enable_history=enable_history,
            default_strategy=default_strategy,
            max_subqueries=max_subqueries,
            show_retrieval_details=show_retrieval_details,
            show_decomposition=show_decomposition,
        )
    else:
        reranked_docs = respond_with_enhanced_naive_rag(
            query=query,
            build_rag_context=build_rag_context,
            retrieval_k=retrieval_k,
            top_rerank=top_rerank,
            multi_retrieval_engine=multi_retrieval_engine,
            session_key=session_key,
            rag_prompt_template=rag_prompt_template,
            **kwargs,
        )

    # Show statistics if requested
    show_retrieval_statistics(
        enable_multi_retrieval,
        show_retrieval_details,
        subquery_answers,
        subqueries,
        reranked_docs,
        detection_info,
    )