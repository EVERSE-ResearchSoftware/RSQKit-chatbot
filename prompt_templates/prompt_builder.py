from prompt_templates.prompts import MULTI_RETRIEVAL_PROMPT_SPACE
from typing import List, Dict
from core_utils.data_types.retrieval_type import RetrievalResult, SubQueryAnswer
import json


def build_rag_context(documents: list[str], **kwargs):
    metadatas = kwargs.get("metadatas", [])
    if len(metadatas) > 0:
        return ("\n\n" + 10 * "----" + "\n\n").join(
            [
                f"Source {idx + 1}:{doc}\n Metadata: Title: {metadatas[idx].get('title', '')}"
                for idx, doc in enumerate(documents)
            ]
        )
    return ("\n\n" + 10 * "----" + "\n\n").join(
        [f"Source {idx + 1}:{doc}" for idx, doc in enumerate(documents)]
    )


def synthesize_prompt(query: str, answers: list[str]) -> str:
    return f"""
        Original question: "{query}"
        
        I have partial answers to that question :
        
        {chr(10).join(answers)}
        
        Please provide a comprehensive, well-structured answer that synthesizes all the above information to fully address the original question. 
        
        If some answers are incomplete, acknowledge this in your final response.
        
        Structure your response clearly and make sure it directly addresses the original question.
        """


def build_detection_prompt(query: str) -> str:
    prompt = MULTI_RETRIEVAL_PROMPT_SPACE["detection_prompt"]
    return prompt.format(query=query)


def build_decomposition_prompt(query: str, detection_info: Dict) -> str:
    prompt = MULTI_RETRIEVAL_PROMPT_SPACE["decomposition_prompt"]
    return prompt.format(
        query=query, detection_info=json.dumps(detection_info, indent=2)
    )


def build_followup_prompt(query: str, initial_result: "RetrievalResult") -> str:
    prompt = MULTI_RETRIEVAL_PROMPT_SPACE["followup_prompt"]
    initial_results = (
        initial_result.documents[0][:200] if initial_result.documents else "No results"
    )
    return prompt.format(query=query, initial_results=initial_results)


def build_synthesis_prompt(
    original_query: str, subquery_answers: List["SubQueryAnswer"]
) -> str:
    prompt = MULTI_RETRIEVAL_PROMPT_SPACE["synthesis_prompt"]
    answers_text = [
        f"{'✅' if answer.has_sufficient_context else '⚠️'} **Q: {answer.question}**\nA: {answer.answer}\n"
        for answer in subquery_answers
    ]
    return prompt.format(
        original_query=original_query, subquery_answers=chr(10).join(answers_text)
    )
