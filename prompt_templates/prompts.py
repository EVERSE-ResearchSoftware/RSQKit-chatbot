ALL_RAG_TEMPLATES = {
    "0": """You are highly trusted Q&A system answering questions from a human during a convesation. Below there is a CONTEXT and a QUESTION.
CONTEXT:
{context}

QUESTION: {query}

Answer the QUESTION using only the given CONTEXT and the previous messages as knowledge. If the context doesn't contain enough information, say "I cannot answer based on the given context.
Always answer in the SAME language as the QUESTION
Answer:""",
    "1": """CONTEXT:
{context}

QUESTION: {query}

Answer the QUESTION using only the given CONTEXT as knowledge. If the context doesn't contain enough information, say "I cannot answer based on the given context."
Always answer in the SAME language as the QUESTION
Answer:""",
    "2": """Given the following conversation, relevant context, and a follow up question, reply with an answer to the current question the user is asking. Return only your response to the question given the above information following the users instructions as needed.
CONTEXT:
{context}

QUESTION: {query}
Answer:""",
    "3": "Given the following conversation, relevant context, and a follow up question, reply with an answer to the current question the user is asking. Return only your response to the question given the above information following the users instructions as needed.",
}

RAG_PROMPT_TEMPLATE = ALL_RAG_TEMPLATES["1"]
RAG_SYSTEM_PROMPT = ALL_RAG_TEMPLATES["3"]

ALL_TASK_TEMPLATES = {
    "mermaid": """Can you generate a Mermaid graph from this sketch? To ensure valid code, make sure that text inside boxes follows the format `letter{…}`. For example `B{Some text}`.""",
    "latex": "Generate latex code for the above picture and render it below.",
}
