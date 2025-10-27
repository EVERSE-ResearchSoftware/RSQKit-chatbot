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
    "4": """ 
You are a retrieval‑augmented generation (RAG) chatbot designed to assist the users.
Given the following conversation, relevant context, and a follow up question, reply with an answer to the current question the user is asking. Return only your response to the question given the above information following the users instructions as needed.
All responses must be grounded solely in the retrieved contexts.

### Role & Behavior
- You are a friendly professional support assistant.
- Respond to questions as clearly and concisely as possible.
- Do **not** discuss unrelated topics.
- Do **not** mention that you are an AI, a language model, or that you are “powered by” any technology.
- Do **not** provide personal data or contact details that are not in the retrieved-context.

### Knowledge Limitations
- Use only the information contained in the retrieved documents.  
- If a user’s question cannot be answered with the available documents, politely say:
  > “I’m sorry, I don’t have that information. Please contact the support team.”  
- Do **not** fabricate or infer details that are not explicitly stated in the context.

### Tone
- Keep the tone courteous, concise, and supportive.  

You are ready to assist.
""",
    "lax-system": """You are highly trusted Q&A system answering questions from a human during a convesation.""",
}

RAG_PROMPT_TEMPLATE = ALL_RAG_TEMPLATES["1"]
RAG_SYSTEM_PROMPT = ALL_RAG_TEMPLATES["lax-system"]

ALL_TASK_TEMPLATES = {
    "mermaid": """Can you generate a Mermaid graph from this sketch? To ensure valid code, make sure that text inside boxes follows the format `letter{…}`. For example `B{Some text}`.""",
    "latex": "Generate latex code for the above picture and render it below.",
}


# Prompt Space
MULTI_RETRIEVAL_PROMPT_SPACE = {
    "detection_prompt": """
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
    """,
    "decomposition_prompt": """
    Break down this complex query into specific subqueries that can be searched independently:

    Original Query: "{query}"
    Detection Info: {detection_info}

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
    """,
    "followup_prompt": """
    Based on these initial search results for the question "{query}",
    generate a refined search query to get more specific information:

    Initial Results Preview: {initial_results}

    Provide a refined search query that would get more targeted information:
    """,
    "synthesis_prompt": """
    Original question: "{original_query}"

    I have answered the following sub-questions:

    {subquery_answers}

    Please provide a comprehensive, well-structured answer that synthesizes all the above information to fully address the original question.

    If some sub-questions couldn't be answered due to insufficient context, acknowledge this in your final response.

    Structure your response clearly and make sure it directly addresses the original question.
    Your answer must be based exclusively from the above answers to sub-questions. No further knowledge is required.
    """,
}
