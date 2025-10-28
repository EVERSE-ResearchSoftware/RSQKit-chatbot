from dataclasses import dataclass
from typing import List, Dict


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
