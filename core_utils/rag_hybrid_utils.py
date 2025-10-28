from core_utils.hybrid_search import HybridSearch
from core_utils.retrieval_utils import create_multi_retrieval_engine
from typing import Callable


def initialize_components(
    collection,
    selected_provider: str,
    chat_function: Callable,
    llm_model: str,
    alpha: int = 0.5,
):
    """Initialize hybrid search and multi-retrieval engine"""
    hybrid_searcher = HybridSearch(collection, alpha=0.5)
    multi_retrieval_engine = create_multi_retrieval_engine(
        collection=collection,
        selected_provider=selected_provider,
        chat_function=chat_function,
        model_name=llm_model,
        alpha=alpha,
    )
    return hybrid_searcher, multi_retrieval_engine
