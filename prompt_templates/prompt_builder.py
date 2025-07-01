def build_rag_context(docs: list[str]):
    return ("\n\n" + 10 * "----" + "\n\n").join(
        [f"Source {idx + 1}:{doc}" for idx, doc in enumerate(docs)]
    )
