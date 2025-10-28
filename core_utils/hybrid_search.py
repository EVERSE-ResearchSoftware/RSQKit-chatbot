import bm25s
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from settings import CUSTOM_STOP_WORDS
import logging




class HybridSearch:
    def __init__(self, chroma_collection, alpha=0.7):
        """
        Initialize hybrid search with ChromaDB collection and BM25 index

        Args:
            chroma_collection: ChromaDB collection object
            alpha: Weight for semantic search (1-alpha will be weight for BM25)
                  alpha=1.0 means only semantic, alpha=0.0 means only BM25
        """
        self.collection = chroma_collection
        self.alpha = alpha
        self.bm25 = None
        self.doc_ids = []
        self.id_to_doc = {}
        self.build_bm25_index()

    def build_bm25_index(self):
        """Build BM25 index from current ChromaDB collection documents"""

        # Get all documents from ChromaDB
        all_results = self.collection.get()
        documents = all_results["documents"]
        ids = all_results["ids"]

        if not documents:
            raise ValueError("No documents found in ChromaDB collection")

        # Prepare corpus for BM25
        self.doc_ids = ids
        self.id_to_doc = {doc_id: doc for doc_id, doc in zip(ids, documents)}

        # Tokenize documents (simple whitespace tokenization

        # Create BM25 index
        self.bm25 = bm25s.BM25(corpus=documents)
        self.bm25.index(bm25s.tokenize(documents))

        # print(
        #     f"BM25 index built from ChromaDB collection. Indexed {len(documents)} documents."
        # )

    def search_bm25(
        self, query: str, retrieval_k: int = 10
    ) -> Tuple[List[str], List[float]]:
        """Search using BM25 and return document IDs and scores"""
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_bm25_index() first.")

        # Tokenize query
        tokenized_query = list(
            bm25s.tokenize(query, stopwords=CUSTOM_STOP_WORDS).vocab.keys()
        )

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top k results
        top_indices = np.argsort(scores)[::-1][:retrieval_k]
        top_scores = scores[top_indices]
        top_doc_ids = [self.doc_ids[i] for i in top_indices]

        return top_doc_ids, top_scores.tolist()

    def search_semantic(
        self, query_embedding: List[float], retrieval_k: int = 10
    ) -> Tuple[List[str], List[float]]:
        """Search using ChromaDB semantic similarity"""
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=retrieval_k
        )

        doc_ids = results["ids"][0]
        # ChromaDB returns distances, convert to similarity scores
        distances = results["distances"][0]
        similarities = [
            1 / (1 + dist)
            for dist in distances  # TODO inspect this with chromadb distance function
        ]  # Convert distance to similarity

        return doc_ids, similarities

    def build_tree(
        self, chroma_results: Dict[str, Any], n_extend: int = 1
        ) -> Dict[str, Any]:
        if n_extend == 0:
            for i, doc in enumerate(chroma_results["documents"]):
                chroma_results["metadatas"][i]["augmented_text"] = doc
            return chroma_results
        
        # Build lookup dictionaries
        id_to_doc = dict(zip(chroma_results["ids"], chroma_results["documents"]))
        id_to_meta = dict(zip(chroma_results["ids"], chroma_results["metadatas"]))
        
        # Helper function to traverse in one direction
        def traverse(start_id, direction_key, n_steps):
            """Returns list of IDs in traversal order"""
            ids = []
            current_id = start_id
            for _ in range(n_steps):
                meta = id_to_meta.get(current_id)
                if not meta or direction_key not in meta:
                    break
                next_id = meta[direction_key]
                ids.append(next_id)
                current_id = next_id
            return ids
        
        # Single pass: collect IDs and cache traversal results
        all_needed_ids = set(chroma_results["ids"])
        traversal_cache = {}
        
        for doc_id in chroma_results["ids"]:
            prev_ids = traverse(doc_id, "previous_id", n_extend)
            next_ids = traverse(doc_id, "next_id", n_extend)
            
            all_needed_ids.update(prev_ids)
            all_needed_ids.update(next_ids)
            
            # Cache the traversal results (reverse prev_ids for correct order)
            traversal_cache[doc_id] = (prev_ids[::-1], next_ids)
        
        # Fetch additional documents
        additional_ids = all_needed_ids - set(chroma_results["ids"])
        if additional_ids:
            try:
                additional_results = self.collection.get(
                    ids=list(additional_ids), include=["documents", "metadatas"]
                )
                id_to_doc.update(zip(additional_results["ids"], additional_results["documents"]))
                id_to_meta.update(zip(additional_results["ids"], additional_results["metadatas"]))
            except Exception as e:
                logging.warning(f"Failed to fetch additional documents: {e}")
        
        # Build augmented text using cached traversals
        for i, doc_id in enumerate(chroma_results["ids"]):
            prev_ids, next_ids = traversal_cache[doc_id]
            
            # Get texts for all IDs (filter out missing ones)
            prev_texts = [id_to_doc[pid] for pid in prev_ids if pid in id_to_doc]
            next_texts = [id_to_doc[nid] for nid in next_ids if nid in id_to_doc]
            
            augmented_text = " ".join(prev_texts + [id_to_doc[doc_id]] + next_texts)
            chroma_results["metadatas"][i]["augmented_text"] = augmented_text
        
        return chroma_results

    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        retrieval_k: int = 10,
        n_extend: int = 1,
    ) -> Dict[str, Any]:
        """
        Perform hybrid search combining semantic and BM25 results

        Args:
            query: Search query string
            query_embedding: Query embedding for semantic search
            retrieval_k: Number of results to return
            n_extend: Number of consecutive previous/next nodes to include in augmented_text

        Returns:
            Dictionary with combined results, scores, metadata (including augmented_text)
        """
        # Get BM25 results
        bm25_ids, bm25_scores = self.search_bm25(
            query, retrieval_k=retrieval_k * 2
        )  # Get more to ensure good coverage

        # Get semantic results
        semantic_ids, semantic_scores = self.search_semantic(
            query_embedding, retrieval_k=retrieval_k * 2
        )

        # Normalize scores to [0, 1] range
        def normalize_scores(scores):
            if not scores or max(scores) == min(scores):
                return [0.0] * len(scores)
            min_score, max_score = min(scores), max(scores)
            return [(score - min_score) / (max_score - min_score) for score in scores]

        norm_bm25_scores = normalize_scores(bm25_scores)
        norm_semantic_scores = normalize_scores(semantic_scores)

        # Combine scores using weighted average
        combined_scores = {}

        # Add BM25 scores
        for doc_id, score in zip(bm25_ids, norm_bm25_scores):
            combined_scores[doc_id] = (1 - self.alpha) * score

        # Add semantic scores
        for doc_id, score in zip(semantic_ids, norm_semantic_scores):
            if doc_id in combined_scores:
                combined_scores[doc_id] += self.alpha * score
            else:
                combined_scores[doc_id] = self.alpha * score

        # Sort by combined score and take top k
        sorted_results = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )[:retrieval_k]

        # Prepare results
        final_ids = [doc_id for doc_id, _ in sorted_results]
        final_scores = [score for _, score in sorted_results]

        # Get documents and metadata from ChromaDB
        chroma_results = self.collection.get(
            ids=final_ids, include=["documents", "metadatas"]
        )

        # Build tree with augmented text
        chroma_results = self.build_tree(chroma_results, n_extend=n_extend)

        # Ensure order matches our sorted results
        id_to_doc_map = {
            doc_id: doc
            for doc_id, doc in zip(chroma_results["ids"], chroma_results["documents"])
        }
        id_to_meta_map = {
            doc_id: meta
            for doc_id, meta in zip(chroma_results["ids"], chroma_results["metadatas"])
        }

        final_docs = [id_to_doc_map.get(doc_id, "") for doc_id in final_ids]
        final_metadatas = [id_to_meta_map.get(doc_id, {}) for doc_id in final_ids]

        return {
            "documents": final_docs,
            "metadatas": final_metadatas,
            "ids": final_ids,
            "scores": final_scores,
            "semantic_weight": self.alpha,
            "bm25_weight": 1 - self.alpha,
        }
