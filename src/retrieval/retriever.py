from typing import List, Dict, Any

import numpy as np

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore

logger = get_logger(__name__)
config = load_config()

RETRIEVAL_CONFIG = config.get("retrieval", {})

class Retriever:
    """
    Production-grade semantic retriever:
    - Embedding-based search
    - Score threshold filtering
    - Metadata-aware results
    - Ready for reranking extensions
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

        self.top_k = RETRIEVAL_CONFIG.get("top_k", 5)
        self.score_threshold = RETRIEVAL_CONFIG.get("score_threshold", 0.5)

    # ==========================================
    # Public API
    # ==========================================

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        score_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.

        Returns:
            List of dicts:
            {
                "text": str,
                "score": float,
                "metadata": dict
            }
        """
        if not query or len(query.strip()) == 0:
            logger.warning("Empty query received")
            return []

        top_k = top_k or self.top_k
        score_threshold = score_threshold or self.score_threshold

        logger.info(f"Retrieving top {top_k} results for query")

        # Step 1: Embed query
        query_embedding = self.embedder.encode([query])

        # Step 2: Search vector store
        raw_results = self.vector_store.search(query_embedding, k=top_k)

        # Step 3: Filter + format
        results = self._filter_results(raw_results, score_threshold)

        logger.info(f"Retrieved {len(results)} relevant results")

        return results

    # ==========================================
    # Internal Methods
    # ==========================================

    def _filter_results(
        self,
        results,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Apply score filtering and format results.
        """
        filtered = []

        for text, score, metadata in results:
            if score < threshold:
                continue

            filtered.append(
                {
                    "text": text,
                    "score": round(score, 4),
                    "metadata": metadata,
                }
            )

        return filtered

    # ==========================================
    # Batch Retrieval (Advanced)
    # ==========================================

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve results for multiple queries efficiently.
        """
        if not queries:
            return {}

        top_k = top_k or self.top_k

        logger.info(f"Batch retrieving for {len(queries)} queries")

        query_embeddings = self.embedder.encode(queries)

        results = {}

        for query, emb in zip(queries, query_embeddings):
            raw = self.vector_store.search(emb.reshape(1, -1), k=top_k)
            filtered = self._filter_results(raw, self.score_threshold)
            results[query] = filtered

        return results