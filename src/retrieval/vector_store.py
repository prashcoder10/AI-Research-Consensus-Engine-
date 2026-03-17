import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

VECTOR_CONFIG = config.get("vector_store", {})

class VectorStore:
    """
    Production-grade FAISS vector store with:
    - Persistence (save/load)
    - Cosine similarity support
    - Incremental updates
    - Metadata storage
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = None
        self.texts: List[str] = []
        self.metadata: List[dict] = []

        self.persist_path = VECTOR_CONFIG.get("persist_path", "./data/faiss_index")
        self.index_type = VECTOR_CONFIG.get("index_type", "flat")

        os.makedirs(self.persist_path, exist_ok=True)

        self.index_file = os.path.join(self.persist_path, "index.faiss")
        self.meta_file = os.path.join(self.persist_path, "metadata.pkl")

        self._init_or_load_index()

    # ==========================================
    # Initialization
    # ==========================================

    def _init_or_load_index(self):
        """
        Load existing index if available, otherwise create new one.
        """
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            try:
                self.index = faiss.read_index(self.index_file)
                with open(self.meta_file, "rb") as f:
                    data = pickle.load(f)
                    self.texts = data["texts"]
                    self.metadata = data["metadata"]

                logger.info("Loaded existing FAISS index from disk")

            except Exception as e:
                logger.error(f"Failed to load index, rebuilding: {e}")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """
        Create new FAISS index.
        """
        if self.index_type == "flat":
            # Cosine similarity → use Inner Product with normalized vectors
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            raise NotImplementedError(f"Index type {self.index_type} not supported")

        self.texts = []
        self.metadata = []

        logger.info("Initialized new FAISS index")

    # ==========================================
    # Add Data
    # ==========================================

    def add(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadata: List[dict] = None,
    ):
        """
        Add new embeddings to the index.
        """
        if len(embeddings) != len(texts):
            raise ValueError("Embeddings and texts must have same length")

        if metadata and len(metadata) != len(texts):
            raise ValueError("Metadata length must match texts")

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self.index.add(embeddings)

        self.texts.extend(texts)
        self.metadata.extend(metadata or [{} for _ in texts])

        logger.info(f"Added {len(texts)} vectors to index")

    # ==========================================
    # Search
    # ==========================================

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[str, float, dict]]:
        """
        Perform similarity search.

        Returns:
            List of (text, score, metadata)
        """
        if self.index.ntotal == 0:
            logger.warning("Search attempted on empty index")
            return []

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # Ensure shape (1, dim)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append(
                (
                    self.texts[idx],
                    float(score),
                    self.metadata[idx],
                )
            )

        return results

    # ==========================================
    # Persistence
    # ==========================================

    def save(self):
        """
        Persist index + metadata to disk.
        """
        try:
            faiss.write_index(self.index, self.index_file)

            with open(self.meta_file, "wb") as f:
                pickle.dump(
                    {
                        "texts": self.texts,
                        "metadata": self.metadata,
                    },
                    f,
                )

            logger.info("Vector store saved to disk")

        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")

    def reset(self):
        """
        Clear index and delete persisted files.
        """
        self._create_new_index()

        if os.path.exists(self.index_file):
            os.remove(self.index_file)

        if os.path.exists(self.meta_file):
            os.remove(self.meta_file)

        logger.warning("Vector store reset")

    # ==========================================
    # Info
    # ==========================================

    def size(self) -> int:
        return self.index.ntotal