import os
import hashlib
import pickle
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

EMBEDDING_CONFIG = config.get("embedding", {})

# ==========================================
# Cache Manager
# ==========================================

class EmbeddingCache:
    """
    Simple disk-based cache for embeddings.
    Keyed by hash(text + model_name)
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.pkl")

    def _hash(self, text: str, model_name: str) -> str:
        return hashlib.md5((text + model_name).encode("utf-8")).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        key = self._hash(text, model_name)
        path = self._get_cache_path(key)

        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        return None

    def set(self, text: str, model_name: str, embedding: np.ndarray):
        key = self._hash(text, model_name)
        path = self._get_cache_path(key)

        try:
            with open(path, "wb") as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

# ==========================================
# Embedder
# ==========================================

class Embedder:
    """
    Production-grade embedder:
    - Batched encoding
    - Disk caching
    - Provider abstraction (local / OpenAI-ready)
    """

    def __init__(self):
        self.provider = EMBEDDING_CONFIG.get("provider", "sentence_transformers")
        self.model_name = EMBEDDING_CONFIG.get("model_name")
        self.batch_size = EMBEDDING_CONFIG.get("batch_size", 32)
        self.normalize = EMBEDDING_CONFIG.get("normalize_embeddings", True)

        self.cache_enabled = EMBEDDING_CONFIG.get("cache_enabled", True)
        self.cache = None

        if self.cache_enabled:
            cache_dir = EMBEDDING_CONFIG.get("cache_dir", "./cache/embeddings")
            self.cache = EmbeddingCache(cache_dir)

        # Load model
        if self.provider == "sentence_transformers":
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        else:
            raise NotImplementedError(f"Provider {self.provider} not supported yet")

    # ==========================================
    # Public API
    # ==========================================

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings with batching + caching.
        """
        if not texts:
            return np.array([])

        all_embeddings = []
        texts_to_encode = []
        cache_hits = {}

        # Step 1: Check cache
        for i, text in enumerate(texts):
            if self.cache_enabled:
                cached = self.cache.get(text, self.model_name)
                if cached is not None:
                    cache_hits[i] = cached
                    continue

            texts_to_encode.append((i, text))

        logger.info(
            f"Embedding {len(texts)} texts | Cache hits: {len(cache_hits)} | To encode: {len(texts_to_encode)}"
        )

        # Step 2: Batch encode missing texts
        if texts_to_encode:
            indices, batch_texts = zip(*texts_to_encode)

            batch_embeddings = self._batch_encode(list(batch_texts))

            # Save to cache
            for idx, emb in zip(indices, batch_embeddings):
                if self.cache_enabled:
                    self.cache.set(texts[idx], self.model_name, emb)
                cache_hits[idx] = emb

        # Step 3: Reconstruct in correct order
        for i in range(len(texts)):
            all_embeddings.append(cache_hits[i])

        embeddings = np.array(all_embeddings)

        # Step 4: Normalize (cosine similarity optimization)
        if self.normalize:
            embeddings = self._normalize(embeddings)

        return embeddings

    # ==========================================
    # Internal Methods
    # ==========================================

    def _batch_encode(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode texts in batches to avoid memory spikes.
        """
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            try:
                batch_emb = self.model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                embeddings.extend(batch_emb)

            except Exception as e:
                logger.error(f"Batch encoding failed: {e}")
                # Fallback: zero vectors
                dim = self.model.get_sentence_embedding_dimension()
                embeddings.extend([np.zeros(dim) for _ in batch])

        return embeddings

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize vectors for cosine similarity.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return embeddings / norms

# ==========================================
# Convenience Function
# ==========================================

def embed_texts(texts: List[str]) -> np.ndarray:
    embedder = Embedder()
    return embedder.encode(texts)