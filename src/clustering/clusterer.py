from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

CLUSTER_CONFIG = config.get("clustering", {})
CONSENSUS_CONFIG = config.get("consensus", {})

# ==========================================
# Clusterer
# ==========================================

class Clusterer:
    """
    Production-grade clustering engine:
    - Supports KMeans and similarity-based clustering
    - Designed for semantic grouping of claims
    """

    def __init__(self):
        self.algorithm = CLUSTER_CONFIG.get("algorithm", "kmeans")
        self.num_clusters = CLUSTER_CONFIG.get("num_clusters", 5)
        self.similarity_threshold = CLUSTER_CONFIG.get("similarity_threshold", 0.75)

    # ==========================================
    # Public API
    # ==========================================

    def cluster(
        self,
        embeddings: np.ndarray,
        texts: List[str],
    ) -> Dict[int, List[str]]:
        """
        Cluster texts based on embeddings.

        Returns:
            {cluster_id: [texts]}
        """
        if len(texts) == 0:
            return {}

        if len(texts) == 1:
            return {0: texts}

        if self.algorithm == "kmeans":
            return self._kmeans_cluster(embeddings, texts)

        elif self.algorithm == "similarity":
            return self._similarity_cluster(embeddings, texts)

        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.algorithm}")

    # ==========================================
    # KMeans Clustering
    # ==========================================

    def _kmeans_cluster(
        self,
        embeddings: np.ndarray,
        texts: List[str],
    ) -> Dict[int, List[str]]:
        k = min(self.num_clusters, len(texts))

        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(embeddings)

        clusters = {}
        for text, label in zip(texts, labels):
            clusters.setdefault(label, []).append(text)

        logger.info(f"KMeans clustering produced {len(clusters)} clusters")

        return clusters

    # ==========================================
    # Similarity-Based Clustering (Better for claims)
    # ==========================================

    def _similarity_cluster(
        self,
        embeddings: np.ndarray,
        texts: List[str],
    ) -> Dict[int, List[str]]:
        """
        Greedy clustering based on cosine similarity.
        More robust for small datasets than KMeans.
        """
        sim_matrix = cosine_similarity(embeddings)

        visited = set()
        clusters = {}
        cluster_id = 0

        for i in range(len(texts)):
            if i in visited:
                continue

            cluster = [texts[i]]
            visited.add(i)

            for j in range(i + 1, len(texts)):
                if j in visited:
                    continue

                if sim_matrix[i][j] >= self.similarity_threshold:
                    cluster.append(texts[j])
                    visited.add(j)

            clusters[cluster_id] = cluster
            cluster_id += 1

        logger.info(f"Similarity clustering produced {len(clusters)} clusters")

        return clusters

# ==========================================
# Consensus Detection
# ==========================================

class ConsensusEngine:
    """
    Detects consensus and conflicting claims from clusters.
    """

    def __init__(self):
        self.min_claims = CONSENSUS_CONFIG.get("min_claims_for_consensus", 2)

    def analyze(
        self,
        clusters: Dict[int, List[str]],
    ) -> Tuple[List[List[str]], List[str]]:
        """
        Returns:
            consensus: list of clusters (each cluster = agreeing claims)
            conflicts: list of standalone or weak claims
        """
        consensus = []
        conflicts = []

        for cluster_id, claims in clusters.items():
            if len(claims) >= self.min_claims:
                consensus.append(claims)
            else:
                conflicts.extend(claims)

        logger.info(
            f"Consensus groups: {len(consensus)} | Conflicts: {len(conflicts)}"
        )

        return consensus, conflicts

# ==========================================
# Advanced Conflict Detection (Optional)
# ==========================================

class ConflictDetector:
    """
    Placeholder for advanced contradiction detection.
    Future: use NLI models (e.g. DeBERTa, GPT)
    """

    def detect(self, claims: List[str]) -> List[Tuple[str, str]]:
        """
        Returns conflicting claim pairs.
        Currently naive (keyword-based placeholder).
        """
        conflicts = []

        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                if self._is_conflict(claims[i], claims[j]):
                    conflicts.append((claims[i], claims[j]))

        return conflicts

    def _is_conflict(self, a: str, b: str) -> bool:
        """
        Naive heuristic:
        Detects negation conflicts (can be replaced with LLM/NLI)
        """
        negations = ["not", "no", "never", "none"]

        a_neg = any(n in a.lower() for n in negations)
        b_neg = any(n in b.lower() for n in negations)

        return a_neg != b_neg