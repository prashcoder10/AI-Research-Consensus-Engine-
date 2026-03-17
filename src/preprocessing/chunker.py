from typing import List
import re

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

PREPROCESSING_CONFIG = config.get("preprocessing", {})
LIMITS = config.get("limits", {})

class TextChunker:
    """
    Production-grade text chunker with:
    - Sentence-aware splitting
    - Overlap support
    - Config-driven chunk sizing
    - Safety limits
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.chunk_size = chunk_size or PREPROCESSING_CONFIG.get("chunk_size", 300)
        self.chunk_overlap = chunk_overlap or PREPROCESSING_CONFIG.get("chunk_overlap", 50)
        self.max_chunks = LIMITS.get("max_chunks", 1000)

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    # ==========================================
    # Public API
    # ==========================================

    def chunk(self, text: str) -> List[str]:
        """
        Main chunking function.
        """
        if not text or len(text.strip()) < 10:
            return []

        sentences = self._split_into_sentences(text)
        chunks = self._build_chunks(sentences)

        if len(chunks) > self.max_chunks:
            logger.warning(
                f"Chunk limit exceeded ({len(chunks)} > {self.max_chunks}). Truncating."
            )
            chunks = chunks[: self.max_chunks]

        return chunks

    # ==========================================
    # Sentence Splitting
    # ==========================================

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Basic sentence tokenizer using regex.
        Can be replaced with spaCy for higher accuracy.
        """
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)

        return [s.strip() for s in sentences if len(s.strip()) > 5]

    # ==========================================
    # Chunk Builder
    # ==========================================

    def _build_chunks(self, sentences: List[str]) -> List[str]:
        """
        Build chunks while preserving sentence boundaries.
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())

            # If adding sentence exceeds chunk size → finalize chunk
            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                current_chunk = self._apply_overlap(current_chunk)
                current_length = sum(len(s.split()) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    # ==========================================
    # Overlap Logic
    # ==========================================

    def _apply_overlap(self, previous_chunk: List[str]) -> List[str]:
        """
        Keeps last N words as overlap.
        """
        if not previous_chunk:
            return []

        words = " ".join(previous_chunk).split()

        overlap_words = words[-self.chunk_overlap :]
        overlap_text = " ".join(overlap_words)

        # Re-split overlap into sentences (approximation)
        sentences = re.split(r"(?<=[.!?])\s+", overlap_text)

        return [s.strip() for s in sentences if s.strip()]

# ==========================================
# Convenience Function
# ==========================================

def chunk_text(text: str) -> List[str]:
    """
    Simple wrapper for quick usage.
    """
    chunker = TextChunker()
    return chunker.chunk(text)