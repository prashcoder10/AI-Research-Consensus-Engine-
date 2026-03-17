import re
from typing import List

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

PREPROCESSING_CONFIG = config.get("preprocessing", {})

class TextCleaner:
    """
    Production-grade text cleaner:
    - Normalizes whitespace
    - Removes URLs (optional)
    - Removes boilerplate artifacts
    - Filters low-quality text
    - Prepares text for embeddings + LLMs
    """

    def __init__(self):
        self.normalize_whitespace = PREPROCESSING_CONFIG.get(
            "normalize_whitespace", True
        )
        self.remove_urls = PREPROCESSING_CONFIG.get("remove_urls", True)
        self.min_length = PREPROCESSING_CONFIG.get("min_text_length", 50)

    # ==========================================
    # Public API
    # ==========================================

    def clean(self, text: str) -> str:
        """
        Clean a single text document.
        """
        if not text:
            return ""

        original_length = len(text)

        text = self._normalize_unicode(text)
        text = self._remove_html_entities(text)

        if self.remove_urls:
            text = self._remove_urls(text)

        text = self._remove_emails(text)
        text = self._remove_extra_symbols(text)

        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        text = text.strip()

        if len(text) < self.min_length:
            logger.warning(
                f"Text too short after cleaning ({len(text)} chars). Skipping."
            )
            return ""

        logger.debug(
            f"Cleaned text from {original_length} → {len(text)} characters"
        )

        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean multiple documents efficiently.
        """
        cleaned = []
        for i, text in enumerate(texts):
            try:
                c = self.clean(text)
                if c:
                    cleaned.append(c)
                else:
                    logger.debug(f"Skipped text index {i}")
            except Exception as e:
                logger.error(f"Cleaning failed at index {i}: {str(e)}")

        return cleaned

    # ==========================================
    # Cleaning Steps
    # ==========================================

    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize weird unicode characters.
        """
        return text.encode("utf-8", "ignore").decode("utf-8")

    def _remove_html_entities(self, text: str) -> str:
        """
        Remove HTML entities like &nbsp; &amp;
        """
        return re.sub(r"&[a-z]+;", " ", text)

    def _remove_urls(self, text: str) -> str:
        return re.sub(r"http\S+|www\S+", " ", text)

    def _remove_emails(self, text: str) -> str:
        return re.sub(r"\S+@\S+", " ", text)

    def _remove_extra_symbols(self, text: str) -> str:
        """
        Remove excessive symbols but keep punctuation.
        """
        return re.sub(r"[^\w\s.,!?;:()\-\'\"]+", " ", text)

    def _normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text)

# ==========================================
# Convenience Functions
# ==========================================

def clean_text(text: str) -> str:
    cleaner = TextCleaner()
    return cleaner.clean(text)

def clean_texts(texts: List[str]) -> List[str]:
    cleaner = TextCleaner()
    return cleaner.clean_batch(texts)