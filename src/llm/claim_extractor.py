import json
from typing import List

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

LLM_CONFIG = config.get("llm", {})

# ==========================================
# Claim Extractor
# ==========================================

class ClaimExtractor:
    """
    Production-grade claim extractor:
    - LLM-based extraction
    - Batch processing
    - Fallback mechanisms
    - Clean structured outputs
    """

    def __init__(self):
        self.provider = LLM_CONFIG.get("provider", "mock")
        self.model = LLM_CONFIG.get("model")
        self.temperature = LLM_CONFIG.get("temperature", 0.2)
        self.prompt_template = LLM_CONFIG.get("claim_extraction_prompt")

        # Lazy init client
        self.client = self._init_client()

    # ==========================================
    # Public API
    # ==========================================

    def extract(self, chunks: List[str]) -> List[str]:
        """
        Extract claims from multiple text chunks.
        """
        if not chunks:
            return []

        logger.info(f"Extracting claims from {len(chunks)} chunks")

        all_claims = []

        for chunk in chunks:
            try:
                claims = self._extract_single(chunk)
                all_claims.extend(claims)
            except Exception as e:
                logger.error(f"Claim extraction failed: {e}")

        # Deduplicate
        unique_claims = list(set(all_claims))

        logger.info(f"Extracted {len(unique_claims)} unique claims")

        return unique_claims

    # ==========================================
    # Core Logic
    # ==========================================

    def _extract_single(self, text: str) -> List[str]:
        """
        Extract claims from a single chunk.
        """
        if self.provider == "mock":
            return self._mock_extract(text)

        elif self.provider == "openai":
            return self._openai_extract(text)

        else:
            raise NotImplementedError(f"Provider {self.provider} not supported")

    # ==========================================
    # Mock Extractor (Fallback / Dev)
    # ==========================================

    def _mock_extract(self, text: str) -> List[str]:
        """
        Fallback extraction using sentence splitting.
        """
        sentences = text.split(".")
        claims = [
            s.strip()
            for s in sentences
            if len(s.strip()) > 20
        ]
        return claims

    # ==========================================
    # OpenAI Extractor (Production)
    # ==========================================

    def _openai_extract(self, text: str) -> List[str]:
        """
        Uses OpenAI API for claim extraction.
        Requires openai package and API key.
        """
        try:
            from openai import OpenAI

            client = OpenAI()

            prompt = f"""
{self.prompt_template}

Text:
{text}

Return output as JSON list of strings.
"""

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You extract factual claims."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )

            content = response.choices[0].message.content.strip()

            return self._parse_json_output(content)

        except Exception as e:
            logger.error(f"OpenAI extraction failed: {e}")
            return self._mock_extract(text)

    # ==========================================
    # Helpers
    # ==========================================

    def _parse_json_output(self, content: str) -> List[str]:
        """
        Safely parse LLM JSON output.
        """
        try:
            data = json.loads(content)

            if isinstance(data, list):
                return [str(x).strip() for x in data if len(str(x).strip()) > 10]

        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON, falling back")

        return []

    def _init_client(self):
        """
        Placeholder for initializing LLM clients.
        """
        if self.provider == "openai":
            try:
                from openai import OpenAI
                return OpenAI()
            except ImportError:
                logger.warning("OpenAI package not installed")
        return None