import json
from typing import List, Dict, Any

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

LLM_CONFIG = config.get("llm", {})

# ==========================================
# Summarizer
# ==========================================

class Summarizer:
    """
    Production-grade summarizer:
    - LLM-based structured summarization
    - Handles consensus + conflicts
    - Fallback-safe
    """

    def __init__(self):
        self.provider = LLM_CONFIG.get("provider", "mock")
        self.model = LLM_CONFIG.get("model")
        self.temperature = LLM_CONFIG.get("temperature", 0.2)
        self.prompt_template = LLM_CONFIG.get("summarization_prompt")

        self.client = self._init_client()

    # ==========================================
    # Public API
    # ==========================================

    def summarize(
        self,
        consensus: List[List[str]],
        conflicts: List[str],
    ) -> Dict[str, Any]:
        """
        Generate final structured summary.
        """
        logger.info("Generating final summary")

        if self.provider == "mock":
            return self._mock_summary(consensus, conflicts)

        elif self.provider == "openai":
            return self._openai_summary(consensus, conflicts)

        else:
            raise NotImplementedError(f"Provider {self.provider} not supported")

    # ==========================================
    # OpenAI Summarization (Production)
    # ==========================================

    def _openai_summary(
        self,
        consensus: List[List[str]],
        conflicts: List[str],
    ) -> Dict[str, Any]:
        try:
            from openai import OpenAI

            client = self.client or OpenAI()

            prompt = f"""
{self.prompt_template}

Consensus Groups:
{json.dumps(consensus, indent=2)}

Conflicting Claims:
{json.dumps(conflicts, indent=2)}

Return JSON with:
{{
  "consensus_summary": [...],
  "conflict_summary": [...],
  "final_summary": "..."
}}
"""

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert research analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )

            content = response.choices[0].message.content.strip()

            return self._parse_json(content)

        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            return self._mock_summary(consensus, conflicts)

    # ==========================================
    # Mock Summary (Fallback)
    # ==========================================

    def _mock_summary(
        self,
        consensus: List[List[str]],
        conflicts: List[str],
    ) -> Dict[str, Any]:
        """
        Simple deterministic summary (safe fallback)
        """
        consensus_summary = [
            f"{len(group)} sources agree: {group[0]}"
            for group in consensus
        ]

        conflict_summary = conflicts

        final_summary = (
            f"Identified {len(consensus)} consensus groups and "
            f"{len(conflicts)} conflicting claims."
        )

        return {
            "consensus_summary": consensus_summary,
            "conflict_summary": conflict_summary,
            "final_summary": final_summary,
        }

    # ==========================================
    # Helpers
    # ==========================================

    def _parse_json(self, content: str) -> Dict[str, Any]:
        """
        Safely parse LLM JSON output.
        """
        try:
            data = json.loads(content)

            if isinstance(data, dict):
                return data

        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON output")

        return {
            "consensus_summary": [],
            "conflict_summary": [],
            "final_summary": content,
        }

    def _init_client(self):
        """
        Initialize LLM client if needed.
        """
        if self.provider == "openai":
            try:
                from openai import OpenAI
                return OpenAI()
            except ImportError:
                logger.warning("OpenAI package not installed")

        return None