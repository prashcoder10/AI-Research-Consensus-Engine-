import asyncio
import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from src.ingestion.web_scraper import AsyncWebScraper
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)
config = load_config()

LIMITS = config.get("limits", {})

# ==========================================
# Data Model
# ==========================================

@dataclass
class Document:
    """
    Standard document object used across pipeline.
    """
    content: str
    source: str
    metadata: Dict[str, Any]

# ==========================================
# Async Document Loader
# ==========================================

class AsyncDocumentLoader:
    """
    Handles ingestion of:
    - URLs
    - Raw text
    - Local files

    Returns standardized Document objects.
    """

    def __init__(self):
        self.scraper = AsyncWebScraper()
        self.max_sources = LIMITS.get("max_input_sources", 100)

    # ==========================================
    # Public API
    # ==========================================

    async def load(
        self,
        texts: Optional[List[str]] = None,
        urls: Optional[List[str]] = None,
        file_paths: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Main entry point for loading all input types.
        """

        texts = texts or []
        urls = urls or []
        file_paths = file_paths or []

        total_sources = len(texts) + len(urls) + len(file_paths)
        if total_sources > self.max_sources:
            raise ValueError(
                f"Too many input sources ({total_sources}). Max allowed: {self.max_sources}"
            )

        logger.info(f"Loading {total_sources} sources...")

        tasks = []

        if texts:
            tasks.append(self._load_texts(texts))

        if urls:
            tasks.append(self._load_urls(urls))

        if file_paths:
            tasks.append(self._load_files(file_paths))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        documents: List[Document] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Loader error: {str(result)}")
                continue
            documents.extend(result)

        logger.info(f"Loaded {len(documents)} documents successfully")

        return documents

    # ==========================================
    # Loaders
    # ==========================================

    async def _load_texts(self, texts: List[str]) -> List[Document]:
        docs = []

        for i, text in enumerate(texts):
            if not text or len(text.strip()) < 10:
                logger.warning(f"Skipping empty/short text at index {i}")
                continue

            docs.append(
                Document(
                    content=text.strip(),
                    source=f"raw_text_{i}",
                    metadata={"type": "text", "index": i},
                )
            )

        return docs

    async def _load_urls(self, urls: List[str]) -> List[Document]:
        texts = await self.scraper.scrape_urls(urls)

        docs = []
        for url, text in zip(urls, texts):
            if not text:
                logger.warning(f"No content extracted from {url}")
                continue

            docs.append(
                Document(
                    content=text,
                    source=url,
                    metadata={"type": "url"},
                )
            )

        return docs

    async def _load_files(self, file_paths: List[str]) -> List[Document]:
        tasks = [self._read_file(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        docs = []
        for path, result in zip(file_paths, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to read file {path}: {str(result)}")
                continue

            if not result:
                continue

            docs.append(
                Document(
                    content=result,
                    source=path,
                    metadata={"type": "file"},
                )
            )

        return docs

    # ==========================================
    # File Reading
    # ==========================================

    async def _read_file(self, path: str) -> str:
        """
        Async file reader (txt only by default).
        Extendable to PDF, DOCX.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        loop = asyncio.get_event_loop()

        return await loop.run_in_executor(None, self._sync_read_file, path)

    def _sync_read_file(self, path: str) -> str:
        """
        Blocking file read (executed in thread pool).
        """
        ext = os.path.splitext(path)[1].lower()

        if ext == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

        # Placeholder for future extensions
        elif ext == ".pdf":
            raise NotImplementedError("PDF support not yet implemented")

        else:
            raise ValueError(f"Unsupported file type: {ext}")

# ==========================================
# Sync Wrapper
# ==========================================

def load_documents_sync(
    texts: Optional[List[str]] = None,
    urls: Optional[List[str]] = None,
    file_paths: Optional[List[str]] = None,
) -> List[Document]:
    loader = AsyncDocumentLoader()
    return asyncio.run(loader.load(texts, urls, file_paths))