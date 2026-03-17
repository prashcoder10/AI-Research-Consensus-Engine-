import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Optional
from urllib.parse import urlparse

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)
config = load_config()

INGESTION_CONFIG = config.get("ingestion", {})

class AsyncWebScraper:
    """
    Production-grade async web scraper with:
    - concurrency control
    - retry logic
    - timeout handling
    - HTML cleaning
    """

    def __init__(self):
        self.timeout = INGESTION_CONFIG.get("timeout_seconds", 10)
        self.max_retries = INGESTION_CONFIG.get("max_retries", 3)
        self.headers = {
            "User-Agent": INGESTION_CONFIG.get(
                "user_agent", "Mozilla/5.0 (AI-Consensus-Engine)"
            )
        }
        self.semaphore = asyncio.Semaphore(
            INGESTION_CONFIG.get("max_concurrent_requests", 10)
        )

    # ==========================================
    # Public API
    # ==========================================

    async def scrape_urls(self, urls: List[str]) -> List[str]:
        """
        Scrape multiple URLs concurrently.

        Returns:
            List of extracted text (empty string for failed URLs)
        """
        tasks = [self._scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        cleaned_results = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to scrape {url}: {str(result)}")
                cleaned_results.append("")
            else:
                cleaned_results.append(result)

        return cleaned_results

    # ==========================================
    # Internal Methods
    # ==========================================

    async def _scrape_with_semaphore(self, url: str) -> str:
        async with self.semaphore:
            return await self._scrape_with_retries(url)

    async def _scrape_with_retries(self, url: str) -> str:
        for attempt in range(self.max_retries):
            try:
                return await self._fetch_and_parse(url)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {url} (attempt {attempt+1})")
            except aiohttp.ClientError as e:
                logger.warning(f"Client error for {url}: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error for {url}: {str(e)}")

            await asyncio.sleep(2 ** attempt)  # exponential backoff

        logger.error(f"All retries failed for {url}")
        return ""

    async def _fetch_and_parse(self, url: str) -> str:
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(timeout=timeout, headers=self.headers) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise aiohttp.ClientError(f"HTTP {response.status}")

                html = await response.text()
                return self._extract_text(html, url)

    # ==========================================
    # HTML Processing
    # ==========================================

    def _extract_text(self, html: str, url: str) -> str:
        """
        Extract meaningful text from HTML.
        Removes scripts, styles, nav, footer, etc.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Extract text from paragraphs
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]

        text = "\n".join(paragraphs)

        # Basic filtering
        if len(text) < 100:
            logger.warning(f"Low content extracted from {urlparse(url).netloc}")

        return text.strip()

# ==========================================
# Convenience Function (Sync Wrapper)
# ==========================================

def scrape_urls_sync(urls: List[str]) -> List[str]:
    """
    Sync wrapper for async scraping (useful for non-async pipelines)
    """
    scraper = AsyncWebScraper()
    return asyncio.run(scraper.scrape_urls(urls))