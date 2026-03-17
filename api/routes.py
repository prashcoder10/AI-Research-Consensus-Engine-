from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

import asyncio

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

from src.ingestion.document_loader import AsyncDocumentLoader
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.chunker import TextChunker
from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.llm.claim_extractor import ClaimExtractor
from src.clustering.clusterer import Clusterer, ConsensusEngine
from src.llm.summarizer import Summarizer

logger = get_logger(__name__)
config = load_config()

router = APIRouter()

LIMITS = config.get("limits", {})

# ==========================================
# Request / Response Models
# ==========================================

class AnalyzeRequest(BaseModel):
    texts: Optional[List[str]] = Field(default_factory=list)
    urls: Optional[List[str]] = Field(default_factory=list)
    file_paths: Optional[List[str]] = Field(default_factory=list)

class AnalyzeResponse(BaseModel):
    consensus_summary: List[str]
    conflict_summary: List[str]
    final_summary: str

# ==========================================
# Pipeline Endpoint
# ==========================================

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Full pipeline:
    ingestion → cleaning → chunking → embedding →
    claim extraction → clustering → consensus → summarization
    """

    try:
        total_inputs = len(request.texts) + len(request.urls) + len(request.file_paths)

        if total_inputs == 0:
            raise HTTPException(status_code=400, detail="No input provided")

        if total_inputs > LIMITS.get("max_input_sources", 100):
            raise HTTPException(
                status_code=400,
                detail=f"Too many inputs. Max allowed: {LIMITS.get('max_input_sources')}"
            )

        logger.info(f"Starting analysis for {total_inputs} sources")

        # ==========================================
        # 1. Ingestion
        # ==========================================
        loader = AsyncDocumentLoader()

        documents = await loader.load(
            texts=request.texts,
            urls=request.urls,
            file_paths=request.file_paths
        )

        if not documents:
            raise HTTPException(status_code=400, detail="No valid content extracted")

        # ==========================================
        # 2. Cleaning
        # ==========================================
        cleaner = TextCleaner()
        cleaned_texts = cleaner.clean_batch([doc.content for doc in documents])

        if not cleaned_texts:
            raise HTTPException(status_code=400, detail="All inputs invalid after cleaning")

        # ==========================================
        # 3. Chunking
        # ==========================================
        chunker = TextChunker()
        chunks = []

        for text in cleaned_texts:
            chunks.extend(chunker.chunk(text))

        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated")

        logger.info(f"Generated {len(chunks)} chunks")

        # ==========================================
        # 4. Embeddings + Vector Store
        # ==========================================
        embedder = Embedder()
        embeddings = embedder.encode(chunks)

        store = VectorStore(dim=embeddings.shape[1])
        store.add(embeddings, chunks)

        # ==========================================
        # 5. Claim Extraction
        # ==========================================
        extractor = ClaimExtractor()
        claims = extractor.extract(chunks)

        if not claims:
            raise HTTPException(status_code=400, detail="No claims extracted")

        # ==========================================
        # 6. Claim Embeddings
        # ==========================================
        claim_embeddings = embedder.encode(claims)

        # ==========================================
        # 7. Clustering
        # ==========================================
        clusterer = Clusterer()
        clusters = clusterer.cluster(claim_embeddings, claims)

        # ==========================================
        # 8. Consensus Detection
        # ==========================================
        consensus_engine = ConsensusEngine()
        consensus, conflicts = consensus_engine.analyze(clusters)

        # ==========================================
        # 9. Summarization
        # ==========================================
        summarizer = Summarizer()
        result = summarizer.summarize(consensus, conflicts)

        logger.info("Analysis completed successfully")

        return result

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Pipeline failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ==========================================
# Health Check
# ==========================================

@router.get("/health")
def health():
    return {"status": "ok"}