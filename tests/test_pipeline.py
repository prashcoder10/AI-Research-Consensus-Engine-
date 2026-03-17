import pytest
import asyncio

from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.chunker import TextChunker
from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.clustering.clusterer import Clusterer, ConsensusEngine
from src.llm.claim_extractor import ClaimExtractor
from src.llm.summarizer import Summarizer
from src.ingestion.document_loader import AsyncDocumentLoader


# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def sample_texts():
    return [
        "AI improves productivity in many industries.",
        "AI increases efficiency and reduces manual work.",
        "Some studies suggest AI reduces productivity in certain workflows."
    ]


# ==========================================
# Unit Tests
# ==========================================

def test_cleaner():
    cleaner = TextCleaner()

    raw = "Visit https://example.com!!!   AI is great \n\n"
    cleaned = cleaner.clean(raw)

    assert "http" not in cleaned
    assert len(cleaned) > 0


def test_chunker():
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)

    text = "AI is transforming industries. It improves productivity significantly."
    chunks = chunker.chunk(text)

    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)


def test_embedder():
    embedder = Embedder()

    texts = ["AI is great", "Machine learning is useful"]
    embeddings = embedder.encode(texts)

    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] > 0


def test_vector_store():
    embedder = Embedder()
    texts = ["AI improves productivity", "AI has risks"]

    embeddings = embedder.encode(texts)
    store = VectorStore(dim=embeddings.shape[1])

    store.add(embeddings, texts)

    query_emb = embedder.encode(["AI benefits"])
    results = store.search(query_emb, k=2)

    assert len(results) > 0


def test_clustering_and_consensus(sample_texts):
    embedder = Embedder()
    embeddings = embedder.encode(sample_texts)

    clusterer = Clusterer()
    clusters = clusterer.cluster(embeddings, sample_texts)

    engine = ConsensusEngine()
    consensus, conflicts = engine.analyze(clusters)

    assert isinstance(consensus, list)
    assert isinstance(conflicts, list)


def test_claim_extraction(sample_texts):
    extractor = ClaimExtractor()

    claims = extractor.extract(sample_texts)

    assert isinstance(claims, list)
    assert len(claims) > 0


def test_summarizer():
    summarizer = Summarizer()

    consensus = [["AI improves productivity", "AI increases efficiency"]]
    conflicts = ["AI reduces productivity"]

    result = summarizer.summarize(consensus, conflicts)

    assert "final_summary" in result
    assert isinstance(result["consensus_summary"], list)


# ==========================================
# Async Tests
# ==========================================

@pytest.mark.asyncio
async def test_document_loader():
    loader = AsyncDocumentLoader()

    docs = await loader.load(
        texts=["AI improves productivity."],
        urls=[],
        file_paths=[]
    )

    assert len(docs) > 0
    assert docs[0].content


@pytest.mark.asyncio
async def test_full_pipeline(sample_texts):
    """
    End-to-end pipeline test (no API).
    """

    # Ingestion
    loader = AsyncDocumentLoader()
    docs = await loader.load(texts=sample_texts)

    # Cleaning
    cleaner = TextCleaner()
    cleaned = cleaner.clean_batch([d.content for d in docs])

    # Chunking
    chunker = TextChunker()
    chunks = []
    for t in cleaned:
        chunks.extend(chunker.chunk(t))

    # Embedding
    embedder = Embedder()
    embeddings = embedder.encode(chunks)

    # Vector Store
    store = VectorStore(dim=embeddings.shape[1])
    store.add(embeddings, chunks)

    # Claim Extraction
    extractor = ClaimExtractor()
    claims = extractor.extract(chunks)

    # Claim Embeddings
    claim_embeddings = embedder.encode(claims)

    # Clustering
    clusterer = Clusterer()
    clusters = clusterer.cluster(claim_embeddings, claims)

    # Consensus
    engine = ConsensusEngine()
    consensus, conflicts = engine.analyze(clusters)

    # Summary
    summarizer = Summarizer()
    result = summarizer.summarize(consensus, conflicts)

    assert isinstance(result, dict)
    assert "final_summary" in result