# 🧪 AI Research Consensus Engine — Testing Guide

---

## ✅ 1. Run Unit + Integration Tests

pytest tests/

This tests:

* Cleaner
* Chunker
* Embedder
* Vector Store
* Clustering
* Claim Extraction
* Summarization
* Full pipeline

---

## ✅ 2. Run API Locally

Start server:

uvicorn main:app --reload

Open Swagger UI:

http://localhost:8000/docs

Test `/analyze` endpoint directly.

---

## ✅ 3. Test Using cURL

curl -X POST "http://localhost:8000/analyze" 
-H "Content-Type: application/json" 
-d '{
"texts": [
"AI improves productivity",
"AI reduces productivity sometimes"
]
}'

---

## ✅ 4. Manual Python Test

from src.ingestion.document_loader import load_documents_sync

docs = load_documents_sync(
texts=[
"AI improves productivity.",
"AI reduces productivity in some cases."
]
)

print(docs)

---

## 💰 Testing WITHOUT OpenAI (IMPORTANT)

You do NOT need OpenAI for development.

---

### ✅ Option 1 (Default): Mock Mode

In config/settings.yaml:

llm:
provider: "mock"

✔ No API key
✔ No cost
✔ Fully functional pipeline

---

### ✅ Option 2: Hybrid Mode (Recommended)

embedding:
provider: "sentence_transformers"

llm:
provider: "mock"

✔ Best dev setup
✔ Real embeddings + fake LLM

---

### ✅ Option 3: Free Local LLM (Advanced)

Install Ollama:

https://ollama.com

Run:

ollama run mistral

Then integrate into:

* claim_extractor.py
* summarizer.py

✔ Fully offline
✔ No cost

---

## ⚠️ When You Need OpenAI

Only for:

* Better summaries
* Better claim extraction
* Production-quality reasoning

---

## 🧠 Dev vs Production Setup

DEV:

* LLM: mock
* Embeddings: local
* Cost: FREE

PROD:

* LLM: OpenAI
* Embeddings: local/OpenAI
* Cost: Paid

---

## 🚀 Recommended Dev Config

llm:
provider: "mock"

embedding:
provider: "sentence_transformers"

---

## 🔥 Pro Tips

* Use small inputs first
* Monitor logs
* Enable DEBUG logging for troubleshooting
* Increase batch sizes gradually

---

## 🎯 Summary

You can:
✔ Run full system locally
✔ Test everything without OpenAI
✔ Scale later without redesign

---

## 🧪 Optional Advanced Testing

* Add API tests using FastAPI TestClient
* Add load testing (locust)
* Add performance benchmarks

---

🚀 You are ready to build and test a real AI system.
