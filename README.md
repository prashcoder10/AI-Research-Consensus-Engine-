# 🧠 AI Research Consensus Engine

An end-to-end AI system that ingests multiple sources (URLs, text, documents), extracts key claims, identifies consensus and conflicts, and generates structured summaries.

---

## 🚀 Features

* 🌐 Async web scraping (50–100+ sources)
* 🧹 Text cleaning & chunking
* 🧠 Semantic embeddings (SentenceTransformers / OpenAI)
* 🔍 Vector search with FAISS
* 🧾 LLM-based claim extraction
* 📊 Clustering & consensus detection
* 🧠 Structured summarization
* ⚡ FastAPI API
* 🧪 Test suite included
* 💾 Caching + persistence

---

## 🏗️ Architecture

Input → Ingestion → Cleaning → Chunking → Embeddings → Vector Store
→ Claim Extraction → Clustering → Consensus → Summary

---

## 📁 Project Structure

ai_consensus_engine/
│
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── embeddings/
│   ├── retrieval/
│   ├── clustering/
│   ├── llm/
│   ├── utils/
│
├── api/
├── config/
├── tests/
├── main.py
├── requirements.txt

---

## ⚙️ Installation

### 1. Clone repo

git clone <your-repo-url>
cd ai_consensus_engine

### 2. Create virtual environment

python -m venv venv
source venv/bin/activate   (Linux/Mac)
venv\Scripts\activate      (Windows)

### 3. Install dependencies

pip install -r requirements.txt

---

## ▶️ Running the Server

python main.py

OR

uvicorn main:app --reload

---

## 📡 API Usage

### Endpoint

POST /analyze

### Example Request

{
"texts": [
"AI improves productivity.",
"AI reduces productivity in some cases."
],
"urls": []
}

### Example Response

{
"consensus_summary": [
"2 sources agree: AI improves productivity"
],
"conflict_summary": [
"AI reduces productivity in some cases"
],
"final_summary": "Mixed evidence with general positive trend."
}

---

## 🧪 Running Tests

pytest tests/

---

## ⚙️ Configuration

Edit:

config/settings.yaml

Example:

embedding:
model_name: "all-MiniLM-L6-v2"

llm:
provider: "mock"

---

## 💡 Development Mode (No OpenAI Required)

Default config:

llm:
provider: "mock"

✔ No API key needed
✔ No cost
✔ Uses fallback logic

---

## 🔌 Using OpenAI (Optional)

1. Install:

pip install openai

2. Set API key:

export OPENAI_API_KEY=your_key_here

3. Update config:

llm:
provider: "openai"

---

## ⚡ Performance Notes

10 sources → Fast
50 sources → Stable
100 sources → Optimized
1000+ → Needs distributed system

---

## 🔮 Future Improvements

* Redis caching
* Celery workers
* HDBSCAN clustering
* LLM contradiction detection
* UI dashboard
* Docker + Kubernetes

---

## 🧠 Use Cases

* Research synthesis
* Fact-checking
* Market intelligence
* News aggregation
* Competitive analysis

---

## 📜 License

MIT License
