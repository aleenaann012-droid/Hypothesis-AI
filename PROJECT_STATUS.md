# Hypothesis-AI: Project Configuration, Running Guide & Verification Status

**Date:** October 2026  
**Status:** All requested features, paper ingestion pipeline, provenance tracing, multi-hop transitive relation discovery, paper isolation filters, and test suites are complete and verified with **34 passing tests** across 6 test suites.

---

## 1. System Requirements & Configuration (`.env`)

Hypothesis-AI uses local models and databases with zero paid APIs.

### Configuration File (`.env`)
Create or verify `.env` in the project root:
```ini
# Neo4j Graph Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=hypothesisai2026

# Local Ollama LLM Service
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```

### Component Status Table
| Component | Specification | Description / Role |
| :--- | :--- | :--- |
| **Python** | `3.11.9` (`.venv`) | Isolated virtual environment with project dependencies |
| **Ollama** | `http://localhost:11434` | Local model server hosting `qwen2.5:7b` |
| **LLM Model** | Local `qwen2.5:7b` | Synthesizes grounded hypotheses, experimental designs & metrics |
| **Embedding Model** | `BAAI/bge-small-en-v1.5` | 384-dimensional dense semantic representations cached locally |
| **Vector Index** | FAISS CPU | Real-time cosine similarity search (14 baseline chunks + dynamic uploads) |
| **NLP Pipeline** | SciSpaCy (`en_core_sci_sm`) | Scientific entity extraction with generic stopword & pronoun filters |
| **Knowledge Graph** | Neo4j 6.4.1 | Evidence-backed claims & direct relationships isolated by `paper_id` |
| **Web Server** | FastAPI & Uvicorn | REST API running on port `8000` |
| **Frontend UI** | HTML5 / CSS / Vanilla JS | Interactive researcher UI running on port `5501` |

---

## 2. Step-by-Step Guide to Run Hypothesis-AI Properly

### Step 1: Start the Backend API (Terminal 1)
In your project root (`Hypothesis-AI`):
```powershell
.venv\Scripts\python -m uvicorn api.main:app --reload --port 8000
```
Open [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) in your browser. It should show:
```json
{
  "status": "healthy",
  "retrieval": "ready",
  "indexed_chunks": 14,
  "neo4j": "connected",
  "working_llm": "qwen2.5:7b"
}
```

### Step 2: Start the Frontend UI (Terminal 2)
In a second terminal:
```powershell
.venv\Scripts\python -m http.server 5501 --directory frontend
```
**Open the UI:** Navigate to **[http://127.0.0.1:5501/index.html](http://127.0.0.1:5501/index.html)** in your browser.  
You should see the top-right status badge: `● Ready (14 chunks, qwen2.5:7b)`.

---

## 3. What to Upload, Where, and How to Ask Questions

### Step 3: What to Upload and Where
1. In the web interface, click the **"Upload Research PDF"** tab.
2. Under **"Select PDF Document"**, click **Choose File** and select any scientific research PDF (e.g., `data/raw/sample.pdf` or any Alzheimer's / biomedical / solar energy paper).
3. *(Optional)* Enter a Paper ID like `AD-PAPER-2026` or `SOLAR-PV-2023`.
4. Click **"Process & Ingest Paper"**.
   - The backend automatically extracts text page-by-page, generates 384-d BGE embeddings, indexes vectors into FAISS, and inserts entities/relations into Neo4j in **~3 to 8 seconds**.
   - The status badge will update to `Ready (14 + N chunks, qwen2.5:7b)`.

### Step 4: Ask Questions & Discover Hypotheses
Click the **"Hypothesis & Discovery"** tab and enter a prompt:

#### A. In-Scope Scientific Question (Example)
- **Prompt:**
  ```text
  How does Donepezil modulate cognitive decline via acetylcholinesterase inhibition?
  ```
- **Output:**
  - Returns an evidence-grounded hypothesis.
  - Confidence score (`HIGH` / `MODERATE`).
  - Top retrieval similarity score.
  - Exact PDF page references and character offsets.
  - Proposed laboratory validation assays with concrete, measurable metrics.

#### B. Out-of-Scope Question (Example)
- **Prompt:**
  ```text
  How does quantum qubit superconducting circuitry perform quantum error correction?
  ```
- **Output:**
  - Returns a **LOW** confidence alert (`scope: out_of_scope`).
  - Explains that the literature lacks empirical evidence for quantum computing, without hallucinating false claims.

---

## 4. Alternative: Run Everything from Terminal (CLI)

You can also run discovery directly from the command line:
```powershell
# Discover top transitive gaps from the knowledge graph
.venv\Scripts\python query_transitive.py --limit 3

# Ask a specific question directly
.venv\Scripts\python query_transitive.py --query "How does forecasting accuracy improve battery storage scheduling?"

# Test out-of-scope query
.venv\Scripts\python query_transitive.py --query "quantum qubit superconducting circuitry error correction"
```

---

## 5. Execute Test Suite
To verify all pipeline logic, fast-path extraction, paper isolation, and prompt protections:
```powershell
.venv\Scripts\python -m pytest tests -q
```
**Results:** **34 passed (100%)** across 6 test modules:
- `tests/test_paper_isolation.py`: 6 passed (paper isolation, generic entity exclusion, fast relation extraction, batch builders).
- `tests/test_api.py`: 9 passed (FastAPI endpoints, paper lists, upload validation, out-of-scope handling).
- `tests/test_hypothesis_generator.py`: 8 passed (anti-echo, measurable metrics, honest confidence, mock LLM provider).
- `tests/test_relation_extractor.py`: 4 passed (dependency extraction, source evidence preservation, offsets).
- `tests/test_chunk_provenance.py`: 3 passed (exact and normalized page provenance).
- `tests/test_importer.py`: 4 passed (entity caching, reuse, stopword filtering).

---

## 6. Performance Benchmarks

| Pipeline Stage | Before Optimization | After Optimization | Improvement |
| :--- | :--- | :--- | :--- |
| **Relation Extraction** | ~8 minutes (serial Ollama calls) | ~1.2 seconds (SciSpaCy fast-path) | **~400x faster** |
| **PDF Ingestion Total** | **~9 to 10 minutes** | **~3 to 8 seconds** | **~100x speedup** |
| **Neo4j DB Writes** | Serial queries | Single `UNWIND $batch` transaction | Bulk efficiency |
| **Paper Isolation** | Unfiltered global graph | Parameterized by `paper_id` | Zero cross-contamination |
| **Test Suite Passing** | 28 tests passed | **34 tests passed** | Comprehensive test coverage |
