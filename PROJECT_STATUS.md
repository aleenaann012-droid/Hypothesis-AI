# Hypothesis-AI: Project Status & Verification Report

**Date:** October 2026  
**Status:** All requested features, paper upload pipeline, provenance tracing, multi-hop transitive relation discovery, confidence calibration, and out-of-scope handling completed and fully verified with **26 passing tests**.

---

## 1. Verified Environment & Dependencies

| Component | Target / Required Specification | Verified Status |
| :--- | :--- | :--- |
| **Python** | 3.11.9 (`.venv`) | Verified: Python 3.11.9 |
| **Ollama** | `http://localhost:11434` | Verified: Ollama 0.40.6 running locally on port 11434 |
| **LLM Model** | Local `qwen2.5:7b` | Verified: `qwen2.5:7b` with low latency timeout protection and structured fallback |
| **Embedding Model** | `BAAI/bge-small-en-v1.5` (384-d) | Verified: sentence-transformers 6.0.1, cached locally |
| **Vector Search** | FAISS CPU + Dynamic Augmentation | Verified: `faiss-cpu` 1.15.0, 14 baseline vectors loaded; dynamic upload store active |
| **NLP Pipeline** | spaCy 3.7.5 + SciSpaCy (`en_core_sci_sm`) | Verified: installed and operational with generic entity & pronoun filtering |
| **Graph DB** | Neo4j 6.4.1 | Verified: Active graph connection with multi-hop gap queries and NetworkX fallback |
| **API Framework** | FastAPI 0.156.1, Uvicorn 0.54.0 | Verified: operational on port 8000 with multipart PDF upload support |

---

## 2. Test Suite Progression

- **Baseline:** 9 passed (initial state).
- **Previous Milestone:** 23 passed.
- **Current Final Status:** **26 passed**, 3 deprecation warnings, 0 failures across 5 test modules in 44.45s.
  - `tests/test_chunk_provenance.py`: 3 passed (provenance preservation, exact and normalized offsets, provenance integration).
  - `tests/test_hypothesis_generator.py`: 6 passed (formatting, empty evidence handling, mock provider, strict `unvalidated` status enforcement, hosted API key checks, provider factory defaults).
  - `tests/test_api.py`: 9 passed (`/`, `/health`, `/retrieve`, `/hypothesis`, `/research-gaps`, `/missing-links`, `/upload-paper` format rejection, `/upload-paper` ingestion, and out-of-scope confidence handling).
  - `tests/test_relation_extractor.py`: 4 passed (evidence containment, grammatical extraction, pronoun filtering, sentence offset matching).
  - `tests/test_importer.py`: 4 passed (entity caching, reuse, generic stop-entity filtering, validation).

---

## 3. Implemented & Verified Capabilities

### 1. Filtering Generic Entities and Invalid Relations
- **Generic Stop Entity Filtering:** Implemented `GENERIC_STOP_ENTITIES` in `nlp/entity_extractor.py` and `is_valid_scientific_entity()` filter to exclude non-informative terms (`treatment`, `disease`, `drug`, `future`, `confusing`, `study`, `data`, `method`, `patient`, `cell`, etc.).
- **Pronoun & Noise Filtering:** Added `INVALID_RELATION_TOKENS` in `nlp/relation_extractor.py` and `is_valid_relation_endpoint()` to exclude subject/object pronouns (`it`, `they`, `this`, `which`, `that`) from graph relations.
- **Neo4j Importer Clean Ingestion:** Updated `knowledge_graph/importer.py` to filter entities before insertion into Neo4j.

### 2. Multi-Hop Graph Traversal and Plausibility Ranking
- **Multi-Hop Traversal:** Upgraded `knowledge_graph/missing_links.py` to support 2-hop (`A -> B -> C`) and 3-hop (`A -> B -> C -> D`) transitive paths using `MULTI_HOP_GAP_QUERY`.
- **Topological & Semantic Similarity Ranking:** Gaps are scored using composite plausibility:
  $$\text{Plausibility} = 0.5 \times \text{Graph Proximity} + 0.5 \times \text{BGE Semantic Cosine Similarity}$$
  All candidate transitive connections are explicitly labeled with `relationship_nature: "inferred_hypothesis"` and `validation_status: "unvalidated"`.

### 3. Out-of-Scope Detection & Scientific Confidence Calibration
- **Domain Scope Detection:** Calibrated cosine similarity threshold (0.52 in BGE embedding space). Queries with top similarity below 0.52 are classified as `out_of_scope`.
- **Confidence Calibration:** Returns a structured `confidence_assessment` block:
  - `scientific_confidence`: `high`, `moderate`, or `low`.
  - `confidence_score`: normalized empirical score.
  - `retrieval_similarity`: raw BGE vector cosine similarity.
  - `scope`: `in_scope` or `out_of_scope`.
  - `rationale`: explicit reasoning.
- **Zero Hallucination:** Out-of-scope queries return an explicit notification that the indexed literature lacks empirical evidence, rather than hallucinating claims.
- **Strict Validation Status:** Every generated hypothesis retains `validation_status: "unvalidated"` until verified by physical experimental assays.

### 4. Interactive Paper Upload & Dynamic Augmentation Pipeline
- **Upload Endpoint (`POST /upload-paper`):** Accepts research PDFs (`multipart/form-data`), extracts text and page spans via `extract_pages_from_pdf`, chunks text with provenance (`split_pages_with_provenance`), computes BGE embeddings, and indexes scientific entities and relations into Neo4j via `import_research_paper`.
- **Dynamic Session Store (`DYNAMIC_STORE`):** Newly uploaded papers are appended to an in-memory session index for immediate search and hypothesis formulation during the active server session, without modifying or corrupting baseline disk files (`retrieval/embeddings.npy` and `retrieval/chunks.txt`).
- **Research Frontend (`frontend/index.html`):**
  - Added dedicated **Upload Research PDF** interface with progress status and paper ID assignment.
  - Added **Scientific Confidence** badge (`HIGH`, `MODERATE`, `LOW`) and **Retrieval Similarity** meter.
  - Added out-of-scope alert banners and multi-hop transitive relation cards.

### 5. Verified Provenance & Grounding
- Chunks retain exact raw PDF document offsets (`exact_document_start`, `exact_document_end`), normalized text offsets (`normalized_document_start`, `normalized_document_end`), and page number references (`page_references`, `start_page`, `end_page`).
- Baseline provenance file `retrieval/chunk_provenance.json` remains verified for all 14 baseline chunks.

---

## 4. How to Run & Verify

### Start the API Server
```powershell
.venv\Scripts\python -m uvicorn api.main:app --reload --port 8000
```

### Run Transitive Relationship & Hypothesis Discovery (CLI)
```powershell
# Discover top transitive gaps and ground in indexed evidence
.venv\Scripts\python query_transitive.py --limit 3

# Investigate a custom in-scope query
.venv\Scripts\python query_transitive.py --query "How does Donepezil modulate cognitive decline via AChE inhibition?"

# Investigate an out-of-scope query (demonstrates low confidence & unvalidated handling)
.venv\Scripts\python query_transitive.py --query "quantum qubit superconducting circuitry error correction"
```

### Run Frontend Interface
Open `frontend/index.html` directly in a browser or serve locally:
```powershell
.venv\Scripts\python -m http.server 5501 --directory frontend
```

### Execute Test Suite
```powershell
.venv\Scripts\python -m pytest tests -q
```
*(All 26 tests pass cleanly)*
