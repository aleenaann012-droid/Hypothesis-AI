# Hypothesis-AI

**Hypothesis-AI** is an AI-powered scientific discovery assistant that helps researchers explore academic literature, retrieve relevant paper passages with exact source provenance, explore knowledge graph relationships, uncover research gaps, and formulate evidence-grounded, testable scientific hypotheses.

---

## Key Features

1. **Provenance-Aware Retrieval**
   - Matches vector similarity (FAISS with `BAAI/bge-small-en-v1.5`) directly to PDF page numbers and character offsets.
   - Distinguishes **normalized document offsets** from **exact original PDF offsets** and **page-specific character spans**.
   - Zero hallucinated citations: all retrieved evidence cites verbatim text chunks from source documents.

2. **Evidence-Grounded, Testable Hypothesis Generation**
   - Synthesizes retrieved evidence into structured scientific hypotheses.
   - Identifies **independent variables**, **dependent variables**, and **mechanistic pathways**.
   - Explicitly cites supporting evidence with page numbers and highlights contradicting or limiting findings.
   - Evaluates **uncertainty level** with qualitative rationale and explicit knowledge gaps.
   - Outlines concrete **proposed validation experiments** (methodology, measured variables, expected outcomes).
   - **Scientific Integrity Safeguard:** All AI-generated hypotheses are strictly tagged as `unvalidated` until verified by genuine empirical validation.

3. **Graph-Based Research Gap Discovery**
   - Discovers open triangles: $(A) \to (B) \to (C)$ where no direct relationship $(A) \to (C)$ is recorded.
   - Formulates mediator hypotheses (e.g. *"Investigate whether APOE modulates Neuroinflammation via intermediate mediator Alzheimer's disease"*).
   - Resilient architecture with `networkx` graph fallback when Neo4j is offline.

4. **Preserved Neo4j Knowledge Graph**
   - Schema links `Paper`, `Entity`, `Claim`, and `Evidence` nodes without destructive migrations.
   - Directly traces entity relationships to sentence-level evidence text and character offsets.

5. **Configurable LLM Provider Abstraction**
   - **Local Ollama** (default: `qwen2.5:7b` at `http://localhost:11434`).
   - **Optional Hosted Provider**: OpenAI-compatible endpoint support (OpenAI, OpenRouter, Groq, local vLLM).
   - **Mock Provider**: Zero-network test provider for continuous integration.

---

## Architecture Overview

```
                          [ Research Paper PDF ]
                                    │
                                    ▼
                         nlp/pdf_extractor.py
                        (Pages & Raw Document)
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        ▼                                                       ▼
nlp/entity_extractor.py                                retrieval/embeddings.py
nlp/relation_extractor.py                              (BAAI/bge-small-en-v1.5)
(SciSpaCy en_core_sci_sm)                                       │
        │                                                       ▼
        ▼                                              retrieval/faiss_search.py
knowledge_graph/graph_builder.py                       retrieval/build_provenance.py
knowledge_graph/importer.py                                     │
        │                                                       ▼
        ▼                                              [ FAISS Index & Provenance ]
 [ Neo4j Graph DB ]                                             │
 (Paper, Entity, Claim, Evidence)                               │
        │                                                       │
        └───────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
                            api/main.py (FastAPI)
                       hypothesis/generator.py
                       hypothesis/llm_provider.py
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            Researcher Web UI                  REST API
           (frontend/index.html)            (Port 8000)
```

---

## Quickstart

### Prerequisites
- Python 3.11.9 (recommended: use `.venv`)
- Ollama installed and running with `qwen2.5:7b`:
  ```bash
  ollama serve
  ollama run qwen2.5:7b
  ```
- (Optional) Neo4j Desktop or Neo4j server (default: `bolt://localhost:7687`)

### Environment Configuration
Configure your `.env` file in the project root:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
LLM_PROVIDER=ollama
```

To use an optional hosted LLM provider:
```env
LLM_PROVIDER=hosted
HOSTED_LLM_API_KEY=your_api_key
HOSTED_LLM_BASE_URL=https://api.openai.com/v1
HOSTED_LLM_MODEL=gpt-4o-mini
```

### Running the API Server
```bash
.venv\Scripts\python -m uvicorn api.main:app --reload --port 8000
```
Interactive API documentation is available at:
`http://127.0.0.1:8000/docs`

### Running the Researcher UI
Open `frontend/index.html` in your browser, or serve it with Python:
```bash
.venv\Scripts\python -m http.server 5501 --directory frontend
```
Visit `http://localhost:5501` to use the researcher interface.

---

## API Reference

### 1. `POST /retrieve`
Retrieve evidence chunks with full page numbers and exact character offsets.
- **Request:**
  ```json
  {
    "query": "amyloid beta and tau proteins in Alzheimer",
    "top_k": 3
  }
  ```
- **Response:**
  ```json
  {
    "query": "amyloid beta and tau proteins in Alzheimer",
    "evidence_count": 3,
    "evidence": [
      {
        "chunk_id": 0,
        "similarity": 0.8841,
        "text": "Alzheimer's Disease: A Comprehensive Review...",
        "start_page": 1,
        "end_page": 1,
        "page_references": [1],
        "normalized_document_start": 0,
        "normalized_document_end": 3134,
        "offset_type": "normalized_document_text",
        "exact_document_start": 0,
        "exact_document_end": 3189,
        "exact_offset_type": "raw_extracted_pdf_text",
        "page_spans": [
          {
            "page_number": 1,
            "page_start_char": 0,
            "page_end_char": 3189,
            "length": 3189
          }
        ]
      }
    ]
  }
  ```

### 2. `POST /hypothesis`
Generate an evidence-grounded, testable hypothesis.
- **Request:**
  ```json
  {
    "query": "How does neuroinflammation relate to amyloid-beta aggregation in Alzheimer's disease?",
    "top_k": 3
  }
  ```
- **Response:**
  ```json
  {
    "query": "...",
    "hypothesis": "Elevated neuroinflammation accelerates amyloid-beta aggregation...",
    "validation_status": "unvalidated",
    "structured_hypothesis": {
      "statement": "Elevated neuroinflammation accelerates amyloid-beta aggregation...",
      "core_variables": {
        "independent_variable": "neuroinflammation",
        "dependent_variable": "amyloid-beta aggregation",
        "proposed_mechanism": "cytokine release impairing microglial phagocytic clearance"
      },
      "supporting_evidence": [
        {
          "quote": "Microglial activation correlates with amyloid plaque deposition.",
          "chunk_id": 0,
          "page_references": [1]
        }
      ],
      "contradicting_or_limiting_evidence": [
        "Early anti-inflammatory interventions did not uniformly reverse existing plaque loads."
      ],
      "uncertainty_assessment": {
        "level": "moderate",
        "rationale": "Causal directionality requires longitudinal animal models.",
        "knowledge_gaps": ["Early vs late stage cytokine dynamics"]
      },
      "proposed_experiments": [
        {
          "title": "Cytokine inhibition assay in transgenic mice",
          "methodology": "Administer selective IL-1beta inhibitors to 3xTg-AD mice...",
          "measured_variable": "Plaque load via PET imaging",
          "expected_outcome": "Statistically significant reduction in plaque accumulation"
        }
      ],
      "validation_status": "unvalidated",
      "summary": "..."
    },
    "evidence_count": 3,
    "evidence": [...]
  }
  ```

### 3. `GET /research-gaps`
Discover unexplored graph-based transitive gaps and mediator hypotheses.
- **Query Parameter:** `limit` (default: 50)
- **Response:**
  ```json
  {
    "count": 2,
    "validation_status": "unvalidated",
    "gaps": [
      {
        "source": "APOE",
        "bridge": "Alzheimer's disease",
        "candidate_target": "Neuroinflammation",
        "bridge_path": "APOE --[ASSOCIATED_WITH]--> Alzheimer's disease --[ASSOCIATED_WITH]--> Neuroinflammation",
        "hypothesized_relationship": "Investigate whether APOE (GENE) modulates Neuroinflammation (PROCESS) via intermediate mediator Alzheimer's disease (DISEASE).",
        "validation_status": "unvalidated"
      }
    ]
  }
  ```

### 4. `GET /health`
System component diagnostic status (retrieval index, vector count, Neo4j, LLM).

---

## Testing

Run the full automated regression suite:
```bash
.venv\Scripts\python -m pytest tests -q
```
Current verification status: **23 passed in 38.57s**, zero failures.
