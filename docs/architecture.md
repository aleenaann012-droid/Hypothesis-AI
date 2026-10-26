# Hypothesis-AI: Architecture & Technical Design

## 1. System Architecture

Hypothesis-AI is built on a hybrid architecture combining dense vector retrieval (FAISS), scientific NLP (SciSpaCy), graph relationship modeling (Neo4j), and language models (Ollama / Hosted APIs) into an evidence-grounded scientific discovery assistant.

```
+-----------------------------------------------------------------------------------------+
|                                     RESEARCH PAPER                                      |
|                             (PDF: test_researchpaper.pdf)                               |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
                         +---------------------------------------+
                         |           nlp/pdf_extractor.py        |
                         |   extract_pages_from_pdf / extract... |
                         +---------------------------------------+
                                             |
             +-------------------------------+-------------------------------+
             |                                                               |
             v                                                               v
+--------------------------+                                     +--------------------------+
|  Scientific NLP Engine   |                                     |    Vector & Provenance   |
|  - entity_extractor.py   |                                     |  - embeddings.py         |
|  - relation_extractor.py |                                     |  - build_provenance.py   |
|  - en_core_sci_sm        |                                     |  - BGE-small-en-v1.5     |
+--------------------------+                                     +--------------------------+
             |                                                               |
             v                                                               v
+--------------------------+                                     +--------------------------+
|  Graph Builder & Importer|                                     |   FAISS Index & Chunks   |
|  - graph_builder.py      |                                     |  - embeddings.npy (384d) |
|  - importer.py           |                                     |  - chunks.txt (14 chunks)|
|  - Paper, Entity,        |                                     |  - chunk_provenance.json |
|    Claim, Evidence       |                                     +--------------------------+
+--------------------------+                                                 |
             |                                                               |
             v                                                               |
+--------------------------+                                                 |
|  Neo4j Knowledge Graph   |                                                 |
|  - bolt://localhost:7687 |                                                 |
|  - queries.py            |                                                 |
|  - missing_links.py      |                                                 |
|  - evidence.py           |                                                 |
+--------------------------+                                                 |
             \                                                              /
              \                                                            /
               v                                                          v
           +------------------------------------------------------------------+
           |                         FastAPI Core                             |
           |                        (api/main.py)                             |
           |   Endpoints:                                                     |
           |     POST /hypothesis     - Structured testable hypothesis        |
           |     POST /retrieve       - Exact provenance & page references    |
           |     GET  /research-gaps  - Graph 2-hop mediator gaps             |
           |     GET  /missing-links  - Open triangle link candidates         |
           |     GET  /graph          - Graph relationships                   |
           |     GET  /health         - Service diagnostics                   |
           +------------------------------------------------------------------+
                                             |
                                             v
                         +---------------------------------------+
                         |      hypothesis/llm_provider.py       |
                         |  - OllamaProvider (qwen2.5:7b)        |
                         |  - HostedLLMProvider (OpenAI-compat)  |
                         |  - MockLLMProvider (Offline CI)       |
                         +---------------------------------------+
```

---

## 2. Provenance & Dual Offset Architecture

A critical innovation in Hypothesis-AI is the distinction between **normalized text offsets** and **exact original PDF offsets**:

### Why Two Coordinate Spaces?
1. **Normalized Document Text (`offset_type: "normalized_document_text"`)**:
   - Hyphens broken across newlines (`re.sub(r"(?<=\w)-\s+(?=\w)", "", text)`) and irregular whitespace are normalized.
   - Enables resilient full-text search and NLP sentence boundary alignment without broken hyphenated tokens.
2. **Exact Original PDF Offsets (`exact_offset_type: "raw_extracted_pdf_text"`)**:
   - Points directly to character indices in the unnormalized PDF text extracted by `pypdf`.
   - Broken down into `page_spans`, allowing frontends and PDF viewers to highlight the exact snippet on Page $X$ from character $A$ to character $B$.

### Data Structure in `chunk_provenance.json`:
```json
{
  "chunk_index": 1,
  "start_page": 1,
  "end_page": 2,
  "page_references": [1, 2],
  "normalized_document_start": 3135,
  "normalized_document_end": 6204,
  "offset_type": "normalized_document_text",
  "exact_document_start": 3190,
  "exact_document_end": 6312,
  "exact_offset_type": "raw_extracted_pdf_text",
  "page_spans": [
    {
      "page_number": 1,
      "page_start_char": 3190,
      "page_end_char": 3827,
      "length": 637
    },
    {
      "page_number": 2,
      "page_start_char": 0,
      "page_end_char": 2484,
      "length": 2484
    }
  ]
}
```

---

## 3. Knowledge Graph Schema

Hypothesis-AI preserves Neo4j relationships while extending evidence traceability:

### Node Labels
- `:Paper`: Represented by `paper_id` and `title`.
- `:Entity`: Represented by `entity_id`, `name`, and `type`.
- `:Claim`: Represented by `claim_id`, `predicate`, `extraction_method`, and `validation_status: "unvalidated"`.
- `:Evidence`: Represented by `evidence_id`, `text`, `sentence_start`, `sentence_end`, `relation_start`, `relation_end`.

### Relationships
- `(p:Paper)-[:MENTIONS]->(e:Entity)`
- `(p:Paper)-[:HAS_EVIDENCE]->(ev:Evidence)`
- `(ev:Evidence)-[:SUPPORTS_CLAIM]->(c:Claim)`
- `(c:Claim)-[:SUBJECT]->(s:Entity)`
- `(c:Claim)-[:OBJECT]->(o:Entity)`
- `(s:Entity)-[:PREDICATE]->(o:Entity)` *(direct edge for traversal and visualization compatibility)*

---

## 4. Hypothesis Generation Schema & Constraints

To ensure scientific rigor, hypotheses generated by Hypothesis-AI conform to the following schema:

```json
{
  "statement": "string (falsifiable scientific claim)",
  "core_variables": {
    "independent_variable": "string",
    "dependent_variable": "string",
    "proposed_mechanism": "string"
  },
  "supporting_evidence": [
    {
      "quote": "string",
      "chunk_id": "integer",
      "page_references": ["integer"]
    }
  ],
  "contradicting_or_limiting_evidence": ["string"],
  "uncertainty_assessment": {
    "level": "low | moderate | high",
    "rationale": "string",
    "knowledge_gaps": ["string"]
  },
  "proposed_experiments": [
    {
      "title": "string",
      "methodology": "string",
      "measured_variable": "string",
      "expected_outcome": "string"
    }
  ],
  "validation_status": "unvalidated"
}
```

### Safety Guarantees
- The `validation_status` field is programmatically enforced to `"unvalidated"`. Even if an LLM returns a hallucinated `"validated"` tag, the parser overrides it before delivering results to the user.
- Citations are restricted to chunks retrieved by the vector index.
