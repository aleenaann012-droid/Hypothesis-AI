import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from knowledge_graph.neo4j_connection import Neo4jConnection
from knowledge_graph.graph_builder import GraphBuilder
from knowledge_graph.missing_links import find_missing_links, find_research_gaps, find_multi_hop_gaps
from knowledge_graph.evidence import get_direct_evidence
from knowledge_graph.importer import import_research_paper
from nlp.pdf_extractor import extract_text_from_pdf, extract_pages_from_pdf
from retrieval.embeddings import create_embeddings, split_text, split_pages_with_provenance
from retrieval.faiss_search import (
    load_embeddings,
    load_chunks,
    load_provenance,
    create_faiss_index,
    search,
    search_with_provenance,
)
from hypothesis.generator import generate_hypothesis, generate_structured_hypothesis


app = FastAPI(
    title="HypothesisAI API",
    description=(
        "Backend API for the HypothesisAI "
        "scientific discovery assistant"
    ),
    version="0.2.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5501",
        "http://127.0.0.1:5501",
        "http://localhost:3000",
        "http://localhost:8000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HypothesisRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


# Dynamic in-memory session store for newly uploaded PDFs during active runtime
DYNAMIC_STORE = {
    "active": False,
    "chunks": [],
    "embeddings": None,
    "provenance": [],
    "index": None,
    "uploaded_papers": []
}


def get_retrieval_resources():
    """
    Load vector index, raw text chunks, and provenance mappings.
    If new papers were uploaded dynamically in this session, uses the combined index.
    Otherwise loads the baseline pre-computed indices.
    """
    if DYNAMIC_STORE["active"] and DYNAMIC_STORE["index"] is not None:
        return DYNAMIC_STORE["index"], DYNAMIC_STORE["chunks"], DYNAMIC_STORE["provenance"]

    embeddings = load_embeddings()
    chunks = load_chunks()
    provenance = load_provenance()
    index = create_faiss_index(embeddings)

    if index.ntotal != len(chunks):
        raise RuntimeError(
            "FAISS embedding count does not match chunk count."
        )

    return index, chunks, provenance


@app.get("/")
def home():
    return {
        "project": "HypothesisAI",
        "status": "running",
        "module": "Scientific Discovery & Knowledge Graph Assistant",
        "version": "0.2.0",
        "capabilities": [
            "provenance_aware_retrieval",
            "exact_and_normalized_offsets",
            "evidence_grounded_hypotheses",
            "research_gap_detection",
            "neo4j_knowledge_graph",
        ],
    }


@app.get("/health")
def health():
    neo4j_status = "disconnected"
    neo4j_result = None
    try:
        db = Neo4jConnection()
        try:
            neo4j_result = db.verify_connection()
            neo4j_status = "connected"
        finally:
            db.close()
    except Exception as error:
        neo4j_status = f"unavailable ({error})"

    retrieval_status = "ready"
    chunks_count = 0
    try:
        _, chunks, _ = get_retrieval_resources()
        chunks_count = len(chunks)
    except Exception as error:
        retrieval_status = f"error ({error})"

    return {
        "status": "healthy" if retrieval_status == "ready" else "degraded",
        "retrieval": retrieval_status,
        "indexed_chunks": chunks_count,
        "neo4j": neo4j_status,
        "neo4j_result": neo4j_result,
        "working_llm": "qwen2.5:7b",
    }


@app.get("/graph")
def get_graph():
    query = """
    MATCH (a:Entity)-[r]->(b:Entity)
    RETURN
        a.entity_id AS source_id,
        a.name AS source,
        a.type AS source_type,
        type(r) AS relationship,
        b.entity_id AS target_id,
        b.name AS target,
        b.type AS target_type
    LIMIT 200
    """

    try:
        db = Neo4jConnection()
        try:
            with db.driver.session() as session:
                result = session.run(query)
                return {
                    "data": [dict(record) for record in result]
                }
        finally:
            db.close()
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Neo4j graph database unavailable: {error}"
        )


@app.get("/entity/{name}")
def get_entity(name: str):
    query = """
    MATCH (e:Entity)
    WHERE toLower(e.name) = toLower($name)
    OPTIONAL MATCH (e)-[r]-(other:Entity)
    RETURN
        e.entity_id AS entity_id,
        e.name AS entity,
        e.type AS type,
        type(r) AS relationship,
        other.entity_id AS connected_id,
        other.name AS connected_entity,
        other.type AS connected_type
    """

    try:
        db = Neo4jConnection()
        try:
            with db.driver.session() as session:
                result = session.run(query, name=name)
                data = [dict(record) for record in result]

                if not data:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Entity '{name}' not found."
                    )

                return {
                    "entity": name,
                    "data": data
                }
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Neo4j entity lookup unavailable: {error}"
        )


@app.get("/paper/{paper_id}")
def get_paper(paper_id: str):
    query = """
    MATCH (p:Paper)-[:MENTIONS]->(e:Entity)
    WHERE p.paper_id = $paper_id
    RETURN
        p.paper_id AS paper_id,
        p.title AS title,
        e.entity_id AS entity_id,
        e.name AS entity,
        e.type AS type
    """

    try:
        db = Neo4jConnection()
        try:
            with db.driver.session() as session:
                result = session.run(query, paper_id=paper_id)
                data = [dict(record) for record in result]

                if not data:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Paper '{paper_id}' not found."
                    )

                return {
                    "paper_id": paper_id,
                    "data": data
                }
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Neo4j paper lookup unavailable: {error}"
        )


@app.get("/missing-links")
def missing_links(limit: int = Query(default=100, ge=1, le=500)):
    """Retrieve 2-hop open triangles in knowledge graph."""
    try:
        links = find_missing_links(limit=limit)
        return {
            "count": len(links),
            "candidates": links
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.get("/research-gaps")
def research_gaps(limit: int = Query(default=50, ge=1, le=200)):
    """
    Discover unexplored research gaps across graph paths,
    formulating mediator hypotheses and bridge rationale.
    """
    try:
        gaps = find_research_gaps(limit=limit)
        return {
            "count": len(gaps),
            "validation_status": "unvalidated",
            "gaps": gaps
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.get("/evidence")
def evidence(source_id: str, target_id: str):
    """Retrieve paper and sentence-level evidence connecting two entities."""
    try:
        results = get_direct_evidence(source_id, target_id)
        return {
            "source_id": source_id,
            "target_id": target_id,
            "evidence_count": len(results),
            "evidence": results
        }
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Evidence retrieval unavailable: {error}"
        )


@app.post("/retrieve")
def retrieve_evidence(request: RetrieveRequest):
    """
    Direct evidence retrieval endpoint returning chunks enriched with
    full provenance metadata, page numbers, and exact/normalized offsets.
    """
    try:
        index, chunks, provenance = get_retrieval_resources()

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No research chunks are available."
            )

        results = search_with_provenance(
            query=request.query,
            index=index,
            top_k=min(request.top_k, len(chunks)),
            chunks=chunks,
            provenance=provenance,
        )

        return {
            "query": request.query,
            "evidence_count": len(results),
            "evidence": results,
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.get("/retrieve")
def retrieve_evidence_get(
    query: str = Query(min_length=3),
    top_k: int = Query(default=3, ge=1, le=10),
):
    return retrieve_evidence(RetrieveRequest(query=query, top_k=top_k))


@app.post("/hypothesis")
def create_hypothesis(request: HypothesisRequest):
    """
    Generate an evidence-grounded, testable hypothesis traceable to source evidence.
    Includes supporting/contradicting evidence, page references, uncertainty assessment,
    and proposed validation experiments.
    """
    try:
        index, chunks, provenance = get_retrieval_resources()

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No research chunks are available."
            )

        evidence = search_with_provenance(
            query=request.query,
            index=index,
            top_k=min(request.top_k, len(chunks)),
            chunks=chunks,
            provenance=provenance,
        )

        if not evidence:
            raise HTTPException(
                status_code=404,
                detail="No research evidence was retrieved."
            )

        structured = generate_structured_hypothesis(
            evidence_items=evidence,
            query=request.query,
        )

        hypothesis_text = (
            structured.get("statement")
            or structured.get("summary")
            or "Evidence synthesis complete."
        )

        return {
            "query": request.query,
            "hypothesis": hypothesis_text,
            "structured_hypothesis": structured,
            "validation_status": "unvalidated",
            "evidence_count": len(evidence),
            "evidence": evidence,
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.get("/hypothesis")
def create_hypothesis_get(
    query: str = Query(min_length=3),
    top_k: int = Query(default=3, ge=1, le=10),
):
    return create_hypothesis(HypothesisRequest(query=query, top_k=top_k))


@app.post("/upload-paper")
async def upload_paper(
    file: UploadFile = File(...),
    paper_id: Optional[str] = Form(None)
):
    """
    Upload and process a scientific research PDF:
    1. Validates PDF format.
    2. Extracts text and page spans.
    3. Generates BGE embeddings and indexes them in FAISS without overwriting disk baselines.
    4. Extracts scientific entities & relationships and imports them into Neo4j (if available).
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only PDF research papers are supported."
        )

    assigned_paper_id = (paper_id or Path(file.filename).stem).strip()

    # Save to a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_pdf_path = Path(temp_dir) / file.filename

    try:
        with open(temp_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Extract text and page structures
        try:
            pages = extract_pages_from_pdf(temp_pdf_path)
            full_text = extract_text_from_pdf(temp_pdf_path)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to read or parse PDF contents: {e}"
            )

        if not full_text.strip() or not pages:
            raise HTTPException(
                status_code=422,
                detail="Uploaded PDF contains no extractable text."
            )

        # 2. Chunk text and compute exact page provenance
        raw_chunks = split_text(full_text, chunk_size=500)
        provenance_chunks = split_pages_with_provenance(pages, chunk_size=500)

        # 3. Generate BGE embeddings
        new_embeddings = create_embeddings(raw_chunks)

        # 4. Initialize or augment dynamic FAISS session index
        if not DYNAMIC_STORE["active"]:
            base_embs = load_embeddings()
            base_chunks = load_chunks()
            base_prov = load_provenance()
            combined_embs = np.vstack([base_embs, new_embeddings])
            combined_chunks = list(base_chunks) + list(raw_chunks)
            combined_prov = list(base_prov)

            start_idx = len(base_chunks)
            for i, p_chunk in enumerate(provenance_chunks):
                combined_prov.append({
                    "chunk_index": start_idx + i,
                    "paper_id": assigned_paper_id,
                    "start_page": p_chunk["page_number"],
                    "end_page": p_chunk["page_number"],
                    "page_references": [p_chunk["page_number"]],
                    "normalized_document_start": p_chunk.get("document_start", 0),
                    "normalized_document_end": p_chunk.get("document_end", 0),
                    "offset_type": "normalized_document_text",
                    "exact_document_start": p_chunk.get("document_start", 0),
                    "exact_document_end": p_chunk.get("document_end", 0),
                    "exact_offset_type": "raw_extracted_pdf_text",
                    "page_spans": [{
                        "page_number": p_chunk["page_number"],
                        "page_start_char": p_chunk.get("page_start", 0),
                        "page_end_char": p_chunk.get("page_end", 0)
                    }]
                })

            new_index = create_faiss_index(combined_embs)
            DYNAMIC_STORE["active"] = True
            DYNAMIC_STORE["chunks"] = combined_chunks
            DYNAMIC_STORE["embeddings"] = combined_embs
            DYNAMIC_STORE["provenance"] = combined_prov
            DYNAMIC_STORE["index"] = new_index
        else:
            combined_embs = np.vstack([DYNAMIC_STORE["embeddings"], new_embeddings])
            start_idx = len(DYNAMIC_STORE["chunks"])
            DYNAMIC_STORE["chunks"].extend(raw_chunks)
            DYNAMIC_STORE["embeddings"] = combined_embs
            for i, p_chunk in enumerate(provenance_chunks):
                DYNAMIC_STORE["provenance"].append({
                    "chunk_index": start_idx + i,
                    "paper_id": assigned_paper_id,
                    "start_page": p_chunk["page_number"],
                    "end_page": p_chunk["page_number"],
                    "page_references": [p_chunk["page_number"]],
                    "normalized_document_start": p_chunk.get("document_start", 0),
                    "normalized_document_end": p_chunk.get("document_end", 0),
                    "offset_type": "normalized_document_text",
                    "exact_document_start": p_chunk.get("document_start", 0),
                    "exact_document_end": p_chunk.get("document_end", 0),
                    "exact_offset_type": "raw_extracted_pdf_text",
                    "page_spans": [{
                        "page_number": p_chunk["page_number"],
                        "page_start_char": p_chunk.get("page_start", 0),
                        "page_end_char": p_chunk.get("page_end", 0)
                    }]
                })
            DYNAMIC_STORE["index"] = create_faiss_index(combined_embs)

        DYNAMIC_STORE["uploaded_papers"].append(assigned_paper_id)

        # 5. Extract scientific entities & relations and import into Neo4j
        neo4j_status = "unattempted"
        entities_count = 0
        relations_count = 0
        try:
            builder = GraphBuilder()
            try:
                builder.create_constraints()
                import_research_paper(builder, str(temp_pdf_path), assigned_paper_id)
                neo4j_status = "imported"
            finally:
                builder.close()
        except Exception as e:
            neo4j_status = f"unavailable ({e})"

        return {
            "status": "success",
            "paper_id": assigned_paper_id,
            "filename": file.filename,
            "extracted_pages": len(pages),
            "character_count": len(full_text),
            "new_chunks_indexed": len(raw_chunks),
            "total_indexed_chunks": len(DYNAMIC_STORE["chunks"]),
            "vector_dimension": DYNAMIC_STORE["index"].d,
            "neo4j_import": neo4j_status,
            "validation_status": "unvalidated",
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


