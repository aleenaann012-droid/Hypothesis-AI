import pytest
from fastapi.testclient import TestClient
from api.main import app
from hypothesis.llm_provider import MockLLMProvider

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "HypothesisAI"
    assert "provenance_aware_retrieval" in data["capabilities"]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["retrieval"] == "ready"
    assert data["indexed_chunks"] == 14
    assert data["working_llm"] == "qwen2.5:7b"


def test_retrieve_endpoint():
    response = client.post(
        "/retrieve",
        json={"query": "amyloid beta and tau proteins in Alzheimer", "top_k": 2},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["evidence_count"] == 2
    assert len(data["evidence"]) == 2

    first = data["evidence"][0]
    assert "chunk_id" in first
    assert "similarity" in first
    assert "text" in first
    assert "start_page" in first
    assert "end_page" in first
    assert "page_references" in first
    assert first["offset_type"] == "normalized_document_text"
    assert first["exact_offset_type"] == "raw_extracted_pdf_text"
    assert "page_spans" in first


def test_hypothesis_endpoint_has_provenance_and_unvalidated_status(monkeypatch):
    import hypothesis.generator as gen_module

    # Inject mock provider into generator to ensure fast deterministic response without live Ollama
    mock_provider = MockLLMProvider()
    monkeypatch.setattr(gen_module, "get_llm_provider", lambda: mock_provider)

    response = client.post(
        "/hypothesis",
        json={"query": "How does amyloid beta affect neural communication?", "top_k": 2},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["validation_status"] == "unvalidated"
    assert data["evidence_count"] == 2
    assert "hypothesis" in data
    assert "structured_hypothesis" in data

    structured = data["structured_hypothesis"]
    assert structured["validation_status"] == "unvalidated"
    assert "statement" in structured
    assert "core_variables" in structured
    assert "supporting_evidence" in structured
    assert "proposed_experiments" in structured
    assert "uncertainty_assessment" in structured

    # Verify provenance attached to each evidence chunk
    for ev in data["evidence"]:
        assert ev["start_page"] is not None
        assert len(ev["page_references"]) >= 1
        assert ev["exact_offset_type"] == "raw_extracted_pdf_text"


def test_research_gaps_endpoint():
    response = client.get("/research-gaps?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "gaps" in data
    assert data["validation_status"] == "unvalidated"

    if data["count"] > 0:
        first_gap = data["gaps"][0]
        assert "bridge_path" in first_gap
        assert "hypothesized_relationship" in first_gap
        assert first_gap["validation_status"] == "unvalidated"


def test_missing_links_endpoint():
    response = client.get("/missing-links?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "candidates" in data


def test_out_of_scope_query_returns_low_confidence():
    # Query completely unrelated to the indexed neuroscience paper
    out_of_scope_query = "Quantum computing entanglement in superconducting qubits at cryogenic temperatures"
    response = client.post(
        "/hypothesis",
        json={"query": out_of_scope_query, "top_k": 2},
    )
    assert response.status_code == 200
    data = response.json()
    structured = data.get("structured_hypothesis", {})
    conf = structured.get("confidence_assessment", {})

    assert conf.get("scope") == "out_of_scope"
    assert conf.get("scientific_confidence") == "low"
    assert "Insufficient" in structured.get("statement", "")
    assert data["validation_status"] == "unvalidated"


def test_upload_invalid_file_format_rejected():
    files = {"file": ("test.txt", b"Invalid text file content", "text/plain")}
    response = client.post("/upload-paper", files=files)
    assert response.status_code == 400
    assert "Only PDF" in response.json()["detail"]


def test_upload_valid_pdf(tmp_path):
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    pdf_path = tmp_path / "sample_test.pdf"
    with open(pdf_path, "wb") as f:
        writer.write(f)

    # Empty blank page with no text should return 422
    with open(pdf_path, "rb") as f:
        response = client.post(
            "/upload-paper",
            files={"file": ("sample_test.pdf", f, "application/pdf")},
            data={"paper_id": "TEST_BLANK"}
        )
    assert response.status_code == 422
    assert "no extractable text" in response.json()["detail"]

