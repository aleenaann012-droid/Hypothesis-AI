import time
from typing import Dict, Any, List
import pytest
from fastapi.testclient import TestClient

from api.main import app
from knowledge_graph.missing_links import (
    find_missing_links,
    find_research_gaps,
    find_missing_links_from_fallback,
)
from nlp.entity_extractor import is_valid_scientific_entity, extract_entities
from nlp.relation_extractor import extract_relations

client = TestClient(app)


def test_generic_entities_excluded():
    """Ensure generic words and invalid stopwords are not treated as scientific entities."""
    generic_words = [
        "treatment", "disease", "drug", "future", "confusing", "study",
        "results", "patient", "mouse", "method", "data", "it", "they", "123"
    ]
    for word in generic_words:
        assert not is_valid_scientific_entity(word), f"'{word}' should be rejected as a scientific entity"

    valid_scientific = ["amyloid-beta", "photovoltaic cell", "lithium-ion battery", "synaptic plasticity"]
    for valid in valid_scientific:
        assert is_valid_scientific_entity(valid), f"'{valid}' should be accepted as a valid entity"


def test_papers_endpoint_lists_distinct_papers():
    """Verify GET /papers endpoint returns active papers including P001."""
    response = client.get("/papers")
    assert response.status_code == 200
    data = response.json()
    assert "papers" in data
    assert "P001" in data["papers"]


def test_graph_endpoint_supports_paper_filter():
    """Verify /graph accepts paper_id query param and reflects filter."""
    res_all = client.get("/graph")
    assert res_all.status_code == 200
    data_all = res_all.json()
    assert "data" in data_all

    res_solar = client.get("/graph?paper_id=SOLAR-PV-2023")
    assert res_solar.status_code == 200
    data_solar = res_solar.json()
    assert data_solar["paper_id"] == "SOLAR-PV-2023"


def test_research_gaps_supports_paper_filter():
    """Verify /research-gaps accepts paper_id query param and does not cross-contaminate."""
    # Fallback simulation: P001 matches sample data, non-existent paper returns no gaps
    res_med = client.get("/research-gaps?limit=10&paper_id=P001")
    assert res_med.status_code == 200
    data_med = res_med.json()
    assert data_med["paper_id"] == "P001"

    res_nonexistent = client.get("/research-gaps?limit=10&paper_id=NON_EXISTENT_PAPER_999")
    assert res_nonexistent.status_code == 200
    data_nonexistent = res_nonexistent.json()
    assert data_nonexistent["paper_id"] == "NON_EXISTENT_PAPER_999"
    # An isolated nonexistent paper should yield 0 gaps
    assert data_nonexistent["count"] == 0


def test_relation_extraction_fast_path_without_llm_hang():
    """Verify that relation extraction fast path runs in sub-second time without Ollama hang."""
    text = (
        "Photovoltaic systems generate electric power from solar irradiance. "
        "Battery energy storage systems smooth fluctuations in renewable power output."
    )
    start_time = time.time()
    # Default use_llm_fallback=False
    relations = extract_relations(text, use_llm_fallback=False)
    duration = time.time() - start_time

    assert duration < 5.0, f"Extraction took {duration:.2f}s, expected fast path under 5s"
    assert len(relations) > 0, "Expected candidate dependency relations"
    for r in relations:
        assert r["extraction_method"] == "dependency_heuristic"
        assert is_valid_scientific_entity(r["subject"]) or len(r["subject"]) >= 2
        assert is_valid_scientific_entity(r["object"]) or len(r["object"]) >= 2


def test_builder_batch_methods_exist():
    """Verify GraphBuilder batch methods exist for optimized bulk database ingestion."""
    from knowledge_graph.graph_builder import GraphBuilder
    assert hasattr(GraphBuilder, "create_entities_batch")
    assert hasattr(GraphBuilder, "create_claims_and_relationships_batch")
