import pytest
from hypothesis.llm_provider import (
    HostedLLMProvider,
    MockLLMProvider,
    OllamaProvider,
    get_llm_provider,
)
from hypothesis.generator import (
    generate_hypothesis,
    generate_structured_hypothesis,
    format_evidence_for_prompt,
)


def test_format_evidence_for_prompt():
    evidence = [
        {
            "chunk_id": 0,
            "text": "Amyloid plaques accumulate in Alzheimer cortex.",
            "page_references": [1, 2],
        },
        {
            "chunk_id": 1,
            "text": "Neuroinflammation impairs microglial clearance.",
            "start_page": 3,
        },
    ]

    formatted = format_evidence_for_prompt(evidence)
    assert "[Chunk ID: 0 | Page(s): 1, 2]" in formatted
    assert "[Chunk ID: 1 | Page(s): 3]" in formatted
    assert "Amyloid plaques accumulate" in formatted


def test_generate_structured_hypothesis_empty_evidence():
    result = generate_structured_hypothesis([])
    assert result["validation_status"] == "unvalidated"
    assert "Insufficient" in result["statement"]
    assert result["supporting_evidence"] == []
    assert result["uncertainty_assessment"]["level"] == "high"


def test_generate_structured_hypothesis_with_mock_provider():
    mock_provider = MockLLMProvider()
    evidence = [
        {
            "chunk_id": 0,
            "text": "Microglial activation correlates with amyloid plaque deposition.",
            "page_references": [1],
        }
    ]

    result = generate_structured_hypothesis(
        evidence_items=evidence,
        query="How does microglial activation relate to plaques?",
        provider=mock_provider,
    )

    assert result["validation_status"] == "unvalidated"
    assert "neuroinflammation" in result["statement"].lower() or "microglial" in result["statement"].lower()
    assert "core_variables" in result
    assert "supporting_evidence" in result
    assert "proposed_experiments" in result
    assert "uncertainty_assessment" in result


def test_validation_status_strictly_unvalidated_even_if_llm_returns_validated():
    # If a malicious or confused LLM returns validated, the system must enforce unvalidated
    tampered_json = (
        '{"statement": "Fact verified.", "validation_status": "validated", '
        '"supporting_evidence": [], "proposed_experiments": []}'
    )
    mock = MockLLMProvider([tampered_json])

    result = generate_structured_hypothesis(
        evidence_items=[{"chunk_id": 0, "text": "Some text"}],
        provider=mock,
    )
    assert result["validation_status"] == "unvalidated"


def test_hosted_llm_provider_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("HOSTED_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = HostedLLMProvider(api_key="")
    with pytest.raises(ValueError, match="Hosted LLM provider selected, but no API key provided"):
        provider.generate("test prompt")


def test_get_llm_provider_defaults_to_ollama(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    provider = get_llm_provider()
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "qwen2.5:7b"


def test_hypothesis_prevents_repeating_question_as_statement():
    # If the LLM echoes the question back verbatim, generator must replace with directional claim
    echo_json = '{"statement": "Does solar forecasting improve battery scheduling?"}'
    mock = MockLLMProvider([echo_json])

    evidence = [{
        "chunk_id": 1,
        "text": "Solar irradiance forecasting methods allow scheduling battery charging cycles.",
        "page_references": [12],
        "similarity": 0.85
    }]

    result = generate_structured_hypothesis(
        evidence_items=evidence,
        query="Does solar forecasting improve battery scheduling?",
        provider=mock,
    )

    # Must NOT end in question mark or repeat question verbatim
    assert not result["statement"].endswith("?")
    assert result["statement"] != "Does solar forecasting improve battery scheduling?"
    assert result["evidence_status"] in ("exploratory", "mechanistically_supported")
    assert "evidence_gap" in result
    assert result["confidence_assessment"]["scientific_confidence"] != "high"  # Conservative: never high prior to empirical assays


def test_experiments_include_measurable_metrics_and_honest_uncertainty():
    evidence = [{
        "chunk_id": 2,
        "text": "Forecasting errors in PV generation propagate into energy storage imbalances.",
        "page_references": [5],
        "similarity": 0.82
    }]

    # In fallback or structured generation, experiments must include measurable metrics
    result = generate_structured_hypothesis(
        evidence_items=evidence,
        query="How does forecasting accuracy affect battery scheduling?",
        provider=MockLLMProvider(["malformed json triggering fallback"])
    )

    assert len(result["proposed_experiments"]) > 0
    exp = result["proposed_experiments"][0]
    assert "measurable_metrics" in exp or "expected_outcome" in exp
    assert result["confidence_assessment"]["retrieval_similarity"] == 0.82
    assert result["confidence_assessment"]["scientific_confidence"] == "moderate"
    assert result["confidence_assessment"]["evidence_status"] == "exploratory"
    assert "knowledge_gaps" in result["uncertainty_assessment"]

