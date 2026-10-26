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
