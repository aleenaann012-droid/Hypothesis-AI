import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import requests
from dotenv import load_dotenv

load_dotenv()


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Generate text or JSON completion from prompt."""
        pass


class OllamaProvider(BaseLLMProvider):
    """Local Ollama LLM provider."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        # Default to working local model qwen2.5:7b
        self.model = (
            model
            or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        )
        self.timeout = timeout or int(os.getenv("OLLAMA_TIMEOUT", "25"))
        self.use_gbnf = os.getenv("OLLAMA_USE_GBNF", "false").lower() in ("true", "1", "yes")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        # Only enable GBNF grammar constrained decoding if explicitly requested,
        # as GBNF on CPU/hybrid GPU causes severe token-generation latency
        if json_mode and self.use_gbnf:
            payload["format"] = "json"

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "").strip()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Ollama request to {self.base_url} ({self.model}) failed: {exc}"
            ) from exc


class HostedLLMProvider(BaseLLMProvider):
    """Hosted OpenAI-compatible provider (OpenAI, OpenRouter, Groq, local vLLM)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("HOSTED_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = (
            base_url
            or os.getenv("HOSTED_LLM_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.model = model or os.getenv("HOSTED_LLM_MODEL", "gpt-4o-mini")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        if not self.api_key:
            raise ValueError(
                "Hosted LLM provider selected, but no API key provided. "
                "Set HOSTED_LLM_API_KEY or OPENAI_API_KEY in environment."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Hosted LLM request to {self.base_url} ({self.model}) failed: {exc}"
            ) from exc


class MockLLMProvider(BaseLLMProvider):
    """Deterministic mock provider for offline testing and verification."""

    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = responses or []
        self._call_count = 0

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        self._call_count += 1
        if self.responses:
            idx = (self._call_count - 1) % len(self.responses)
            return self.responses[idx]

        if json_mode:
            return json.dumps({
                "statement": "Elevated neuroinflammation may accelerate amyloid-beta aggregation in cortical neurons.",
                "core_variables": {
                    "independent_variable": "neuroinflammation",
                    "dependent_variable": "amyloid-beta aggregation",
                    "mechanism": "microglial cytokine release impairing clearance pathways"
                },
                "supporting_evidence": [
                    {
                        "quote": "Microglial activation correlates with amyloid plaque deposition.",
                        "page_references": [1],
                        "chunk_id": 0
                    }
                ],
                "contradicting_or_limiting_evidence": [
                    "Early anti-inflammatory interventions did not uniformly reverse existing plaque loads."
                ],
                "uncertainty_assessment": "Moderate uncertainty: association demonstrated, but causal timing across early vs late disease stages requires longitudinal verification.",
                "proposed_experiments": [
                    {
                        "title": "Cytokine inhibition in transgenic murine models",
                        "methodology": "Administer selective IL-1beta inhibitors to 3xTg-AD mice and quantify plaque burden via PET imaging.",
                        "expected_outcome": "Reduced plaque accumulation compared to untreated control cohorts."
                    }
                ],
                "validation_status": "unvalidated"
            })
        return "Hypothesis: Elevated neuroinflammation impairs clearance of amyloid-beta via microglial dysfunction."


def get_llm_provider(provider_type: Optional[str] = None) -> BaseLLMProvider:
    """Factory to instantiate configured LLM provider."""
    p_type = (provider_type or os.getenv("LLM_PROVIDER", "ollama")).lower().strip()

    if p_type == "hosted":
        return HostedLLMProvider()
    elif p_type == "mock":
        return MockLLMProvider()
    else:
        return OllamaProvider()
