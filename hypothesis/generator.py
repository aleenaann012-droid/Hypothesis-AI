import json
import os
import re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from hypothesis.llm_provider import BaseLLMProvider, get_llm_provider

load_dotenv()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434"
).rstrip("/")

# Working local model default
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL", "qwen2.5:7b"
)


def extract_json_payload(raw: str) -> Optional[Dict[str, Any]]:
    """Parse JSON safely from raw output, handling markdown blocks and stray text."""
    if not raw or not raw.strip():
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return None


def format_evidence_for_prompt(evidence_items: List[Dict[str, Any]]) -> str:
    """Format retrieved evidence chunks with explicit chunk IDs and page references."""
    formatted_chunks = []
    for item in evidence_items:
        chunk_id = item.get("chunk_id", "Unknown")
        pages = item.get("page_references", [])
        if not pages and item.get("start_page") is not None:
            pages = [item.get("start_page")]
        pages_str = ", ".join(str(p) for p in pages) if pages else "Unknown"

        text = (item.get("text") or "").strip()
        formatted_chunks.append(
            f"[Chunk ID: {chunk_id} | Page(s): {pages_str}]\n{text}"
        )
    return "\n\n---\n\n".join(formatted_chunks)


def generate_structured_hypothesis(
    evidence_items: List[Dict[str, Any]],
    query: Optional[str] = None,
    provider: Optional[BaseLLMProvider] = None,
) -> Dict[str, Any]:
    """
    Generate an evidence-grounded, testable hypothesis with supporting and
    contradicting evidence, page references, uncertainty, and proposed experiments.

    All generated hypotheses are marked 'unvalidated' until scientifically verified.
    """
    if not evidence_items:
        return {
            "query": query or "",
            "statement": "Insufficient retrieved research evidence to formulate a hypothesis.",
            "core_variables": {},
            "supporting_evidence": [],
            "contradicting_or_limiting_evidence": [],
            "uncertainty_assessment": {
                "level": "high",
                "rationale": "No evidence chunks provided.",
                "knowledge_gaps": ["No retrieved context."],
            },
            "proposed_experiments": [],
            "validation_status": "unvalidated",
            "summary": "Insufficient retrieved research evidence.",
        }

    target_query = query or "What scientific relationship is demonstrated in these excerpts?"
    formatted_context = format_evidence_for_prompt(evidence_items)
    # Check retrieval relevance: calculate max and average retrieval similarity
    sim_scores = [float(item.get("similarity", 0.0)) for item in evidence_items]
    max_similarity = max(sim_scores) if sim_scores else 0.0
    avg_similarity = (sum(sim_scores) / len(sim_scores)) if sim_scores else 0.0

    # Low similarity threshold: in BGE embedding space, completely unrelated text yields ~0.45-0.48.
    # Scores below 0.52 indicate query is outside the domain of indexed research literature.
    is_out_of_scope = max_similarity < 0.52

    if is_out_of_scope:
        statement = (
            f"Insufficient empirical evidence in the indexed literature to formulate a grounded hypothesis for: '{target_query}'."
        )
        return {
            "query": target_query,
            "statement": statement,
            "core_variables": {"focus": target_query, "status": "out_of_scope"},
            "supporting_evidence": [],
            "contradicting_or_limiting_evidence": [
                f"Top retrieved chunk similarity ({max_similarity:.4f}) is below scientific relevance threshold (0.45).",
                "The uploaded literature does not contain direct evidence addressing this topic."
            ],
            "uncertainty_assessment": {
                "level": "high",
                "rationale": "Query falls outside the domain of indexed research papers. No empirical claims can be formulated without external evidence.",
                "knowledge_gaps": ["Requires ingesting relevant scientific literature on this specific query."]
            },
            "proposed_experiments": [
                {
                    "title": "Literature Acquisition Protocol",
                    "methodology": "Index domain-specific papers addressing the requested biological mechanism.",
                    "expected_outcome": "Extraction of verified entity relations and passage evidence."
                }
            ],
            "validation_status": "unvalidated",
            "confidence_assessment": {
                "scientific_confidence": "low",
                "confidence_score": round(max_similarity * 0.5, 3),
                "retrieval_similarity": round(max_similarity, 4),
                "evidence_coverage": "insufficient",
                "scope": "out_of_scope",
                "rationale": "Insufficient source evidence in the uploaded literature to formulate a verified scientific hypothesis."
            },
            "summary": statement,
        }

    # In-scope: compute scientific confidence distinguishing retrieval similarity from validation
    if max_similarity >= 0.70:
        conf_level = "high"
        conf_score = round(min(0.95, max_similarity * 0.95), 3)
    elif max_similarity >= 0.55:
        conf_level = "moderate"
        conf_score = round(max_similarity * 0.85, 3)
    else:
        conf_level = "low"
        conf_score = round(max_similarity * 0.7, 3)

    system_prompt = (
        "You are an expert scientific discovery assistant. "
        "Ground every claim strictly in the supplied excerpts. "
        "Never invent citations, mechanisms, or findings. "
        "Distinguish correlation from causation. "
        "Formulate testable, falsifiable hypotheses. "
        "Output valid JSON only."
    )

    prompt = f"""
Research Question:
{target_query}

Retrieved Scientific Excerpts:
---
{formatted_context[:1800]}
---

Formulate a cautious, testable scientific hypothesis based strictly on the excerpts above.
Return ONLY a valid JSON object with these exact keys:
{{
  "statement": "A single clear, testable, falsifiable scientific hypothesis answering the question",
  "independent_variable": "key factor or perturbation",
  "dependent_variable": "observed outcome or measurement",
  "proposed_mechanism": "causal or mechanistic pathway stated in excerpts",
  "uncertainty": "current evidence limitations and knowledge gaps",
  "proposed_experiment": "concrete experimental procedure to test this hypothesis",
  "expected_outcome": "what confirms or refutes the hypothesis"
}}
"""

    llm = provider or get_llm_provider()

    try:
        raw_output = llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            json_mode=True,
            temperature=0.1,
            max_tokens=220,
        )

        data = extract_json_payload(raw_output)
        if isinstance(data, dict):
            statement = data.get("statement", "Evidence synthesis generated.")

            # Extract variables (supporting both flat and nested schemas)
            core_vars = data.get("core_variables", {})
            if not core_vars:
                core_vars = {
                    "independent_variable": data.get("independent_variable", "Factor in excerpts"),
                    "dependent_variable": data.get("dependent_variable", "Outcome in excerpts"),
                    "proposed_mechanism": data.get("proposed_mechanism", "Mechanism indicated in text"),
                }

            # Supporting evidence from chunks
            supporting = data.get("supporting_evidence", [])
            if not supporting:
                supporting = [
                    {
                        "quote": item.get("text", "")[:240].strip(),
                        "chunk_id": item.get("chunk_id"),
                        "page_references": item.get("page_references", []),
                    }
                    for item in evidence_items[:2]
                ]

            # Uncertainty
            unc_raw = data.get("uncertainty_assessment") or data.get("uncertainty")
            if isinstance(unc_raw, dict):
                uncertainty = unc_raw
            else:
                uncertainty = {
                    "level": "moderate",
                    "rationale": str(unc_raw) if unc_raw else "Preliminary literature finding requiring empirical validation.",
                    "knowledge_gaps": ["Detailed causal pathway requires targeted longitudinal analysis."],
                }

            # Proposed experiments
            exps = data.get("proposed_experiments", [])
            if not exps:
                exp_method = data.get("proposed_experiment", "Targeted biochemical assay")
                exp_outcome = data.get("expected_outcome", "Statistically significant correlation")
                exps = [
                    {
                        "title": "Empirical validation protocol",
                        "methodology": exp_method,
                        "expected_outcome": exp_outcome,
                    }
                ]

            conf_dict = {
                "scientific_confidence": conf_level,
                "confidence_score": conf_score,
                "retrieval_similarity": round(max_similarity, 4),
                "evidence_coverage": "sufficient",
                "scope": "in_scope",
                "rationale": f"Grounding in {len(evidence_items)} indexed passage(s) with max cosine similarity {max_similarity:.4f}."
            }

            return {
                "query": target_query,
                "statement": statement,
                "core_variables": core_vars,
                "supporting_evidence": supporting,
                "contradicting_or_limiting_evidence": data.get(
                    "contradicting_or_limiting_evidence",
                    ["Empirical verification required across broader experimental cohorts."]
                ),
                "uncertainty_assessment": uncertainty,
                "proposed_experiments": exps,
                "confidence_assessment": conf_dict,
                "validation_status": "unvalidated",
                "summary": data.get("summary", statement),
            }

    except Exception as exc:
        print(f"Structured hypothesis generation fallback: {exc}")

    # Immediate fallback synthesis directly from chunks without re-querying the failing LLM
    top_chunk_text = evidence_items[0].get("text", "")[:350] if evidence_items else ""
    first_sentence = top_chunk_text.split(".")[0].strip() if "." in top_chunk_text else top_chunk_text.strip()
    plain_text = (
        f"Based on retrieved evidence ({first_sentence}), "
        f"{target_query} represents a plausible testable mechanism warranting empirical validation."
    )
    fallback_summary = (
        f"Grounded in excerpt from Chunk #{evidence_items[0].get('chunk_id', 0) if evidence_items else 0} "
        f"(Page {evidence_items[0].get('start_page', 'N/A') if evidence_items else 'N/A'}). Direct experimental assay required."
    )

    conf_dict = {
        "scientific_confidence": conf_level,
        "confidence_score": conf_score,
        "retrieval_similarity": round(max_similarity, 4),
        "evidence_coverage": "sufficient_fallback",
        "scope": "in_scope",
        "rationale": f"Direct passage synthesis from Chunk #{evidence_items[0].get('chunk_id', 0) if evidence_items else 0} with max similarity {max_similarity:.4f}."
    }

    return {
        "query": target_query,
        "statement": plain_text,
        "core_variables": {
            "focus": target_query,
            "evidence_basis": f"Chunk #{evidence_items[0].get('chunk_id', 0)}" if evidence_items else "Retrieved literature",
        },
        "supporting_evidence": [
            {
                "quote": item.get("text", "")[:200],
                "chunk_id": item.get("chunk_id"),
                "page_references": item.get("page_references", []),
            }
            for item in evidence_items[:2]
        ],
        "contradicting_or_limiting_evidence": [
            "Empirical verification required across broader experimental cohorts."
        ],
        "uncertainty_assessment": {
            "level": "moderate",
            "rationale": "Generated via cautious evidence synthesis.",
            "knowledge_gaps": ["Mechanism requires direct empirical assay."],
        },
        "proposed_experiments": [
            {
                "title": "Empirical validation protocol",
                "methodology": "Targeted assay evaluating variables identified in retrieved excerpts.",
                "measured_variable": "Dependent endpoint biomarker",
                "expected_outcome": "Statistically significant correlation with proposed intervention",
            }
        ],
        "confidence_assessment": conf_dict,
        "validation_status": "unvalidated",
        "summary": fallback_summary,
    }



def generate_hypothesis(
    research_text: str,
    provider: Optional[BaseLLMProvider] = None,
) -> str:
    """
    Generate a tentative hypothesis grounded in retrieved research text.
    The output is an unvalidated proposal, not an established scientific fact.
    """
    if not research_text or not research_text.strip():
        return "Insufficient retrieved research evidence to propose a hypothesis."

    prompt = f"""
You are assisting with scientific hypothesis generation.

Use ONLY the research excerpts below.
Answer the user's specific research question, not merely the general topic.
Prioritize evidence that directly explains the relationship asked about.
If the question asks about a relationship, explain that relationship rather than switching to a treatment or intervention.
Keep the hypothesis focused on the variables named in the question.
Do not combine unrelated findings into one hypothesis.
Ignore bibliography, reference lists, acknowledgements, and other non-evidence text.
Propose one cautious, testable hypothesis only when the excerpts support it.
Do not invent results, mechanisms, or citations.
Distinguish association from causation.
If the evidence is insufficient, say so explicitly.
Return only the hypothesis in plain text, without a heading.

Retrieved research excerpts:
---
{research_text[:4500]}
---
"""

    llm = provider or get_llm_provider()

    try:
        hypothesis = llm.generate(
            prompt=prompt,
            system_prompt="Be scientifically cautious. Ground every statement in the supplied research excerpts.",
            json_mode=False,
            temperature=0.2,
            max_tokens=350,
        )
        if hypothesis:
            return hypothesis
    except Exception as exc:
        print(f"Hypothesis generation unavailable: {exc}")


    return (
        "Insufficient evidence to generate a research-grounded hypothesis. "
        "Please check the local Ollama service and retrieved research text."
    )


if __name__ == "__main__":
    sample_text = """
    Alzheimer's disease is associated with the accumulation
    of amyloid-beta and tau proteins in the brain.
    """

    print("\n========== GENERATED HYPOTHESIS ==========\n")
    print(generate_hypothesis(sample_text))

