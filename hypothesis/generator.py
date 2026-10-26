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

    # Distinguish retrieval similarity from scientific confidence:
    # High retrieval similarity only means the text is topically relevant;
    # Scientific confidence in an unvalidated hypothesis must remain conservative (never HIGH before empirical testing).
    # We assign:
    # - "exploratory" status when retrieved evidence establishes topic relevance but does not directly prove the causal mechanism
    # - "corroborated_proposal" when direct mechanistic linkages are mentioned in text
    evidence_status = "exploratory"
    if max_similarity >= 0.75:
        sci_conf_level = "moderate"
        sci_conf_score = round(min(0.65, max_similarity * 0.70), 3)
        sci_rationale = (
            f"Retrieved passages exhibit strong topical relevance (similarity {max_similarity:.4f}), "
            "but the proposed causal relationship is unvalidated and requires formal empirical testing."
        )
    elif max_similarity >= 0.60:
        sci_conf_level = "moderate"
        sci_conf_score = round(max_similarity * 0.60, 3)
        sci_rationale = (
            f"Moderate literature relevance (similarity {max_similarity:.4f}). "
            "Evidence supports related concepts, but causal directionality remains an exploratory hypothesis."
        )
    else:
        sci_conf_level = "low"
        sci_conf_score = round(max_similarity * 0.45, 3)
        sci_rationale = (
            f"Marginal retrieval similarity ({max_similarity:.4f}). "
            "Literature provides conceptual context but lacks direct mechanism evidence."
        )

    system_prompt = (
        "You are an expert scientific discovery assistant. "
        "Formulate a specific, testable, falsifiable candidate scientific hypothesis. "
        "CRITICAL RULES:\n"
        "1. NEVER simply repeat or rephrase the research question. Propose a SPECIFIC causal mechanism or directional relationship (e.g. 'Factor X may alter Outcome Y by mediating Pathway Z').\n"
        "2. Ground the claim strictly in the concepts present in the supplied excerpts.\n"
        "3. If the excerpts discuss the topic but do not directly prove the causal mechanism, label the evidence_status as 'exploratory' and state the evidence gap clearly.\n"
        "4. Propose a concrete experiment specifying measurable independent/dependent variables and quantitative evaluation metrics.\n"
        "5. Output valid JSON only."
    )

    prompt = f"""
Research Question:
{target_query}

Retrieved Scientific Excerpts:
---
{formatted_context[:2200]}
---

Formulate a specific, testable candidate scientific hypothesis answering the question above.
Do NOT repeat the question. Propose a directional relationship or mechanism supported or suggested by the excerpts.

Return ONLY a valid JSON object with these exact keys:
{{
  "statement": "A single specific, testable candidate hypothesis (e.g. 'Factor A may improve Metric B by reducing Mechanism C')",
  "evidence_status": "exploratory" or "mechanistically_supported",
  "evidence_gap": "Explicit statement of what the excerpts establish vs what they do not prove",
  "independent_variable": "exact perturbation or factor tested (with units/levels if applicable)",
  "dependent_variable": "exact measurable outcome metrics",
  "proposed_mechanism": "specific operational or biochemical mechanism linking variables",
  "uncertainty": "evidence limitations, potential confounding variables, and knowledge gaps",
  "proposed_experiment": "concrete comparative experimental protocol under controlled conditions",
  "measured_metrics": ["metric 1", "metric 2"],
  "expected_outcome": "specific quantitative outcome that confirms or refutes the hypothesis"
}}
"""

    llm = provider or get_llm_provider()

    try:
        raw_output = llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            json_mode=True,
            temperature=0.1,
            max_tokens=450,
        )

        data = extract_json_payload(raw_output)
        if isinstance(data, dict):
            raw_statement = data.get("statement", "").strip()

            # Prevent repeating or echoing the question as the hypothesis
            clean_query = target_query.lower().strip("?. ")
            clean_stmt = raw_statement.lower().strip("?. ")
            if not raw_statement or clean_stmt == clean_query or raw_statement.endswith("?"):
                # Formulate a specific directional claim using core extracted concepts
                top_text = evidence_items[0].get("text", "") if evidence_items else ""
                first_sent = top_text.split(".")[0].strip() if "." in top_text else "evidence in literature"
                raw_statement = (
                    f"Targeted modulation of {target_query.rstrip('?')} may exert a measurable effect "
                    f"by optimizing intermediate system dynamics as suggested by: '{first_sent}'."
                )

            evidence_gap = data.get(
                "evidence_gap",
                "Retrieved excerpts establish conceptual relevance, but do not directly confirm the causal link."
            )
            ev_status = data.get("evidence_status", "exploratory")

            # Extract variables (supporting both flat and nested schemas)
            core_vars = data.get("core_variables", {})
            if not core_vars:
                core_vars = {
                    "independent_variable": data.get("independent_variable", "Intervention / Operational parameter"),
                    "dependent_variable": data.get("dependent_variable", "Measured efficiency / Performance endpoint"),
                    "proposed_mechanism": data.get("proposed_mechanism", "Mediating pathway indicated in excerpts"),
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
                    "rationale": str(unc_raw) if unc_raw else (
                        "Literature establishes topic context, but direct causal mechanism is unverified. "
                        f"Evidence gap: {evidence_gap}"
                    ),
                    "knowledge_gaps": [
                        evidence_gap,
                        "Requires controlled comparative trials with standardized baseline metrics."
                    ],
                }

            # Proposed experiments with measurable metrics
            exps = data.get("proposed_experiments", [])
            if not exps:
                exp_method = data.get(
                    "proposed_experiment",
                    "Comparative evaluation under controlled baseline conditions."
                )
                exp_outcome = data.get(
                    "expected_outcome",
                    "Statistically significant improvement in primary measured metrics."
                )
                metrics = data.get("measured_metrics", [
                    core_vars.get("dependent_variable", "System performance metric")
                ])
                exps = [
                    {
                        "title": "Controlled comparative validation experiment",
                        "methodology": exp_method,
                        "measurable_metrics": metrics,
                        "expected_outcome": exp_outcome,
                    }
                ]

            conf_dict = {
                "scientific_confidence": sci_conf_level,
                "confidence_score": sci_conf_score,
                "retrieval_similarity": round(max_similarity, 4),
                "evidence_status": ev_status,
                "evidence_gap": evidence_gap,
                "scope": "in_scope",
                "rationale": sci_rationale,
            }

            return {
                "query": target_query,
                "statement": raw_statement,
                "evidence_status": ev_status,
                "evidence_gap": evidence_gap,
                "core_variables": core_vars,
                "supporting_evidence": supporting,
                "contradicting_or_limiting_evidence": data.get(
                    "contradicting_or_limiting_evidence",
                    [f"Excerpts establish topical relevance but do not directly confirm causal direction: {evidence_gap}"]
                ),
                "uncertainty_assessment": uncertainty,
                "proposed_experiments": exps,
                "confidence_assessment": conf_dict,
                "validation_status": "unvalidated",
                "summary": data.get("summary", raw_statement),
            }

    except Exception as exc:
        print(f"Structured hypothesis generation fallback: {exc}")

    # Immediate, grounded fallback synthesis without repeating the question
    top_chunk_text = evidence_items[0].get("text", "")[:350] if evidence_items else ""
    first_sentence = top_chunk_text.split(".")[0].strip() if "." in top_chunk_text else top_chunk_text.strip()

    # Build a specific, directional candidate claim
    plain_text = (
        f"Targeted modulation of parameters related to '{target_query.rstrip('?')}' "
        f"may systematically improve operational outcomes by mediating mechanisms highlighted in literature: '{first_sentence}'."
    )
    fallback_gap = (
        "Retrieved excerpts establish relevance to this domain, but do not report controlled trials directly proving the causal link."
    )
    fallback_summary = (
        f"Exploratory candidate hypothesis grounded in Chunk #{evidence_items[0].get('chunk_id', 0) if evidence_items else 0} "
        f"(Page {evidence_items[0].get('start_page', 'N/A') if evidence_items else 'N/A'}). Direct comparative testing required."
    )

    conf_dict = {
        "scientific_confidence": sci_conf_level,
        "confidence_score": sci_conf_score,
        "retrieval_similarity": round(max_similarity, 4),
        "evidence_status": "exploratory",
        "evidence_gap": fallback_gap,
        "scope": "in_scope",
        "rationale": sci_rationale,
    }

    return {
        "query": target_query,
        "statement": plain_text,
        "evidence_status": "exploratory",
        "evidence_gap": fallback_gap,
        "core_variables": {
            "independent_variable": f"Controlled modulation of {target_query.rstrip('?')}",
            "dependent_variable": "System efficiency / primary response endpoint",
            "proposed_mechanism": f"Mechanistic interaction referenced in: '{first_sentence}'",
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
            f"Excerpts establish topic relevance, but do not establish the causal effect directly: {fallback_gap}"
        ],
        "uncertainty_assessment": {
            "level": "moderate",
            "rationale": (
                "Exploratory hypothesis synthesized from literature context. "
                "The source documents discuss these variables but lack direct empirical confirmation of the causal relationship."
            ),
            "knowledge_gaps": [
                fallback_gap,
                "Requires controlled comparative trials with standardized baseline metrics."
            ],
        },
        "proposed_experiments": [
            {
                "title": "Controlled comparative evaluation under standardized conditions",
                "methodology": "Test system across varying levels of the independent variable while holding operating conditions constant.",
                "measurable_metrics": [
                    "Operational efficiency / loss percentage",
                    "Response variance relative to baseline",
                ],
                "expected_outcome": "Statistically significant improvement in primary measured metrics under controlled intervention",
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

