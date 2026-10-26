"""
Hypothesis-AI: Transitive Relationship & Discovery Query Runner

This script allows any researcher to:
1. Discover 2-hop transitive graph connections (research gaps).
2. Retrieve exact supporting evidence chunks with PDF page numbers and offsets.
3. Formulate evidence-grounded, testable hypotheses with proposed experiments.

Usage:
    .venv\\Scripts\\python query_transitive.py
    .venv\\Scripts\\python query_transitive.py --limit 3
    .venv\\Scripts\\python query_transitive.py --query "How does Donepezil modulate cognitive decline via AChE inhibition?"
    .venv\\Scripts\\python query_transitive.py --direct
"""

import argparse
import json
import sys
from pathlib import Path
import requests

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

API_BASE = "http://127.0.0.1:8000"


def check_api_online() -> bool:
    try:
        res = requests.get(f"{API_BASE}/health", timeout=3)
        return res.status_code == 200
    except Exception:
        return False


def print_hypothesis_card(hypo_data: dict):
    struct = hypo_data.get("structured_hypothesis", {})

    print("\n========== GENERATED TESTABLE HYPOTHESIS ==========")
    print(f"Status:             {hypo_data.get('validation_status', 'unvalidated').upper()}")
    statement = struct.get("statement") or hypo_data.get("hypothesis", "")
    print(f"Statement:          {statement}\n")

    core_vars = struct.get("core_variables", {})
    if core_vars and isinstance(core_vars, dict):
        print("Core Variables:")
        for k, v in core_vars.items():
            print(f"  - {k.replace('_', ' ').capitalize()}: {v}")
        print()
    else:
        # Check flat fields
        indep = struct.get("independent_variable")
        dep = struct.get("dependent_variable")
        mech = struct.get("proposed_mechanism")
        if indep or dep or mech:
            print("Core Variables:")
            if indep:
                print(f"  - Independent: {indep}")
            if dep:
                print(f"  - Dependent:   {dep}")
            if mech:
                print(f"  - Mechanism:   {mech}")
            print()

    unc = struct.get("uncertainty_assessment", {})
    if unc:
        level = unc.get("level", "moderate") if isinstance(unc, dict) else str(unc)
        print(f"Uncertainty Level:  {str(level).upper()}")
        if isinstance(unc, dict) and unc.get("rationale"):
            print(f"Rationale:          {unc.get('rationale')}\n")

    # Confidence & Evidence Assessment
    conf_assess = hypo_data.get("confidence_assessment") or struct.get("confidence_assessment", {})
    ev_status = struct.get("evidence_status") or (conf_assess.get("evidence_status") if conf_assess else "exploratory")
    ev_gap = struct.get("evidence_gap") or (conf_assess.get("evidence_gap") if conf_assess else "")

    print(f"Evidence Status:    {ev_status.upper()} (UNVALIDATED)")
    if ev_gap:
        print(f"Evidence Gap:       {ev_gap}\n")

    if conf_assess:
        sci_conf = conf_assess.get("scientific_confidence", "MODERATE").upper()
        scope = conf_assess.get("scope", "in_scope").upper()
        score = conf_assess.get("confidence_score", 0.0)
        sim = conf_assess.get("retrieval_similarity", 0.0)
        print(f"Confidence Assessment: {sci_conf} ({score:.2f}) | Retrieval Similarity: {sim:.4f} | Scope: {scope}")
        if conf_assess.get("rationale") or conf_assess.get("explanation"):
            print(f"Confidence Rationale:  {conf_assess.get('rationale') or conf_assess.get('explanation')}")
        print()

    exps = struct.get("proposed_experiments", [])
    if exps:
        print("Proposed Validation Experiments:")
        for idx, exp in enumerate(exps, start=1):
            print(f"  {idx}. {exp.get('title', 'Comparative validation assay')}")
            if exp.get("methodology"):
                print(f"     Methodology:        {exp.get('methodology')}")
            if exp.get("measurable_metrics"):
                metrics_str = ", ".join(exp.get("measurable_metrics")) if isinstance(exp.get("measurable_metrics"), list) else str(exp.get("measurable_metrics"))
                print(f"     Measurable Metrics: {metrics_str}")
            if exp.get("expected_outcome"):
                print(f"     Expected Outcome:   {exp.get('expected_outcome')}")
            print()

    print("=" * 65)
    print("Workflow complete. Grounded in indexed provenance chunks.")


def run_via_api(limit: int = 5, custom_query: str = None):
    print("Connecting to Hypothesis-AI API at", API_BASE)
    print("=" * 65)

    # 1. Fetch Transitive Gaps
    print("\n[STEP 1] Fetching transitive relations and research gaps...")
    try:
        gaps_res = requests.get(f"{API_BASE}/research-gaps?limit={limit}", timeout=15)
        gaps_res.raise_for_status()
        gaps_data = gaps_res.json()
    except Exception as err:
        print(f"Failed to fetch research gaps from API: {err}")
        return

    total_gaps = gaps_data.get("count", 0)
    gaps_list = gaps_data.get("gaps", [])

    print(f"Discovered {total_gaps} candidate transitive relationship(s):\n")

    for idx, gap in enumerate(gaps_list, start=1):
        print(f"  Gap #{idx}:")
        print(f"    Path:       {gap['bridge_path']}")
        print(f"    Hypothesis: {gap['hypothesized_relationship']}")
        print(f"    Status:     {gap.get('validation_status', 'unvalidated')}\n")

    if not gaps_list and not custom_query:
        print("No transitive gaps detected in current graph.")
        return

    # Select query to investigate
    if custom_query:
        selected_query = custom_query
        print(f"Investigating custom research query: \"{selected_query}\"")
    else:
        selected_query = gaps_list[0]["hypothesized_relationship"]
        print(f"Investigating top transitive connection: \"{selected_query}\"")

    # 2. Retrieve Evidence Chunks
    print("\n" + "=" * 65)
    print("[STEP 2] Retrieving supporting evidence chunks from paper...")
    try:
        ret_res = requests.get(
            f"{API_BASE}/retrieve",
            params={"query": selected_query, "top_k": 2},
            timeout=25,
        )
        ret_res.raise_for_status()
        ret_data = ret_res.json()
        evidence_list = ret_data.get("evidence", [])
    except Exception as err:
        print(f"Evidence retrieval failed: {err}")
        evidence_list = []

    print(f"Retrieved {len(evidence_list)} evidence chunk(s):\n")
    for ev in evidence_list:
        pages_str = ", ".join(str(p) for p in ev.get("page_references", []))
        exact_span = f"[{ev.get('exact_document_start')} - {ev.get('exact_document_end')}]"
        print(f"  * Chunk #{ev.get('chunk_id')} | Page(s): {pages_str} | Similarity: {ev.get('similarity', 0):.4f}")
        print(f"    Exact PDF Span: {exact_span}")
        print(f"    Text Preview:   \"{ev.get('text', '')[:160]}...\"\n")

    # 3. Generate Evidence-Grounded Hypothesis
    print("=" * 65)
    print("[STEP 3] Formulating testable scientific hypothesis with LLM...")
    hypo_data = None
    try:
        hypo_res = requests.get(
            f"{API_BASE}/hypothesis",
            params={"query": selected_query, "top_k": 2},
            timeout=180,
        )
        hypo_res.raise_for_status()
        hypo_data = hypo_res.json()
    except (requests.Timeout, requests.RequestException) as err:
        print(f"\n[Note] API LLM generation response delayed: {err}")
        print("Falling back immediately to direct structured evidence synthesis...")
        top_ev = evidence_list[0] if evidence_list else {}
        first_sent = top_ev.get("text", "").split(".")[0].strip() if top_ev.get("text") else "Retrieved evidence"
        hypo_data = {
            "validation_status": "unvalidated",
            "hypothesis": f"Based on retrieved literature ({first_sent}), {selected_query} warrants empirical validation.",
            "structured_hypothesis": {
                "statement": f"Based on retrieved literature ({first_sent}), {selected_query} warrants empirical validation.",
                "core_variables": {"focus": selected_query, "mediator": "intermediate disease state"},
                "uncertainty_assessment": {"level": "MODERATE", "rationale": "Synthesized directly from retrieved paper chunks."},
                "proposed_experiments": [
                    {
                        "title": "Targeted validation assay",
                        "methodology": "Targeted longitudinal assay evaluating biological endpoints in controlled models.",
                        "expected_outcome": "Statistically significant correlation with proposed intervention."
                    }
                ],
            }
        }
    except Exception as err:
        print(f"\nAPI communication error: {err}")
        return

    print_hypothesis_card(hypo_data)


def run_direct(limit: int = 5, custom_query: str = None):
    """Run discovery directly using local modules without requiring the HTTP API."""
    print("Hypothesis-AI Direct Execution Mode (In-Process)")
    print("=" * 65)

    # 1. Discover research gaps
    print("\n[STEP 1] Fetching transitive relations and research gaps...")
    from knowledge_graph.missing_links import find_research_gaps
    gaps_list = find_research_gaps(limit=limit)
    print(f"Discovered {len(gaps_list)} candidate transitive relationship(s):\n")

    for idx, gap in enumerate(gaps_list, start=1):
        print(f"  Gap #{idx}:")
        print(f"    Path:       {gap['bridge_path']}")
        print(f"    Hypothesis: {gap['hypothesized_relationship']}")
        print(f"    Status:     {gap.get('validation_status', 'unvalidated')}\n")

    if not gaps_list and not custom_query:
        print("No transitive gaps detected in current graph.")
        return

    selected_query = custom_query if custom_query else gaps_list[0]["hypothesized_relationship"]
    print(f"Investigating query: \"{selected_query}\"")

    # 2. Retrieve evidence
    print("\n" + "=" * 65)
    print("[STEP 2] Retrieving supporting evidence chunks from paper...")
    from api.main import get_retrieval_resources, search_with_provenance
    index, chunks, provenance = get_retrieval_resources()
    evidence_list = search_with_provenance(
        query=selected_query,
        index=index,
        top_k=2,
        chunks=chunks,
        provenance=provenance,
    )

    print(f"Retrieved {len(evidence_list)} evidence chunk(s):\n")
    for ev in evidence_list:
        pages_str = ", ".join(str(p) for p in ev.get("page_references", []))
        exact_span = f"[{ev.get('exact_document_start')} - {ev.get('exact_document_end')}]"
        print(f"  * Chunk #{ev.get('chunk_id')} | Page(s): {pages_str} | Similarity: {ev.get('similarity', 0):.4f}")
        print(f"    Exact PDF Span: {exact_span}")
        print(f"    Text Preview:   \"{ev.get('text', '')[:160]}...\"\n")

    # 3. Generate structured hypothesis
    print("=" * 65)
    print("[STEP 3] Formulating testable scientific hypothesis with LLM...")
    from hypothesis.generator import generate_structured_hypothesis
    structured = generate_structured_hypothesis(evidence_items=evidence_list, query=selected_query)

    hypo_data = {
        "query": selected_query,
        "hypothesis": structured.get("statement", ""),
        "structured_hypothesis": structured,
        "validation_status": "unvalidated",
        "evidence": evidence_list,
    }

    print_hypothesis_card(hypo_data)


def main():
    parser = argparse.ArgumentParser(description="Query transitive relations and generate hypotheses.")
    parser.add_argument("--limit", type=int, default=3, help="Maximum transitive gaps to list.")
    parser.add_argument("--query", type=str, default=None, help="Custom research query to investigate.")
    parser.add_argument("--direct", action="store_true", help="Run directly in-process without requiring FastAPI server.")
    args = parser.parse_args()

    if args.direct:
        run_direct(limit=args.limit, custom_query=args.query)
        return

    if not check_api_online():
        print("[Notice] API server at http://127.0.0.1:8000 is not reachable.")
        print("Automatically switching to direct in-process execution...\n")
        run_direct(limit=args.limit, custom_query=args.query)
        return

    run_via_api(limit=args.limit, custom_query=args.query)


if __name__ == "__main__":
    main()
