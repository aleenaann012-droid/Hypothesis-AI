import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_graph.neo4j_connection import Neo4jConnection



MISSING_LINK_QUERY = """

MATCH
    (a:Entity)-[r1]->(b:Entity),
    (b)-[r2]->(c:Entity)

WHERE
    a <> c

    AND NOT (
        (a)-[]->(c)
    )

RETURN

    a.entity_id AS source_id,
    a.name AS source,
    a.type AS source_type,

    type(r1) AS first_relationship,

    b.entity_id AS bridge_id,
    b.name AS bridge,
    b.type AS bridge_type,

    type(r2) AS second_relationship,

    c.entity_id AS target_id,
    c.name AS candidate_target,
    c.type AS target_type

LIMIT 100

"""


from nlp.entity_extractor import is_valid_scientific_entity


MULTI_HOP_GAP_QUERY = """
MATCH path = (a:Entity)-[r*2..3]->(c:Entity)
WHERE a <> c
  AND NOT (a)-[]->(c)
  AND ALL(node IN nodes(path) WHERE NOT toLower(node.name) IN [
    'treatment', 'disease', 'drug', 'future', 'confusing', 'study', 'results', 'patient', 'patients', 'mouse', 'mice', 'level', 'rate', 'time', 'method', 'data'
  ])
WITH a, c, path,
     [rel in relationships(path) | type(rel)] as rel_types,
     [node in nodes(path) | node.name] as node_names,
     [node in nodes(path) | node.type] as node_types,
     [node in nodes(path) | node.entity_id] as node_ids
RETURN
    a.entity_id AS source_id,
    a.name AS source,
    a.type AS source_type,
    c.entity_id AS target_id,
    c.name AS candidate_target,
    c.type AS target_type,
    rel_types,
    node_names,
    node_types,
    node_ids,
    length(path) AS hop_count
LIMIT 150
"""


def find_multi_hop_gaps(min_hops: int = 2, max_hops: int = 3, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Find 2-hop and 3-hop transitive gaps in Neo4j with fallback.
    """
    try:
        db = Neo4jConnection()
        try:
            with db.driver.session() as session:
                query = MULTI_HOP_GAP_QUERY.replace("LIMIT 150", f"LIMIT {limit * 3}")
                result = session.run(query)
                records = [dict(record) for record in result]
                if records:
                    return records
        finally:
            db.close()
    except Exception as exc:
        print(f"Neo4j multi-hop query unavailable ({exc}); falling back to in-memory graph.")

    return find_missing_links_from_fallback(limit=limit, max_hops=max_hops)


def find_missing_links(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Find open transitive paths in the knowledge graph.
    If Neo4j is offline, falls back gracefully to processed local graph data.
    """
    try:
        db = Neo4jConnection()
        try:
            with db.driver.session() as session:
                query = MISSING_LINK_QUERY.replace("LIMIT 100", f"LIMIT {limit}")
                result = session.run(query)
                records = [dict(record) for record in result]
                # Filter out generic entities
                filtered = [
                    r for r in records
                    if is_valid_scientific_entity(r.get("source"))
                    and is_valid_scientific_entity(r.get("bridge"))
                    and is_valid_scientific_entity(r.get("candidate_target"))
                ]
                return filtered if filtered else records
        finally:
            db.close()
    except Exception as exc:
        print(f"Neo4j missing links query unavailable ({exc}); using graph fallback.")
        return find_missing_links_from_fallback(limit=limit)


def find_missing_links_from_fallback(limit: int = 100, max_hops: int = 3) -> List[Dict[str, Any]]:
    """
    Extract missing links from processed JSON graph files using networkx.
    Used when Neo4j service is offline or unreachable.
    """
    sample_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "sample.json"
    if not sample_path.exists():
        return []

    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        G = nx.DiGraph()

        for ent in data.get("entities", []):
            eid = f"{ent.get('type', 'ENTITY')}:{ent['text'].lower()}"
            G.add_node(eid, name=ent["text"], type=ent.get("type", "UNKNOWN"))

        for rel in data.get("relations", []):
            s = rel.get("source")
            t = rel.get("target")
            r = rel.get("relation", "RELATED_TO")
            if s and t:
                G.add_edge(s, t, relationship=r)

        candidates = []
        # Multi-hop search up to max_hops
        for a in G.nodes():
            for b in G.successors(a):
                r1 = G[a][b].get("relationship", "RELATED_TO")
                for c in G.successors(b):
                    if a != c and not G.has_edge(a, c):
                        r2 = G[b][c].get("relationship", "RELATED_TO")
                        node_a = G.nodes[a]
                        node_b = G.nodes[b]
                        node_c = G.nodes[c]

                        # Check generic filter
                        src_name = node_a.get("name", a)
                        bridge_name = node_b.get("name", b)
                        tgt_name = node_c.get("name", c)

                        candidates.append({
                            "source_id": a,
                            "source": src_name,
                            "source_type": node_a.get("type", "UNKNOWN"),
                            "first_relationship": r1,
                            "bridge_id": b,
                            "bridge": bridge_name,
                            "bridge_type": node_b.get("type", "UNKNOWN"),
                            "second_relationship": r2,
                            "target_id": c,
                            "candidate_target": tgt_name,
                            "target_type": node_c.get("type", "UNKNOWN"),
                            "hop_count": 2,
                            "node_names": [src_name, bridge_name, tgt_name],
                            "rel_types": [r1, r2],
                        })
                        if len(candidates) >= limit * 2:
                            break

        return candidates[:limit]
    except Exception as err:
        print(f"Fallback graph processing error: {err}")
        return []


def calculate_semantic_similarity(text_a: str, text_b: str) -> float:
    """Calculate cosine similarity between two entity/concept strings using BGE embeddings."""
    try:
        from retrieval.embeddings import model
        emb = model.encode([text_a, text_b], normalize_embeddings=True)
        # Dot product of normalized vectors = cosine similarity
        sim = float(emb[0] @ emb[1])
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))
    except Exception:
        # Fallback lexical Jaccard
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        union = len(words_a | words_b)
        return len(words_a & words_b) / union if union > 0 else 0.5


def find_research_gaps(limit: int = 50, filter_generic: bool = True) -> List[Dict[str, Any]]:
    """
    Synthesize missing links into researcher-facing discovery gaps,
    proposing testable mediator hypotheses, bridge paths, and ranking
    them by combining graph topology and semantic plausibility.
    """
    raw_links = find_missing_links(limit=limit * 2)
    gaps = []

    for item in raw_links:
        src = item["source"]
        bridge = item.get("bridge")
        tgt = item["candidate_target"]

        # Filter generic non-informative entities
        if filter_generic:
            if not is_valid_scientific_entity(src) or not is_valid_scientific_entity(tgt):
                continue
            if bridge and not is_valid_scientific_entity(bridge):
                continue

        r1 = item.get("first_relationship", "RELATED_TO")
        r2 = item.get("second_relationship", "RELATED_TO")

        node_names = item.get("node_names") or [src, bridge, tgt]
        rel_types = item.get("rel_types") or [r1, r2]

        # Construct path representation
        path_parts = []
        for i in range(len(rel_types)):
            path_parts.append(f"{node_names[i]} --[{rel_types[i]}]--> ")
        path_parts.append(node_names[-1])
        path_str = "".join(path_parts)

        hypothesis_prompt = (
            f"Investigate whether {src} ({item.get('source_type', 'Entity')}) "
            f"modulates {tgt} ({item.get('target_type', 'Entity')}) "
            f"via intermediate mediator {bridge} ({item.get('bridge_type', 'Entity')})."
        )

        # Scientific plausibility & confidence score:
        # Combining graph proximity (hop penalty) and semantic relatedness
        hop_count = item.get("hop_count", 2)
        graph_topo_score = 1.0 / hop_count
        semantic_sim = calculate_semantic_similarity(src, tgt)
        plausibility_score = round(0.5 * graph_topo_score + 0.5 * semantic_sim, 4)

        gaps.append({
            "source_id": item["source_id"],
            "source": src,
            "source_type": item.get("source_type"),
            "bridge_id": item.get("bridge_id"),
            "bridge": bridge,
            "bridge_type": item.get("bridge_type"),
            "target_id": item["target_id"],
            "candidate_target": tgt,
            "target_type": item.get("target_type"),
            "first_relationship": r1,
            "second_relationship": r2,
            "hop_count": hop_count,
            "bridge_path": path_str,
            "hypothesized_relationship": hypothesis_prompt,
            "plausibility_score": plausibility_score,
            "semantic_similarity": round(semantic_sim, 4),
            "gap_type": "transitive_mediator_gap",
            "relationship_nature": "inferred_hypothesis",
            "validation_status": "unvalidated",
        })

    # Sort gaps by descending plausibility score
    gaps.sort(key=lambda g: g["plausibility_score"], reverse=True)
    return gaps[:limit]


if __name__ == "__main__":
    gaps = find_research_gaps(limit=10)

    print()
    print("Potential Research Gaps")
    print("=======================")

    for gap in gaps:
        print()
        print("Path:", gap["bridge_path"])
        print("Gap Hypothesis:", gap["hypothesized_relationship"])