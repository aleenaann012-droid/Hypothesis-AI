from typing import Any, Dict, List
from knowledge_graph.neo4j_connection import Neo4jConnection


DIRECT_EVIDENCE_QUERY = """

MATCH
    (p:Paper)-[:MENTIONS]->(a:Entity),
    (p)-[:MENTIONS]->(b:Entity)

WHERE
    a.entity_id = $source_id
    AND b.entity_id = $target_id

OPTIONAL MATCH (c:Claim)-[:SUBJECT]->(a)
WHERE (c)-[:OBJECT]->(b)

OPTIONAL MATCH (ev:Evidence)-[:SUPPORTS_CLAIM]->(c)
WHERE (p)-[:HAS_EVIDENCE]->(ev)

RETURN
    p.paper_id AS paper_id,
    p.title AS title,
    c.claim_id AS claim_id,
    c.predicate AS relationship,
    ev.evidence_id AS evidence_id,
    ev.text AS evidence_text,
    ev.sentence_start AS sentence_start,
    ev.sentence_end AS sentence_end,
    c.extraction_method AS extraction_method,
    c.validation_status AS validation_status

ORDER BY p.paper_id

"""


def get_direct_evidence(
    source_id: str,
    target_id: str
) -> List[Dict[str, Any]]:
    """
    Retrieve direct evidence connecting two entities from Neo4j.
    Returns paper mentions and detailed sentence-level Evidence/Claim records if present.
    """
    db = Neo4jConnection()

    try:
        with db.driver.session() as session:
            result = session.run(
                DIRECT_EVIDENCE_QUERY,
                source_id=source_id,
                target_id=target_id
            )

            records = []
            for record in result:
                d = dict(record)
                # Ensure validation_status defaults to unvalidated if absent
                if not d.get("validation_status"):
                    d["validation_status"] = "unvalidated"
                records.append(d)
            return records

    finally:
        db.close()