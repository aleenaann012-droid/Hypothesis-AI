from knowledge_graph.neo4j_connection import Neo4jConnection


DIRECT_EVIDENCE_QUERY = """

MATCH
    (p:Paper)-[:MENTIONS]->(a:Entity),
    (p)-[:MENTIONS]->(b:Entity)

WHERE
    a.entity_id = $source_id
    AND b.entity_id = $target_id

RETURN
    p.paper_id AS paper_id,
    p.title AS title

ORDER BY p.paper_id

"""


def get_direct_evidence(
    source_id,
    target_id
):

    db = Neo4jConnection()

    try:

        with db.driver.session() as session:

            result = session.run(
                DIRECT_EVIDENCE_QUERY,
                source_id=source_id,
                target_id=target_id
            )

            return [
                dict(record)
                for record in result
            ]

    finally:

        db.close()