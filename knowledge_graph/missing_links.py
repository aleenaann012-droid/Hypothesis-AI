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


def find_missing_links():

    db = Neo4jConnection()

    try:

        with db.driver.session() as session:

            result = session.run(
                MISSING_LINK_QUERY
            )

            return [
                dict(record)
                for record in result
            ]

    finally:

        db.close()


if __name__ == "__main__":

    links = find_missing_links()

    print()
    print("Potential Missing Connections")
    print("=============================")

    for link in links:

        print()

        print(
            f"{link['source']} "
            f"--{link['first_relationship']}--> "
            f"{link['bridge']}"
        )

        print(
            f"{link['bridge']} "
            f"--{link['second_relationship']}--> "
            f"{link['candidate_target']}"
        )

        print(
            "Candidate:",
            link["source"],
            "->",
            link["candidate_target"]
        )