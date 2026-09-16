import re

from knowledge_graph.neo4j_connection import Neo4jConnection


class GraphBuilder:

    def __init__(self):
        self.db = Neo4jConnection()

    # -----------------------------------------
    # CREATE DATABASE CONSTRAINTS
    # -----------------------------------------

    def create_constraints(self):

        queries = [

            """
            CREATE CONSTRAINT paper_id_unique IF NOT EXISTS
            FOR (p:Paper)
            REQUIRE p.paper_id IS UNIQUE
            """,

            """
            CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
            FOR (e:Entity)
            REQUIRE e.entity_id IS UNIQUE
            """

        ]

        with self.db.driver.session() as session:

            for query in queries:
                session.run(query)

    # -----------------------------------------
    # CREATE PAPER NODE
    # -----------------------------------------

    def create_paper(self, paper_id, title):

        query = """

        MERGE (p:Paper {
            paper_id: $paper_id
        })

        SET p.title = $title

        """

        with self.db.driver.session() as session:

            session.run(
                query,
                paper_id=paper_id,
                title=title
            )

    # -----------------------------------------
    # CREATE ENTITY NODE
    # -----------------------------------------

    def create_entity(
        self,
        entity_id,
        name,
        entity_type,
        paper_id
    ):

        query = """

        MERGE (e:Entity {
            entity_id: $entity_id
        })

        SET
            e.name = $name,
            e.type = $entity_type

        WITH e

        MATCH (p:Paper {
            paper_id: $paper_id
        })

        MERGE (p)-[:MENTIONS]->(e)

        """

        with self.db.driver.session() as session:

            session.run(
                query,
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
                paper_id=paper_id
            )

    # -----------------------------------------
    # CREATE RELATIONSHIP
    # -----------------------------------------

    def create_relationship(
        self,
        source_id,
        target_id,
        relationship
    ):

        # Make relationship safe for Neo4j
        relationship = re.sub(
            r"[^A-Z0-9_]",
            "_",
            relationship.upper()
        )

        if not relationship:
            relationship = "RELATED_TO"

        query = f"""

        MATCH (a:Entity {{
            entity_id: $source_id
        }})

        MATCH (b:Entity {{
            entity_id: $target_id
        }})

        MERGE (a)-[:{relationship}]->(b)

        """

        with self.db.driver.session() as session:

            session.run(
                query,
                source_id=source_id,
                target_id=target_id
            )

    # -----------------------------------------
    # CLOSE DATABASE
    # -----------------------------------------

    def close(self):

        self.db.close()