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
            """,
            """
            CREATE CONSTRAINT claim_id_unique IF NOT EXISTS
            FOR (c:Claim)
            REQUIRE c.claim_id IS UNIQUE
            """,
        
            """
            CREATE CONSTRAINT evidence_id_unique IF NOT EXISTS
            FOR (ev:Evidence)
            REQUIRE ev.evidence_id IS UNIQUE
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
    
        
    def create_evidence_claim(
        self,
        claim_id,
        evidence_id,
        paper_id,
        subject_id,
        object_id,
        predicate,
        evidence_text,
        sentence_start,
        sentence_end,
        relation_start,
        relation_end,
        subject_label,
        object_label,
        extraction_method,
        validation_status
    ):
        query = """
        MATCH (p:Paper {paper_id: $paper_id})
        MATCH (s:Entity {entity_id: $subject_id})
        MATCH (o:Entity {entity_id: $object_id})

        MERGE (c:Claim {claim_id: $claim_id})
        SET c.predicate = $predicate,
        c.subject_label = $subject_label,
        c.object_label = $object_label,
        c.extraction_method = $extraction_method,
        c.extractor_version = $extractor_version,
        c.validation_status = $validation_status

        MERGE (ev:Evidence {evidence_id: $evidence_id})
        SET ev.text = $evidence_text,
            ev.sentence_start = $sentence_start,
            ev.sentence_end = $sentence_end,
            ev.relation_start = $relation_start,
            ev.relation_end = $relation_end

        MERGE (p)-[:HAS_EVIDENCE]->(ev)
        MERGE (ev)-[:SUPPORTS_CLAIM]->(c)
        MERGE (c)-[:SUBJECT]->(s)
        MERGE (c)-[:OBJECT]->(o)
        """
        

        with self.db.driver.session() as session:
            session.run(
                query,
                claim_id=claim_id,
                evidence_id=evidence_id,
                paper_id=paper_id,
                subject_id=subject_id,
                object_id=object_id,
                predicate=predicate,
                evidence_text=evidence_text,
                sentence_start=sentence_start,
                sentence_end=sentence_end,
                relation_start=relation_start,
                relation_end=relation_end,
                subject_label=subject_label,
                object_label=object_label,
                extraction_method=extraction_method,
                validation_status=validation_status,
                extractor_version="dependency_v2"
            )


    # -----------------------------------------
    # CLOSE DATABASE
    # -----------------------------------------

    def close(self):

        self.db.close()