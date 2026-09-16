import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()


class Neo4jConnection:

    def __init__(self):

        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")

        if not uri:
            raise ValueError("NEO4J_URI is missing in .env")

        if not username:
            raise ValueError("NEO4J_USER is missing in .env")

        if not password:
            raise ValueError("NEO4J_PASSWORD is missing in .env")

        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password)
        )

    def verify_connection(self):

        with self.driver.session() as session:

            result = session.run(
                "RETURN 1 AS result"
            )

            return result.single()["result"]

    def close(self):

        self.driver.close()