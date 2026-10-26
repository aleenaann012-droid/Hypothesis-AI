from knowledge_graph.neo4j_connection import Neo4jConnection


if __name__ == "__main__":
    db = Neo4jConnection()

    try:
        print("Testing Neo4j connection...")
        result = db.verify_connection()
        print("Neo4j connection successful!")
        print("Result:", result)
    finally:
        db.close()