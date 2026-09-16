from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from knowledge_graph.neo4j_connection import Neo4jConnection
from knowledge_graph.missing_links import find_missing_links
from knowledge_graph.evidence import get_direct_evidence


app = FastAPI(
    title="HypothesisAI API",
    description=(
        "Backend API for the HypothesisAI "
        "scientific discovery assistant"
    ),
    version="0.1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def home():

    return {
        "project": "HypothesisAI",
        "status": "running",
        "module": "Knowledge Graph Backend"
    }


@app.get("/health")
def health():

    db = Neo4jConnection()

    try:

        result = db.verify_connection()

        return {
            "status": "healthy",
            "neo4j": "connected",
            "result": result
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        db.close()


@app.get("/graph")
def get_graph():

    query = """

    MATCH
        (a:Entity)-[r]->(b:Entity)

    RETURN

        a.entity_id AS source_id,
        a.name AS source,
        a.type AS source_type,

        type(r) AS relationship,

        b.entity_id AS target_id,
        b.name AS target,
        b.type AS target_type

    LIMIT 200

    """

    db = Neo4jConnection()

    try:

        with db.driver.session() as session:

            result = session.run(query)

            return {
                "data": [
                    dict(record)
                    for record in result
                ]
            }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        db.close()


@app.get("/entity/{name}")
def get_entity(name: str):

    query = """

    MATCH
        (e:Entity)

    WHERE
        toLower(e.name)
        = toLower($name)

    OPTIONAL MATCH
        (e)-[r]-(other:Entity)

    RETURN

        e.entity_id AS entity_id,
        e.name AS entity,
        e.type AS type,

        type(r) AS relationship,

        other.entity_id AS connected_id,
        other.name AS connected_entity,
        other.type AS connected_type

    """

    db = Neo4jConnection()

    try:

        with db.driver.session() as session:

            result = session.run(
                query,
                name=name
            )

            data = [
                dict(record)
                for record in result
            ]

            if not data:

                raise HTTPException(
                    status_code=404,
                    detail="Entity not found"
                )

            return {
                "entity": name,
                "data": data
            }

    finally:

        db.close()


@app.get("/paper/{paper_id}")
def get_paper(paper_id: str):

    query = """

    MATCH
        (p:Paper)-[:MENTIONS]->(e:Entity)

    WHERE
        p.paper_id = $paper_id

    RETURN

        p.paper_id AS paper_id,
        p.title AS title,

        e.entity_id AS entity_id,
        e.name AS entity,
        e.type AS type

    """

    db = Neo4jConnection()

    try:

        with db.driver.session() as session:

            result = session.run(
                query,
                paper_id=paper_id
            )

            data = [
                dict(record)
                for record in result
            ]

            if not data:

                raise HTTPException(
                    status_code=404,
                    detail="Paper not found"
                )

            return {
                "paper_id": paper_id,
                "data": data
            }

    finally:

        db.close()


@app.get("/missing-links")
def missing_links():

    try:

        links = find_missing_links()

        return {
            "count": len(links),
            "candidates": links
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.get("/evidence")
def evidence(
    source_id: str,
    target_id: str
):

    try:

        results = get_direct_evidence(
            source_id,
            target_id
        )

        return {
            "source_id": source_id,
            "target_id": target_id,
            "evidence_count": len(results),
            "evidence": results
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )