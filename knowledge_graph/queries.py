GET_ALL_ENTITIES = """

MATCH (e:Entity)

RETURN
    e.entity_id AS entity_id,
    e.name AS name,
    e.type AS type

ORDER BY e.type, e.name

"""


GET_ENTITY_CONNECTIONS = """

MATCH (e:Entity)

WHERE toLower(e.name)
      = toLower($name)

OPTIONAL MATCH
    (e)-[r]-(other:Entity)

RETURN
    e.name AS source,
    e.type AS source_type,
    type(r) AS relationship,
    other.name AS target,
    other.type AS target_type

"""


GET_GRAPH = """

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


GET_PAPER_ENTITIES = """

MATCH
    (p:Paper)-[:MENTIONS]->(e:Entity)

WHERE
    p.paper_id = $paper_id

RETURN
    e.entity_id AS entity_id,
    e.name AS name,
    e.type AS type

ORDER BY e.type, e.name

"""