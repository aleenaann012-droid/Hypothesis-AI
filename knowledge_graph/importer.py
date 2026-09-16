from pathlib import Path

from nlp.pdf_extractor import extract_text_from_pdf
from nlp.entity_extractor import extract_entities
from nlp.relation_extractor import extract_relations

from knowledge_graph.graph_builder import GraphBuilder


def create_entity_id(name):
    """
    Create a unique Neo4j ID for an extracted entity.
    """
    return f"ENTITY:{name.lower().strip()}"


def import_research_paper(builder, pdf_path, paper_id):
    """
    Run the NLP pipeline on a PDF and import
    the extracted entities and relationships into Neo4j.
    """

    print()
    print("========================================")
    print("IMPORTING RESEARCH PAPER")
    print("========================================")

    print("PDF:", pdf_path)
    print("Paper ID:", paper_id)

    # -------------------------------------------------
    # STEP 1: Extract PDF text
    # -------------------------------------------------

    print()
    print("Step 1: Extracting PDF text...")

    text = extract_text_from_pdf(pdf_path)

    print("Text extracted successfully!")
    print("Characters:", len(text))

    # -------------------------------------------------
    # STEP 2: Extract entities
    # -------------------------------------------------

    print()
    print("Step 2: Extracting entities...")

    entities = extract_entities(text)

    print("Entities extracted:", len(entities))

    # -------------------------------------------------
    # STEP 3: Extract relations
    # -------------------------------------------------

    print()
    print("Step 3: Extracting relations...")

    relations = extract_relations(text)

    print("Relations extracted:", len(relations))

    # -------------------------------------------------
    # STEP 4: Create Paper node
    # -------------------------------------------------

    title = Path(pdf_path).stem

    builder.create_paper(
        paper_id=paper_id,
        title=title
    )

    # -------------------------------------------------
    # STEP 5: Create Entity nodes
    # -------------------------------------------------

    entity_lookup = {}

    imported_entities = 0

    for entity in entities:

        name = entity.get("text", "").strip()

        if not name:
            continue

        entity_id = create_entity_id(name)

        # Store the entity ID using lowercase text
        entity_lookup[name.lower()] = entity_id

        builder.create_entity(
            entity_id=entity_id,
            name=name,
            entity_type="ENTITY",
            paper_id=paper_id
        )

        imported_entities += 1

    # -------------------------------------------------
    # STEP 6: Create relationships
    # -------------------------------------------------

    imported_relations = 0
    skipped_relations = 0

    for relation in relations:

        subject = relation.get("subject", "").strip()
        relation_type = relation.get("relation", "RELATED_TO").strip()
        object_name = relation.get("object", "").strip()

        if not subject or not object_name:
            continue

        source_id = entity_lookup.get(subject.lower())
        target_id = entity_lookup.get(object_name.lower())

        # Only create relation if both entities
        # were found by the entity extractor.
        if not source_id or not target_id:

            skipped_relations += 1

            print(
                "Skipping relation because entity was not found:",
                subject,
                "->",
                object_name
            )

            continue

        builder.create_relationship(
            source_id=source_id,
            target_id=target_id,
            relationship=relation_type
        )

        imported_relations += 1

    # -------------------------------------------------
    # SUMMARY
    # -------------------------------------------------

    print()
    print("========================================")
    print("IMPORT SUMMARY")
    print("========================================")

    print("Entities extracted:", len(entities))
    print("Entities imported:", imported_entities)

    print("Relations extracted:", len(relations))
    print("Relations imported:", imported_relations)
    print("Relations skipped:", skipped_relations)

    print()
    print("Neo4j import completed successfully!")


def main():

    # Research paper
    pdf_path = "test_researchpaper.pdf"

    # ID used for the Paper node in Neo4j
    paper_id = "P001"

    builder = GraphBuilder()

    try:

        builder.create_constraints()

        import_research_paper(
            builder,
            pdf_path,
            paper_id
        )

    finally:

        builder.close()


if __name__ == "__main__":
    main()