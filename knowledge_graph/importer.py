import hashlib
from pathlib import Path

from nlp.pdf_extractor import extract_text_from_pdf
from nlp.entity_extractor import extract_entities, is_valid_scientific_entity
from nlp.relation_extractor import extract_relations

from knowledge_graph.graph_builder import GraphBuilder


def create_entity_id(name):
    """
    Create a unique Neo4j ID for an extracted entity.
    """
    return f"ENTITY:{name.lower().strip()}"

def resolve_or_create_entity(
    builder,
    entity_lookup,
    name,
    entity_type,
    paper_id,
    filter_generic=True
):
    """
    Resolve an existing entity or create a missing one.
    Returns the entity ID, or None if the name is empty or invalid.
    """
    name = (name or "").strip()

    if not name:
        return None

    if filter_generic and not is_valid_scientific_entity(name):
        return None

    lookup_key = name.lower()

    # Reuse an entity already extracted from the paper.
    if lookup_key in entity_lookup:
        return entity_lookup[lookup_key]

    # Create a stable ID using the existing project convention.
    entity_id = create_entity_id(name)

    builder.create_entity(
        entity_id=entity_id,
        name=name,
        entity_type=entity_type or "UNKNOWN",
        paper_id=paper_id
    )

    entity_lookup[lookup_key] = entity_id

    return entity_id

def create_evidence_id(paper_id, sentence_start, sentence_end):
    raw = f"{paper_id}:{sentence_start}:{sentence_end}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"EVIDENCE:{digest}"


def create_claim_id(
    paper_id,
    sentence_start,
    subject,
    predicate,
    object_name
):
    raw = (
        f"{paper_id}:{sentence_start}:"
        f"{subject.lower().strip()}:"
        f"{predicate.lower().strip()}:"
        f"{object_name.lower().strip()}"
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"CLAIM:{digest}"

def import_research_paper(builder, pdf_path, paper_id, text=None, use_llm_fallback: bool = False):
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

    if text is None:
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

    relations = extract_relations(text, use_llm_fallback=use_llm_fallback)

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
    batch_entities = []

    imported_entities = 0

    for entity in entities:

        name = entity.get("text", "").strip()

        if not name or not is_valid_scientific_entity(name):
            continue

        entity_id = create_entity_id(name)

        # Store the entity ID using lowercase text
        entity_lookup[name.lower()] = entity_id

        batch_entities.append({
            "entity_id": entity_id,
            "name": name,
            "entity_type": entity.get("label", "UNKNOWN")
        })

    if hasattr(builder, "create_entities_batch"):
        builder.create_entities_batch(batch_entities, paper_id=paper_id)
        imported_entities = len(batch_entities)
    else:
        for ent in batch_entities:
            builder.create_entity(
                entity_id=ent["entity_id"],
                name=ent["name"],
                entity_type=ent["entity_type"],
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
        predicate = relation.get("relation", "RELATED_TO").strip()
        object_name = relation.get("object", "").strip()

        if not subject or not predicate or not object_name:
            skipped_relations += 1
            continue

        subject_id = resolve_or_create_entity(
            builder=builder,
            entity_lookup=entity_lookup,
            name=subject,
            entity_type=relation.get("subject_label", "UNKNOWN"),
            paper_id=paper_id
        )

        object_id = resolve_or_create_entity(
            builder=builder,
            entity_lookup=entity_lookup,
            name=object_name,
            entity_type=relation.get("object_label", "UNKNOWN"),
            paper_id=paper_id
        )

        if subject_id is None or object_id is None:
            skipped_relations += 1
            print(
                "Skipping relation because an entity name is empty:",
                subject, "->", object_name
            )
            continue

        evidence_text = relation.get("evidence_text", "").strip()
        sentence_start = relation.get("sentence_start")
        sentence_end = relation.get("sentence_end")
        relation_start = relation.get("relation_start")
        relation_end = relation.get("relation_end")

        # Do not store a claim without its supporting sentence and offsets.
        if (
            not evidence_text
            or sentence_start is None
            or sentence_end is None
            or relation_start is None
            or relation_end is None
        ):
            skipped_relations += 1
            print("Skipping relation with missing evidence:", relation)
            continue

        evidence_id = create_evidence_id(
            paper_id, sentence_start, sentence_end
        )

        claim_id = create_claim_id(
            paper_id,
            sentence_start,
            subject,
            predicate,
            object_name
        )

        builder.create_evidence_claim(
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
            subject_label=relation.get("subject_label", "UNKNOWN"),
            object_label=relation.get("object_label", "UNKNOWN"),
            extraction_method=relation.get(
                "extraction_method", "unknown"
            ),
            validation_status=relation.get(
                "validation_status", "unvalidated"
            )
        )

        # Keep the direct edge with paper_id for paper-isolated graph queries
        builder.create_relationship(
            source_id=subject_id,
            target_id=object_id,
            relationship=predicate,
            paper_id=paper_id
        )

        imported_relations += 1

    print("Imported evidence-backed claims:", imported_relations)
    print("Skipped relations:", skipped_relations)


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