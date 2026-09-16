import json
from pathlib import Path

from knowledge_graph.graph_builder import GraphBuilder


def create_entity_id(name, entity_type):

    return (
        f"{entity_type.upper().strip()}:"
        f"{name.lower().strip()}"
    )


def import_paper(builder, paper):

    paper_id = paper["paper_id"]

    title = paper.get(
        "title",
        "Unknown Title"
    )

    print()
    print(
        f"Importing paper: {paper_id}"
    )

    # -----------------------------------------
    # CREATE PAPER
    # -----------------------------------------

    builder.create_paper(
        paper_id,
        title
    )

    # -----------------------------------------
    # CREATE ENTITIES
    # -----------------------------------------

    entity_lookup = {}

    for entity in paper.get(
        "entities",
        []
    ):

        name = entity.get(
            "text",
            ""
        ).strip()

        entity_type = entity.get(
            "type",
            "ENTITY"
        ).strip()

        if not name:
            continue

        entity_id = create_entity_id(
            name,
            entity_type
        )

        entity_lookup[
            name.lower()
        ] = entity_id

        builder.create_entity(
            entity_id,
            name,
            entity_type,
            paper_id
        )

    # -----------------------------------------
    # CREATE RELATIONSHIPS
    # -----------------------------------------

    for relation in paper.get(
        "relations",
        []
    ):

        source = relation.get(
            "source",
            ""
        ).strip()

        target = relation.get(
            "target",
            ""
        ).strip()

        relation_type = relation.get(
            "relation",
            "RELATED_TO"
        ).strip()

        if not source or not target:
            continue

        # Use entity ID if provided
        if source in entity_lookup.values():

            source_id = source

        else:

            source_id = entity_lookup.get(
                source.lower()
            )

        if target in entity_lookup.values():

            target_id = target

        else:

            target_id = entity_lookup.get(
                target.lower()
            )

        if not source_id or not target_id:

            print(
                "Warning: entity not found:",
                source,
                "->",
                target
            )

            continue

        builder.create_relationship(
            source_id,
            target_id,
            relation_type
        )

    print(
        f"Finished importing {paper_id}"
    )


def import_directory():

    directory = Path(
        "data/processed"
    )

    files = list(
        directory.glob("*.json")
    )

    if not files:

        print(
            "No JSON files found in "
            "data/processed/"
        )

        return

    builder = GraphBuilder()

    try:

        builder.create_constraints()

        print(
            f"Found {len(files)} JSON files."
        )

        for file in files:

            try:

                with open(
                    file,
                    "r",
                    encoding="utf-8"
                ) as f:

                    paper = json.load(f)

                import_paper(
                    builder,
                    paper
                )

            except Exception as error:

                print(
                    f"Error importing {file.name}:",
                    error
                )

        print()
        print("================================")
        print("IMPORT COMPLETED")
        print("================================")

    finally:

        builder.close()


if __name__ == "__main__":

    import_directory()