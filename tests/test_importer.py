from knowledge_graph.importer import resolve_or_create_entity


class FakeBuilder:
    def __init__(self):
        self.created_entities = []

    def create_entity(
        self,
        entity_id,
        name,
        entity_type,
        paper_id
    ):
        self.created_entities.append({
            "entity_id": entity_id,
            "name": name,
            "entity_type": entity_type,
            "paper_id": paper_id
        })


def test_missing_entity_is_created_and_cached():
    builder = FakeBuilder()
    entity_lookup = {
        "communication": "ENTITY:communication"
    }

    entity_id = resolve_or_create_entity(
        builder,
        entity_lookup,
        "buildup",
        "ENTITY",
        "P001"
    )

    assert entity_id == "ENTITY:buildup"
    assert entity_lookup["buildup"] == "ENTITY:buildup"
    assert len(builder.created_entities) == 1


def test_existing_entity_is_reused():
    builder = FakeBuilder()
    entity_lookup = {
        "communication": "ENTITY:communication"
    }

    entity_id = resolve_or_create_entity(
        builder,
        entity_lookup,
        "communication",
        "ENTITY",
        "P001"
    )

    assert entity_id == "ENTITY:communication"
    assert builder.created_entities == []


def test_empty_name_is_rejected():
    builder = FakeBuilder()

    entity_id = resolve_or_create_entity(
        builder,
        {},
        "   ",
        "ENTITY",
        "P001"
    )

    assert entity_id is None
    assert builder.created_entities == []

def test_missing_subject_and_existing_object_are_resolved():
    builder = FakeBuilder()
    entity_lookup = {
        "communication": "ENTITY:communication"
    }

    subject_id = resolve_or_create_entity(
        builder,
        entity_lookup,
        "buildup",
        "ENTITY",
        "P001"
    )

    object_id = resolve_or_create_entity(
        builder,
        entity_lookup,
        "communication",
        "ENTITY",
        "P001"
    )

    assert subject_id == "ENTITY:buildup"
    assert object_id == "ENTITY:communication"
    assert len(builder.created_entities) == 1
