
from nlp.relation_extractor import extract_relations


def test_relation_contains_evidence():
    text = (
        "Solar radiation increases the temperature "
        "of spacecraft materials."
    )

    relations = extract_relations(text)

    assert relations, "Expected a candidate relationship."

    relation = relations[0]

    assert relation["evidence_text"] == text
    assert relation["subject_label"]
    assert relation["object_label"]
    assert relation["validation_status"] == "unvalidated"
    assert relation["relation_start"] >= 0


def test_extracts_grammatical_relationship():
    text = (
        "Excess protein buildup limits communication "
        "between nerve cells."
    )

    relations = extract_relations(text)

    assert relations, "Expected a candidate relationship."

    matching_relations = [
        relation
        for relation in relations
        if relation["subject"].lower() == "buildup"
        and relation["relation"].lower() == "limit"
        and relation["object"].lower() == "communication"
    ]

    assert matching_relations, (
        "Expected buildup --limit--> communication, "
        "using the grammatical subject and object."
    )

    relation = matching_relations[0]

    assert relation["evidence_text"] == text
    assert relation["extraction_method"] == "dependency_heuristic"
    assert relation["validation_status"] == "unvalidated"
