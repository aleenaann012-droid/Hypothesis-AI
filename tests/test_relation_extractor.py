
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

def test_llm_relation_offsets_use_correct_sentence_occurrence(monkeypatch):
    from types import SimpleNamespace
    from nlp import relation_extractor as extractor

    sentence_text = "Protein limits protein activity."
    sentence = SimpleNamespace(
    text=sentence_text,
    start_char=100,
    end_char=100 + len(sentence_text),
)

    monkeypatch.setattr(
        extractor.requests,
        "post",
        lambda *args, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "message": {
                    "content": (
                        '{"relations":[{"subject":"protein",'
                        '"relation":"limits","object":"protein"}]}'
                    )
                }
            },
        ),
    )

    relations = extractor._llm_relations(sentence)

    assert len(relations) == 1
    relation = relations[0]
    assert relation["sentence_start"] == 100
    assert relation["sentence_end"] == 100 + len(sentence_text)
    assert relation["relation_start"] == 108
    assert relation["relation_end"] == 114
    
def test_llm_relation_preserves_subject_and_object_offsets(monkeypatch):
    from types import SimpleNamespace
    from nlp import relation_extractor as extractor

    sentence_text = "Protein limits protein activity."
    sentence = SimpleNamespace(
        text=sentence_text,
        start_char=100,
        end_char=100 + len(sentence_text),
    )

    monkeypatch.setattr(
        extractor.requests,
        "post",
        lambda *args, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "message": {
                    "content": (
                        '{"relations":[{"subject":"Protein",'
                        '"relation":"limits","object":"activity"}]}'
                    )
                }
            },
        ),
    )

    relation = extractor._llm_relations(sentence)[0]

    assert sentence_text[
        relation["subject_start"] - sentence.start_char:
        relation["subject_end"] - sentence.start_char
    ] == relation["subject"]

    assert sentence_text[
        relation["relation_start"] - sentence.start_char:
        relation["relation_end"] - sentence.start_char
    ] == relation["relation"]

    assert sentence_text[
        relation["object_start"] - sentence.start_char:
        relation["object_end"] - sentence.start_char
    ] == relation["object"]
