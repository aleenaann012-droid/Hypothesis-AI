
import spacy

# Load the existing scientific NLP model.
nlp = spacy.load("en_core_sci_sm")

RELATION_WORDS = {
    "cause", "contribute", "increase", "decrease",
    "associate", "affect", "lead", "result",
    "involve", "produce", "develop", "progress",
    "reduce", "predict", "indicate", "contain",
    "form", "limit", "improve", "inhibit",
    "activate", "regulate", "correlate"
}

SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj"}
OBJECT_DEPS = {"dobj", "obj", "attr", "oprd"}




def find_entity_for_token(token, entities):
    # Prefer an existing entity containing the token.
    candidates = [
        ent for ent in entities
        if ent.start <= token.i < ent.end
    ]

    if candidates:
        return min(candidates, key=lambda ent: len(ent))

    # Find the noun phrase containing the grammatical argument.
    noun_phrase = None

    for chunk in token.doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            noun_phrase = chunk
            break

    if noun_phrase is None:
        return None

    # Reuse an existing entity if it matches the noun phrase.
    phrase_text = noun_phrase.text.strip().lower()

    for ent in entities:
        if ent.text.strip().lower() == phrase_text:
            return ent

    # Create a temporary spaCy span for this candidate.
    # It is NOT automatically inserted into the knowledge graph.
    return noun_phrase

def extract_relations(text):
    """
    Extract candidate relationships using dependency parsing.

    Relationships retain source evidence and remain unvalidated.
    Arguments are resolved against existing entities or grammatical
    noun phrases when entity recognition misses a valid argument.
    """
    doc = nlp(text)
    relations = []
    seen = set()

    for sent in doc.sents:
        entities = list(sent.ents)

        if not entities:
            continue

        for token in sent:
            # Consider only recognized relationship verbs.
            if token.pos_ != "VERB":
                continue

            relation = token.lemma_.lower()

            if relation not in RELATION_WORDS:
                continue

            # Find grammatical subjects and objects attached
            # to this verb, rather than pairing nearby entities.
            subjects = [
                child for child in token.children
                if child.dep_ in SUBJECT_DEPS]

            objects = [
                child for child in token.children
                if child.dep_ in OBJECT_DEPS]

            if not subjects or not objects:
                continue

            for subj_token in subjects:
                for obj_token in objects:
                    subject_ent = find_entity_for_token(
                        subj_token, entities
                    )
                    object_ent = find_entity_for_token(
                        obj_token, entities
                    )

                    if subject_ent is None or object_ent is None:
                        continue

                    # Avoid duplicate candidate relationships.
def extract_relations(text):
    """
    Extract candidate relationships using dependency parsing.

    Relationships retain source evidence and remain unvalidated.
    Only relationships whose grammatical subject and object
    match existing scientific entities are returned.
    """
    doc = nlp(text)
    relations = []
    seen = set()

    for sent in doc.sents:
        entities = list(sent.ents)

        if len(entities) < 2:
            continue

        for token in sent:
            # Consider only recognized relationship verbs.
            if token.pos_ != "VERB":
                continue

            relation = token.lemma_.lower()

            if relation not in RELATION_WORDS:
                continue

            # Find grammatical subjects and objects attached
            # to this verb, rather than pairing nearby entities.
            subjects = [
                child for child in token.children
                if child.dep_ in SUBJECT_DEPS]

            objects = [
                child for child in token.children
                if child.dep_ in OBJECT_DEPS]

            if not subjects or not objects:
                continue

            for subj_token in subjects:
                for obj_token in objects:
                    subject_ent = find_entity_for_token(
                        subj_token, entities
                    )
                    object_ent = find_entity_for_token(
                        obj_token, entities
                    )

                    if subject_ent is None or object_ent is None:
                        continue

                    # Avoid duplicate candidate relationships.
                    key = (
                        subject_ent.start,
                        relation,
                        object_ent.start,
                        token.idx
                    )

                    if key in seen:
                        continue

                    seen.add(key)

                    relations.append({
                       "subject": (
                            subj_token.text
                            if subject_ent.label_ == "NP"
                            else subject_ent.text
                        ),
                        "subject_label": (
                            "UNKNOWN" if subject_ent.label_ == "NP"
                            else subject_ent.label_
                        ),

                        # Preserve original source evidence.
                        "evidence_text": sent.text,
                        "sentence_start": sent.start_char,
                        "sentence_end": sent.end_char,

                        # Character offsets of the relation verb.
                        "relation_start": token.idx,
                        "relation_end": token.idx + len(token.text),

                        "extraction_method": "dependency_heuristic",
                        "validation_status": "unvalidated"
                    })

    return relations


                       