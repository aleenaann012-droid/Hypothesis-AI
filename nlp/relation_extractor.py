import spacy
from pathlib import Path
import sys

# Add nlp folder to Python path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from nlp.pdf_extractor import extract_text_from_pdf


# Load SciSpaCy model
nlp = spacy.load("en_core_sci_sm")


def extract_relations(text):
    """
    Extract relationships between scientific entities
    using dependency parsing.
    """

    doc = nlp(text)
    relations = []

    # Words that are useful as scientific relationships
    relation_words = {
        "cause",
        "cause",
        "contribute",
        "increase",
        "decrease",
        "associate",
        "associated",
        "affect",
        "lead",
        "result",
        "involve",
        "produce",
        "develop",
        "progress",
        "reduce",
        "predict",
        "indicate",
        "contain",
        "form",
        "limit"
    }

    for sent in doc.sents:

        entities = list(sent.ents)

        if len(entities) < 2:
            continue

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):

                entity1 = entities[i]
                entity2 = entities[j]

                # Ignore entities that are too far apart
                distance = abs(entity1.start - entity2.start)

                if distance > 15:
                    continue

                relation = None

                # Look for verbs between the two entities
                start = min(entity1.end, entity2.end)
                end = max(entity1.start, entity2.start)

                for token in doc[start:end]:

                    if token.pos_ == "VERB":

                        lemma = token.lemma_.lower()

                        if lemma in relation_words:
                            relation = lemma
                            break

                if relation:

                    relations.append({
                        "subject": entity1.text,
                        "relation": relation,
                        "object": entity2.text
                    })

    return relations


if __name__ == "__main__":

    # Research paper
    pdf_path = "test_researchpaper.pdf"

    # Extract PDF text
    text = extract_text_from_pdf(pdf_path)

    # Extract relations
    relations = extract_relations(text)

    print("\n========== EXTRACTED RELATIONS ==========\n")

    for relation in relations[:100]:

        print(
            relation["subject"],
            "->",
            relation["relation"],
            "->",
            relation["object"]
        )

    print("\nTotal relations:", len(relations))