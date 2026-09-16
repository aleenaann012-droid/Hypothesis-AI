import spacy
from nlp.pdf_extractor import extract_text_from_pdf


# Load SciSpaCy scientific NLP model
nlp = spacy.load("en_core_sci_sm")


def extract_entities(text):
    """
    Extract scientific entities from text.
    """

    doc = nlp(text)

    entities = []

    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })

    return entities


if __name__ == "__main__":

    # PDF file
    pdf_path = "test_researchpaper.pdf"

    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    # Extract entities
    entities = extract_entities(text)

    print("\n========== EXTRACTED ENTITIES ==========\n")

    for entity in entities:
        print(entity["text"], "->", entity["label"])

    print("\nTotal entities:", len(entities))


import json

# Save extracted entities to JSON
with open("nlp/entity.json", "w", encoding="utf-8") as f:
    json.dump(entities, f, indent=4, ensure_ascii=False)

print("\nEntities saved to nlp/entity.json")