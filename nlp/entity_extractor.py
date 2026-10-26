import spacy
from nlp.pdf_extractor import extract_text_from_pdf


# Load SciSpaCy scientific NLP model
nlp = spacy.load("en_core_sci_sm")


# Generic or stop words that are not meaningful scientific entities
GENERIC_STOP_ENTITIES = {
    "treatment", "disease", "drug", "future", "confusing", "study", "studies",
    "result", "results", "paper", "data", "figure", "table", "graph", "analysis",
    "author", "authors", "patient", "patients", "human", "humans", "mouse", "mice",
    "approach", "method", "methods", "trial", "trials", "effect", "effects",
    "level", "levels", "group", "groups", "sample", "samples", "case", "cases",
    "rate", "time", "year", "years", "day", "days", "week", "weeks", "month", "months",
    "normal", "control", "finding", "findings", "state", "factor", "factors",
    "function", "functions", "role", "type", "types", "evidence", "hypothesis",
    "model", "models", "conclusion", "conclusions", "test", "tests", "ability",
    "progress", "progression", "presence", "absence", "change", "changes", "increase", "decrease",
    "it", "they", "this", "that", "these", "those", "which", "who", "whom", "whose",
    "there", "here", "what", "we", "i", "you", "he", "she", "one", "all", "both"
}


def is_valid_scientific_entity(name: str) -> bool:
    """
    Check if an extracted entity string is scientifically meaningful and not a generic noun/stopword.
    """
    if not name or not isinstance(name, str):
        return False
    cleaned = name.strip().lower()
    # Length check: single characters and pure punctuation/digits are not valid entities
    if len(cleaned) < 2:
        return False
    if cleaned in GENERIC_STOP_ENTITIES:
        return False
    # If it's pure numbers or symbols
    if cleaned.isdigit() or all(not c.isalnum() for c in cleaned):
        return False
    return True


def extract_entities(text, filter_generic: bool = True):
    """
    Extract scientific entities from text, with optional generic entity filtering.
    """
    if not text or not text.strip():
        return []

    doc = nlp(text)

    entities = []
    seen = set()

    for ent in doc.ents:
        raw_name = ent.text.strip()
        if not raw_name:
            continue
        if filter_generic and not is_valid_scientific_entity(raw_name):
            continue
        key = (raw_name.lower(), ent.label_)
        if key not in seen:
            seen.add(key)
            entities.append({
                "text": raw_name,
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


# import json

# # Save extracted entities to JSON
# with open("nlp/entity.json", "w", encoding="utf-8") as f:
#     json.dump(entities, f, indent=4, ensure_ascii=False)

# print("\nEntities saved to nlp/entity.json")