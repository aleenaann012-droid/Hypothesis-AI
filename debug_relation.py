
import spacy

nlp = spacy.load("en_core_sci_sm")

text = (
    "Excess protein buildup limits communication "
    "between nerve cells."
)

doc = nlp(text)

print("\n--- TOKENS AND DEPENDENCIES ---")
for token in doc:
    print(
        f"{token.text:18} "
        f"POS={token.pos_:8} "
        f"DEP={token.dep_:12} "
        f"HEAD={token.head.text}"
    )

print("\n--- RECOGNIZED ENTITIES ---")
for entity in doc.ents:
    print(
        f"{entity.text!r:35} "
        f"LABEL={entity.label_:15} "
        f"TOKEN_RANGE=({entity.start}, {entity.end})"
    )

print("\n--- RELATION EXTRACTOR OUTPUT ---")
from nlp.relation_extractor import extract_relations

print(extract_relations(text))
