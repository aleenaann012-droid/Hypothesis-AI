
import spacy

nlp = spacy.load("en_core_sci_sm")

sentence = (
    "Excess protein buildup limits communication "
    "between nerve cells."
)

doc = nlp(sentence)

print("\nTOKEN DEPENDENCY CHECK")
print("-" * 60)
print(f"{'TOKEN':<20} {'POS':<10} {'DEP':<15} {'HEAD':<20}")

for token in doc:
    print(
        f"{token.text:<20} "
        f"{token.pos_:<10} "
        f"{token.dep_:<15} "
        f"{token.head.text:<20}"
    )

print("\nROOT VERBS:")
for token in doc:
    if token.pos_ == "VERB" and token.dep_ == "ROOT":
        print(token.text)
