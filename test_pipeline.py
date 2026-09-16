from nlp.pdf_extractor import extract_text_from_pdf
from nlp.entity_extractor import extract_entities
from nlp.relation_extractor import extract_relations

from retrieval.faiss_search import (
    load_embeddings,
    load_chunks,
    create_faiss_index,
    search
)

from hypothesis.generator import generate_hypothesis


# ==========================================
# TEST PIPELINE
# ==========================================

PDF_PATH = "test_researchpaper.pdf"


print("\n========== HYPOTHESIS-AI PIPELINE ==========\n")


# ------------------------------------------
# 1. PDF TEXT EXTRACTION
# ------------------------------------------

print("STEP 1: Extracting PDF text...")

text = extract_text_from_pdf(PDF_PATH)

print("PDF text extracted successfully!")
print("Total characters:", len(text))


# ------------------------------------------
# 2. ENTITY EXTRACTION
# ------------------------------------------

print("\nSTEP 2: Extracting entities...")

entities = extract_entities(text)

print("Total entities:", len(entities))

print("\nSample entities:")

for entity in entities[:20]:
    print(entity)


# ------------------------------------------
# 3. RELATION EXTRACTION
# ------------------------------------------

print("\nSTEP 3: Extracting relations...")

relations = extract_relations(text)

print("Total relations:", len(relations))

print("\nSample relations:")

for relation in relations[:20]:
    print(
        relation["subject"],
        "->",
        relation["relation"],
        "->",
        relation["object"]
    )


# ------------------------------------------
# 4. LOAD EMBEDDINGS
# ------------------------------------------

print("\nSTEP 4: Loading embeddings...")

embeddings = load_embeddings()

print("Embedding shape:", embeddings.shape)


# ------------------------------------------
# 5. LOAD TEXT CHUNKS
# ------------------------------------------

print("\nSTEP 5: Loading text chunks...")

chunks = load_chunks()

print("Total chunks:", len(chunks))


# ------------------------------------------
# 6. CREATE FAISS INDEX
# ------------------------------------------

print("\nSTEP 6: Creating FAISS index...")

index = create_faiss_index(embeddings)

print("FAISS index created successfully!")
print("Total vectors:", index.ntotal)
print("Vector dimension:", index.d)


# ------------------------------------------
# 7. SEARCH
# ------------------------------------------

query = input(
    "\nEnter your research question: "
)

print("\nSTEP 7: Searching research paper...")

scores, indices = search(
    query,
    index,
    top_k=3
)


print("\n========== SEARCH RESULTS ==========")


# Store retrieved text
retrieved_text = ""


for rank, (idx, score) in enumerate(
    zip(indices, scores),
    start=1
):

    print(f"\n--- Result {rank} ---")

    print("Chunk:", idx)

    print(
        f"Similarity Score: {score:.4f}"
    )

    if idx < len(chunks):

        print("\nText:")

        print(chunks[idx][:1000])

        retrieved_text += chunks[idx] + "\n"


# ------------------------------------------
# 8. HYPOTHESIS GENERATION
# ------------------------------------------

print("\nSTEP 8: Generating hypothesis...")

if retrieved_text.strip():

    hypothesis = generate_hypothesis(
        retrieved_text
    )

    print(
        "\n========== GENERATED HYPOTHESIS ==========\n"
    )

    print(hypothesis)

else:

    print("No relevant research text found.")


print("\n========== PIPELINE COMPLETE ==========\n")