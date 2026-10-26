import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis.generator import generate_hypothesis

# Paths
BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_FILE = BASE_DIR / "embeddings.npy"
CHUNKS_FILE = BASE_DIR / "chunks.txt"
PROVENANCE_FILE = BASE_DIR / "chunk_provenance.json"


# Load BGE model
print("Loading BGE model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def load_embeddings():
    """
    Load saved research paper embeddings.
    """
    embeddings = np.load(EMBEDDING_FILE)
    return embeddings.astype("float32")


def load_chunks():
    """
    Load the original text chunks.
    """
    text = CHUNKS_FILE.read_text(encoding="utf-8")

    chunks = text.split("---CHUNK---")

    # Remove empty chunks and extra spaces
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    return chunks


def load_provenance():
    """
    Load chunk provenance metadata mapping chunks to original PDF pages and offsets.
    """
    if PROVENANCE_FILE.exists():
        import json
        with open(PROVENANCE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def create_faiss_index(embeddings):
    """
    Create FAISS index using cosine similarity.
    """

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def search(query, index, top_k=3):
    """
    Convert query into an embedding and search FAISS.
    """

    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    ).astype("float32")

    query_embedding = query_embedding.reshape(1, -1)

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    return scores[0], indices[0]


def search_with_provenance(query, index, top_k=3, chunks=None, provenance=None):
    """
    Search FAISS and return retrieved chunks enriched with full provenance metadata.
    Distinguishes normalized offsets from exact original PDF offsets.
    """
    if chunks is None:
        chunks = load_chunks()
    if provenance is None:
        provenance = load_provenance()

    prov_lookup = {p.get("chunk_index"): p for p in provenance}
    scores, indices = search(query, index, top_k=top_k)

    results = []
    for idx, score in zip(indices, scores):
        idx = int(idx)
        if 0 <= idx < len(chunks):
            prov = prov_lookup.get(idx, {})
            start_p = prov.get("start_page")
            end_p = prov.get("end_page")
            page_refs = prov.get("page_references", [start_p] if start_p is not None else [])

            results.append({
                "chunk_id": idx,
                "similarity": float(score),
                "text": chunks[idx],
                "start_page": start_p,
                "end_page": end_p,
                "page_references": page_refs,
                "normalized_document_start": prov.get("normalized_document_start"),
                "normalized_document_end": prov.get("normalized_document_end"),
                "offset_type": prov.get("offset_type", "normalized_document_text"),
                "exact_document_start": prov.get("exact_document_start"),
                "exact_document_end": prov.get("exact_document_end"),
                "exact_offset_type": prov.get("exact_offset_type", "raw_extracted_pdf_text"),
                "page_spans": prov.get("page_spans", []),
            })

    return results


if __name__ == "__main__":

    print("\nLoading embeddings...")

    embeddings = load_embeddings()

    print("Embedding shape:", embeddings.shape)

    print("\nLoading text chunks...")

    chunks = load_chunks()

    print("Total chunks:", len(chunks))

    print("\nCreating FAISS index...")

    index = create_faiss_index(embeddings)

    print("FAISS index created successfully!")
    print("Total vectors:", index.ntotal)
    print("Vector dimension:", index.d)

    # Get user query
    query = input("\nEnter your research question: ")

    scores, indices = search(
        query,
        index,
        top_k=3
    )

    print("\n========== SEARCH RESULTS ==========")

    for rank, (idx, score) in enumerate(
        zip(indices, scores),
        start=1
    ):

        print(f"\n--- Result {rank} ---")
        print(f"Chunk: {idx}")
        print(f"Similarity Score: {score:.4f}")

        if idx < len(chunks):
            print("\nText:")
            print(chunks[idx][:1000])
        # Generate hypothesis from the top retrieved research chunk
    if len(indices) > 0:
        research_text = chunks[indices[0]]
        hypothesis = generate_hypothesis(research_text)
        print("\n========== GENERATED HYPOTHESIS ==========")
        print(hypothesis)
