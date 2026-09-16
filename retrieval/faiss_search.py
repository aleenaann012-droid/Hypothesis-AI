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
