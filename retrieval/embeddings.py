from sentence_transformers import SentenceTransformer
import numpy as np
import sys
from pathlib import Path


# Add the nlp folder to Python's path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "nlp"))

from pdf_extractor import extract_text_from_pdf


# Load BGE embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def create_embeddings(texts):
    """
    Convert text passages into numerical vectors.
    """

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return np.array(embeddings)


def split_text(text, chunk_size=500):
    """
    Split extracted PDF text into smaller chunks.
    """

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


if __name__ == "__main__":

    # Research paper
    pdf_path = "test_researchpaper.pdf"

    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    print("\nPDF text extracted.")
    print("Total characters:", len(text))

    # Split text into chunks
    chunks = split_text(text)

    print("Total chunks:", len(chunks))

    # Create BGE embeddings
    embeddings = create_embeddings(chunks)

    print("\n========== BGE EMBEDDINGS ==========\n")

    print("Embedding shape:", embeddings.shape)

    print("\nFirst embedding:")
    print(embeddings[0][:10])

    # Save embeddings
    np.save("retrieval/embeddings.npy", embeddings)

    # Save chunks
    with open("retrieval/chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.replace("\n", " ") + "\n---CHUNK---\n")

    print("\nEmbeddings saved successfully.")