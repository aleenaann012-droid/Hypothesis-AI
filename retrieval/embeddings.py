from sentence_transformers import SentenceTransformer
import numpy as np
import sys
from pathlib import Path
from nlp.pdf_extractor import extract_pages_from_pdf
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "nlp"))

from pdf_extractor import extract_text_from_pdf



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

def split_pages_with_provenance(pages, chunk_size=500):
    """Split page text into chunks with page-local and document offsets."""
    chunks = []
    document_offset = 0

    for page in pages:
        page_text = page["text"]
        page_number = page["page_number"]

        words = list(re.finditer(r"\S+", page_text))

        for i in range(0, len(words), chunk_size):
            selected = words[i:i + chunk_size]
            if not selected:
                continue

            start = selected[0].start()
            end = selected[-1].end()

            chunks.append({
                "page_number": page_number,
                "page_start": start,
                "page_end": end,
                "document_start": document_offset + start,
                "document_end": document_offset + end,
                "text": page_text[start:end],
            })

        # Account for the newline between pages.
        document_offset += len(page_text) + 1

    return chunks


if __name__ == "__main__":

   
    pdf_path = "test_researchpaper.pdf"

   
    text = extract_text_from_pdf(pdf_path)

    print("\nPDF text extracted.")
    print("Total characters:", len(text))

    
    chunks = split_text(text)

    print("Total chunks:", len(chunks))

    
    embeddings = create_embeddings(chunks)

    print("\n========== BGE EMBEDDINGS ==========\n")

    print("Embedding shape:", embeddings.shape)

    print("\nFirst embedding:")
    print(embeddings[0][:10])

    
    np.save("retrieval/embeddings.npy", embeddings)

    
    with open("retrieval/chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.replace("\n", " ") + "\n---CHUNK---\n")

    print("\nEmbeddings saved successfully.")
    
