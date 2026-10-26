from retrieval.embeddings import split_pages_with_provenance
from retrieval.faiss_search import load_provenance, search_with_provenance


def test_split_pages_with_provenance_preserves_offsets():
    pages = [
        {"page_number": 1, "text": "Alpha beta gamma"},
        {"page_number": 2, "text": "Delta epsilon"},
    ]

    chunks = split_pages_with_provenance(pages, chunk_size=2)

    assert len(chunks) == 3

    first = chunks[0]
    assert first["page_number"] == 1
    assert first["page_start"] == 0
    assert first["page_end"] == 10
    assert first["text"] == "Alpha beta"

    second_page = chunks[2]
    assert second_page["page_number"] == 2
    assert second_page["text"] == "Delta epsilon"
    assert second_page["page_start"] == 0

    for chunk in chunks:
        page = next(
            p for p in pages
            if p["page_number"] == chunk["page_number"]
        )
        assert page["text"][chunk["page_start"]:chunk["page_end"]] == chunk["text"]
        assert chunk["document_end"] - chunk["document_start"] == (
            chunk["page_end"] - chunk["page_start"]
        )


def test_chunk_provenance_file_has_exact_and_normalized_offsets():
    records = load_provenance()
    assert len(records) == 14, "Expected 14 chunk provenance records."

    for rec in records:
        assert "chunk_index" in rec
        assert "start_page" in rec
        assert "end_page" in rec
        assert rec["start_page"] <= rec["end_page"]
        assert "page_references" in rec
        assert len(rec["page_references"]) >= 1

        # Normalized offsets
        assert rec.get("offset_type") == "normalized_document_text"
        assert rec["normalized_document_start"] >= 0
        assert rec["normalized_document_end"] > rec["normalized_document_start"]

        # Exact original PDF offsets
        assert rec.get("exact_offset_type") == "raw_extracted_pdf_text"
        assert rec["exact_document_start"] >= 0
        assert rec["exact_document_end"] > rec["exact_document_start"]
        assert "page_spans" in rec
        assert len(rec["page_spans"]) >= 1


def test_search_with_provenance_integrates_metadata(monkeypatch):
    from retrieval import faiss_search

    dummy_chunks = ["First chunk about proteins", "Second chunk about neurons"]
    dummy_provenance = [
        {
            "chunk_index": 0,
            "start_page": 1,
            "end_page": 1,
            "page_references": [1],
            "normalized_document_start": 0,
            "normalized_document_end": 26,
            "offset_type": "normalized_document_text",
            "exact_document_start": 0,
            "exact_document_end": 26,
            "exact_offset_type": "raw_extracted_pdf_text",
            "page_spans": [{"page_number": 1, "page_start_char": 0, "page_end_char": 26}],
        },
        {
            "chunk_index": 1,
            "start_page": 2,
            "end_page": 2,
            "page_references": [2],
            "normalized_document_start": 27,
            "normalized_document_end": 53,
            "offset_type": "normalized_document_text",
            "exact_document_start": 27,
            "exact_document_end": 53,
            "exact_offset_type": "raw_extracted_pdf_text",
            "page_spans": [{"page_number": 2, "page_start_char": 0, "page_end_char": 26}],
        },
    ]

    monkeypatch.setattr(
        faiss_search,
        "search",
        lambda query, index, top_k=3: ([0.92, 0.81], [0, 1]),
    )

    results = search_with_provenance(
        "query",
        index=None,
        top_k=2,
        chunks=dummy_chunks,
        provenance=dummy_provenance,
    )

    assert len(results) == 2
    first = results[0]
    assert first["chunk_id"] == 0
    assert first["similarity"] == 0.92
    assert first["start_page"] == 1
    assert first["page_references"] == [1]
    assert first["offset_type"] == "normalized_document_text"
    assert first["exact_offset_type"] == "raw_extracted_pdf_text"
    assert len(first["page_spans"]) == 1

