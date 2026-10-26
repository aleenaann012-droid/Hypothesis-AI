import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nlp.pdf_extractor import extract_pages_from_pdf, extract_text_from_pdf


PDF_FILE = PROJECT_ROOT / "test_researchpaper.pdf"
CHUNKS_FILE = PROJECT_ROOT / "retrieval" / "chunks.txt"


def normalize(text):
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def build_provenance_records(pdf_path=PDF_FILE, chunks_file=CHUNKS_FILE):
    pages = extract_pages_from_pdf(pdf_path)
    raw_document = extract_text_from_pdf(pdf_path)
    chunks = [
        c.strip()
        for c in Path(chunks_file).read_text(encoding="utf-8").split("---CHUNK---")
        if c.strip()
    ]

    # Build normalized document text and normalized page ranges.
    document_parts = []
    normalized_page_ranges = []
    cursor = 0

    for page in pages:
        normalized_page = normalize(page["text"])
        if not normalized_page:
            continue

        if document_parts:
            cursor += 1  # separator space

        start = cursor
        document_parts.append(normalized_page)
        cursor += len(normalized_page)

        normalized_page_ranges.append({
            "page_number": page["page_number"],
            "start": start,
            "end": cursor,
        })

    document_text = " ".join(document_parts)

    # Build raw document page ranges
    raw_page_ranges = []
    raw_cursor = 0
    for page in pages:
        page_len = len(page["text"])
        raw_page_ranges.append({
            "page_number": page["page_number"],
            "start": raw_cursor,
            "end": raw_cursor + page_len,
            "text": page["text"],
        })
        raw_cursor += page_len + 1

    raw_words = list(re.finditer(r"\S+", raw_document))

    failures = []
    records = []

    for index, chunk in enumerate(chunks):
        target = normalize(chunk)
        matches = [
            m.start()
            for m in re.finditer(re.escape(target), document_text)
        ]

        if len(matches) != 1:
            failures.append((index, "match_count", len(matches)))
            continue

        start = matches[0]
        end = start + len(target)

        matched_pages = [
            p["page_number"]
            for p in normalized_page_ranges
            if start < p["end"] and end > p["start"]
        ]

        if not matched_pages:
            failures.append((index, "no_page_range", None))
            continue

        # Exact raw PDF text offsets and page spans
        start_w = index * 500
        end_w = min((index + 1) * 500, len(raw_words))
        exact_doc_start = raw_words[start_w].start()
        exact_doc_end = raw_words[end_w - 1].end()

        page_spans = []
        for pr in raw_page_ranges:
            if exact_doc_start < pr["end"] and exact_doc_end > pr["start"]:
                p_start = max(exact_doc_start, pr["start"]) - pr["start"]
                p_end = min(exact_doc_end, pr["end"]) - pr["start"]
                page_spans.append({
                    "page_number": pr["page_number"],
                    "page_start_char": p_start,
                    "page_end_char": p_end,
                    "length": p_end - p_start,
                })

        page_refs = [s["page_number"] for s in page_spans] if page_spans else matched_pages

        records.append({
            "chunk_index": index,
            "start_page": matched_pages[0],
            "end_page": matched_pages[-1],
            "page_references": page_refs,
            "normalized_document_start": start,
            "normalized_document_end": end,
            "offset_type": "normalized_document_text",
            "exact_document_start": exact_doc_start,
            "exact_document_end": exact_doc_end,
            "exact_offset_type": "raw_extracted_pdf_text",
            "page_spans": page_spans,
        })

    if failures:
        raise ValueError(f"Mapping failed with {len(failures)} failures: {failures}")

    return records


def main():
    records = build_provenance_records(PDF_FILE, CHUNKS_FILE)
    print(f"\nTotal chunks: {len(records)}")

    output_file = PROJECT_ROOT / "retrieval" / "chunk_provenance.json"
    output_file.write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    print("Saved provenance records:", len(records))
    print("Output:", output_file)


if __name__ == "__main__":
    main()

