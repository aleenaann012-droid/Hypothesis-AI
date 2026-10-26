from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))

    text = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text.append(page_text)

    return "\n".join(text)


def extract_pages_from_pdf(pdf_path):
    """Return extracted PDF text page by page, preserving page numbers."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append({
                "page_number": page_number,
                "text": page_text,
            })

    return pages

if __name__ == "__main__":

    pdf_path = input("Enter PDF path: ").strip()

    try:
        extracted_text = extract_text_from_pdf(pdf_path)

        print("\n========== EXTRACTED TEXT ==========\n")
        print(extracted_text[:5000])

        print("\n====================================")
        print("Total characters:", len(extracted_text))

    except Exception as e:
        print("Error:", e)