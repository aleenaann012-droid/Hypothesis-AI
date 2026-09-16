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