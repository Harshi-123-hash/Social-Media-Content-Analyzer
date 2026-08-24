from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(
    file_path: str | Path,
) -> str:
    """
    Extract text from a PDF file.

    Returns an empty string when the PDF contains no
    machine-readable text.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Unsupported file format: {path.suffix}"
        )

    try:
        reader = PdfReader(str(path))

        pages = []

        for page in reader.pages:
            text = page.extract_text() or ""

            cleaned = " ".join(
                text.split()
            ).strip()

            if cleaned:
                pages.append(cleaned)

        return "\n\n".join(pages)

    except Exception as exc:
        raise RuntimeError(
            f"Unable to extract text from PDF: {exc}"
        ) from exc