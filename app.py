from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request

from utils.analyzer import analyze_content
from utils.ocr import extract_text_from_image
from utils.pdf_extractor import extract_text_from_pdf


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}


app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return (
        Path(filename).suffix.lower()
        in ALLOWED_EXTENSIONS
    )


def extract_text(file_path: Path) -> str:
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)

    if extension in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
    }:
        return extract_text_from_image(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    extracted_text = ""
    error = None

    if request.method == "POST":

        uploaded_file = request.files.get("file")

        if uploaded_file is None:
            error = "Please choose a PDF or image."

            return render_template(
                "index.html",
                result=result,
                extracted_text=extracted_text,
                error=error,
            )

        if not uploaded_file.filename:
            error = "Please choose a file."

            return render_template(
                "index.html",
                result=result,
                extracted_text=extracted_text,
                error=error,
            )

        if not allowed_file(
            uploaded_file.filename
        ):
            error = (
                "Unsupported file type. "
                "Upload PDF, PNG, JPG, JPEG, WEBP or BMP."
            )

            return render_template(
                "index.html",
                result=result,
                extracted_text=extracted_text,
                error=error,
            )

        extension = Path(
            uploaded_file.filename
        ).suffix.lower()

        safe_filename = (
            f"{uuid4().hex}{extension}"
        )

        file_path = (
            UPLOAD_FOLDER / safe_filename
        )

        try:
            uploaded_file.save(file_path)

            extracted_text = extract_text(
                file_path
            )

            if not extracted_text.strip():
                error = (
                    "No readable caption text was found. "
                    "Make sure the PDF/image contains clear "
                    "social-media text."
                )

            else:
                result = analyze_content(
                    extracted_text
                )

        except Exception as exc:
            error = (
                f"Analysis failed: {exc}"
            )

        finally:
            if file_path.exists():
                try:
                    file_path.unlink()
                except OSError:
                    pass

    return render_template(
        "index.html",
        result=result,
        extracted_text=extracted_text,
        error=error,
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )