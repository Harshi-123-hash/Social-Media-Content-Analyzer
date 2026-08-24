from __future__ import annotations

from pathlib import Path

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def _prepare_image(image: Image.Image) -> Image.Image:
    """
    Prepare an image for OCR.

    The image is converted to grayscale, enlarged, enhanced,
    and lightly sharpened to improve text recognition.
    """

    image = image.convert("L")

    # Upscale small images.
    width, height = image.size

    if width < 1200:
        scale = 1200 / max(width, 1)

        image = image.resize(
            (
                int(width * scale),
                int(height * scale),
            ),
            Image.Resampling.LANCZOS,
        )

    # Improve contrast.
    image = ImageOps.autocontrast(image)

    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.5)

    # Improve sharpness.
    image = image.filter(
        ImageFilter.SHARPEN
    )

    return image


def extract_text_from_image(
    file_path: str | Path,
) -> str:
    """
    Extract readable text from an image using Tesseract OCR.

    Returns an empty string when no readable text is found.
    It does not raise an error simply because an image contains
    no text.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Image file not found: {path}"
        )

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    }

    if path.suffix.lower() not in supported_extensions:
        raise ValueError(
            f"Unsupported image format: {path.suffix}"
        )

    try:
        with Image.open(path) as image:
            image.load()

            prepared_image = _prepare_image(
                image
            )

            text = pytesseract.image_to_string(
                prepared_image,
                config="--psm 6",
            )

    except Exception as exc:
        raise RuntimeError(
            f"Unable to perform OCR on image: {exc}"
        ) from exc

    # Normalize OCR output.
    lines = []

    for line in text.splitlines():
        cleaned = " ".join(
            line.split()
        ).strip()

        if cleaned:
            lines.append(cleaned)

    return "\n".join(lines)
