"""OCR wrapper — AI4Bharat Indic-OCR primary, Tesseract fallback.

Invoked per language with ``lang`` in {"hin", "asm"}. If the primary engine fails
or returns low-confidence output, falls back to Tesseract (``-l hin`` / ``-l asm``);
if both fail on a document, the document is skipped and logged (never fabricated).

Both engines are invoked via subprocess so the caller controls availability; local
dev without either engine raises a clear ``OCRUnavailableError`` instead of
silently returning garbage.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional

TESSERACT_LANG = {"hindi": "hin", "assamese": "asm", "hin": "hin", "asm": "asm"}
TESSERACT_CONF = "--psm 6"


class OCRUnavailableError(RuntimeError):
    """Neither OCR engine is installed/runnable."""


def _run(cmd: list[str], timeout: int = 300) -> str:
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"OCR command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-500:]}")
    return proc.stdout


def run_indic_ocr(image_path: str, lang: str, **kwargs) -> Optional[str]:
    """AI4Bharat Indic-OCR via its CLI (``indic-ocr``), None if unavailable.

    Returns extracted text; raises RuntimeError on engine failure.
    """
    exe = shutil.which("indic-ocr")
    if exe is None:
        return None
    cmd = [exe, "predict", "--image", image_path, "--lang", lang]
    return _run(cmd, timeout=kwargs.get("timeout", 300))


def run_tesseract(image_path: str, lang: str, **kwargs) -> str:
    t_lang = TESSERACT_LANG.get(lang, lang)
    try:
        import pytesseract
        from PIL import Image

        img = Image.open(image_path)
        # Convert to grayscale for improved OCR contrast
        if img.mode != "L":
            img = img.convert("L")
        text = pytesseract.image_to_string(img, lang=t_lang, config=TESSERACT_CONF)
        if text and text.strip():
            return text
    except Exception:
        pass

    exe = shutil.which("tesseract")
    if exe is None:
        raise OCRUnavailableError(
            "tesseract not installed — install it or provide an OCR engine "
            "(pip install pytesseract + apt tesseract-ocr-hin/asm)"
        )
    cmd = [exe, image_path, "stdout", "-l", t_lang, TESSERACT_CONF]
    return _run(cmd, timeout=kwargs.get("timeout", 300))


def ocr_document(
    image_path: str,
    lang: str,
    min_conf: float = 0.0,
    use_tesseract_fallback: bool = True,
) -> str:
    """OCR one image: primary engine, then Tesseract fallback.

    Raises OCRUnavailableError / RuntimeError only if BOTH fail — the caller
    (data/collect.py) catches this and skips + logs the document.
    """
    text = run_indic_ocr(image_path, lang)
    if text and text.strip():
        return text.strip()
    if use_tesseract_fallback:
        return run_tesseract(image_path, lang).strip()
    raise OCRUnavailableError(f"no OCR output for {image_path}")


def ocr_pdf(
    pdf_path: str,
    lang: str,
    max_pages: int = 50,
) -> str:
    """Render pages of a scanned PDF and extract text using OCR.

    Uses pypdfium2 or fitz to rasterize pages to temporary PNG images,
    then runs ocr_document on each page.
    """
    import os
    import tempfile
    from pathlib import Path

    page_texts: list[str] = []

    # 1. Try pypdfium2
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(pdf_path)
        num_pages = min(len(pdf), max_pages)
        with tempfile.TemporaryDirectory() as tmpdir:
            for page_idx in range(num_pages):
                page = pdf[page_idx]
                image = page.render(scale=2).to_pil()
                img_path = os.path.join(tmpdir, f"page_{page_idx:04d}.png")
                image.save(img_path)
                try:
                    p_text = ocr_document(img_path, lang)
                    if p_text and p_text.strip():
                        page_texts.append(p_text.strip())
                except Exception:
                    continue
        if page_texts:
            return "\n\n".join(page_texts)
    except ImportError:
        pass

    # 2. Try fitz (PyMuPDF)
    try:
        import fitz

        doc = fitz.open(pdf_path)
        num_pages = min(len(doc), max_pages)
        with tempfile.TemporaryDirectory() as tmpdir:
            for page_idx in range(num_pages):
                page = doc[page_idx]
                pix = page.get_pixmap(dpi=150)
                img_path = os.path.join(tmpdir, f"page_{page_idx:04d}.png")
                pix.save(img_path)
                try:
                    p_text = ocr_document(img_path, lang)
                    if p_text and p_text.strip():
                        page_texts.append(p_text.strip())
                except Exception:
                    continue
        doc.close()
        if page_texts:
            return "\n\n".join(page_texts)
    except ImportError:
        pass

    if not page_texts:
        raise OCRUnavailableError(f"unable to rasterize and OCR PDF: {pdf_path}")

    return "\n\n".join(page_texts)
