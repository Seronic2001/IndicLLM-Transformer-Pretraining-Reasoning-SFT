"""PDF / ZIP corpus plumbing shared by both languages (pipelinemanual fraction).

Behind the NCERT/SCERT fetchers in ``hindi/data/sources.py`` /
``assamese/data/sources.py``:

  * ``download_pdf_or_zip`` — fetch a URL that is a PDF, an HTML page with an
    embedded ``.pdf`` link, or a ZIP of chapter PDFs (the NCERT official format:
    ``https://ncert.nic.in/textbook/pdf/<code>dd.zip``); returns the local PDF
    paths (one for a direct PDF, several for an archive). Downloads are cached
    under ``cache_dir``, so re-runs are free and interrupted runs resume.
  * ``extract_pdf_text``     — pull text out of a PDF via pypdf / PyPDF2 /
    PyMuPDF (first one installed).

Optional dependencies (pypdf, PyPDF2, fitz) are imported lazily so this module
imports offline; a missing extractor raises :class:`PDFError` with a pointer to
``scripts/ocr.py`` for scanned books. The HTML-viewer → PDF-link extraction
pattern follows the approach of https://github.com/aayushdutt/ncert-downloader
(parsing NCERT's public textbook pages).
"""

from __future__ import annotations

import hashlib
import re
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

PDF_LINK_RE = re.compile(
    r'<(?:a|iframe)\s[^>]*?(?:href|src)\s*=\s*["\']([^"\']*\.pdf[^"\']*)["\']',
    re.I,
)
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LM-corpus-collector"}
_TIMEOUT = 180


class PDFError(RuntimeError):
    """Download or extraction failure for a PDF/archive."""


def _fetch(url: str, timeout: int) -> Optional[bytes]:
    """Fetch the full body; None on any network failure."""
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


def _cache_name(url: str) -> str:
    """Deterministic, collision-free cache stem.

    The last path segment alone is ambiguous (e.g. Google Drive's
    ``/uc?export=download&id=<id>`` all share the segment ``uc``), so a short
    URL hash is appended: ``<segment>-<sha256[:8]>``.
    """
    segment = re.sub(r"[^A-Za-z0-9._-]", "_", url.split("?")[0].rsplit("/", 1)[-1])
    if not segment:
        segment = "book"
    return f"{segment}-{hashlib.sha256(url.encode('utf-8')).hexdigest()[:8]}"


def download_pdf_or_zip(url: str, cache_dir: str) -> list[Path]:
    """Download ``url`` and return local PDF paths.

    Accepts three payloads:
      * a direct PDF            -> [pdf_path]
      * a ZIP of chapter PDFs   -> [chapter1.pdf, chapter2.pdf, ...] (extracted)
      * an HTML viewer page     -> follows the first embedded ``.pdf`` link
    Returns [] when the download fails or yields no PDF (caller reports an
    honest shortfall). Cached: a completed download is never re-fetched.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    stem = _cache_name(url)
    dest = cache / f"{stem}.bin"
    try:
        if not (dest.exists() and dest.stat().st_size > 0):
            data = _fetch(url, _TIMEOUT)
            if data is None:
                return []
            # HTML viewer page -> extract the first embedded PDF link or Google Drive confirmation link.
            if not data.startswith(b"%PDF") and not data.startswith(b"PK\x03\x04"):
                html_text = data.decode("utf-8", "replace")
                match = PDF_LINK_RE.search(html_text)
                if not match and "drive.google.com" in url:
                    gdrive_match = re.search(r'href=["\'](/uc\?export=download[^"\']*)["\']', html_text)
                    if gdrive_match:
                        match = gdrive_match
                if not match:
                    return []
                pdf_url = match.group(1)
                if pdf_url.startswith("/"):
                    from urllib.parse import urlsplit

                    parts = urlsplit(url)
                    pdf_url = f"{parts.scheme}://{parts.netloc}{pdf_url}"
                data = _fetch(pdf_url, _TIMEOUT)
                if data is None:
                    return []
            dest.write_bytes(data)
        else:
            data = dest.read_bytes()
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return []  # flaky servers must not kill the run — honest shortfall

    if data.startswith(b"%PDF"):
        pdf = cache / (stem if stem.lower().endswith(".pdf") else f"{stem}.pdf")
        pdf.write_bytes(data)
        return [pdf]

    if data.startswith(b"PK\x03\x04"):  # ZIP of chapter PDFs
        try:
            with zipfile.ZipFile(dest) as z:
                z.extractall(cache / stem)
        except zipfile.BadZipFile:
            return []
        return sorted((cache / stem).rglob("*.pdf"))

    return []  # neither PDF nor ZIP — treat as a failed entry


def extract_pdf_text(path: str) -> str:
    """Extract text from a PDF; '' for unreadable files; PDFError if no extractor."""
    for modname in ("pypdf", "PyPDF2", "fitz"):
        try:
            if modname == "pypdf":
                from pypdf import PdfReader

                reader = PdfReader(path)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            if modname == "PyPDF2":
                from PyPDF2 import PdfReader

                reader = PdfReader(path)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            import fitz

            doc = fitz.open(path)
            try:
                return "\n".join(page.get_text() for page in doc)
            finally:
                doc.close()
        except ImportError:
            continue
        except Exception:
            return ""  # corrupt/unreadable page — treat as empty
    raise PDFError(
        "no PDF text extractor installed (pip install pypdf); scanned books need "
        "scripts/ocr.py instead"
    )
