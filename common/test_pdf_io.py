"""Offline tests for common/pdf_io.py — downloads are faked, PDF/ZIP handling is real."""

import io
import zipfile
from pathlib import Path

import pytest


def make_text_pdf(path: Path) -> None:
    """Write a minimal valid single-page PDF whose text extracts as 'Hello world'."""
    content = b"BT /F1 12 Tf 72 720 Td (Hello world) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(out))


def _pdf_bytes() -> bytes:
    tmp = Path("__pdf_io_probe.pdf")
    try:
        make_text_pdf(tmp)
        return tmp.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)


@pytest.fixture
def pdfio(monkeypatch, tmp_path):
    import common.pdf_io as p

    monkeypatch.setattr(p, "_fetch", lambda url, timeout: _pdf_bytes())
    return p, tmp_path


def test_direct_pdf_download(pdfio, tmp_path):
    p, _ = pdfio
    pdfs = p.download_pdf_or_zip("https://example.org/book.pdf", str(tmp_path / "c"))
    assert len(pdfs) == 1
    assert pdfs[0].suffix == ".pdf"
    assert pdfs[0].read_bytes().startswith(b"%PDF")


def test_drive_urls_do_not_collide_cache(monkeypatch, tmp_path):
    """Google Drive /uc URLs share a path segment; the URL hash keeps them apart."""
    import common.pdf_io as p

    monkeypatch.setattr(p, "_fetch", lambda url, timeout: _pdf_bytes())
    a = p.download_pdf_or_zip(
        "https://drive.google.com/uc?export=download&id=AAAA", str(tmp_path / "c")
    )
    b = p.download_pdf_or_zip(
        "https://drive.google.com/uc?export=download&id=BBBB", str(tmp_path / "c")
    )
    assert len(a) == 1 and len(b) == 1
    assert a[0].name != b[0].name  # distinct cache files despite identical path segment


def test_download_caches_on_second_call(pdfio, monkeypatch, tmp_path):
    p, _ = pdfio
    cache = str(tmp_path / "c")
    p.download_pdf_or_zip("https://example.org/book.pdf", cache)
    calls = {"n": 0}

    def boom(url, timeout):
        calls["n"] += 1
        raise AssertionError("network hit after cache fill")

    monkeypatch.setattr(p, "_fetch", boom)
    pdfs = p.download_pdf_or_zip("https://example.org/book.pdf", cache)
    assert calls["n"] == 0  # served from cache
    assert len(pdfs) == 1


def test_zip_of_chapter_pdfs(monkeypatch, tmp_path):
    import common.pdf_io as p

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("ch1.pdf", _pdf_bytes())
        z.writestr("ch2.pdf", _pdf_bytes())

    monkeypatch.setattr(p, "_fetch", lambda url, timeout: buf.getvalue())
    pdfs = p.download_pdf_or_zip(
        "https://ncert.nic.in/textbook/pdf/jhks1dd.zip", str(tmp_path / "c")
    )
    assert len(pdfs) == 2
    assert {x.name for x in pdfs} == {"ch1.pdf", "ch2.pdf"}
    assert all(x.read_bytes().startswith(b"%PDF") for x in pdfs)


def test_html_viewer_follows_pdf_link(monkeypatch, tmp_path):
    import common.pdf_io as p

    responses = iter(
        [
            b'<html><a href="/books/real.pdf">download</a></html>',
            _pdf_bytes(),
        ]
    )

    def fake_fetch(url, timeout):
        return next(responses)

    monkeypatch.setattr(p, "_fetch", fake_fetch)
    pdfs = p.download_pdf_or_zip("https://ncert.nic.in/textbook.php?x=1", str(tmp_path / "c"))
    assert len(pdfs) == 1
    assert pdfs[0].read_bytes().startswith(b"%PDF")


def test_unrecognized_payload_returns_empty(monkeypatch, tmp_path):
    import common.pdf_io as p

    monkeypatch.setattr(p, "_fetch", lambda url, timeout: b"<html>no pdf link here</html>")
    assert p.download_pdf_or_zip("https://example.org/page", str(tmp_path / "c")) == []


def test_network_failure_returns_empty(monkeypatch, tmp_path):
    import common.pdf_io as p

    def fail(url, timeout):
        raise TimeoutError("ncert is flaky")

    monkeypatch.setattr(p, "_fetch", fail)
    assert p.download_pdf_or_zip("https://ncert.nic.in/x.pdf", str(tmp_path / "c")) == []


def test_extract_pdf_text_real(tmp_path):
    import common.pdf_io as p

    pdf = tmp_path / "book.pdf"
    make_text_pdf(pdf)
    assert "Hello world" in p.extract_pdf_text(str(pdf))


def test_extract_pdf_text_missing_extractor(monkeypatch, tmp_path):
    import common.pdf_io as p

    pdf = tmp_path / "book.pdf"
    make_text_pdf(pdf)

    def no_pypdf(name):
        raise ImportError(f"no {name}")

    # Force every extractor import to fail -> PDFError.
    for mod in ("pypdf", "PyPDF2", "fitz"):
        monkeypatch.setitem(__import__("sys").modules, mod, None)
    with pytest.raises(p.PDFError, match="pypdf"):
        p.extract_pdf_text(str(pdf))
