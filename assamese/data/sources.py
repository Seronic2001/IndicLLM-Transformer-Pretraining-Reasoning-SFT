"""Real Assamese corpus sources for ``data/collect.py`` (Agent-A, Kaggle-ready).

Mirror of ``hindi/data/sources.py`` for Assamese — same infrastructure, different
corpora and language paths (no imports from ``hindi``).

Sources and where they were verified (Aug 2026):

  * ``sangraha_verified``    — HF ``ai4bharat/sangraha``, ``verified/asm/data-*.parquet``
    (3 shards). Open (not gated). Human-verified subset (ACL 2024 Blueprint).
  * ``sangraha_unverified``  — same repo, ``unverified/asm/data-0.parquet`` (1).
  * ``sangraha_synthetic``   — same repo, ``synthetic/asm_Beng/wiki_asm_Beng_*.parquet``
    (63 shards; synthetic wiki-style data, lowest quality tier).
  * ``indiccorp_v2``         — HF ``ai4bharat/IndicCorpV2``, ``data/as.txt``. Open.
    One sentence per line.
  * ``oscar_assamese``       — HF ``oscar-corpus/OSCAR-2301``, ``as/`` parquet.
    **Gated**: accept the dataset terms on huggingface.co and set ``HF_TOKEN``;
    otherwise this source fails cleanly (honest shortfall, logged and skipped).
  * ``wikipedia_assamese``   — HF ``wikimedia/wikipedia``, ``20231101.as/``
    parquet. Open. Cleanest Wikipedia dump for ``as`` (no Kaggle mirror found).
  * ``news_portals``         — Kaggle ``krishnabhdas/assamese-news-article-dataset``
    (CLI-verified to exist).
  * ``ncert_assamese_pdfs``  — manual fraction: NCERT Assamese-medium textbooks
    from ``data/ncert_pdfs.json`` (29 books, regenerated with
    ``python -m scripts.update_ncert_manifest --medium a --out assamese/data/ncert_pdfs.json``;
    NCERT book codes use 'a' = Assamese medium). Official chapter-ZIP format,
    one doc per chapter PDF.
  * ``seba_pdfs``            — manual fraction: 43 verified Assamese-medium
    state-board books (classes 1-10) from devlibrary.in's index, in
    ``data/seba_pdfs.json`` (regenerated with
    ``python -m scripts.update_seba_manifest``; links are probe-verified at
    generation time, English-medium excluded).
  * ``scert_pdfs``           — manual fraction: SCERT Assam official textbooks
    from ``data/scert_pdfs.json`` (regenerated with
    ``python -m scripts.update_scert_manifest``).

All PDF manifest sources share one contract: entries may be direct PDFs, ZIPs
of chapter PDFs, HTML pages embedding a PDF link, or Google Drive exports; the
fetcher downloads, extracts, and yields one doc per chapter/page.

Optional control knobs (identical to the Hindi module): ``COLLECT_CACHE_DIR``,
``COLLECT_MAX_SHARDS`` (HF shard cap per source, default 6 — the Collector's
disk-space guard stops the run safely when the disk fills), ``COLLECT_MAX_DOCS``
(cap + single-shard smoke mode), ``HF_TOKEN``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.hf_io import (  # noqa: E402
    HFError,
    hf_download,
    iter_parquet_text,
    iter_text_lines,
    list_repo_files,
)
from common.pdf_io import download_pdf_or_zip, extract_pdf_text  # noqa: E402
# Language-qualified import so tests, the collector CLI, and this module share
# ONE instance of the Source classes (a bare ``from data.collect import ...``
# would create a second module for the same file and break isinstance checks).
from assamese.data.collect import KaggleDatasetSource, Source, SourceError  # noqa: E402

LANG = "assamese"
_CACHE = Path(__file__).resolve().parent / "cache"

SCRIPT_FRACTION_MIN = 0.2  # below this the PDF likely needs OCR, not plain extraction


def _max_docs() -> Optional[int]:
    raw = os.environ.get("COLLECT_MAX_DOCS")
    return int(raw) if raw else None


# ---------------------------------------------------------------- HF corpus sources

def _clean_hf_cache(cache_dir: Path) -> None:
    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)
    hf_home = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_home.exists():
        shutil.rmtree(hf_home, ignore_errors=True)


def hf_parquet_source(
    name: str,
    repo_id: str,
    prefix: str,
    cache_root: str,
    priority: int,
    manual: bool = False,
    text_keys: tuple = ("text", "sentence", "content"),
) -> Source:
    """A Source streaming ``*.parquet`` files from a HF repo (resumable, cached, disk-bounded)."""
    max_docs = _max_docs()

    def fetch() -> list[tuple[str, str]]:
        cache = Path(cache_root) / "hf" / repo_id.replace("/", "__") / prefix.strip("/").replace("/", "_")
        try:
            files = [f for f in list_repo_files(repo_id, prefix=prefix) if f.endswith(".parquet")]
            if not files:
                # Try finding parquet files in repo if prefix structure differed
                all_files = list_repo_files(repo_id)
                files = [f for f in all_files if f.endswith(".parquet") and prefix.strip("/") in f]
            if not files:
                raise SourceError(f"{name}: no .parquet files under {repo_id}:{prefix}")
            start_shard = int(os.environ.get("COLLECT_START_SHARD", "0"))
            max_shards = int(os.environ.get("COLLECT_MAX_SHARDS", "6"))
            if max_docs:
                files = files[start_shard : start_shard + 1]  # one shard is plenty for a smoke test
            else:
                files = files[start_shard : max_shards]
            cache.mkdir(parents=True, exist_ok=True)
            docs: list[tuple[str, str]] = []
            for f in files:
                shard_path = hf_download(repo_id, f, str(cache))
                for doc_id, text in iter_parquet_text([str(shard_path)], text_keys):
                    docs.append((doc_id, text))
                    if max_docs and len(docs) >= max_docs:
                        break
                try:
                    p = Path(shard_path)
                    if p.is_file():
                        p.unlink(missing_ok=True)
                except OSError:
                    pass
                if max_docs and len(docs) >= max_docs:
                    break
        except Exception as exc:
            _clean_hf_cache(cache)
            raise SourceError(f"{name}: {exc}") from exc
        finally:
            _clean_hf_cache(cache)
        if not docs:
            raise SourceError(f"{name}: no text rows found in {repo_id}:{prefix}")
        return docs

    return Source(name, fetch, priority=priority, manual=manual)


def hf_txt_source(
    name: str,
    repo_id: str,
    prefix: str,
    cache_root: str,
    priority: int,
    manual: bool = False,
    max_file_size_bytes: int = 2_000_000_000,
) -> Source:
    """A Source streaming ``*.txt`` files from a HF repo (one doc per line, disk-bounded)."""
    max_docs = _max_docs()

    def fetch() -> list[tuple[str, str]]:
        cache = Path(cache_root) / "hf" / repo_id.replace("/", "__") / prefix.strip("/").replace("/", "_")
        try:
            files = [f for f in list_repo_files(repo_id, prefix=prefix) if f.endswith(".txt")]
            if not files:
                raise SourceError(f"{name}: no .txt files under {repo_id}:{prefix}")
            start_shard = int(os.environ.get("COLLECT_START_SHARD", "0"))
            max_shards = int(os.environ.get("COLLECT_MAX_SHARDS", "6"))
            if max_docs:
                files = files[start_shard : start_shard + 1]
            else:
                files = files[start_shard : max_shards]
            cache.mkdir(parents=True, exist_ok=True)
            docs: list[tuple[str, str]] = []
            for f in files:
                # File size check to prevent downloading 20GB+ monolithic files
                try:
                    from huggingface_hub import get_hf_file_metadata, hf_hub_url
                    url = hf_hub_url(repo_id, f, repo_type="dataset")
                    meta = get_hf_file_metadata(url)
                    if meta.size and meta.size > max_file_size_bytes:
                        print(f"[{name}] Skipping {f} ({meta.size / 1e9:.2f} GB > {max_file_size_bytes / 1e9:.1f} GB disk cap)")
                        continue
                except Exception:
                    pass

                shard_path = hf_download(repo_id, f, str(cache))
                for doc_id, text in iter_text_lines([str(shard_path)]):
                    docs.append((doc_id, text))
                    if max_docs and len(docs) >= max_docs:
                        break
                try:
                    p = Path(shard_path)
                    if p.is_file():
                        p.unlink(missing_ok=True)
                except OSError:
                    pass
                if max_docs and len(docs) >= max_docs:
                    break
        except Exception as exc:
            _clean_hf_cache(cache)
            raise SourceError(f"{name}: {exc}") from exc
        finally:
            _clean_hf_cache(cache)
        if not docs:
            raise SourceError(f"{name}: no text rows in {repo_id}:{prefix}")
        return docs

    return Source(name, fetch, priority=priority, manual=manual)


# ---------------------------------------------------------------- NCERT PDFs (manual)

def pdf_manifest_source(
    name: str,
    manifest_path: str,
    cache_root: str,
    priority: int,
    manual: bool = True,
) -> Source:
    """Manual-fraction source: download each URL in a JSON manifest and extract text.

    Each entry's ``url`` may be a direct PDF, a ZIP of chapter PDFs (the NCERT
    official format, e.g. ``https://ncert.nic.in/textbook/pdf/<code>dd.zip``),
    or an HTML page embedding a PDF link. One doc is yielded per chapter PDF
    with doc_id ``<title>/<chapter>.pdf`` for traceability.
    """
    max_docs = _max_docs()

    def fetch() -> list[tuple[str, str]]:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not manifest:
            raise SourceError(f"{name}: empty manifest {manifest_path}")
        cache = Path(cache_root) / "pdfs"
        cache.mkdir(parents=True, exist_ok=True)
        docs: list[tuple[str, str]] = []
        failures: list[str] = []
        import shutil
        try:
            for entry in manifest:
                title = entry.get("title", entry["url"])
                pdfs = download_pdf_or_zip(entry["url"], str(cache))
                if not pdfs:
                    failures.append(f"{entry['url']} (download failed)")
                    continue
                for pdf in pdfs:
                    text = extract_pdf_text(str(pdf)).strip()
                    if not text or not _script_fraction_ok(text):
                        # Try OCR fallback on scanned PDF
                        try:
                            try:
                                from common.ocr import ocr_pdf
                            except ImportError:
                                from scripts.ocr import ocr_pdf

                            if ocr_pdf is not None:
                                ocr_text = ocr_pdf(str(pdf), lang=LANG).strip()
                                if ocr_text and _script_fraction_ok(ocr_text):
                                    text = ocr_text
                        except Exception:
                            pass

                    if not text:
                        failures.append(f"{pdf.name} (empty extract — needs OCR?)")
                        continue
                    if not _script_fraction_ok(text):
                        failures.append(f"{pdf.name} (low script fraction — scanned, needs OCR)")
                        continue
                    docs.append((f"{title}/{pdf.name}", text))
                    if max_docs and len(docs) >= max_docs:
                        return docs
                for pdf in pdfs:
                    try:
                        pdf.unlink(missing_ok=True)
                    except OSError:
                        pass
        finally:
            shutil.rmtree(cache, ignore_errors=True)
        if not docs:
            raise SourceError(
                f"{name}: all {len(manifest)} manifest URLs failed: {failures[:3]}"
            )
        if failures:
            print(f"[{name}] {len(failures)}/{len(manifest)} URLs skipped: {failures[:3]}")
        return docs

    return Source(name, fetch, priority=priority, manual=manual)


def _script_fraction_ok(text: str) -> bool:
    from common.script_utils import script_fraction

    in_script, total = script_fraction(text, LANG)
    return total > 0 and in_script / total >= SCRIPT_FRACTION_MIN


# ---------------------------------------------------------------- Wikipedia Dump (manual)

def clean_wiki_markup(text: str) -> str:
    """Strip MediaWiki markup to extract plain readable text from raw XML dumps."""
    import re

    if not text:
        return ""
    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Remove ref tags and contents
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<ref[^/>]*/>", "", text, flags=re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove file/image links: [[File:...]] or [[Image:...]] or [[চিত্র:...]] or [[ফাইল:...]]
    text = re.sub(r"\[\[(?:File|Image|চিত্র|ফাইল):[^\]]+\]\]", "", text, flags=re.IGNORECASE)
    # Remove category links [[Category:...]] / [[শ্ৰেণী:...]]
    text = re.sub(r"\[\[(?:Category|শ্ৰেণী):[^\]]+\]\]", "", text, flags=re.IGNORECASE)
    # Remove templates {{...}} (handles nested templates up to 4 levels)
    for _ in range(4):
        text = re.sub(r"\{\{[^{}]*\}\}", "", text, flags=re.DOTALL)
    # Remove tables {| ... |}
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)
    # Convert internal links [[Target|Anchor]] -> Anchor, [[Target]] -> Target
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    # Remove external links [http... text] -> text or [http...] -> ''
    text = re.sub(r"\[https?://[^\s\]]+(?:\s+([^\]]+))?\]", r"\1", text)
    # Remove headings == Heading ==
    text = re.sub(r"={2,6}(.*?)(?:={2,6})", r"\1", text)
    # Remove bold/italic markup
    text = re.sub(r"'{2,5}", "", text)
    # Clean up empty brackets / parens left from template removal
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\[\s*\]", "", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def wikipedia_dump_source(
    name: str = "wikipedia_dump_manual",
    dump_url: str = "https://dumps.wikimedia.org/aswiki/20260801/aswiki-20260801-pages-articles.xml.bz2",
    cache_root: Optional[str] = None,
    priority: int = 11,
    manual: bool = True,
) -> Source:
    """Manual-fraction source: stream raw XML dump of Assamese Wikipedia and clean text.

    Uses memory-efficient streaming XML iterparse. MediaWiki wikitext is manually
    stripped of templates, tables, citations, images, and formatting into clean Assamese text.
    """
    max_docs = _max_docs()

    def fetch() -> list[tuple[str, str]]:
        import bz2
        import shutil
        import urllib.request
        import xml.etree.ElementTree as ET

        cache = Path(cache_root or _CACHE) / "wiki_dump"
        cache.mkdir(parents=True, exist_ok=True)
        dump_filename = Path(dump_url).name
        dump_file = cache / dump_filename

        try:
            is_http = dump_url.startswith("http://") or dump_url.startswith("https://")
            if not dump_file.exists() or dump_file.stat().st_size == 0:
                if is_http:
                    print(f"[{name}] Downloading Assamese Wikipedia XML dump from {dump_url}...", flush=True)
                    req = urllib.request.Request(
                        dump_url,
                        headers={"User-Agent": "LMA-Project/1.0 (academic; student)"},
                    )
                    with urllib.request.urlopen(req, timeout=180) as resp, open(dump_file, "wb") as f_out:
                        shutil.copyfileobj(resp, f_out, length=1024 * 1024)
                elif Path(dump_url).exists():
                    shutil.copy2(dump_url, dump_file)
                else:
                    raise SourceError(f"{name}: dump URL/path not found: {dump_url}")

            docs: list[tuple[str, str]] = []
            print(f"[{name}] Streaming and extracting articles from {dump_filename}...", flush=True)
            with bz2.open(dump_file, "rt", encoding="utf-8", errors="ignore") as f:
                context = ET.iterparse(f, events=("end",))
                for _event, elem in context:
                    tag = elem.tag.split("}")[-1]
                    if tag == "page":
                        ns_elem = elem.find("{*}ns")
                        if ns_elem is not None and ns_elem.text != "0":
                            elem.clear()
                            continue
                        if elem.find("{*}redirect") is not None:
                            elem.clear()
                            continue
                        title_elem = elem.find("{*}title")
                        title = title_elem.text if title_elem is not None else ""
                        rev = elem.find("{*}revision")
                        if rev is not None:
                            text_elem = rev.find("{*}text")
                            if text_elem is not None and text_elem.text:
                                raw_text = text_elem.text
                                cleaned = clean_wiki_markup(raw_text)
                                if cleaned and _script_fraction_ok(cleaned):
                                    docs.append((f"{title}", cleaned))
                                    if max_docs and len(docs) >= max_docs:
                                        elem.clear()
                                        break
                        elem.clear()
        except Exception as exc:
            shutil.rmtree(cache, ignore_errors=True)
            raise SourceError(f"{name}: {exc}") from exc
        finally:
            shutil.rmtree(cache, ignore_errors=True)

        if not docs:
            raise SourceError(f"{name}: no valid articles extracted from {dump_url}")
        return docs

    return Source(name, fetch, priority=priority, manual=manual)


# ---------------------------------------------------------------- Web Crawler (manual)

def crawler_manual_source(
    name: str,
    crawler_source_key: str,
    priority: int = 12,
    manual: bool = True,
) -> Source:
    """Manual-fraction source: stream scraped documents from Assamese Web Crawler."""
    max_docs = _max_docs()

    def fetch() -> list[tuple[str, str]]:
        from assamese.data.crawler import AssameseCrawler

        clean_dir = Path(__file__).resolve().parent / "clean"
        jsonl_path = clean_dir / f"crawler_{crawler_source_key}.jsonl"

        # If already crawled into clean_dir, load directly
        docs: list[tuple[str, str]] = []
        if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
            with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            docs.append((d.get("doc_id", f"{crawler_source_key}/{len(docs)}"), d.get("text", "")))
                            if max_docs and len(docs) >= max_docs:
                                break
                        except Exception:
                            pass
            if docs:
                return docs

        # Otherwise execute the crawler for this specific source
        crawler = AssameseCrawler(
            out_dir=clean_dir,
            target_tokens=100_000_000,
            max_docs=max_docs,
            workers=6,
        )
        crawler.run(sources=[crawler_source_key])

        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            docs.append((d.get("doc_id", f"{crawler_source_key}/{len(docs)}"), d.get("text", "")))
                            if max_docs and len(docs) >= max_docs:
                                break
                        except Exception:
                            pass

        if not docs:
            raise SourceError(f"{name}: no documents retrieved from crawler source {crawler_source_key}")
        return docs

    return Source(name, fetch, priority=priority, manual=manual)


# ---------------------------------------------------------------- module wiring

def build_sources(cache_root: Optional[str] = None) -> list[Source]:
    cache = cache_root or os.environ.get("COLLECT_CACHE_DIR") or str(_CACHE)
    dump_url = os.environ.get(
        "WIKI_DUMP_URL",
        "https://dumps.wikimedia.org/aswiki/20260801/aswiki-20260801-pages-articles.xml.bz2",
    )
    return [
        hf_parquet_source(
            "sangraha_verified", "ai4bharat/sangraha", "verified/asm/",
            cache, priority=1,
        ),
        hf_parquet_source(
            "sangraha_unverified", "ai4bharat/sangraha", "unverified/asm/",
            cache, priority=2,
        ),
        hf_txt_source(
            "indiccorp_v2", "ai4bharat/IndicCorpV2", "data/as",
            cache, priority=3,
        ),
        hf_parquet_source(
            "oscar_assamese", "oscar-corpus/OSCAR-2301", "as/",
            cache, priority=4,
        ),
        hf_parquet_source(
            "wikipedia_assamese", "wikimedia/wikipedia", "20231101.as/",
            cache, priority=5, manual=True,
        ),
        hf_parquet_source(
            "asm_corpus", "ananddey/asm-corpus", "data/train/",
            cache, priority=6,
        ),
        KaggleDatasetSource(
            name="news_portals",
            dataset_ref="krishnabhdas/assamese-news-article-dataset",
            cache_dir=cache, priority=7,
        ),
        pdf_manifest_source(
            "ncert_assamese_pdfs",
            str(Path(__file__).resolve().parent / "ncert_pdfs.json"),
            cache, priority=8, manual=True,
        ),
        pdf_manifest_source(
            "seba_pdfs",
            str(Path(__file__).resolve().parent / "seba_pdfs.json"),
            cache, priority=9, manual=True,
        ),
        pdf_manifest_source(
            "scert_pdfs",
            str(Path(__file__).resolve().parent / "scert_pdfs.json"),
            cache, priority=10, manual=True,
        ),
        wikipedia_dump_source(
            "wikipedia_dump_manual",
            dump_url,
            cache, priority=11, manual=True,
        ),
        crawler_manual_source(
            "crawler_cc100_as", "cc100_as", priority=12, manual=True,
        ),
        crawler_manual_source(
            "crawler_niyomiya_barta", "niyomiya_barta", priority=13, manual=True,
        ),
        crawler_manual_source(
            "crawler_nenow_asm", "nenow_asm", priority=14, manual=True,
        ),
        crawler_manual_source(
            "crawler_asomiya_pratidin", "asomiya_pratidin", priority=15, manual=True,
        ),
        crawler_manual_source(
            "crawler_news18_asm", "news18_asm", priority=16, manual=True,
        ),
        crawler_manual_source(
            "crawler_wikisource_as", "wikisource_as", priority=17, manual=True,
        ),
        crawler_manual_source(
            "crawler_vikaspedia_as", "vikaspedia_as", priority=18, manual=True,
        ),
        crawler_manual_source(
            "crawler_agradoot", "agradoot", priority=19, manual=True,
        ),
        crawler_manual_source(
            "crawler_xahitya", "xahitya", priority=20, manual=True,
        ),
    ]


SOURCES = build_sources()


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m assamese.data.sources",
        description="Inspect the real Assamese corpus sources (no downloads).",
    )
    parser.add_argument("--list", action="store_true", help="print the source table")
    args = parser.parse_args(argv)
    if args.list:
        print(f"{'name':<22}{'type':<12}{'pri':<5}notes")
        for s in SOURCES:
            print(f"{s.name:<22}{s.type:<12}{s.priority:<5}{s.__class__.__name__}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
