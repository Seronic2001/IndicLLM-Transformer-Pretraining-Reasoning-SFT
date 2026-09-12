"""Real Hindi corpus sources for ``data/collect.py`` (Data Pipeline, Kaggle-ready).

``--source-module hindi.data.sources`` loads ``SOURCES`` below. Every source is
either a :class:`KaggleDatasetSource` (public Kaggle Dataset, pulled via the
authenticated Kaggle CLI) or a :class:`Source` whose ``fetch`` streams from
HuggingFace / the web into a local cache dir. All downloads are cached and the
collector's manifest makes the whole run resumable across 12-hour Kaggle
sessions.

Sources and where they were verified (Aug 2026):

  * ``sangraha_verified``    — HF ``ai4bharat/sangraha``, ``verified/hin/data-*.parquet``
    (100 shards). Open (not gated). Human-verified, high-quality subset
    (ACL 2024 Blueprint: web + PDF data with manual verification).
  * ``sangraha_unverified``  — same repo, ``unverified/hin/data-*.parquet`` (113).
  * ``sangraha_synthetic``   — same repo, ``synthetic/hin_Deva/wiki_*.parquet``
    (synthetic wiki-style data; lowest quality tier).
  * ``indiccorp_v2``         — HF ``ai4bharat/IndicCorpV2``, ``data/hi-*.txt``.
    Open. One sentence per line.
  * ``oscar_hindi``          — Kaggle ``abhishek/hindi-oscar-corpus`` (OSCAR dump
    mirror; CLI-verified to exist).
  * ``wikipedia_hindi``      — Kaggle ``disisbig/hindi-wikipedia-articles-172k``
    (CLI-verified). CSV of Hindi Wikipedia articles.
  * ``ncert_pdfs``           — manual fraction: NCERT Hindi-medium textbooks from
    ``data/ncert_pdfs.json`` (URL manifest). NCERT publishes each book as a ZIP
    of chapter PDFs at ``https://ncert.nic.in/textbook/pdf/<code>dd.zip`` (the
    pattern used by github.com/aayushdutt/ncert-downloader); the fetcher
    downloads the archive, extracts every chapter PDF, and yields one doc per
    chapter. Regenerate the manifest with ``update_ncert_manifest.py`` (parses
    the live ``textbook.php`` catalog, keeps Hindi-medium books: code[1]=='h').
    Scanned books need ``scripts/ocr.py``.

Optional control knobs (Kaggle smoke runs):

  * ``COLLECT_CACHE_DIR`` — raw download cache root (default ``<lang>/data/cache``).
  * ``COLLECT_MAX_SHARDS`` — cap on HF parquet/txt shards per source (default 6;
    the Collector's disk-space guard stops the run when the disk fills, so a
    higher cap is safe).
  * ``COLLECT_MAX_DOCS``  — cap docs per source; when set, only the first shard
    of each HF corpus is downloaded, giving a fast end-to-end smoke test.
  * ``HF_TOKEN``          — required only for gated repos (none of the Hindi
    sources above are gated; keep it set anyway for future mirrors).
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
from hindi.data.collect import KaggleDatasetSource, Source, SourceError  # noqa: E402

LANG = "hindi"
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


# ---------------------------------------------------------------- Web Crawler (manual)

def crawler_manual_source(
    name: str,
    crawler_source_key: str,
    priority: int = 7,
    manual: bool = True,
) -> Source:
    """Manual-fraction source: stream scraped documents from Hindi Web Crawler."""
    max_docs = _max_docs()

    def fetch() -> list[tuple[str, str]]:
        from hindi.data.crawler import HindiCrawler

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
        crawler = HindiCrawler(
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
    return [
        hf_parquet_source(
            "sangraha_verified", "ai4bharat/sangraha", "verified/hin/",
            cache, priority=1,
        ),
        hf_parquet_source(
            "sangraha_unverified", "ai4bharat/sangraha", "unverified/hin/",
            cache, priority=2,
        ),
        hf_txt_source(
            "indiccorp_v2", "ai4bharat/IndicCorpV2", "data/hi-",
            cache, priority=3,
        ),
        KaggleDatasetSource(
            name="oscar_hindi", dataset_ref="abhishek/hindi-oscar-corpus",
            cache_dir=cache, priority=4,
        ),
        KaggleDatasetSource(
            name="wikipedia_hindi",
            dataset_ref="disisbig/hindi-wikipedia-articles-172k",
            cache_dir=cache, priority=5, manual=True,
        ),
        pdf_manifest_source(
            "ncert_pdfs",
            str(Path(__file__).resolve().parent / "ncert_pdfs.json"),
            cache, priority=6, manual=True,
        ),
        crawler_manual_source(
            "crawler_cc100_hi", "cc100_hi", priority=7, manual=True,
        ),
        crawler_manual_source(
            "crawler_wikisource_hi", "wikisource_hi", priority=8, manual=True,
        ),
        crawler_manual_source(
            "crawler_jansatta", "jansatta", priority=9, manual=True,
        ),
        crawler_manual_source(
            "crawler_prabhatkhabar", "prabhatkhabar", priority=10, manual=True,
        ),
        crawler_manual_source(
            "crawler_news18_hi", "news18_hi", priority=11, manual=True,
        ),
        crawler_manual_source(
            "crawler_ndtv_hi", "ndtv_hi", priority=12, manual=True,
        ),
        crawler_manual_source(
            "crawler_webdunia_hi", "webdunia_hi", priority=13, manual=True,
        ),
        crawler_manual_source(
            "crawler_gadyakosh", "gadyakosh", priority=14, manual=True,
        ),
        crawler_manual_source(
            "crawler_kavitakosh", "kavitakosh", priority=15, manual=True,
        ),
        crawler_manual_source(
            "crawler_vikaspedia_hi", "vikaspedia_hi", priority=16, manual=True,
        ),
        crawler_manual_source(
            "crawler_amarujala", "amarujala", priority=17, manual=True,
        ),
        crawler_manual_source(
            "crawler_bbc_hi", "bbc_hi", priority=18, manual=True,
        ),
    ]


SOURCES = build_sources()


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m hindi.data.sources",
        description="Inspect the real Hindi corpus sources (no downloads).",
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
