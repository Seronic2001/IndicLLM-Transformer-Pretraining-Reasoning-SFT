"""Resumable corpus collection orchestrator (Agent-A).

Every long-running collection job on Kaggle must survive interruption, so this
module's core contract is:

  * **Idempotent + resumable**: a ``manifest.jsonl`` records every document that
    was fetched/cleaned/written. Re-running the job skips already-completed
    documents *and* already-completed sources (a source with a completion marker
    is never re-fetched). Kaggle sessions can interrupt freely and restart.
  * **Dead source**: log a warning, skip, continue to the next source in the
    priority list — one failure never kills the run.
  * **Rate-limited (403/429)**: exponential backoff, max 3 retries, then skip the
    source for this run and record it in ``failed_sources``.
  * **OCR fallback**: primary engine -> Tesseract -> skip the document and log.
  * **Shortfall honesty**: never fabricate/pad data to hit 500M tokens or 20%
    manual fraction — record exact counts in dataset_stats.json + the report.

Real fetchers (AI4Bharat Sangraha, IndicCorp, OSCAR, Wikipedia dumps, PDF/OCR
scrapers) are pluggable ``fetch`` callables; the local tests use fake sources.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.clean import clean_document, dedup_documents, write_clean_outputs  # noqa: E402

MANIFEST_NAME = "manifest.jsonl"
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2.0


class SourceError(RuntimeError):
    """Raised for a dead/blocked source; the collector skips and moves on."""


class RateLimitedError(SourceError):
    """HTTP 403/429 — retried with exponential backoff before skipping."""


def _max_docs() -> Optional[int]:
    import os
    raw = os.environ.get("COLLECT_MAX_DOCS")
    return int(raw) if raw else None


@dataclass
class Source:
    """A pluggable corpus source.

    ``fetch`` must be a callable taking no arguments and yielding documents as
    ``(doc_id, text)`` strings. ``manual=True`` marks human-curated material
    (NCERT/SCERT PDFs, OCR'd scans) for the manual-token fraction.
    """

    name: str
    fetch: Callable[[], "list[tuple[str, str]]"]
    priority: int = 10
    manual: bool = False

    @property
    def type(self) -> str:
        return "manual" if self.manual else "downloaded"


@dataclass
class KaggleDatasetSource(Source):
    """A corpus source backed by a public Kaggle Dataset, pulled via the CLI.

    ``fetch()`` downloads the dataset once (``datasets download --unzip``) into a
    per-source cache dir under ``cache_dir``, then yields ``(doc_id, text)`` from
    every ``*.txt`` (one doc per non-empty line), ``*.jsonl`` (one doc per line,
    ``text``/``sentence`` key) and ``*.csv`` (``text``/``article``/... column)
    file. Downloads are cached: re-running the collector reuses the cache, and
    the collector's manifest skips the source once completed, so the whole flow
    stays resumable across Kaggle sessions.

    Requires the authenticated Kaggle CLI (see common/kaggle_cli.py). A missing
    CLI or missing credentials raises :class:`SourceError` (dead source — the
    collector logs and continues, never fabricating data).
    """

    fetch: Optional[Callable] = None  # redeclared with default; replaced in __post_init__
    dataset_ref: str = ""  # "owner/dataset" as shown on kaggle.com
    cache_dir: str = ""  # where downloaded files land (per-source subdir)

    def __post_init__(self) -> None:
        if not self.dataset_ref:
            raise ValueError("KaggleDatasetSource requires dataset_ref='owner/dataset'")
        self.fetch = self._fetch  # method becomes the pluggable callable

    def _find_mounted_input(self) -> Optional[Path]:
        """Check if dataset is already mounted in /kaggle/input (zero disk space)."""
        import os
        input_root = Path("/kaggle/input")
        if not input_root.exists():
            return None
        slug = self.dataset_ref.split("/")[-1]
        owner = self.dataset_ref.split("/")[0] if "/" in self.dataset_ref else ""
        candidates = [
            input_root / "datasets" / owner / slug,
            input_root / "datasets" / slug,
            input_root / slug,
            input_root / self.dataset_ref,
        ]
        for c in candidates:
            if c.is_dir():
                return c
        for root, dirs, _files in os.walk(input_root):
            if slug in dirs:
                return Path(root) / slug
        return None

    def _fetch(self) -> list[tuple[str, str]]:
        from common.hf_io import iter_csv_text, iter_jsonl_text, iter_text_lines

        max_docs = _max_docs() or int(os.environ.get("COLLECT_MAX_SOURCE_DOCS", "300000"))
        mounted = self._find_mounted_input()
        if mounted is not None:
            docs: list[tuple[str, str]] = []
            for doc_id, text in iter_text_lines(str(p) for p in mounted.rglob("*.txt")):
                docs.append((f"{self.dataset_ref}/{doc_id}", text))
                if max_docs and len(docs) >= max_docs:
                    return docs
            for doc_id, text in iter_jsonl_text(str(p) for p in mounted.rglob("*.jsonl")):
                docs.append((f"{self.dataset_ref}/{doc_id}", text))
                if max_docs and len(docs) >= max_docs:
                    return docs
            for doc_id, text in iter_csv_text(str(p) for p in mounted.rglob("*.csv")):
                docs.append((f"{self.dataset_ref}/{doc_id}", text))
                if max_docs and len(docs) >= max_docs:
                    return docs
            if docs:
                return docs

        from common.kaggle_cli import (  # lazy import: offline runs stay fast
            KaggleCLIError,
            cli_available,
            datasets_download,
            has_credentials,
        )

        if not cli_available() or not has_credentials():
            raise SourceError(
                "kaggle CLI unavailable or unauthenticated (run `kaggle auth login` "
                "or place kaggle.json in ~/.kaggle/) — cannot fetch "
                f"dataset {self.dataset_ref}"
            )

        cache = Path(self.cache_dir) / self.dataset_ref.replace("/", "__")
        try:
            datasets_download(self.dataset_ref, str(cache), unzip=True, quiet=True)
        except KaggleCLIError as exc:
            raise SourceError(f"kaggle dataset download failed: {exc}") from exc

        docs: list[tuple[str, str]] = []
        for doc_id, text in iter_text_lines(str(p) for p in cache.rglob("*.txt")):
            docs.append((f"{self.dataset_ref}/{doc_id}", text))
            if max_docs and len(docs) >= max_docs:
                return docs
        for doc_id, text in iter_jsonl_text(str(p) for p in cache.rglob("*.jsonl")):
            docs.append((f"{self.dataset_ref}/{doc_id}", text))
            if max_docs and len(docs) >= max_docs:
                return docs
        for doc_id, text in iter_csv_text(str(p) for p in cache.rglob("*.csv")):
            docs.append((f"{self.dataset_ref}/{doc_id}", text))
            if max_docs and len(docs) >= max_docs:
                return docs
        return docs


def _clean_worker_task(args: tuple[str, str, str]) -> tuple[str, Optional[str], dict]:
    doc_id, text, lang = args
    cleaned, cstats = clean_document(text, lang, doc_id=doc_id)
    return doc_id, cleaned, cstats


def get_free_disk_gb(path: Path) -> float:
    """Return free disk space in GB for the path's filesystem."""
    import shutil
    try:
        usage = shutil.disk_usage(str(path if path.exists() else path.parent))
        return usage.free / (1024 ** 3)
    except Exception:
        return 999.0


class Collector:
    def __init__(self, data_dir: str, lang: str, seed: int = 1337):
        self.data_dir = Path(data_dir)
        self.lang = lang
        self.manifest_path = self.data_dir / MANIFEST_NAME
        self.done_sources: set[str] = set()
        self.seen_hashes: set[str] = set()
        self.failed_sources: list[tuple[str, str]] = []
        self.new_docs = 0
        self.manifest: list[dict] = self._load_manifest()

    # ------------------------------------------------------------------ manifest

    def _load_manifest(self) -> list[dict]:
        if not self.manifest_path.exists():
            return []
        rows = []
        with open(self.manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        for row in rows:
            if row.get("kind") == "source_done":
                self.done_sources.add(row["source"])
            elif row.get("kind") == "doc" and row.get("hash"):
                self.seen_hashes.add(row["hash"])
        return rows

    def _append_manifest(self, row: dict) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.manifest.append(row)
        if row.get("kind") == "doc" and row.get("hash"):
            self.seen_hashes.add(row["hash"])

    def _append_manifest_records(self, records: list[dict]) -> None:
        if not records:
            return
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest.extend(records)
        lines = [json.dumps(r, ensure_ascii=False) + "\n" for r in records]
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.writelines(lines)
        for r in records:
            if r.get("kind") == "doc" and r.get("hash"):
                self.seen_hashes.add(r["hash"])

    def _seen_hashes(self) -> set[str]:
        return self.seen_hashes

    # ------------------------------------------------------------------ fetch

    def fetch_source(self, source: Source) -> list[tuple[str, str]]:
        """Fetch + clean + dedup one source, honoring backoff and resume rules.

        Returns the cleaned unique (doc_id, text) pairs for this source.
        """
        import gc
        import shutil

        if source.name in self.done_sources:
            print(f"[skip] source {source.name}: already completed")
            return []

        # Disk space guard: check available space before starting a source
        free_gb = get_free_disk_gb(self.data_dir)
        if free_gb < 4.0:
            print(f"[!] Warning: Low disk space detected ({free_gb:.2f} GB free). Purging caches...", flush=True)
            shutil.rmtree(self.data_dir / "cache", ignore_errors=True)
            shutil.rmtree(Path.home() / ".cache" / "huggingface", ignore_errors=True)
            free_gb = get_free_disk_gb(self.data_dir)
            if free_gb < 3.0:
                self.failed_sources.append((source.name, f"disk space guard: {free_gb:.2f} GB free remaining (safely preserved for downstream splits/tokenization)"))
                print(f"[skip] source {source.name}: low disk space ({free_gb:.2f} GB free), skipping to safely preserve space for splits and token binaries.", flush=True)
                return []

        # Check if clean jsonl output already exists with valid data (e.g. restored from previous run)
        clean_jsonl = self.data_dir / "clean" / f"{source.name}.jsonl"
        clean_txt = self.data_dir / "clean" / f"{source.name}.txt"
        if clean_jsonl.exists() and clean_jsonl.stat().st_size > 0:
            if not clean_txt.exists() or clean_txt.stat().st_size == 0:
                with open(clean_jsonl, "r", encoding="utf-8") as f_in, open(clean_txt, "w", encoding="utf-8") as f_out:
                    for line in f_in:
                        try:
                            doc = json.loads(line)
                            f_out.write(doc.get("text", "").replace("\n", " ") + "\n")
                        except Exception:
                            pass
            print(f"[cached] source {source.name}: clean data already present on disk, reusing!", flush=True)
            self.done_sources.add(source.name)
            return []

        print(f"[fetch] source {source.name} (priority {source.priority}, {source.type})")
        attempt = 0
        while True:
            try:
                raw_docs = source.fetch()
                break
            except RateLimitedError:
                attempt += 1
                if attempt >= MAX_RETRIES:
                    self.failed_sources.append((source.name, f"rate-limited after {MAX_RETRIES} retries"))
                    print(f"[skip] source {source.name}: rate-limited, skipping for this run")
                    return []
                delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.random()
                print(f"[backoff] {source.name}: retry {attempt}/{MAX_RETRIES} in {delay:.1f}s")
                time.sleep(delay)
            except SourceError as exc:
                self.failed_sources.append((source.name, str(exc)))
                print(f"[skip] source {source.name}: dead/blocked ({exc}); continuing")
                return []

        print(f"[fetch] source {source.name} (priority {source.priority}, {source.type})...", flush=True)
        seen = self._seen_hashes()
        fresh: list[tuple[str, str, str]] = []
        stats = {"script_filter_removed_chars": 0}
        total_raw = len(raw_docs)
        print(f"[{source.name}] Fetched {total_raw} raw docs. Cleaning and deduplicating...", flush=True)

        from data.clean import doc_hash
        from concurrent.futures import ProcessPoolExecutor

        t0 = time.time()
        num_workers = min(os.cpu_count() or 1, 4)
        processed_count = 0
        log_interval = max(5000, total_raw // 20)

        if num_workers > 1 and len(raw_docs) > 1000:
            tasks = [(doc_id, text, self.lang) for doc_id, text in raw_docs]
            chunk_size = max(100, len(tasks) // (num_workers * 16))
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for doc_id, cleaned, cstats in executor.map(_clean_worker_task, tasks, chunksize=chunk_size):
                    processed_count += 1
                    stats["script_filter_removed_chars"] += cstats["script_filter_removed_chars"]
                    if cleaned is not None:
                        h = doc_hash(cleaned)
                        if h not in seen:
                            fresh.append((doc_id, source.name, cleaned))
                    if processed_count % log_interval == 0 or processed_count == total_raw:
                        elapsed = time.time() - t0
                        speed = processed_count / max(elapsed, 0.001)
                        pct = (processed_count / total_raw) * 100
                        print(
                            f"[{source.name}] Cleaned {processed_count:,}/{total_raw:,} ({pct:.1f}%) "
                            f"| {speed:.0f} docs/sec | Kept: {len(fresh):,} docs",
                            flush=True,
                        )
            del tasks
        else:
            for doc_id, text in raw_docs:
                processed_count += 1
                cleaned, cstats = clean_document(text, self.lang, doc_id=doc_id)
                stats["script_filter_removed_chars"] += cstats["script_filter_removed_chars"]
                if cleaned is not None:
                    h = doc_hash(cleaned)
                    if h not in seen:
                        fresh.append((doc_id, source.name, cleaned))
                if processed_count % log_interval == 0 or processed_count == total_raw:
                    elapsed = time.time() - t0
                    speed = processed_count / max(elapsed, 0.001)
                    pct = (processed_count / total_raw) * 100
                    print(
                        f"[{source.name}] Cleaned {processed_count:,}/{total_raw:,} ({pct:.1f}%) "
                        f"| {speed:.0f} docs/sec | Kept: {len(fresh):,} docs",
                        flush=True,
                    )

        # Free raw docs memory before deduplication and output writing
        del raw_docs
        gc.collect()

        unique, exact_removed, _near_pairs = dedup_documents(fresh, minhash_threshold=0.6)
        del fresh
        gc.collect()

        for r in unique:
            seen.add(r["hash"])

        # Write clean outputs (idempotent: per-source file, appended rows only once).
        records = [
            {"doc_id": r["doc_id"], "source": r["source"], "text": r["text"],
             "source_type": source.type, "hash": r["hash"]}
            for r in unique
        ]
        if records:
            write_clean_outputs(records, str(self.data_dir), source.name)

        manifest_entries = [
            {"kind": "doc", "doc_id": r["doc_id"], "source": r["source"], "hash": r["hash"]}
            for r in records
        ]
        self._append_manifest_records(manifest_entries)
        self.new_docs += len(records)
        self.done_sources.add(source.name)
        print(f"[{source.name}] Saved {len(records)} clean docs (dropped {exact_removed} exact duplicates).", flush=True)
        # Completion marker makes re-runs skip this source entirely.
        self._append_manifest({"kind": "source_done", "source": source.name})
        # Clean temporary caches after saving clean outputs
        shutil.rmtree(self.data_dir / "cache", ignore_errors=True)
        shutil.rmtree(Path.home() / ".cache" / "huggingface", ignore_errors=True)
        free_gb = get_free_disk_gb(self.data_dir)
        print(f"[ok] source {source.name}: {len(records)} new docs "
              f"({exact_removed} exact-dup removed, {stats['script_filter_removed_chars']} chars filtered) "
              f"| {free_gb:.2f} GB free disk remaining", flush=True)
        return [(r["doc_id"], r["text"]) for r in records]

    # ------------------------------------------------------------------ orchestration

    def run(self, sources: list[Source]) -> dict:
        """Run all sources in priority order (lowest number first)."""
        for source in sorted(sources, key=lambda s: s.priority):
            try:
                self.fetch_source(source)
            except Exception as exc:  # never let one source kill the run
                self.failed_sources.append((source.name, f"unexpected: {exc}"))
                print(f"[skip] source {source.name}: unexpected error {exc}; continuing")
        self._write_sources_jsonl(sources)
        return {
            "new_docs": self.new_docs,
            "failed_sources": self.failed_sources,
            "done_sources": sorted(self.done_sources),
        }

    def _write_sources_jsonl(self, sources: list[Source]) -> None:
        """Companion sources.jsonl (Agent-A): one line per source, tagged type."""
        path = self.data_dir / "sources.jsonl"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for source in sorted(sources, key=lambda s: s.priority):
                f.write(json.dumps({
                    "name": source.name,
                    "type": source.type,
                    "priority": source.priority,
                    "completed": source.name in self.done_sources,
                    "failed": source.name in dict(self.failed_sources),
                }, ensure_ascii=False) + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the resumable corpus collection (Agent-A). Fetchers are "
                    "defined in a notebook/module that imports this; the CLI "
                    "--source-module option loads them."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--lang", required=True, choices=["hindi", "assamese"])
    parser.add_argument("--source-module", default=None,
                        help="python module path (dotted) exposing SOURCES: list[Source]")
    parser.add_argument("--sources", nargs="*", default=None,
                        help="optional subset of source names to collect (e.g. --sources wikipedia_dump_manual asm_corpus)")
    args = parser.parse_args(argv)

    if args.source_module:
        mod = __import__(args.source_module, fromlist=["SOURCES"])
        sources = list(mod.SOURCES)
    else:
        sources = example_sources()

    filter_names = args.sources
    if filter_names is None and "COLLECT_ONLY_SOURCES" in os.environ:
        raw_env = os.environ["COLLECT_ONLY_SOURCES"].strip()
        if raw_env:
            filter_names = [s.strip() for s in raw_env.replace(",", " ").split() if s.strip()]

    if filter_names:
        filter_set = set(filter_names)
        sources = [s for s in sources if s.name in filter_set]
        print(f"[*] Filtered collection to {len(sources)} source(s): {[s.name for s in sources]}", flush=True)

    collector = Collector(args.data_dir, args.lang)
    result = collector.run(sources)
    print(json.dumps(result, indent=2))
    return 0


def example_sources() -> list[Source]:
    """Placeholder sources documenting the real corpus fetchers (Kaggle-time).

    Each real fetch should be a resumable downloader (e.g. huggingface_hub /
    requests streaming to a cache dir) yielding (doc_id, text). Implementations
    live per-language (Agent-A notebooks) and are out of scope for local runs.
    """

    def _not_implemented() -> list[tuple[str, str]]:
        raise SourceError("fetcher not configured for local runs (see notebook setup)")

    return [
        Source("wikipedia_dump", _not_implemented, priority=1, manual=False),
        Source("oscar", _not_implemented, priority=2, manual=False),
        Source("indiccorp_v2", _not_implemented, priority=3, manual=False),
        Source("sangraha", _not_implemented, priority=4, manual=False),
        Source("ncert_scert_pdfs", _not_implemented, priority=5, manual=True),
        Source("archive_dli_ocr", _not_implemented, priority=6, manual=True),
        Source("news_portals", _not_implemented, priority=7, manual=False),
    ]


if __name__ == "__main__":
    raise SystemExit(main())
