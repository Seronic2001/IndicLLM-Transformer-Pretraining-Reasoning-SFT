"""HuggingFace corpus plumbing shared by both languages (Agent-A sources).

Language-agnostic helpers behind the real corpus fetchers in
``hindi/data/sources.py`` / ``assamese/data/sources.py``:

  * ``list_repo_files``  — discover corpus files under a repo + prefix.
  * ``hf_download``      — resumable download to a local cache dir (uses
    ``huggingface_hub.hf_hub_download``, which resumes partial files, so
    interrupted Kaggle sessions continue where they left off).
  * ``iter_*_text``      — yield ``(doc_id, text)`` from ``.parquet`` /
    ``.txt`` / ``.jsonl`` / ``.csv`` files.

Every optional dependency (``huggingface_hub``, ``pyarrow``, ``pandas``) is
imported lazily so this module imports offline; a missing dependency raises
:class:`HFError` at call time with a clear message. Gated repositories (e.g.
OSCAR-2301) require a logged-in token (``HF_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN``)
that has accepted the dataset terms — the error message says exactly that.

This module deliberately imports nothing from the per-language packages.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

TEXT_KEYS = ("text", "sentence", "content", "article", "body")


class HFError(RuntimeError):
    """Raised for any HF download/listing/reading failure (missing dep, gated
    repo, network error, corrupt file). Callers (per-language sources.py) wrap
    it in a SourceError so the collector logs and continues."""


def hf_token() -> Optional[str]:
    """HF token from env (honours the same vars the hub library reads)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def hf_available() -> bool:
    try:
        import huggingface_hub  # noqa: F401

        return True
    except ImportError:
        return False


def _hub_error(exc: Exception, action: str, repo_id: str) -> HFError:
    msg = str(exc)
    hint = ""
    low = msg.lower()
    if "gated" in low or "401" in msg or "403" in msg:
        hint = (
            " (gated repo: accept the dataset terms on huggingface.co and set "
            "HF_TOKEN so huggingface_hub can authenticate)"
        )
    elif "404" in msg:
        hint = " (404: repo or file does not exist — check the id/path)"
    return HFError(f"{action} {repo_id} failed: {msg}{hint}")


def list_repo_files(
    repo_id: str,
    prefix: Optional[str] = None,
    token: Optional[str] = None,
) -> list[str]:
    """All file names in ``repo_id`` (optionally filtered by prefix), sorted."""
    if not hf_available():
        raise HFError("huggingface_hub not installed — pip install huggingface_hub")
    try:
        from huggingface_hub import list_repo_files

        # Corpora live in DATASET repos (Sangraha, IndicCorpV2, OSCAR-2301,
        # wikimedia/wikipedia); the hub defaults to model repos and 401s.
        # token=False forces anonymous access unless the caller (or HF_TOKEN)
        # explicitly opts into auth — a stale cached token must not break
        # downloads of open repos.
        files = list_repo_files(
            repo_id=repo_id, repo_type="dataset", token=token or hf_token() or False
        )
    except Exception as exc:  # network / gating / auth — wrap with a hint
        raise _hub_error(exc, "listing files in", repo_id) from exc
    if prefix:
        files = [f for f in files if f.startswith(prefix)]
    return sorted(files)


def hf_download(
    repo_id: str,
    filename: str,
    cache_dir: str,
    token: Optional[str] = None,
    timeout: int = 120,
) -> Path:
    """Resume-capable download of one repo file into ``cache_dir``.

    Returns the local path. ``hf_hub_download`` caches by (repo, filename,
    revision), so re-runs are free; partial downloads resume.
    """
    if not hf_available():
        raise HFError("huggingface_hub not installed — pip install huggingface_hub")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        # NB: huggingface_hub >= 1.0 dropped `resume_download` (always resumes)
        # and `timeout`; `etag_timeout` bounds only the metadata probe.
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=str(cache),
            token=token or hf_token() or False,  # anonymous unless explicitly authed
            etag_timeout=timeout,
        )
    except Exception as exc:
        raise _hub_error(exc, f"downloading {filename} from", repo_id) from exc
    return Path(local)


# ---------------------------------------------------------------- text extraction

def _pick_text_column(columns: Sequence[str]) -> Optional[str]:
    for key in TEXT_KEYS:
        if key in columns:
            return key
    return None


def iter_parquet_text(
    paths: Iterable[str],
    text_keys: Sequence[str] = TEXT_KEYS,
    batch_size: int = 8192,
) -> Iterator[tuple[str, str]]:
    """Yield ``(doc_id, text)`` rows from parquet files.

    Column chosen as the first of ``text_keys`` present, else the first
    string-typed column. Batched reads keep memory bounded on big corpora.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise HFError("pyarrow not installed — pip install pyarrow") from None

    for path in sorted(Path(p) for p in paths):
        table = pq.read_table(str(path))
        cols = table.column_names
        key = _pick_text_column(cols)
        if key is None:
            for c in cols:
                if table.schema.field(c).type in (pa.string(), pa.large_string()):
                    key = c
                    break
        if key is None:
            raise HFError(f"no text column in {path} (columns: {cols})")
        for i, batch in enumerate(table.to_batches(max_chunksize=batch_size)):
            base = i * batch_size
            for j, value in enumerate(batch.column(key).to_pylist()):
                if isinstance(value, str) and value.strip():
                    yield f"{Path(path).name}#{base + j}", value.strip()


def iter_text_lines(paths: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Yield ``(doc_id, text)`` from plain-text files, one doc per non-empty line."""
    for path in sorted(Path(p) for p in paths):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        yield f"{path.name}#{i}", line
        except OSError as exc:
            raise HFError(f"cannot read {path}: {exc}") from exc


def iter_jsonl_text(paths: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Yield ``(doc_id, text)`` from JSONL files (``text`` / ``sentence`` keys)."""
    for path in sorted(Path(p) for p in paths):
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = rec.get("text") or rec.get("sentence") or ""
                if isinstance(text, str) and text.strip():
                    yield f"{path.name}#{i}", text.strip()


def iter_csv_text(paths: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Yield ``(doc_id, text)`` from CSV files (``text``/``article``/... column)."""
    try:
        import pandas as pd
    except ImportError:
        raise HFError("pandas not installed — pip install pandas") from None

    for path in sorted(Path(p) for p in paths):
        try:
            df = pd.read_csv(str(path), dtype=str, encoding="utf-8")
        except OSError as exc:
            raise HFError(f"cannot read {path}: {exc}") from exc
        key = _pick_text_column(df.columns)
        if key is None:
            key = df.columns[0]
        for i, value in enumerate(df[key].tolist()):
            if isinstance(value, str) and value.strip():
                yield f"{path.name}#{i}", value.strip()
