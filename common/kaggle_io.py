"""Kaggle / Google-Drive plumbing shared by both languages.

Every function here is infra with no opinion about language content: it locates the
highest-step valid checkpoint in a mounted Dataset input dir, stages a checkpoint
snapshot into ``/kaggle/working`` with a manifest entry, mirrors files to a
mounted Google Drive path, and pushes staged snapshots as new versions of a
Kaggle Dataset via the authenticated CLI (:func:`cli_commit`). All commits are
explicit (opt-in flags) and recorded in the checkpoint manifest so nothing is
pushed silently.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Optional

from common.checkpoint import checkpoint_paths_in_dir, validate_checkpoint

KAGGLE_WORKING = "/kaggle/working"
KAGGLE_INPUT = "/kaggle/input"
MANIFEST_NAME = "checkpoint_manifest.jsonl"


def find_latest_checkpoint(dataset_input_dir: str) -> Optional[str]:
    """Return the highest-step valid checkpoint in a mounted Kaggle Dataset input.

    Scans ``dataset_input_dir`` recursively for ``ckpt_<step>.pt`` files, validates
    each, and returns the one with the largest step, or ``None`` if none is valid.
    """
    if not os.path.isdir(dataset_input_dir):
        return None

    candidates: list[tuple[int, str]] = []
    for root, _dirs, files in os.walk(dataset_input_dir):
        for f in files:
            if f.startswith("ckpt_") and f.endswith(".pt"):
                step_str = f[len("ckpt_") : -len(".pt")]
                if step_str.isdigit():
                    candidates.append((int(step_str), os.path.join(root, f)))

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    for _step, path in candidates:
        if validate_checkpoint(path):
            return path
    return None


def commit_checkpoint_snapshot(
    local_ckpt_path: str, notes: str, manifest_dir: Optional[str] = None
) -> str:
    """Stage a checkpoint into Kaggle working and append a manifest entry.

    Copies ``local_ckpt_path`` to ``<KAGGLE_WORKING or cwd>/checkpoints/`` and
    appends a JSONL manifest line recording what was staged and when, so the
    "Save Version" UI action has a paper trail. Returns the staged destination
    path. No-op-safe outside Kaggle (works in any cwd).
    """
    if not os.path.exists(local_ckpt_path):
        raise FileNotFoundError(f"Cannot stage missing checkpoint: {local_ckpt_path}")

    working = KAGGLE_WORKING if os.path.isdir(KAGGLE_WORKING) else os.getcwd()
    dest_dir = os.path.join(working, "checkpoints")
    os.makedirs(dest_dir, exist_ok=True)

    dest = os.path.join(dest_dir, os.path.basename(local_ckpt_path))
    shutil.copy2(local_ckpt_path, dest)

    if manifest_dir is None:
        manifest_dir = working
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, MANIFEST_NAME)
    entry = {
        "file": dest,
        "source": os.path.abspath(local_ckpt_path),
        "notes": notes,
        "staged_at": _now_iso(),
    }
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return dest


def cli_commit(
    payload_dir: str,
    owner_slug: str,
    dataset_slug: str,
    title: str,
    message: str = "checkpoint snapshot",
    create_if_missing: bool = False,
    manifest_dir: Optional[str] = None,
) -> dict:
    """Version a staged checkpoint payload to Kaggle via the authenticated CLI.

    Complements :func:`commit_checkpoint_snapshot`: that stages files locally,
    this pushes them as a new version of ``owner_slug/dataset_slug`` using
    ``python -m kaggle datasets version`` (creating the dataset on the first push
    when ``create_if_missing``). The commit is recorded in ``checkpoint_manifest.jsonl``
    (kind ``kaggle_commit``) so every pushed snapshot has a local paper trail.

    Raises
    ------
    common.kaggle_cli.KaggleCLIError
        When the CLI is missing/unauthenticated, the dataset does not exist and
        ``create_if_missing`` is False, or the API call fails.
    """
    from common.kaggle_cli import (  # lazy import keeps local offline use cheap
        KaggleCLIError,
        cli_available,
        datasets_version,
        ensure_dataset,
        has_credentials,
        write_datasets_metadata,
    )

    if not cli_available():
        raise KaggleCLIError(
            "kaggle CLI not available on this interpreter — install with "
            "`pip install kaggle` and authenticate with `kaggle auth login` "
            "(or kaggle.json credentials)."
        )
    if not has_credentials():
        raise KaggleCLIError(
            "no Kaggle credentials found — run `kaggle auth login` or place "
            "kaggle.json in ~/.kaggle/"
        )

    payload = os.path.abspath(payload_dir)
    os.makedirs(payload, exist_ok=True)
    write_datasets_metadata(
        payload, title, owner_slug=owner_slug, dataset_slug=dataset_slug,
    )
    if create_if_missing:
        action = ensure_dataset(payload, title, owner_slug, dataset_slug, quiet=True)
    else:
        datasets_version(payload, message=message)
        action = "versioned"

    if manifest_dir is None:
        manifest_dir = payload
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, MANIFEST_NAME)
    entry = {
        "kind": "kaggle_commit",
        "dataset": f"{owner_slug}/{dataset_slug}",
        "action": action,
        "message": message,
        "payload_dir": payload,
        "at": _now_iso(),
    }
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def mirror_to_drive(local_path: str, drive_subpath: str) -> Optional[str]:
    """Copy ``local_path`` to ``<Drive>/<drive_subpath>``.

    Google Drive on Kaggle mounts at ``/kaggle/input/google-drive/MyDrive``. Outside
    Kaggle, a ``DRIVE_MOUNT`` env var can point at a local Drive mirror. No-op with a
    warning (and returns None) when no Drive mount is present.
    """
    mount = None
    for candidate in ("/kaggle/input/google-drive/MyDrive",):
        if os.path.isdir(candidate):
            mount = candidate
            break
    if mount is None:
        mount = os.environ.get("DRIVE_MOUNT")
    if not mount or not os.path.isdir(mount):
        print(
            f"[kaggle_io] WARNING: Drive not mounted; skipping mirror of {local_path} "
            f"(expected mount at /kaggle/input/google-drive/MyDrive or $DRIVE_MOUNT)."
        )
        return None

    dest_dir = os.path.join(mount, drive_subpath)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(local_path))
    shutil.copy2(local_path, dest)
    return dest


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
