"""Kaggle CLI wrapper (common infra — no language-specific logic).

The ``kaggle`` CLI is a pip package and may not be on PATH (e.g. Git Bash on
Windows), so every command is invoked as ``python -m kaggle ...`` with the current
interpreter. Credentials are read from ``~/.kaggle/kaggle.json`` (API key) or
``~/.kaggle/credentials.json`` (OAuth — used by ``kaggle auth login``).

Used for two things in this project:
  * ``datasets download``  — pull public corpora (Agent-A data collection).
  * ``datasets version``   — persist training checkpoints across Kaggle sessions
    by committing them as new versions of a Kaggle Dataset (Agent-D). Commits are
    always explicit (opt-in flag) and recorded in the checkpoint manifest so
    nothing is pushed silently (spec §3 Agent-D / §5.2).

All wrappers are pure subprocess plumbing — easily unit-tested offline by
monkeypatching ``subprocess.run`` / the credentials lookup.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

KAGGLE_CREDENTIAL_DIR = Path.home() / ".kaggle"
KAGGLE_JSON = KAGGLE_CREDENTIAL_DIR / "kaggle.json"
CREDENTIALS_JSON = KAGGLE_CREDENTIAL_DIR / "credentials.json"


class KaggleCLIError(RuntimeError):
    """Raised when the Kaggle CLI is missing, unauthenticated, or a command fails."""


def _base_cmd() -> list[str]:
    return [sys.executable, "-m", "kaggle"]


def cli_available() -> bool:
    """True if ``python -m kaggle`` runs on this interpreter (offline check)."""
    try:
        proc = subprocess.run(
            _base_cmd() + ["--version"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def credential_files() -> list[Path]:
    return [p for p in (KAGGLE_JSON, CREDENTIALS_JSON) if p.exists()]


def has_credentials() -> bool:
    """Offline check: an API key (kaggle.json) or OAuth credentials exist."""
    return bool(credential_files())


def verify_auth(timeout: int = 60) -> bool:
    """Live probe: run an authenticated, read-only CLI command.

    ``kaggle datasets list`` needs valid credentials; exit code 0 means the CLI is
    authenticated. Returns False (never raises) when the CLI is missing, creds are
    absent, or the API rejects the request.
    """
    if not cli_available() or not has_credentials():
        return False
    try:
        # NOTE: `datasets list` takes no -q flag in kaggle CLI 2.x; the probe
        # stays read-only, minimal, and quiet by construction.
        proc = subprocess.run(
            _base_cmd() + ["datasets", "list", "--page-size", "1"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def auth_status() -> str:
    """One of: "cli_missing" | "no_credentials" | "unauthenticated" | "authenticated".

    Uses the cheap offline checks; the live probe is only run when both the CLI
    and credentials are present.
    """
    if not cli_available():
        return "cli_missing"
    if not has_credentials():
        return "no_credentials"
    return "authenticated" if verify_auth() else "unauthenticated"


def _run(args: list[str], quiet: bool = True, timeout: int = 600) -> subprocess.CompletedProcess:
    if quiet:
        args = args + ["-q"] if "-q" not in args else args
    return subprocess.run(
        args, capture_output=True, text=True, timeout=timeout, check=False
    )


def datasets_download(
    dataset_ref: str,
    target_dir: str,
    file_name: Optional[str] = None,
    unzip: bool = True,
    force: bool = False,
    quiet: bool = True,
) -> Path:
    """``kaggle datasets download <owner/dataset> -p <dir> [--unzip]``.

    Returns the target directory. Raises KaggleCLIError on failure (network,
    invalid ref, unauthenticated) with the CLI's stderr attached.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    args = _base_cmd() + ["datasets", "download", dataset_ref, "-p", str(target)]
    if file_name:
        args += ["-f", file_name]
    if unzip:
        args += ["--unzip"]
    if force:
        args += ["-o"]
    if quiet:
        args += ["-q"]
    proc = _run(args, quiet=quiet)
    if proc.returncode != 0:
        raise KaggleCLIError(
            f"kaggle datasets download {dataset_ref} failed ({proc.returncode}): "
            f"{proc.stderr.strip()[-500:] or proc.stdout.strip()[-500:]}"
        )
    return target


# ---------------------------------------------------------------- dataset metadata

def write_datasets_metadata(
    folder: str,
    title: str,
    subtitle: Optional[str] = None,
    description: Optional[str] = None,
    owner_slug: Optional[str] = None,
    dataset_slug: Optional[str] = None,
    license_name: str = "other",
    is_private: bool = True,
    extra: Optional[dict] = None,
) -> Path:
    """Write ``dataset-metadata.json`` in the payload folder (kaggle CLI format).

    ``id``/``id_no`` identify an *existing* dataset (required for ``version``);
    omit them when creating a brand-new dataset (``datasets create``).
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "title": title,
        "subtitle": subtitle or "",
        "description": description or title,
        "licenses": [{"name": license_name}],
        "isPrivate": is_private,
    }
    if owner_slug and dataset_slug:
        meta["id"] = f"{owner_slug}/{dataset_slug}"
    if extra:
        meta.update(extra)
    path = folder / "dataset-metadata.json"
    content = json.dumps(meta, ensure_ascii=False, indent=2)
    path.write_text(content, encoding="utf-8")
    (folder / "datasets-metadata.json").write_text(content, encoding="utf-8")
    return path


def datasets_create(folder: str, quiet: bool = True) -> subprocess.CompletedProcess:
    """Create a new Kaggle Dataset from a payload folder (needs metadata file)."""
    proc = _run(_base_cmd() + ["datasets", "create", "-p", str(folder)], quiet=quiet)
    if proc.returncode != 0:
        raise KaggleCLIError(
            f"kaggle datasets create failed ({proc.returncode}): "
            f"{proc.stderr.strip()[-500:] or proc.stdout.strip()[-500:]}"
        )
    return proc


def datasets_version(
    folder: str,
    message: str,
    dir_mode: str = "tar",
    ignore_patterns: Optional[list[str]] = None,
    quiet: bool = True,
) -> subprocess.CompletedProcess:
    """Upload a new version of an existing Kaggle Dataset (checkpoint snapshot).

    The payload folder must contain ``datasets-metadata.json`` with ``id``/``id_no``
    (see write_datasets_metadata). Raises KaggleCLIError on failure.
    """
    args = _base_cmd() + ["datasets", "version", "-p", str(folder), "-m", message]
    if dir_mode:
        args += ["-r", dir_mode]
    if ignore_patterns:
        args += ["--ignore-patterns", ",".join(ignore_patterns)]
    if quiet:
        args += ["-q"]
    proc = _run(args, quiet=quiet)
    if proc.returncode != 0:
        raise KaggleCLIError(
            f"kaggle datasets version failed ({proc.returncode}): "
            f"{proc.stderr.strip()[-500:] or proc.stdout.strip()[-500:]}"
        )
    return proc


def ensure_dataset(
    payload_dir: str,
    title: str,
    owner_slug: str,
    dataset_slug: str,
    description: Optional[str] = None,
    is_private: bool = True,
    quiet: bool = True,
) -> str:
    """Create the dataset on first commit, then upload a version.

    Tries ``datasets version``; if that fails because the dataset does not exist
    yet (error text contains 'not found'), writes metadata with no ``id`` and runs
    ``datasets create``, then retries the version. Returns the CLI action taken:
    "created" or "versioned".
    """
    version_dir = Path(payload_dir)
    write_datasets_metadata(
        str(version_dir), title, owner_slug=owner_slug, dataset_slug=dataset_slug,
        description=description, is_private=is_private,
    )
    try:
        datasets_version(str(version_dir), message=f"checkpoint snapshot: {title}", quiet=quiet)
        return "versioned"
    except KaggleCLIError as exc:
        if "not found" not in str(exc).lower():
            raise
        # Dataset doesn't exist yet — create it (no id in metadata), then version.
        write_datasets_metadata(
            str(version_dir), title, owner_slug=None, dataset_slug=None,
            description=description, is_private=is_private,
        )
        datasets_create(str(version_dir), quiet=quiet)
        write_datasets_metadata(
            str(version_dir), title, owner_slug=owner_slug, dataset_slug=dataset_slug,
            description=description, is_private=is_private,
        )
        datasets_version(str(version_dir), message=f"checkpoint snapshot: {title}", quiet=quiet)
        return "created"


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry: python -m common.kaggle_cli <command>

    Commands:
      auth                     print auth status
      download <owner/dataset> --path DIR [--file NAME] [--no-unzip]
      push-checkpoints DIR --title TITLE --owner OWNER --slug SLUG [--message M]
    """
    import argparse

    parser = argparse.ArgumentParser(prog="python -m common.kaggle_cli")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("auth")
    dl = sub.add_parser("download")
    dl.add_argument("dataset_ref")
    dl.add_argument("--path", default=".")
    dl.add_argument("--file", dest="file_name", default=None)
    dl.add_argument("--no-unzip", action="store_true")

    push = sub.add_parser("push-checkpoints")
    push.add_argument("payload_dir")
    push.add_argument("--title", required=True)
    push.add_argument("--owner", required=True)
    push.add_argument("--slug", required=True)
    push.add_argument("--message", default="checkpoint snapshot")
    push.add_argument("--create-if-missing", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "auth":
        status = auth_status()
        print(f"auth status: {status}")
        if status == "authenticated":
            print("  (CLI is authenticated - dataset download / version commands will work)")
        return 0 if status == "authenticated" else 1
    if args.command == "download":
        datasets_download(
            args.dataset_ref, args.path, file_name=args.file_name,
            unzip=not args.no_unzip,
        )
        print(f"downloaded {args.dataset_ref} -> {args.path}")
        return 0
    if args.command == "push-checkpoints":
        write_datasets_metadata(
            args.payload_dir, args.title,
            owner_slug=args.owner, dataset_slug=args.slug,
        )
        if args.create_if_missing:
            action = ensure_dataset(args.payload_dir, args.title, args.owner, args.slug)
            print(f"dataset {args.owner}/{args.slug}: {action}")
        else:
            datasets_version(args.payload_dir, args.message)
            print(f"pushed version of {args.owner}/{args.slug} from {args.payload_dir}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
