"""Resolve per-language config paths / YAML dicts.

Pure infra: knows the repo layout, has no opinion about model architecture or
language content, and never imports model code — so it is safe to share between
the two languages (spec §2: common/ holds only cross-cutting infra).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

REPO_ROOT = Path(__file__).resolve().parents[1]


def language_dir(lang: str) -> Path:
    """Return the language root (hindi/ or assamese/)."""
    root = REPO_ROOT / lang
    if not root.is_dir():
        raise FileNotFoundError(f"No language directory found at {root}")
    return root


def model_config_path(lang: str, name: str) -> Path:
    return language_dir(lang) / "configs" / name


def train_config_path(lang: str, name: str) -> Path:
    return language_dir(lang) / "configs" / name


def tokenizer_config_path(lang: str, name: str) -> Path:
    return language_dir(lang) / "configs" / name


def load_yaml_dict(path: Union[str, Path]) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} is not a YAML mapping")
    return data
