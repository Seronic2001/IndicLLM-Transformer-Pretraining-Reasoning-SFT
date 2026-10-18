"""Matplotlib fonts for Devanagari (Hindi) and Eastern Nagari (Assamese) text.

Matplotlib's default font, DejaVu Sans, has neither script, so token labels render as
empty boxes (the Kaggle images have no Indic fonts either). The Noto Sans fonts bundled
in common/fonts (SIL Open Font License, OFL.txt) are registered as fallbacks after
DejaVu Sans, so Latin text keeps its usual look. Matplotlib >= 3.11 shapes complex
scripts (conjuncts, pre-base vowel signs); older versions draw the glyphs unshaped.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
from matplotlib import font_manager

FONT_DIR = Path(__file__).resolve().parent / "fonts"
INDIC_FAMILIES = ["Noto Sans Devanagari", "Noto Sans Bengali"]


def use_indic_fonts() -> None:
    """Register the bundled fonts and make them fallbacks for every plot."""
    for path in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(path))
    matplotlib.rcParams["font.family"] = ["DejaVu Sans", *INDIC_FAMILIES]
    major, minor = (int(p) for p in matplotlib.__version__.split(".")[:2])
    if (major, minor) < (3, 11):
        warnings.warn(f"matplotlib {matplotlib.__version__} does not shape Indic text; "
                      "install matplotlib>=3.11 for correct conjuncts and vowel signs")
