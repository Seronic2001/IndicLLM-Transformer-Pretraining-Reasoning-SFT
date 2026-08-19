"""Unicode script-range helpers — shared infra (no language-specific logic).

Used by the data pipeline (script filtering), the reasoning generator's purity
check, and the acceptance tests. The Indic dandas (। U+0964, ॥ U+0965) are shared
Indic punctuation valid in both scripts and are always allowed.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional, Tuple

SCRIPT_RANGES = {
    "hindi": (0x0900, 0x097F),      # Devanagari
    "assamese": (0x0980, 0x09FF),   # Bengali-Assamese
}
# Shared Indic sentence punctuation, valid in both scripts.
SHARED_PUNCT = {"\u0964", "\u0965"}  # । ॥

# -------------------------------------------------------------
# UNICODE RANGES & PATTERNS FOR INDIC SCRIPTS
# -------------------------------------------------------------
# Hindi (Devanagari)
HI_MATRAS = r"[\u093E-\u094C\u094E-\u094F\u0955-\u0957\u0962-\u0963]"
HI_SIGNS = r"[\u0901-\u0903\u093C]"  # Candrabindu, Anusvara, Visarga, Nukta
HI_HALANT = r"\u094D"
HI_CONSONANTS = r"[\u0915-\u0939\u0958-\u095F]"
HI_ALL_DIACRITICS = r"[\u0901-\u0903\u093C\u093E-\u094F\u0955-\u0957\u0962-\u0963]"

# Assamese (Eastern Nagari / Bengali-Assamese)
AS_MATRAS = r"[\u09BE-\u09CC\u09D7\u09E2-\u09E3]"
AS_SIGNS = r"[\u0981-\u0983\u09BC\u09CE]"  # Candrabindu, Anusvara, Visarga, Nukta, Khanda-Ta
AS_HALANT = r"\u09CD"
AS_CONSONANTS = r"[\u0995-\u09B9\u09DC-\u09DF\u09F0\u09F1]"
AS_ALL_DIACRITICS = r"[\u0981-\u0983\u09BC\u09BE-\u09CD\u09D7\u09E2-\u09E3]"

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_INVISIBLE_RE = re.compile(r"[\u200B-\u200F\uFEFF]")


def is_script_char(ch: str, lang: str) -> bool:
    """True if ``ch`` is in the language's script range or is shared Indic punct."""
    if ch in SHARED_PUNCT:
        return True
    lo, hi = SCRIPT_RANGES[lang]
    return lo <= ord(ch) <= hi


def script_fraction(text: str, lang: str) -> tuple[int, int]:
    """(chars in script, total non-space chars)."""
    in_script = total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if is_script_char(ch, lang):
            in_script += 1
    return in_script, total


def clean_indic_text(text: str, lang: str = "hindi") -> str:
    """Deeply cleans and repairs OCR/crawler-damaged Hindi & Assamese text.

    Heals spaced-out matras, fixes broken conjuncts / halants, strips orphan
    diacritics, removes HTML tags / URLs, and collapses whitespace.
    """
    if not text or not text.strip():
        return ""

    # 1. Unicode Normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Strip HTML tags, URLs, control chars, and zero-width markers
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _CONTROL_RE.sub("", text)
    text = _INVISIBLE_RE.sub("", text)

    if lang == "hindi":
        c, m, h, all_d = HI_CONSONANTS, HI_MATRAS, HI_HALANT, HI_ALL_DIACRITICS

        # 3. Fix intra-word split matras where OCR inserted spaces on both sides of a matra inside a word
        # (e.g. "शिक्षणसंक े त" -> "शिक्षणसंकेत")
        text = re.sub(rf"([\u0900-\u097F]{{2,}}{c})\s+({all_d})\s+({c})\b", r"\1\2\3", text)

        # 4. Fix OCR spaced-out matras & diacritics (e.g. "क े" -> "के", "म ें" -> "में", "ह ैं" -> "हैं")
        for _ in range(2):
            text = re.sub(rf"(\S)\s+({all_d})", r"\1\2", text)

        # 5. Fix broken conjuncts / halants (e.g. "बच ्चे" -> "बच्चे")
        text = re.sub(rf"({c})\s*({h})\s*({c})", r"\1\2\3", text)

        # 6. Fix invalid stacked duplicate matras (e.g. "उन्हेंै" -> "उन्हें")
        text = re.sub(rf"({m}){m}+", r"\1", text)

    else:
        c, m, h, all_d = AS_CONSONANTS, AS_MATRAS, AS_HALANT, AS_ALL_DIACRITICS

        # 3. Fix intra-word split matras where OCR inserted spaces on both sides
        text = re.sub(rf"([\u0980-\u09FF]{{2,}}{c})\s+({all_d})\s+({c})\b", r"\1\2\3", text)

        # 4. Fix OCR spaced-out matras & diacritics
        for _ in range(2):
            text = re.sub(rf"(\S)\s+({all_d})", r"\1\2", text)

        # 5. Fix broken conjuncts / halants (e.g. "প ্ৰ" -> "প্ৰ")
        text = re.sub(rf"({c})\s*({h})\s*({c})", r"\1\2\3", text)

        # 6. Fix Assamese genitive case suffix 'ৰ' and common verb suffixes detached by OCR (e.g. "অসম ৰ" -> "অসমৰ", "হৈ ছে" -> "হৈছে")
        text = re.sub(r"([\u0980-\u09FF]+)\s+ৰ(?:\s|[।॥]|$)", r"\1ৰ ", text)
        text = re.sub(r"([\u0980-\u09FF]+)\s+(ছে|ছিল|লে|ছেন)(?:\s|[।॥]|$)", r"\1\2 ", text)

        # 7. Fix invalid stacked duplicate matras
        text = re.sub(rf"({m}){m}+", r"\1", text)

    # 8. Remove orphan diacritics (stray matras floating alone at start of word/line)
    text = re.sub(rf"(?:^|\s){all_d}+", " ", text)

    # 9. Collapse whitespace
    text = " ".join(text.split())
    return text


def is_high_quality_indic_document(text: str, lang: str = "hindi") -> Tuple[bool, str]:
    """Rejects severely corrupted OCR / crawled documents before corpus ingestion.

    Returns:
        (is_valid, reason_if_invalid)
    """
    words = text.split()
    if len(words) < 5:
        return False, "too_short"

    all_d = HI_ALL_DIACRITICS if lang == "hindi" else AS_ALL_DIACRITICS
    m = HI_MATRAS if lang == "hindi" else AS_MATRAS
    h = HI_HALANT if lang == "hindi" else AS_HALANT

    # Rule A: Check for excessive orphan matras or split tokens
    orphan_matras = len(re.findall(rf"(?:^|\s){all_d}", text))
    if (orphan_matras / max(1, len(words))) > 0.03:
        return False, "high_orphan_diacritics"

    # Rule B: Single-Character Token Ratio
    valid_singletons = {"व", "न", "त", "य", "ও", "এ", "বা"}
    single_char_words = sum(1 for w in words if len(w) == 1 and w not in valid_singletons)
    if (single_char_words / max(1, len(words))) > 0.15:
        return False, "excessive_single_char_fragments"

    # Rule C: Consecutive Invalid Matras or Halants
    invalid_diacritics = len(re.findall(rf"({m}{{2,}}|{h}{{2,}})", text))
    if (invalid_diacritics / max(1, len(words))) > 0.02:
        return False, "invalid_diacritic_stacking"

    return True, "ok"

