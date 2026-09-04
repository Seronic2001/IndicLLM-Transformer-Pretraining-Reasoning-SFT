"""Shared utilities for web crawlers (/ Manual Data Collection).

Provides:
  - Robust HTTP fetching with retries, timeout, and connection reuse
  - High-precision HTML text extraction (boilerplate / navigation / ad stripping)
  - MediaWiki wikitext cleaner (template / table / category stripping)
  - Recursive XML sitemap parser
  - Script purity check and token count estimation
"""

from __future__ import annotations

import html
import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

# Create a permissive SSL context for government / educational sites with legacy certificates
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "hi,as,en-US,en;q=0.9",
}


def fetch_url(
    url: str,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 12.0,
    max_retries: int = 3,
    backoff_factor: float = 1.5,
) -> Optional[bytes]:
    """Fetch raw bytes from a URL with exponential backoff and retry."""
    req_headers = dict(DEFAULT_HEADERS)
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, headers=req_headers)
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, context=_SSL_CTX, timeout=timeout) as resp:
                if resp.status == 200:
                    return resp.read()
                elif resp.status in (404, 410):
                    return None
        except urllib.error.HTTPError as err:
            if err.code in (404, 410):
                return None
            if attempt == max_retries - 1:
                return None
            time.sleep(backoff_factor * (attempt + 1))
        except Exception:
            if attempt == max_retries - 1:
                return None
            time.sleep(backoff_factor * (attempt + 1))
    return None


def fetch_json(url: str, headers: Optional[dict[str, str]] = None, timeout: float = 12.0) -> Optional[dict]:
    """Fetch and parse JSON from a URL."""
    data = fetch_url(url, headers=headers, timeout=timeout)
    if not data:
        return None
    try:
        return json.loads(data.decode("utf-8", errors="replace"))
    except Exception:
        return None


def clean_html_text(raw_html: str) -> str:
    """Extract clean, coherent body text from raw HTML, stripping boilerplate."""
    if not raw_html:
        return ""

    text = raw_html

    # 1. Remove scripts, styles, noscript, svg, nav, header, footer, iframe, form
    for tag in ("script", "style", "noscript", "svg", "nav", "header", "footer", "iframe", "form", "aside"):
        text = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", " ", text, flags=re.DOTALL | re.IGNORECASE)

    # 2. Remove HTML comments
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)

    # 3. Replace paragraph / block tags with newlines
    text = re.sub(r"</?(?:p|div|br|h[1-6]|li|tr|blockquote|section|article|dt|dd)[^>]*>", "\n", text, flags=re.IGNORECASE)

    # 4. Strip all remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 5. Decode HTML entities
    text = html.unescape(text)

    # 6. Clean and filter lines
    cleaned_lines = []
    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        # Filter out trivial navigation strings or single symbols
        if len(line) >= 10 and not _is_boilerplate_line(line):
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _is_boilerplate_line(line: str) -> bool:
    """Detect typical website boilerplate, social share prompts, or cookie notices."""
    lower = line.lower()
    boilerplate_fragments = (
        "cookie policy", "terms of use", "privacy policy", "all rights reserved",
        "copyright ©", "follow us on", "subscribe to", "whatsapp group", "click here to read",
        "share this article", "advertisement", "sponsored link", "published by",
        "comment below", "latest news", "breaking news", "trending now",
    )
    for frag in boilerplate_fragments:
        if frag in lower and len(line) < 80:
            return True
    return False


def clean_wikitext(text: str) -> str:
    """Strip MediaWiki markup to extract plain readable prose from MediaWiki API responses."""
    if not text:
        return ""

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Remove ref tags and contents
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<ref[^/>]*/>", "", text, flags=re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove file/image links
    text = re.sub(r"\[\[(?:File|Image|चित्र|ফাইল):[^\]]+\]\]", "", text, flags=re.IGNORECASE)
    # Remove category links
    text = re.sub(r"\[\[(?:Category|श्रेणी|শ্রেণী):[^\]]+\]\]", "", text, flags=re.IGNORECASE)
    # Remove nested templates {{...}}
    for _ in range(5):
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
    # Clean up empty brackets / parens
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\[\s*\]", "", text)
    # Filter empty / boilerplate lines
    lines = [re.sub(r"\s+", " ", l).strip() for l in text.split("\n")]
    lines = [l for l in lines if len(l) >= 20]
    return "\n".join(lines).strip()


def parse_sitemap_urls(sitemap_url: str, max_urls: int = 10000, timeout: float = 12.0) -> list[str]:
    """Parse XML sitemap or sitemap index and return list of article URLs."""
    raw = fetch_url(sitemap_url, timeout=timeout)
    if not raw:
        return []

    text = raw.decode("utf-8", errors="replace")
    all_locs = re.findall(r"<loc>\s*(https?://[^\s<]+)\s*</loc>", text, flags=re.IGNORECASE)

    urls: list[str] = []
    sub_sitemaps: list[str] = []

    for loc in all_locs:
        loc = loc.strip()
        if "sitemap" in loc.lower() and (loc.endswith(".xml") or loc.endswith(".xml.gz")):
            sub_sitemaps.append(loc)
        else:
            urls.append(loc)

    # If sitemap index, recursively fetch sub-sitemaps
    for sub in sub_sitemaps:
        if len(urls) >= max_urls:
            break
        sub_raw = fetch_url(sub, timeout=timeout)
        if sub_raw:
            sub_text = sub_raw.decode("utf-8", errors="replace")
            sub_locs = re.findall(r"<loc>\s*(https?://[^\s<]+)\s*</loc>", sub_text, flags=re.IGNORECASE)
            for sl in sub_locs:
                if not ("sitemap" in sl.lower() and sl.endswith(".xml")):
                    urls.append(sl.strip())
                    if len(urls) >= max_urls:
                        break

    return urls[:max_urls]


def compute_script_fraction(text: str, lang: str) -> tuple[int, int, float]:
    """Return (in_script_chars, total_chars, script_fraction)."""
    if not text:
        return 0, 0, 0.0

    if lang.lower() == "hindi":
        # Devanagari Unicode Block: U+0900 - U+097F
        pattern = r"[\u0900-\u097F]"
    elif lang.lower() == "assamese":
        # Bengali-Assamese Unicode Block: U+0980 - U+09FF
        pattern = r"[\u0980-\u09FF]"
    else:
        pattern = r"\w"

    in_script = len(re.findall(pattern, text))
    total_non_ws = len(re.sub(r"\s+", "", text))
    fraction = (in_script / total_non_ws) if total_non_ws > 0 else 0.0
    return in_script, total_non_ws, fraction


def estimate_tokens(text: str) -> int:
    """Estimate token count for Indic text (approx 1 word ≈ 1.3 - 1.5 tokens)."""
    words = len(text.split())
    # Average of word-based heuristic for subword tokenizers
    return max(1, int(words * 1.35))
