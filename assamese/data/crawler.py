"""Assamese Web Crawler for Manual Corpus.

Target: Collect authentic, high-quality Assamese text (~100M tokens, ~2-3 GB)
strictly avoiding Wikipedia (already collected in downloaded/manual dumps).

Sources Crawled:
  1. CC-100 Assamese Web Corpus (statmt.org/cc-100) - Clean web-crawled Assamese text
  2. Niyomiya Barta Deep Category Archives (niyomiyabarta.com) - Extensive pagination (pages 1-750)
  3. Northeast Now Deep Category Archives (assam.nenow.in) - Extensive pagination (pages 1-1000)
  4. Asomiya Pratidin Deep Sitemaps (asomiyapratidin.in) - Sitemaps 1-1500
  5. News18 Assam (assam.news18.com) - State & national news sections
  6. Vikaspedia Assamese (as.vikaspedia.in) - Government knowledge encyclopedia
  7. Dainik Agradoot (dainikagradoot.in) - Daily Assamese news portal
  8. Xahitya.org (xahitya.org) - Monthly Assamese literary webzine
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import re
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.crawler_utils import (
    clean_html_text,
    compute_script_fraction,
    estimate_tokens,
    fetch_json,
    fetch_url,
    parse_sitemap_urls,
)

LANG = "assamese"
MIN_SCRIPT_FRACTION = 0.65
MIN_DOC_CHARS = 120
BATCH_SIZE = 32

# Distinctive Assamese Unicode characters (ৰ: U+09F0, ৱ: U+09F1)
ASSAMESE_SPECIFIC_CHARS = re.compile(r"[\u09F0\u09F1]")


class AssameseCrawler:
    """Multi-source concurrent crawler for Assamese manual corpus."""

    def __init__(
        self,
        out_dir: Path,
        target_tokens: int = 100_000_000,
        max_docs: Optional[int] = None,
        max_bytes: Optional[int] = None,
        workers: int = 12,
        prev_dirs: Optional[list[Path]] = None,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.target_tokens = target_tokens
        self.max_docs = max_docs
        self.max_bytes = max_bytes
        self.workers = workers
        self.prev_dirs = [Path(p) for p in (prev_dirs or []) if Path(p).exists()]

        self.seen_hashes: set[str] = set()
        self.seen_urls: set[str] = set()
        self.total_tokens_collected = 0
        self.total_bytes_collected = 0
        self.total_docs_collected = 0
        self.stats: dict[str, dict] = {}

        self._load_existing_hashes()

    def _load_existing_hashes(self) -> None:
        """Pre-populate seen hashes and URLs from existing JSONL files to prevent duplicate scraping."""
        scan_dirs = [self.out_dir] + self.prev_dirs

        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            for root, dirs, files in os.walk(kaggle_input):
                if any(f.endswith(".jsonl") for f in files):
                    p = Path(root)
                    if p not in scan_dirs:
                        scan_dirs.append(p)

        for sdir in scan_dirs:
            if not sdir.exists():
                continue
            for jsonl_file in sdir.glob("**/*.jsonl"):
                try:
                    with open(jsonl_file, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    doc = json.loads(line)
                                    h = doc.get("hash")
                                    if h:
                                        self.seen_hashes.add(h)
                                    u = doc.get("url")
                                    if u:
                                        self.seen_urls.add(u)
                                except Exception:
                                    pass
                except Exception:
                    pass

        if self.seen_hashes or self.seen_urls:
            print(f"[AssameseCrawler] Pre-loaded {len(self.seen_hashes):,} existing doc hashes and {len(self.seen_urls):,} URLs for deduplication.", flush=True)

    def _is_limit_reached(self) -> bool:
        if self.total_tokens_collected >= self.target_tokens:
            return True
        if self.max_docs and self.total_docs_collected >= self.max_docs:
            return True
        if self.max_bytes and self.total_bytes_collected >= self.max_bytes:
            return True
        return False

    def _save_doc(self, source_name: str, doc_id: str, text: str, url: str = "") -> bool:
        """Filter, hash, and persist a single document."""
        if not text or len(text) < MIN_DOC_CHARS:
            return False

        if url and url in self.seen_urls:
            return False

        _, _, purity = compute_script_fraction(text, LANG)
        if purity < MIN_SCRIPT_FRACTION:
            return False

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash in self.seen_hashes:
            return False

        self.seen_hashes.add(text_hash)
        if url:
            self.seen_urls.add(url)

        tokens = estimate_tokens(text)
        byte_len = len(text.encode("utf-8"))
        has_asm_distinctive = bool(ASSAMESE_SPECIFIC_CHARS.search(text))

        doc_record = {
            "doc_id": doc_id,
            "url": url,
            "source": f"crawler_{source_name}",
            "type": "manual",
            "hash": text_hash,
            "token_estimate": tokens,
            "char_count": len(text),
            "script_fraction": round(purity, 4),
            "has_assamese_distinctive_chars": has_asm_distinctive,
            "text": text,
        }

        out_file = self.out_dir / f"crawler_{source_name}.jsonl"
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(doc_record, ensure_ascii=False) + "\n")

        self.total_tokens_collected += tokens
        self.total_bytes_collected += byte_len
        self.total_docs_collected += 1

        if source_name not in self.stats:
            self.stats[source_name] = {"docs": 0, "tokens": 0, "bytes": 0}
        self.stats[source_name]["docs"] += 1
        self.stats[source_name]["tokens"] += tokens
        self.stats[source_name]["bytes"] += byte_len

        return True

    def _crawl_url_batches(self, source_name: str, urls: list[str]) -> None:
        """Process URLs in small batches to exit immediately on reaching quotas."""
        urls = [u for u in urls if u and u not in self.seen_urls]
        if not urls:
            return

        def _fetch_article(url):
            raw = fetch_url(url, timeout=10.0)
            if not raw:
                return None
            html = raw.decode("utf-8", errors="replace")
            cleaned = clean_html_text(html)
            return (url, cleaned)

        for i in range(0, len(urls), BATCH_SIZE):
            if self._is_limit_reached():
                break
            batch = urls[i : i + BATCH_SIZE]
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [pool.submit(_fetch_article, u) for u in batch]
                for fut in as_completed(futures):
                    res = fut.result()
                    if res:
                        u, text = res
                        doc_id = f"{source_name}/" + u.strip("/").split("/")[-1]
                        self._save_doc(source_name, doc_id, text, u)
                        if self._is_limit_reached():
                            break

    # ---------------------------------------------------------------- Sources

    def crawl_cc100_assamese(self, max_docs: int = 50000) -> None:
        """Stream clean web-crawled Assamese corpus from CC-100 repository."""
        print("\n[AssameseCrawler] Starting CC-100 Assamese Web Corpus stream...", flush=True)
        url = "https://data.statmt.org/cc-100/as.txt.xz"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                raw_bytes = resp.read()
            decomp = lzma.decompress(raw_bytes).decode("utf-8", errors="ignore")
            docs = [d.strip() for d in decomp.split("\n\n") if len(d.strip()) >= MIN_DOC_CHARS]
            print(f"[AssameseCrawler] Decompressed {len(docs):,} Assamese candidate docs from CC-100.", flush=True)
            for idx, doc_text in enumerate(docs[:max_docs]):
                if self._is_limit_reached():
                    break
                doc_id = f"cc100_as/doc_{idx}"
                self._save_doc("cc100_as", doc_id, doc_text, "")
        except Exception as exc:
            print(f"[AssameseCrawler] CC-100 Assamese stream error: {exc}", flush=True)

        print(f"[AssameseCrawler] CC-100 Assamese complete. Saved docs: {self.stats.get('cc100_as', {}).get('docs', 0)}", flush=True)

    def crawl_niyomiya_barta_deep(self, start_page: int = 1, max_pages_per_cat: int = 750) -> None:
        """Crawl Niyomiya Barta daily newspaper via deep category pagination."""
        print(f"\n[AssameseCrawler] Starting Niyomiya Barta Deep Category crawler (pages {start_page}-{max_pages_per_cat})...", flush=True)
        categories = ["assam", "editorial", "lifestyle", "sports", "national", "business", "international"]
        article_links: list[str] = []

        for cat in categories:
            if self._is_limit_reached():
                break
            print(f"  + Scanning category '{cat}'...", flush=True)
            for p in range(start_page, max_pages_per_cat + 1):
                cat_url = f"https://niyomiyabarta.com/category/{cat}/page/{p}/"
                raw = fetch_url(cat_url, timeout=10.0)
                if not raw:
                    break
                html = raw.decode("utf-8", errors="replace")
                found = re.findall(r'<h[23][^>]*>\s*<a\s+[^>]*href=[\'"](https?://niyomiyabarta\.com/[^\'"#]+/)[\'"]', html)
                if not found:
                    break
                valid = [l for l in found if not any(k in l for k in ("/category/", "/page/", "/tag/", "/author/", "/feed/", "/wp-json/"))]
                article_links.extend(valid)
                if len(article_links) >= 30000:
                    break

        article_links = list(dict.fromkeys(article_links))
        print(f"[AssameseCrawler] Discovered {len(article_links)} Niyomiya Barta article URLs.", flush=True)
        self._crawl_url_batches("niyomiya_barta", article_links)
        print(f"[AssameseCrawler] Niyomiya Barta Deep crawl complete. Saved docs: {self.stats.get('niyomiya_barta', {}).get('docs', 0)}", flush=True)

    def crawl_nenow_deep(self, start_page: int = 1, max_pages_per_cat: int = 800) -> None:
        """Crawl Northeast Now Assamese via deep category pagination."""
        print(f"\n[AssameseCrawler] Starting Northeast Now Deep Category crawler (pages {start_page}-{max_pages_per_cat})...", flush=True)
        categories = ["assam", "editorial", "sports", "literature-culture", "business", "lifestyle", "entertainment", "north-east-india"]
        article_links: list[str] = []

        for cat in categories:
            if self._is_limit_reached():
                break
            print(f"  + Scanning category '{cat}'...", flush=True)
            for p in range(start_page, max_pages_per_cat + 1):
                cat_url = f"https://assam.nenow.in/category/{cat}/page/{p}/"
                raw = fetch_url(cat_url, timeout=10.0)
                if not raw:
                    break
                html = raw.decode("utf-8", errors="replace")
                found = re.findall(r'<h[23][^>]*>\s*<a\s+[^>]*href=[\'"](https?://assam\.nenow\.in/[^\'"#]+/)[\'"]', html)
                if not found:
                    break
                valid = [l for l in found if not any(k in l for k in ("/category/", "/page/", "/tag/", "/author/", "/feed/", "/wp-json/"))]
                article_links.extend(valid)
                if len(article_links) >= 30000:
                    break

        article_links = list(dict.fromkeys(article_links))
        print(f"[AssameseCrawler] Discovered {len(article_links)} NENow article URLs.", flush=True)
        self._crawl_url_batches("nenow_asm", article_links)
        print(f"[AssameseCrawler] NENow Deep crawl complete. Saved docs: {self.stats.get('nenow_asm', {}).get('docs', 0)}", flush=True)

    def crawl_asomiya_pratidin_deep(self, start_sitemap: int = 1, max_sitemaps: int = 1500) -> None:
        """Crawl Asomiya Pratidin news portal via deep sitemaps (sitemaps 1-1500)."""
        print(f"\n[AssameseCrawler] Starting Asomiya Pratidin Deep crawler (sitemaps {start_sitemap}-{max_sitemaps})...", flush=True)
        sitemap_url = "https://www.asomiyapratidin.in/sitemap.xml"
        raw_index = fetch_url(sitemap_url, timeout=12.0)
        if not raw_index:
            print("[AssameseCrawler] Failed to fetch Asomiya Pratidin sitemap index.")
            return

        index_text = raw_index.decode("utf-8", errors="replace")
        sub_sitemaps = re.findall(r"<loc>\s*(https?://www\.asomiyapratidin\.in/[^\s<]+)\s*</loc>", index_text, flags=re.IGNORECASE)
        post_sitemaps = [s for s in sub_sitemaps if any(k in s for k in ("post", "article", "news", "sitemap-news", "homepage"))]
        target_sitemaps = post_sitemaps if post_sitemaps else sub_sitemaps

        print(f"[AssameseCrawler] Found {len(target_sitemaps)} Asomiya Pratidin sitemaps. Processing slice [{start_sitemap}:{max_sitemaps}]...", flush=True)

        for sm in target_sitemaps[start_sitemap:max_sitemaps]:
            if self._is_limit_reached():
                break
            urls = parse_sitemap_urls(sm, max_urls=300)
            if not urls:
                continue
            self._crawl_url_batches("asomiya_pratidin", urls)

        print(f"[AssameseCrawler] Asomiya Pratidin Deep crawl complete. Saved docs: {self.stats.get('asomiya_pratidin', {}).get('docs', 0)}", flush=True)

    def crawl_news18_assam(self, max_articles: int = 25000) -> None:
        """Crawl News18 Assam edition news, state, and culture sections."""
        print("\n[AssameseCrawler] Starting News18 Assam crawler...", flush=True)
        sitemaps = [
            "https://assam.news18.com/sitemap.xml",
            "https://assam.news18.com/sitemap-news.xml",
            "https://assam.news18.com/commonfeeds/v1/asm/rss/latest.xml",
        ]
        article_urls: list[str] = []
        for sm in sitemaps:
            urls = parse_sitemap_urls(sm, max_urls=max_articles)
            article_urls.extend(urls)
            if len(article_urls) >= max_articles:
                break

        # Also probe section pages
        sections = ["news/assam", "news/state", "news/nation", "news/sports", "photogallery/assam"]
        for sec in sections:
            sec_raw = fetch_url(f"https://assam.news18.com/{sec}/", timeout=10.0)
            if sec_raw:
                html = sec_raw.decode("utf-8", errors="replace")
                found = re.findall(r'href=[\'"](https?://assam\.news18\.com/[^\'"#\s]+\.html)[\'"]', html)
                article_urls.extend(found)

        article_urls = list(dict.fromkeys(article_urls))[:max_articles]
        print(f"[AssameseCrawler] Discovered {len(article_urls)} News18 Assam URLs to crawl.", flush=True)
        self._crawl_url_batches("news18_asm", article_urls)
        print(f"[AssameseCrawler] News18 Assam crawl complete. Saved docs: {self.stats.get('news18_asm', {}).get('docs', 0)}", flush=True)

    def crawl_vikaspedia_assamese(self, max_articles: int = 20000) -> None:
        """Crawl Vikaspedia Assamese portal via sitemaps across all domains."""
        print("\n[AssameseCrawler] Starting Vikaspedia Assamese crawler...", flush=True)
        sitemap_urls = [
            "https://as.vikaspedia.in/sitemap.xml",
            "https://as.vikaspedia.in/agriculture/sitemap.xml",
            "https://as.vikaspedia.in/health/sitemap.xml",
            "https://as.vikaspedia.in/education/sitemap.xml",
            "https://as.vikaspedia.in/social-welfare/sitemap.xml",
            "https://as.vikaspedia.in/energy/sitemap.xml",
            "https://as.vikaspedia.in/e-governance/sitemap.xml",
        ]

        article_urls: list[str] = []
        for sm in sitemap_urls:
            urls = parse_sitemap_urls(sm, max_urls=max_articles)
            article_urls.extend(urls)
            if len(article_urls) >= max_articles:
                break

        article_urls = list(dict.fromkeys(article_urls))[:max_articles]
        print(f"[AssameseCrawler] Discovered {len(article_urls)} Vikaspedia Assamese URLs to crawl.", flush=True)
        self._crawl_url_batches("vikaspedia_as", article_urls)
        print(f"[AssameseCrawler] Vikaspedia Assamese crawl complete. Saved docs: {self.stats.get('vikaspedia_as', {}).get('docs', 0)}", flush=True)

    def crawl_dainik_agradoot(self, max_articles: int = 20000) -> None:
        """Crawl Dainik Agradoot Assamese newspaper archive."""
        print("\n[AssameseCrawler] Starting Dainik Agradoot crawler...", flush=True)
        sitemap_urls = [
            "https://dainikagradoot.in/sitemap.xml",
            "https://dainikagradoot.in/sitemap_index.xml",
        ]
        article_urls: list[str] = []
        for sm in sitemap_urls:
            urls = parse_sitemap_urls(sm, max_urls=max_articles)
            article_urls.extend(urls)

        article_urls = list(dict.fromkeys(article_urls))[:max_articles]
        print(f"[AssameseCrawler] Discovered {len(article_urls)} Dainik Agradoot URLs to crawl.", flush=True)
        self._crawl_url_batches("agradoot", article_urls)
        print(f"[AssameseCrawler] Dainik Agradoot crawl complete. Saved docs: {self.stats.get('agradoot', {}).get('docs', 0)}", flush=True)

    def crawl_asomiya_pratidin_deep(self, max_articles: int = 30000) -> None:
        """Crawl Asomiya Pratidin (authentic Assamese newspaper) via daily sitemap archive."""
        print("\n[AssameseCrawler] Starting Asomiya Pratidin Deep crawler...", flush=True)
        main_sitemap = "https://www.asomiyapratidin.in/sitemap.xml"
        raw = fetch_url(main_sitemap, timeout=12.0)
        daily_sitemaps: list[str] = []
        if raw:
            text = raw.decode("utf-8", errors="replace")
            daily_sitemaps = re.findall(r"<loc>\s*(https?://[^\s<]+sitemap_\d{4}-\d{2}-\d{2}\.xml)\s*</loc>", text, flags=re.IGNORECASE)

        article_urls: list[str] = []
        print(f"[AssameseCrawler] Found {len(daily_sitemaps)} daily sitemaps on Asomiya Pratidin.", flush=True)
        # Scan recent daily sitemaps
        for sm in daily_sitemaps[:120]:
            if len(article_urls) >= max_articles:
                break
            sm_raw = fetch_url(sm, timeout=10.0)
            if sm_raw:
                sm_text = sm_raw.decode("utf-8", errors="replace")
                locs = re.findall(r"<loc>\s*(https?://www\.asomiyapratidin\.in/[^\s<]+-\d+)\s*</loc>", sm_text, flags=re.IGNORECASE)
                article_urls.extend(locs)

        article_urls = list(dict.fromkeys(article_urls))[:max_articles]
        print(f"[AssameseCrawler] Discovered {len(article_urls)} Asomiya Pratidin Assamese URLs to crawl.", flush=True)
        self._crawl_url_batches("asomiya_pratidin", article_urls)
        print(f"[AssameseCrawler] Asomiya Pratidin crawl complete. Saved docs: {self.stats.get('asomiya_pratidin', {}).get('docs', 0)}", flush=True)

    def crawl_nenow_deep(self, max_articles: int = 30000) -> None:
        """Crawl Northeast Now Assamese edition (assam.nenow.in) via deep sitemaps."""
        print("\n[AssameseCrawler] Starting Northeast Now Assamese crawler...", flush=True)
        index_url = "https://assam.nenow.in/sitemap_index.xml"
        raw = fetch_url(index_url, timeout=12.0)
        sub_sitemaps: list[str] = []
        if raw:
            text = raw.decode("utf-8", errors="replace")
            sub_sitemaps = re.findall(r"<loc>\s*(https?://assam\.nenow\.in/post-sitemap\d*\.xml)\s*</loc>", text, flags=re.IGNORECASE)

        article_urls: list[str] = []
        print(f"[AssameseCrawler] Found {len(sub_sitemaps)} post sitemaps on assam.nenow.in.", flush=True)
        for sm in sub_sitemaps[:35]:
            if len(article_urls) >= max_articles:
                break
            urls = parse_sitemap_urls(sm, max_urls=1000)
            article_urls.extend(urls)

        article_urls = list(dict.fromkeys(article_urls))[:max_articles]
        print(f"[AssameseCrawler] Discovered {len(article_urls)} Northeast Now Assamese URLs.", flush=True)
        self._crawl_url_batches("nenow_asm", article_urls)
        print(f"[AssameseCrawler] Northeast Now Assamese complete. Saved docs: {self.stats.get('nenow_asm', {}).get('docs', 0)}", flush=True)

    def crawl_eastmojo_assamese(self, max_articles: int = 20000) -> None:
        """Crawl EastMojo Assamese edition via sitemaps."""
        print("\n[AssameseCrawler] Starting EastMojo Assamese crawler...", flush=True)
        sitemap_urls = [
            "https://assam.eastmojo.com/sitemap_index.xml",
            "https://assam.eastmojo.com/post-sitemap.xml",
            "https://assam.eastmojo.com/sitemap.xml",
        ]
        article_urls: list[str] = []
        for sm in sitemap_urls:
            urls = parse_sitemap_urls(sm, max_urls=max_articles)
            article_urls.extend(urls)
            if len(article_urls) >= max_articles:
                break

        article_urls = list(dict.fromkeys(article_urls))[:max_articles]
        print(f"[AssameseCrawler] Discovered {len(article_urls)} EastMojo Assamese URLs.", flush=True)
        self._crawl_url_batches("eastmojo_asm", article_urls)
        print(f"[AssameseCrawler] EastMojo Assamese complete. Saved docs: {self.stats.get('eastmojo_asm', {}).get('docs', 0)}", flush=True)

    def crawl_etvbharat_assamese(self, max_articles: int = 20000) -> None:
        """Crawl ETV Bharat Assamese section."""
        print("\n[AssameseCrawler] Starting ETV Bharat Assamese crawler...", flush=True)
        sitemap_urls = [
            "https://www.etvbharat.com/assamese/assam/sitemap.xml",
            "https://www.etvbharat.com/assamese/assam/sitemap-news.xml",
        ]
        article_urls: list[str] = []
        for sm in sitemap_urls:
            urls = parse_sitemap_urls(sm, max_urls=max_articles)
            article_urls.extend(urls)

        article_urls = list(dict.fromkeys(article_urls))[:max_articles]
        print(f"[AssameseCrawler] Discovered {len(article_urls)} ETV Bharat Assamese URLs.", flush=True)
        self._crawl_url_batches("etvbharat_asm", article_urls)
        print(f"[AssameseCrawler] ETV Bharat Assamese complete. Saved docs: {self.stats.get('etvbharat_asm', {}).get('docs', 0)}", flush=True)

    def crawl_xahitya_literary(self, max_articles: int = 15000) -> None:
        """Crawl Xahitya.org monthly Assamese literary webzine."""
        print("\n[AssameseCrawler] Starting Xahitya.org Literary Archive crawler...", flush=True)
        sitemap_urls = [
            "https://xahitya.org/sitemap.xml",
            "https://xahitya.org/sitemap_index.xml",
            "https://xahitya.org/post-sitemap.xml",
        ]
        article_urls: list[str] = []
        for sm in sitemap_urls:
            urls = parse_sitemap_urls(sm, max_urls=max_articles)
            article_urls.extend(urls)

        article_urls = list(dict.fromkeys(article_urls))[:max_articles]
        print(f"[AssameseCrawler] Discovered {len(article_urls)} Xahitya.org URLs.", flush=True)
        self._crawl_url_batches("xahitya_literary", article_urls)
        print(f"[AssameseCrawler] Xahitya.org complete. Saved docs: {self.stats.get('xahitya_literary', {}).get('docs', 0)}", flush=True)

    def run(self, sources: Optional[Sequence[str]] = None) -> dict:
        """Run all enabled crawlers in sequence until target tokens/limits are satisfied."""
        print("=" * 70, flush=True)
        print("Assamese Web Crawler Pipeline (20% Manual Effort, Target: 100M Tokens)", flush=True)
        print(f"Target Tokens: {self.target_tokens:,} | Output Dir: {self.out_dir}", flush=True)
        print("=" * 70, flush=True)

        available_sources = {
            "asomiya_pratidin": self.crawl_asomiya_pratidin_deep,
            "asomiya_pratidin_deep": self.crawl_asomiya_pratidin_deep,
            "pratidin": self.crawl_asomiya_pratidin_deep,
            "nenow": self.crawl_nenow_deep,
            "nenow_deep": self.crawl_nenow_deep,
            "eastmojo": self.crawl_eastmojo_assamese,
            "eastmojo_asm": self.crawl_eastmojo_assamese,
            "etvbharat": self.crawl_etvbharat_assamese,
            "etvbharat_asm": self.crawl_etvbharat_assamese,
            "xahitya": self.crawl_xahitya_literary,
            "xahitya_literary": self.crawl_xahitya_literary,
            "cc100_as": self.crawl_cc100_assamese,
            "niyomiya_barta_deep": self.crawl_niyomiya_barta_deep,
            "news18_asm": self.crawl_news18_assam,
            "vikaspedia_as": self.crawl_vikaspedia_assamese,
            "agradoot": self.crawl_dainik_agradoot,
        }

        to_run = list(sources) if sources else list(available_sources.keys())
        for s_name in to_run:
            if self._is_limit_reached():
                print(f"[*] Limit reached ({self.total_tokens_collected:,} tokens). Stopping crawler.")
                break
            crawler_fn = available_sources.get(s_name)
            if crawler_fn:
                try:
                    crawler_fn()
                except Exception as exc:
                    print(f"[!] Source {s_name} encountered an error: {exc}", flush=True)

        print("\n" + "=" * 70, flush=True)
        print("Assamese Web Crawling Summary:", flush=True)
        print(f"  * Total Documents Collected:  {self.total_docs_collected:,}", flush=True)
        print(f"  * Total Estimated Tokens:     {self.total_tokens_collected:,}", flush=True)
        print(f"  * Total Raw Text Size:        {self.total_bytes_collected / (1024 * 1024):.2f} MB", flush=True)
        for s, info in self.stats.items():
            print(f"    - {s:<24}: {info['docs']:>6} docs | {info['tokens']:>10,} tokens | {info['bytes'] / (1024*1024):>6.2f} MB", flush=True)
        print("=" * 70, flush=True)

        return {
            "lang": "assamese",
            "total_docs": self.total_docs_collected,
            "total_tokens": self.total_tokens_collected,
            "total_bytes": self.total_bytes_collected,
            "sources": self.stats,
        }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Assamese Web Crawler for Manual Corpus")
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "assamese" / "data" / "clean"), help="Output directory for clean JSONL files")
    parser.add_argument("--target-tokens", type=int, default=100_000_000, help="Target total tokens to collect (default: 100M)")
    parser.add_argument("--max-docs", type=int, default=None, help="Cap maximum docs to collect")
    parser.add_argument("--max-bytes", type=int, default=None, help="Cap maximum raw bytes to collect")
    parser.add_argument("--workers", type=int, default=12, help="Number of concurrent worker threads")
    parser.add_argument("--sources", nargs="+", help="Specific sources to crawl")
    parser.add_argument("--prev-dirs", nargs="*", default=[], help="Directories with existing JSONL files to deduplicate against")
    args = parser.parse_args(argv)

    crawler = AssameseCrawler(
        out_dir=Path(args.out_dir),
        target_tokens=args.target_tokens,
        max_docs=args.max_docs,
        max_bytes=args.max_bytes,
        workers=args.workers,
        prev_dirs=[Path(p) for p in args.prev_dirs],
    )
    crawler.run(sources=args.sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
