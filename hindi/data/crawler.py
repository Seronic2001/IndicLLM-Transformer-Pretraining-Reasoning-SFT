"""Hindi Web Crawler for Manual Corpus (pipeline/ 20% Manual Effort).

Target: Collect authentic, high-quality Hindi text (~100M tokens, ~2-3 GB)
strictly avoiding Wikipedia (already collected in downloaded/manual dumps).

Sources Crawled:
  1. CC-100 Hindi Web Corpus (statmt.org/cc-100) - Clean web-crawled Hindi text
  2. Hindi Wikisource (hi.wikisource.org) - Out-of-copyright literature, prose, epics
  3. Jansatta (jansatta.com) - National journalism, editorials, and literature
  4. Prabhat Khabar (prabhatkhabar.com) - Leading Hindi daily and cultural reporting
  5. News18 Hindi (hindi.news18.com) - Comprehensive daily reporting & features
  6. NDTV Hindi (ndtv.in) - National news & editorials
  7. Amar Ujala Deep Archives (amarujala.com) - Extensive archive sitemaps
  8. Webdunia Hindi (hindi.webdunia.com) - Literature & daily news
  9. Vikaspedia Hindi (hi.vikaspedia.in) - Government knowledge encyclopedia
  10. Gadya Kosh (gadyakosh.org) - Extended prose archives
  11. Kavita Kosh (kavitakosh.org) - Extended poetry archives
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
    clean_wikitext,
    compute_script_fraction,
    estimate_tokens,
    fetch_json,
    fetch_url,
    parse_sitemap_urls,
)

LANG = "hindi"
MIN_SCRIPT_FRACTION = 0.65
MIN_DOC_CHARS = 120
BATCH_SIZE = 32


class HindiCrawler:
    """Multi-source concurrent crawler for Hindi manual corpus."""

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
            print(f"[HindiCrawler] Pre-loaded {len(self.seen_hashes):,} existing doc hashes and {len(self.seen_urls):,} URLs for deduplication.", flush=True)

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

        doc_record = {
            "doc_id": doc_id,
            "url": url,
            "source": f"crawler_{source_name}",
            "type": "manual",
            "hash": text_hash,
            "token_estimate": tokens,
            "char_count": len(text),
            "script_fraction": round(purity, 4),
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

    def crawl_cc100_hindi(self, max_docs: int = 150000) -> None:
        """Stream clean web-crawled Hindi corpus from CC-100 repository."""
        print("\n[HindiCrawler] Starting CC-100 Hindi Web Corpus stream...", flush=True)
        url = "https://data.statmt.org/cc-100/hi.txt.xz"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                decompressor = lzma.LZMADecompressor()
                buffer = ""
                doc_idx = 0
                while not self._is_limit_reached() and doc_idx < max_docs:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    try:
                        decompressed = decompressor.decompress(chunk)
                        buffer += decompressed.decode("utf-8", errors="ignore")
                    except Exception:
                        continue

                    while "\n\n" in buffer:
                        doc_text, buffer = buffer.split("\n\n", 1)
                        doc_text = doc_text.strip()
                        if len(doc_text) >= MIN_DOC_CHARS:
                            doc_id = f"cc100_hi/doc_{doc_idx}"
                            self._save_doc("cc100_hi", doc_id, doc_text, "")
                            doc_idx += 1
                            if self._is_limit_reached() or doc_idx >= max_docs:
                                break
        except Exception as exc:
            print(f"[HindiCrawler] CC-100 Hindi stream error: {exc}", flush=True)

        print(f"[HindiCrawler] CC-100 Hindi complete. Saved docs: {self.stats.get('cc100_hi', {}).get('docs', 0)}", flush=True)

    def crawl_hi_wikisource(self, max_pages: int = 25000) -> None:
        """Crawl Hindi Wikisource out-of-copyright literature, prose, and poetry."""
        print("\n[HindiCrawler] Starting Hindi Wikisource crawler...", flush=True)
        base_api = "https://hi.wikisource.org/w/api.php"
        apcontinue = None
        pages_processed = 0

        while not self._is_limit_reached() and pages_processed < max_pages:
            params = {
                "action": "query",
                "list": "allpages",
                "aplimit": "50",
                "format": "json",
            }
            if apcontinue:
                params["apcontinue"] = apcontinue

            url = f"{base_api}?{urllib.parse.urlencode(params)}"
            data = fetch_json(url, timeout=12.0)
            if not data:
                break

            pages = data.get("query", {}).get("allpages", [])
            if not pages:
                break

            def _fetch_page(page_meta):
                pid = page_meta["pageid"]
                title = page_meta["title"]
                c_url = f"{base_api}?action=query&prop=revisions&rvprop=content&pageids={pid}&format=json"
                c_data = fetch_json(c_url, timeout=10.0)
                if not c_data:
                    return None
                revs = c_data.get("query", {}).get("pages", {}).get(str(pid), {}).get("revisions", [])
                if revs:
                    raw_wiki = revs[0].get("*", "")
                    cleaned = clean_wikitext(raw_wiki)
                    return (title, cleaned, f"https://hi.wikisource.org/wiki/{urllib.parse.quote(title)}")
                return None

            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [pool.submit(_fetch_page, p) for p in pages]
                for fut in as_completed(futures):
                    res = fut.result()
                    if res:
                        title, text, page_url = res
                        if text:
                            self._save_doc("wikisource_hi", f"wikisource/{title}", text, page_url)
                            pages_processed += 1
                            if self._is_limit_reached():
                                break

            if "continue" in data and "apcontinue" in data["continue"]:
                apcontinue = data["continue"]["apcontinue"]
            else:
                break

        print(f"[HindiCrawler] Hindi Wikisource crawl complete. Saved docs: {self.stats.get('wikisource_hi', {}).get('docs', 0)}", flush=True)

    def crawl_jansatta(self, max_articles: int = 25000) -> None:
        """Crawl Jansatta news, editorials, and literature sections."""
        print("\n[HindiCrawler] Starting Jansatta crawler...", flush=True)
        sitemaps = [
            "https://www.jansatta.com/sitemap.xml",
            "https://www.jansatta.com/sitemap/sitemap-index.xml",
        ]
        article_urls: list[str] = []
        for sm in sitemaps:
            urls = parse_sitemap_urls(sm, max_urls=max_articles)
            article_urls.extend(urls)
            if len(article_urls) >= max_articles:
                break

        article_urls = list(dict.fromkeys(article_urls))[:max_articles]
        print(f"[HindiCrawler] Discovered {len(article_urls)} Jansatta URLs to crawl.", flush=True)
        self._crawl_url_batches("jansatta", article_urls)
        print(f"[HindiCrawler] Jansatta crawl complete. Saved docs: {self.stats.get('jansatta', {}).get('docs', 0)}", flush=True)

    def crawl_prabhatkhabar(self, max_articles: int = 25000) -> None:
        """Crawl Prabhat Khabar news portal and features."""
        print("\n[HindiCrawler] Starting Prabhat Khabar crawler...", flush=True)
        sitemaps = [
            "https://www.prabhatkhabar.com/sitemap/sitemap-index.xml",
            "https://www.prabhatkhabar.com/sitemap.xml",
        ]
        article_urls: list[str] = []
        for sm in sitemaps:
            urls = parse_sitemap_urls(sm, max_urls=max_articles)
            article_urls.extend(urls)
            if len(article_urls) >= max_articles:
                break

        article_urls = list(dict.fromkeys(article_urls))[:max_articles]
        print(f"[HindiCrawler] Discovered {len(article_urls)} Prabhat Khabar URLs to crawl.", flush=True)
        self._crawl_url_batches("prabhatkhabar", article_urls)
        print(f"[HindiCrawler] Prabhat Khabar crawl complete. Saved docs: {self.stats.get('prabhatkhabar', {}).get('docs', 0)}", flush=True)

    def crawl_news18_hindi(self, max_articles: int = 25000) -> None:
        """Crawl News18 Hindi news and cultural features."""
        print("\n[HindiCrawler] Starting News18 Hindi crawler...", flush=True)
        sitemaps = [
            "https://hindi.news18.com/sitemap.xml",
            "https://hindi.news18.com/sitemap-news.xml",
        ]
        article_urls: list[str] = []
        for sm in sitemaps:
            urls = parse_sitemap_urls(sm, max_urls=max_articles)
            article_urls.extend(urls)
            if len(article_urls) >= max_articles:
                break

        article_urls = list(dict.fromkeys(article_urls))[:max_articles]
        print(f"[HindiCrawler] Discovered {len(article_urls)} News18 Hindi URLs to crawl.", flush=True)
        self._crawl_url_batches("news18_hi", article_urls)
        print(f"[HindiCrawler] News18 Hindi crawl complete. Saved docs: {self.stats.get('news18_hi', {}).get('docs', 0)}", flush=True)

    def crawl_ndtv_hindi(self, max_articles: int = 20000) -> None:
        """Crawl NDTV Hindi news and features."""
        print("\n[HindiCrawler] Starting NDTV Hindi crawler...", flush=True)
        sitemaps = [
            "https://ndtv.in/sitemap_index.xml",
            "https://khabar.ndtv.com/sitemap.xml",
        ]
        article_urls: list[str] = []
        for sm in sitemaps:
            urls = parse_sitemap_urls(sm, max_urls=max_articles)
            article_urls.extend(urls)
            if len(article_urls) >= max_articles:
                break

        article_urls = list(dict.fromkeys(article_urls))[:max_articles]
        print(f"[HindiCrawler] Discovered {len(article_urls)} NDTV Hindi URLs to crawl.", flush=True)
        self._crawl_url_batches("ndtv_hi", article_urls)
        print(f"[HindiCrawler] NDTV Hindi crawl complete. Saved docs: {self.stats.get('ndtv_hi', {}).get('docs', 0)}", flush=True)

    def crawl_webdunia_hindi(self, max_articles: int = 20000) -> None:
        """Crawl Webdunia Hindi news and literature."""
        print("\n[HindiCrawler] Starting Webdunia Hindi crawler...", flush=True)
        sitemaps = [
            "https://hindi.webdunia.com/sitemap.xml",
            "https://hindi.webdunia.com/rss/feed.xml",
        ]
        article_urls: list[str] = []
        for sm in sitemaps:
            urls = parse_sitemap_urls(sm, max_urls=max_articles)
            article_urls.extend(urls)
            if len(article_urls) >= max_articles:
                break

        article_urls = list(dict.fromkeys(article_urls))[:max_articles]
        print(f"[HindiCrawler] Discovered {len(article_urls)} Webdunia Hindi URLs to crawl.", flush=True)
        self._crawl_url_batches("webdunia_hi", article_urls)
        print(f"[HindiCrawler] Webdunia Hindi crawl complete. Saved docs: {self.stats.get('webdunia_hi', {}).get('docs', 0)}", flush=True)

    def crawl_amarujala_deep(self, max_articles: int = 25000) -> None:
        """Crawl Amar Ujala deep archive sitemaps."""
        print("\n[HindiCrawler] Starting Amar Ujala Deep Archive crawler...", flush=True)
        sitemaps = [
            "https://www.amarujala.com/sitemap.xml",
            "https://www.amarujala.com/rss/national-news.xml",
            "https://www.amarujala.com/rss/kavya.xml",
            "https://www.amarujala.com/rss/editorial.xml",
        ]
        article_urls: list[str] = []
        for sm in sitemaps:
            urls = parse_sitemap_urls(sm, max_urls=max_articles)
            article_urls.extend(urls)
            if len(article_urls) >= max_articles:
                break

        article_urls = list(dict.fromkeys(article_urls))[:max_articles]
        print(f"[HindiCrawler] Discovered {len(article_urls)} Amar Ujala Deep URLs to crawl.", flush=True)
        self._crawl_url_batches("amarujala", article_urls)
        print(f"[HindiCrawler] Amar Ujala Deep crawl complete. Saved docs: {self.stats.get('amarujala', {}).get('docs', 0)}", flush=True)

    def crawl_vikaspedia_hindi(self, max_articles: int = 20000) -> None:
        """Crawl Vikaspedia Hindi knowledge portal across all domains."""
        print("\n[HindiCrawler] Starting Vikaspedia Hindi crawler...", flush=True)
        sitemap_urls = [
            "https://hi.vikaspedia.in/sitemap.xml",
            "https://hi.vikaspedia.in/agriculture/sitemap.xml",
            "https://hi.vikaspedia.in/health/sitemap.xml",
            "https://hi.vikaspedia.in/education/sitemap.xml",
            "https://hi.vikaspedia.in/social-welfare/sitemap.xml",
            "https://hi.vikaspedia.in/energy/sitemap.xml",
            "https://hi.vikaspedia.in/e-governance/sitemap.xml",
        ]
        article_urls: list[str] = []
        for sm in sitemap_urls:
            urls = parse_sitemap_urls(sm, max_urls=max_articles)
            article_urls.extend(urls)
            if len(article_urls) >= max_articles:
                break

        article_urls = list(dict.fromkeys(article_urls))[:max_articles]
        print(f"[HindiCrawler] Discovered {len(article_urls)} Vikaspedia Hindi URLs to crawl.", flush=True)
        self._crawl_url_batches("vikaspedia_hi", article_urls)
        print(f"[HindiCrawler] Vikaspedia Hindi crawl complete. Saved docs: {self.stats.get('vikaspedia_hi', {}).get('docs', 0)}", flush=True)

    def crawl_gadyakosh(self, max_pages: int = 50000) -> None:
        """Crawl Hindi prose literature from Gadya Kosh via MediaWiki API."""
        print("\n[HindiCrawler] Starting Gadya Kosh crawler...", flush=True)
        base_api = "http://gadyakosh.org/gk/api.php"
        apcontinue = None
        pages_processed = 0

        while not self._is_limit_reached() and pages_processed < max_pages:
            params = {
                "action": "query",
                "list": "allpages",
                "aplimit": "50",
                "format": "json",
            }
            if apcontinue:
                params["apcontinue"] = apcontinue

            url = f"{base_api}?{urllib.parse.urlencode(params)}"
            data = fetch_json(url, timeout=12.0)
            if not data:
                break

            pages = data.get("query", {}).get("allpages", [])
            if not pages:
                break

            def _fetch_page(page_meta):
                pid = page_meta["pageid"]
                title = page_meta["title"]
                c_url = f"{base_api}?action=query&prop=revisions&rvprop=content&pageids={pid}&format=json"
                c_data = fetch_json(c_url, timeout=10.0)
                if not c_data:
                    return None
                revs = c_data.get("query", {}).get("pages", {}).get(str(pid), {}).get("revisions", [])
                if revs:
                    raw_wiki = revs[0].get("*", "")
                    cleaned = clean_wikitext(raw_wiki)
                    return (title, cleaned, f"http://gadyakosh.org/gk/{urllib.parse.quote(title)}")
                return None

            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [pool.submit(_fetch_page, p) for p in pages]
                for fut in as_completed(futures):
                    res = fut.result()
                    if res:
                        title, text, page_url = res
                        if text:
                            self._save_doc("gadyakosh", f"gadyakosh/{title}", text, page_url)
                            pages_processed += 1
                            if self._is_limit_reached():
                                break

            if "continue" in data and "apcontinue" in data["continue"]:
                apcontinue = data["continue"]["apcontinue"]
            else:
                break

        print(f"[HindiCrawler] Gadya Kosh crawl complete. Saved docs: {self.stats.get('gadyakosh', {}).get('docs', 0)}", flush=True)

    def crawl_kavitakosh(self, max_pages: int = 50000) -> None:
        """Crawl Hindi poetry literature from Kavita Kosh via MediaWiki API."""
        print("\n[HindiCrawler] Starting Kavita Kosh crawler...", flush=True)
        base_api = "http://kavitakosh.org/kk/api.php"
        apcontinue = None
        pages_processed = 0

        while not self._is_limit_reached() and pages_processed < max_pages:
            params = {
                "action": "query",
                "list": "allpages",
                "aplimit": "50",
                "format": "json",
            }
            if apcontinue:
                params["apcontinue"] = apcontinue

            url = f"{base_api}?{urllib.parse.urlencode(params)}"
            data = fetch_json(url, timeout=12.0)
            if not data:
                break

            pages = data.get("query", {}).get("allpages", [])
            if not pages:
                break

            def _fetch_page(page_meta):
                pid = page_meta["pageid"]
                title = page_meta["title"]
                c_url = f"{base_api}?action=query&prop=revisions&rvprop=content&pageids={pid}&format=json"
                c_data = fetch_json(c_url, timeout=10.0)
                if not c_data:
                    return None
                revs = c_data.get("query", {}).get("pages", {}).get(str(pid), {}).get("revisions", [])
                if revs:
                    raw_wiki = revs[0].get("*", "")
                    cleaned = clean_wikitext(raw_wiki)
                    return (title, cleaned, f"http://kavitakosh.org/kk/{urllib.parse.quote(title)}")
                return None

            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [pool.submit(_fetch_page, p) for p in pages]
                for fut in as_completed(futures):
                    res = fut.result()
                    if res:
                        title, text, page_url = res
                        if text:
                            self._save_doc("kavitakosh", f"kavitakosh/{title}", text, page_url)
                            pages_processed += 1
                            if self._is_limit_reached():
                                break

            if "continue" in data and "apcontinue" in data["continue"]:
                apcontinue = data["continue"]["apcontinue"]
            else:
                break

        print(f"[HindiCrawler] Kavita Kosh crawl complete. Saved docs: {self.stats.get('kavitakosh', {}).get('docs', 0)}", flush=True)

    def run(self, sources: Optional[Sequence[str]] = None) -> dict:
        """Run all enabled crawlers in sequence until target tokens/limits are satisfied."""
        print("=" * 70, flush=True)
        print("Hindi Web Crawler Pipeline (20% Manual Effort, Target: 100M Tokens)", flush=True)
        print(f"Target Tokens: {self.target_tokens:,} | Output Dir: {self.out_dir}", flush=True)
        print("=" * 70, flush=True)

        available_sources = {
            "cc100_hi": self.crawl_cc100_hindi,
            "wikisource_hi": self.crawl_hi_wikisource,
            "jansatta": self.crawl_jansatta,
            "prabhatkhabar": self.crawl_prabhatkhabar,
            "news18_hi": self.crawl_news18_hindi,
            "ndtv_hi": self.crawl_ndtv_hindi,
            "webdunia_hi": self.crawl_webdunia_hindi,
            "amarujala": self.crawl_amarujala_deep,
            "vikaspedia_hi": self.crawl_vikaspedia_hindi,
            "gadyakosh": self.crawl_gadyakosh,
            "kavitakosh": self.crawl_kavitakosh,
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
        print("Hindi Web Crawling Summary:", flush=True)
        print(f"  * Total Documents Collected:  {self.total_docs_collected:,}", flush=True)
        print(f"  * Total Estimated Tokens:     {self.total_tokens_collected:,}", flush=True)
        print(f"  * Total Raw Text Size:        {self.total_bytes_collected / (1024 * 1024):.2f} MB", flush=True)
        for s, info in self.stats.items():
            print(f"    - {s:<18}: {info['docs']:>6} docs | {info['tokens']:>10,} tokens | {info['bytes'] / (1024*1024):>6.2f} MB", flush=True)
        print("=" * 70, flush=True)

        return {
            "lang": "hindi",
            "total_docs": self.total_docs_collected,
            "total_tokens": self.total_tokens_collected,
            "total_bytes": self.total_bytes_collected,
            "sources": self.stats,
        }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Hindi Web Crawler for Manual Corpus")
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "hindi" / "data" / "clean"), help="Output directory for clean JSONL files")
    parser.add_argument("--target-tokens", type=int, default=100_000_000, help="Target total tokens to collect (default: 100M)")
    parser.add_argument("--max-docs", type=int, default=None, help="Cap maximum docs to collect")
    parser.add_argument("--max-bytes", type=int, default=None, help="Cap maximum raw bytes to collect")
    parser.add_argument("--workers", type=int, default=12, help="Number of concurrent worker threads")
    parser.add_argument("--sources", nargs="+", help="Specific sources to crawl")
    parser.add_argument("--prev-dirs", nargs="*", default=[], help="Directories with existing JSONL files to deduplicate against")
    args = parser.parse_args(argv)

    crawler = HindiCrawler(
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
