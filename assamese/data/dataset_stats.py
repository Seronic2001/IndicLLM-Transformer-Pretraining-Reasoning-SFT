"""Dataset statistics — the locked dataset_stats.json schema (Agent-A).

Schema (AGENT_BUILD_SPEC §3 Agent-A):
  {
    "total_tokens_estimate": int,
    "manual_tokens": int,
    "downloaded_tokens": int,
    "manual_fraction": float,
    "sources": [{"name", "type", "doc_count", "token_estimate"}],
    "dedup_removed_docs": int,
    "script_filter_removed_chars": int,
    "failed_sources": [...]
  }

Every number traces to an artifact on disk — the report (Agent-I) reads this JSON,
it never re-types numbers (spec §5.3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

TOKENS_PER_WORD = 1.2  # BPE-ish estimate; documented in the report, not a claim


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate: whitespace words * TOKENS_PER_WORD."""
    return max(1, round(len(text.split()) * TOKENS_PER_WORD))


def compute_stats(
    clean_dir: str,
    failed_sources: Optional[list[str]] = None,
    dedup_removed_docs: int = 0,
    script_filter_removed_chars: int = 0,
) -> dict:
    """Aggregate stats from the clean/<source>.jsonl records."""
    sources = []
    manual_tokens = 0
    downloaded_tokens = 0

    for path in sorted(Path(clean_dir).glob("*.jsonl")):
        source_name = path.stem
        doc_count = 0
        tokens = 0
        types: set[str] = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = json.loads(line)
                    doc_count += 1
                    tokens += estimate_tokens(doc.get("text", ""))
                    raw_t = doc.get("source_type") or doc.get("type")
                    if raw_t:
                        types.add(raw_t)

        name_lower = source_name.lower()
        manual_keywords = (
            "crawler", "kavitakosh", "gadyakosh", "amarujala", "bbc", "nenow",
            "pratidin", "agradoot", "xahitya", "wiki", "wikipedia", "ncert",
            "scert", "seba", "dump", "niyomiya", "cc100", "jansatta",
            "prabhatkhabar", "news18", "ndtv", "webdunia", "vikaspedia",
            "sangraha_manual", "literature"
        )
        stype = "manual" if ("manual" in types or any(k in name_lower for k in manual_keywords)) else "downloaded"
        if stype == "manual":
            manual_tokens += tokens
        else:
            downloaded_tokens += tokens

        sources.append({
            "name": source_name,
            "type": stype,
            "doc_count": doc_count,
            "token_estimate": tokens,
        })

    total = manual_tokens + downloaded_tokens
    return {
        "total_tokens_estimate": total,
        "manual_tokens": manual_tokens,
        "downloaded_tokens": downloaded_tokens,
        "manual_fraction": round(manual_tokens / total, 6) if total else 0.0,
        "sources": sources,
        "dedup_removed_docs": int(dedup_removed_docs),
        "script_filter_removed_chars": int(script_filter_removed_chars),
        "failed_sources": sorted(failed_sources or []),
        "token_estimator": f"words * {TOKENS_PER_WORD} (BPE-ish, documented in report)",
    }


def write_report(stats: dict, out_path: str) -> None:
    """Render dataset_stats.json as the phase-1 markdown tables (report/phase1/)."""
    lines = ["# Dataset statistics", ""]
    lines.append(f"- Total tokens (estimate): **{stats['total_tokens_estimate']:,}**")
    lines.append(f"- Manual tokens: **{stats['manual_tokens']:,}** "
                 f"({stats['manual_fraction'] * 100:.2f}%)")
    lines.append(f"- Downloaded tokens: **{stats['downloaded_tokens']:,}**")
    lines.append(f"- Dedup removed docs: {stats['dedup_removed_docs']}")
    lines.append(f"- Script-filter removed chars: {stats['script_filter_removed_chars']}")
    lines.append(f"- Failed sources: {', '.join(stats['failed_sources']) or 'none'}")
    lines.append("")
    lines.append("| Source | Type | Docs | Token estimate |")
    lines.append("|---|---|---|---|")
    for s in stats["sources"]:
        lines.append(
            f"| {s['name']} | {s['type']} | {s['doc_count']} | {s['token_estimate']:,} |"
        )
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compute dataset_stats.json (Agent-A)")
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--report-md", default=None)
    parser.add_argument("--dedup-removed", type=int, default=0)
    parser.add_argument("--script-filtered", type=int, default=0)
    parser.add_argument("--failed-sources", nargs="*", default=[])
    args = parser.parse_args(argv)
    stats = compute_stats(
        args.clean_dir,
        failed_sources=args.failed_sources,
        dedup_removed_docs=args.dedup_removed,
        script_filter_removed_chars=args.script_filtered,
    )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.report_md:
        write_report(stats, args.report_md)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
