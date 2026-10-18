"""How much of each reasoning eval split can be answered from the training split alone.

usage: python -m common.leakage_check --data-dir <dir with train.jsonl and eval splits> --out leakage.json

Per eval split:
  name_overlap         eval entity names that also occur in train (should be 0)
  exact_text_dup       eval prompts that occur verbatim in train (should be 0)
  structure_seen       share of eval examples whose structure (premise graph with names
                       relabelled by first appearance, query, attribute) occurs in train
  structure_lookup_acc accuracy of answering with the majority train answer for that
                       structure (the train-majority relation when unseen): a model that
                       memorised structures, without reasoning, would score this
  majority_acc         always answering the train-majority relation
A high structure_lookup_acc means a split rewards memorising structures; only a model that
beats it on splits where it is near chance gives evidence of reasoning.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def relation(ex: dict) -> str:
    return ex["answer_rel"].split()[1]


def structure(ex: dict) -> tuple:
    """Premises + query + attribute with entity names relabelled by first appearance."""
    order = list(dict.fromkeys([x for p in ex["premises"] for x in (p[0], p[2])] + list(ex["query"])))
    ids = {e: i for i, e in enumerate(order)}
    return (tuple((ids[a], r, ids[b]) for a, r, b in ex["premises"]),
            tuple(ids[q] for q in ex["query"]), ex["attribute"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    data = Path(args.data_dir)
    train = load(data / "train.jsonl")
    names = {n for ex in train for n in ex["entities"]}
    texts = {ex["text"] for ex in train}
    answers = collections.defaultdict(collections.Counter)
    for ex in train:
        answers[structure(ex)][relation(ex)] += 1
    majority = collections.Counter(relation(ex) for ex in train).most_common(1)[0][0]

    report = {"train_examples": len(train), "train_names": len(names),
              "train_structures": len(answers), "train_majority_relation": majority, "splits": {}}
    for path in sorted(data.glob("*.jsonl")):
        if path.stem == "train":
            continue
        ev = load(path)
        seen = [structure(ex) in answers for ex in ev]
        lookup = [(answers[structure(ex)].most_common(1)[0][0] if s else majority) == relation(ex)
                  for ex, s in zip(ev, seen)]
        report["splits"][path.stem] = {
            "n": len(ev),
            "name_overlap": len({n for ex in ev for n in ex["entities"]} & names),
            "exact_text_dup": sum(ex["text"] in texts for ex in ev),
            "structure_seen": round(sum(seen) / len(ev), 4),
            "structure_lookup_acc": round(sum(lookup) / len(ev), 4),
            "majority_acc": round(sum(relation(ex) == majority for ex in ev) / len(ev), 4),
        }
        print(f"{path.stem:22s} " + " ".join(f"{k}={v}" for k, v in report["splits"][path.stem].items()), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
