"""Evaluation: data splits, leakage guards and metrics.

Run the split with `make split` (or `python -m raredx.evaluate split`).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
SPLITS = ("train", "val", "test")


def load_settings() -> dict:
    return yaml.safe_load((ROOT / "settings.yaml").read_text(encoding="utf-8"))


def load_cases() -> list[dict]:
    with open(PROCESSED / "cases.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def load_pub_dates() -> dict[str, str]:
    with open(PROCESSED / "pmid_dates.csv", newline="") as fh:
        return {r["pmid"]: r["pubdate"] for r in csv.DictReader(fh)}


# =============================================================================
# SPLIT BY PAPER
# All cases from the same paper go to the same split. Otherwise the model could
# be tested on patients from a paper it already saw while training (leakage).
# Example: the 462 patients of one cohort paper all land in train, or all in test.
# =============================================================================

def split_by_paper(cases: list[dict], fractions: dict[str, float], seed: int) -> dict[str, str]:
    """Return {pmid: split}. Papers are shuffled and filled into splits until each
    split reaches its share of cases (not of papers: papers have very different sizes)."""
    per_paper = Counter(c["pmid"] for c in cases)
    papers = sorted(per_paper)
    random.Random(seed).shuffle(papers)

    total = len(cases)
    targets = {s: fractions[s] * total for s in SPLITS}
    filled = {s: 0 for s in SPLITS}
    assignment = {}
    for pmid in papers:
        # put the paper where it is most needed relative to its target
        split = min(SPLITS, key=lambda s: filled[s] / targets[s])
        assignment[pmid] = split
        filled[split] += per_paper[pmid]
    return assignment


# =============================================================================
# SAMPLES
# Only a sample of each split is sent to the LLMs (each case = 4 paid LLM calls).
# At most `cap` cases per paper, so the sample covers many papers and diseases.
# =============================================================================

def sample_cases(cases: list[dict], n: int, cap: int, seed: int) -> list[str]:
    shuffled = sorted(cases, key=lambda c: c["case_id"])
    random.Random(seed).shuffle(shuffled)
    taken, per_paper = [], Counter()
    for c in shuffled:
        if per_paper[c["pmid"]] < cap:
            taken.append(c["case_id"])
            per_paper[c["pmid"]] += 1
            if len(taken) == n:
                break
    return sorted(taken)


def make_splits() -> dict:
    cfg = load_settings()
    ev = cfg["eval"]
    cases, dates = load_cases(), load_pub_dates()

    assignment = split_by_paper(cases, ev["split_fractions"], ev["seed"])
    by_split = defaultdict(list)
    for c in cases:
        by_split[assignment[c["pmid"]]].append(c)

    samples = {
        s: set(sample_cases(by_split[s], ev["sample_size"][s], ev["max_cases_per_paper"], ev["seed"]))
        for s in SPLITS
    }

    # Full assignment (in data/, not committed) ...
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED / "splits.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "pmid", "split", "in_sample"])
        for c in cases:
            s = assignment[c["pmid"]]
            w.writerow([c["case_id"], c["pmid"], s, int(c["case_id"] in samples[s])])

    # ... and the sampled case ids (committed, so the exact samples are versioned)
    (RESULTS / "samples").mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        (RESULTS / "samples" / f"{s}.txt").write_text("\n".join(sorted(samples[s])) + "\n", encoding="utf-8")

    # Summary
    train_diseases = {c["disease_id"] for c in by_split["train"]}
    cutoffs = {m["model"]: str(m["cutoff"]) for m in cfg["llm"]["ensemble"] if m.get("cutoff")}
    summary = {}
    for s in SPLITS:
        all_c = by_split[s]
        smp = [c for c in all_c if c["case_id"] in samples[s]]
        summary[s] = {
            "cases": len(all_c),
            "papers": len({c["pmid"] for c in all_c}),
            "diseases": len({c["disease_id"] for c in all_c}),
            "sample_cases": len(smp),
            "sample_papers": len({c["pmid"] for c in smp}),
            "sample_diseases": len({c["disease_id"] for c in smp}),
            "sample_diseases_not_in_train": len({c["disease_id"] for c in smp} - train_diseases) if s != "train" else None,
            "sample_cases_after_cutoff": {m: sum(dates[c["pmid"]] > cut for c in smp) for m, cut in cutoffs.items()},
        }
    (RESULTS / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


# =============================================================================
# METRICS
# For each case we know the position of the correct disease in the ranking.
#   top-k = share of cases where it is among the first k
#   MRR   = average of 1/position (1 if always first, 0.5 if always second...)
# 95% confidence intervals by bootstrap: resample the cases 1,000 times.
# =============================================================================

def metrics(ranks: list[float | None], seed: int = 42, n_boot: int = 1000) -> dict:
    import numpy as np
    r = np.array([x if x is not None else np.inf for x in ranks], dtype=float)

    def summary(v):
        return {
            "top1": float(np.mean(v <= 1)), "top5": float(np.mean(v <= 5)),
            "top10": float(np.mean(v <= 10)), "mrr": float(np.mean(1 / v)),
        }

    point = summary(r)
    rng = np.random.default_rng(seed)
    boots = [summary(r[rng.integers(0, len(r), len(r))]) for _ in range(n_boot)]
    out = {"cases": len(r), "unknown_disease": int(np.isinf(r).sum())}
    for k, v in point.items():
        lo, hi = np.percentile([b[k] for b in boots], [2.5, 97.5])
        out[k] = {"value": round(v, 3), "ci95": [round(float(lo), 3), round(float(hi), 3)]}
    return out


def sample_cases_of(split: str) -> list[dict]:
    ids = set((RESULTS / "samples" / f"{split}.txt").read_text().split())
    return [c for c in load_cases() if c["case_id"] in ids]


def eval_ranker(split: str) -> dict:
    """Ontology ranker alone, with and without leakage guard 1."""
    from .ranker import OntologyRanker
    ranker = OntologyRanker()
    cases = sample_cases_of(split)
    result = {}
    for name, guard in [("with_guard", True), ("without_guard", False)]:
        ranks = [
            ranker.rank_of(c["disease_id"], c["hpo_present"], c["hpo_excluded"],
                           exclude_ref=c["pmid"] if guard else None)
            for c in cases
        ]
        result[name] = metrics(ranks)
    (RESULTS / "baselines").mkdir(parents=True, exist_ok=True)
    (RESULTS / "baselines" / f"ranker_{split}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["split", "ranker"])
    parser.add_argument("--split", default="val", choices=SPLITS, help="which sample to evaluate on")
    args = parser.parse_args()
    if args.command == "split":
        print(json.dumps(make_splits(), indent=2))
    elif args.command == "ranker":
        print(json.dumps(eval_ranker(args.split), indent=2))


if __name__ == "__main__":
    main()
