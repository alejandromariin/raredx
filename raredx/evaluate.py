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
    out = {"cases": len(r), "correct_not_ranked": int(np.isinf(r).sum())}
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


def eval_llm(model: str, split: str, limit: int | None = None) -> dict:
    """One LLM alone: position of the correct disease in its top-10 list."""
    from collections import Counter
    from .ensemble import Ensemble
    ens = Ensemble()
    cases = sample_cases_of(split)
    if limit:   # quick runs: a random, spread-out subset (the sample list is sorted by paper)
        cases = random.Random(42).sample(cases, min(limit, len(cases)))
    ranks, how, usd, cached = [], Counter(), 0.0, 0
    for i, c in enumerate(cases, 1):
        r = ens.diagnose(c, model)
        ranks.append(Ensemble.rank_of(c["disease_id"], r))
        how.update(x["how"] for x in r["candidates"])
        usd += r["usd"]; cached += r["cached"]
        print(f"  {i}/{len(cases)} rank={ranks[-1]} cost so far={usd:.4f} $", flush=True)
    result = {
        "model": model, "split": split,
        "metrics": metrics(ranks),
        "not_in_top10": sum(r is None for r in ranks),
        "name_translation": dict(how),              # how each proposed name was turned into codes
        "usd_this_run": round(usd, 4), "cached_calls": cached,
    }
    slug = model.replace("/", "_") + f"_{split}" + (f"_first{limit}" if limit else "")
    (RESULTS / "baselines").mkdir(parents=True, exist_ok=True)
    (RESULTS / "baselines" / f"llm_{slug}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _subset(ds: dict, case_idx: list[int]) -> dict:
    """The rows of some patients only (to train on 250 / 500 / 1,000 of them)."""
    import numpy as np
    starts = np.cumsum([0] + ds["groups"])
    rows = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in case_idx])
    return {"X": ds["X"][rows], "y": ds["y"][rows], "groups": [ds["groups"][i] for i in case_idx],
            "candidates": [ds["candidates"][i] for i in case_idx], "gold": [ds["gold"][i] for i in case_idx]}


def _fusion_tables(models: list[str]):
    """Build the candidate tables of the train and validation samples (see fusion.py)."""
    from .ensemble import Ensemble
    from .fusion import FeatureBuilder
    from .ranker import OntologyRanker
    cfg = load_settings()["fusion"]
    ranker = OntologyRanker()
    fb = FeatureBuilder(ranker, Ensemble(onto=ranker.onto), models, cfg["ranker_top"])
    print("building tables (train)...", flush=True)
    train = fb.dataset(sample_cases_of("train"))
    print("building tables (val)...", flush=True)
    val = fb.dataset(sample_cases_of("val"))
    return fb, train, val


# =============================================================================
# ABLATIONS (RE9): remove one group of columns at a time, retrain the fusion and
# measure on validation. A big drop means that group matters.
# Note: removing the "ranker" columns keeps the ranker's 50 candidates in the table;
# it measures what its SCORES add to the ordering, not the candidates it proposes.
# =============================================================================

COLUMN_GROUPS = {
    "ranker": lambda c: c.startswith("ranker_"),
    "llm": lambda c: ":" in c or c.startswith("llm_"),
    "prevalence": lambda c: c.startswith("prevalence_"),
    "disease_info": lambda c: c.startswith("disease_"),
    "patient_info": lambda c: c.startswith("case_"),
}


def eval_ablation(models: list[str] | None = None) -> dict:
    from .fusion import LightGBMFusion, LogRegFusion, ranks_from_scores
    try:
        import lightgbm  # noqa: F401
        Learner = LightGBMFusion
    except ImportError:
        print("lightgbm not installed: using logistic regression")
        Learner = LogRegFusion

    models = models or load_settings()["fusion"]["models"]
    fb, train, val = _fusion_tables(models)
    cols = fb.columns

    def run(keep: list[int]) -> dict:
        tr, va = {**train, "X": train["X"][:, keep]}, {**val, "X": val["X"][:, keep]}
        m = metrics(ranks_from_scores(va, Learner().fit(tr).score(va)))
        return {k: m[k]["value"] for k in ("top1", "top5", "top10", "mrr")}

    result = {"model": Learner.name, "models_used": models, "all_columns": run(list(range(len(cols)))), "without": {}}
    for group, is_member in COLUMN_GROUPS.items():
        removed = [c for c in cols if is_member(c)]
        keep = [i for i, c in enumerate(cols) if not is_member(c)]
        result["without"][group] = {"removed_columns": removed, **run(keep)}
        print(f"  without {group}: top1 {result['without'][group]['top1']}", flush=True)

    (RESULTS / "ablation_val.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


# =============================================================================
# COST AND TIME PER PATIENT (RE7)
# Diagnoses a few validation patients WITHOUT the cache (real LLM calls) and times
# each step: ranker, each LLM, fusion. LLMs can be asked in parallel, so we report
# the time in series (sum of LLMs) and in parallel (slowest LLM).
# It also reports the average cost per patient of every LLM over all calls so far.
# =============================================================================

def eval_timing(n: int = 10) -> dict:
    import sqlite3
    import time
    import numpy as np
    from .llm import CACHE_DB

    models = load_settings()["fusion"]["models"]
    fb, train, _ = _fusion_tables(models)
    try:
        from .fusion import LightGBMFusion as Learner
        import lightgbm  # noqa: F401
    except ImportError:
        from .fusion import LogRegFusion as Learner
    fusion = Learner().fit(train)

    cases = random.Random(7).sample(sample_cases_of("val"), n)
    steps = {"ranker": [], "fusion": [], **{m: [] for m in models}}
    for i, c in enumerate(cases, 1):
        t = time.perf_counter()
        fb.ranker.score(c["hpo_present"], c["hpo_excluded"])
        steps["ranker"].append(time.perf_counter() - t)
        for m in models:
            steps[m].append(fb.ensemble.diagnose(c, m, use_cache=False)["seconds"])
        t = time.perf_counter()
        names, table = fb.case_table(c)          # answers are cached now: this times building + scoring
        fusion.score({"X": table})
        steps["fusion"].append(time.perf_counter() - t)
        print(f"  {i}/{n}", flush=True)

    mean = {k: round(float(np.mean(v)), 3) for k, v in steps.items()}
    llm_series = sum(mean[m] for m in models)
    llm_parallel = max((mean[m] for m in models), default=0.0)

    cost = {m: (0.0, 0) for m in models}
    if CACHE_DB.exists():   # average cost of every real call recorded so far, per model
        db = sqlite3.connect(CACHE_DB)
        cost = {m: db.execute("SELECT AVG(usd), COUNT(*) FROM spend WHERE model = ?", (m,)).fetchone() for m in models}
    result = {
        "patients_timed": n, "models": models,
        "seconds_per_patient": {
            **{f"step:{k}": v for k, v in mean.items()},
            "total_llms_in_series": round(mean["ranker"] + llm_series + mean["fusion"], 2),
            "total_llms_in_parallel": round(mean["ranker"] + llm_parallel + mean["fusion"], 2),
        },
        "usd_per_patient": {m: {"mean": round(c[0] or 0, 6), "calls_measured": c[1]} for m, c in cost.items()},
        "usd_per_patient_total": round(sum(c[0] or 0 for c in cost.values()), 6),
    }
    (RESULTS / "cost_time.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def eval_fusion(models: list[str] | None = None) -> dict:
    """Train the fusion on the train sample, evaluate on validation, against the references."""
    import numpy as np
    from .fusion import LightGBMFusion, LogRegFusion, NOT_LISTED, ranks_from_scores, rrf_scores

    cfg = load_settings()["fusion"]
    models = models or cfg["models"]
    fb, train, val = _fusion_tables(models)
    cols = fb.columns

    covered = sum(g in names for g, names in zip(val["gold"], val["candidates"]))
    result = {"models_used": models, "train_cases": len(train["groups"]), "val_cases": len(val["groups"]),
              "val_correct_among_candidates": round(covered / len(val["groups"]), 3), "results": {}}

    # References without learning
    result["results"]["ranker_alone"] = metrics(ranks_from_scores(val, val["X"][:, cols.index("ranker_score")]))
    for m in models:
        pos = val["X"][:, cols.index(f"{m}:pos")]
        result["results"][f"llm_alone:{m}"] = metrics(ranks_from_scores(val, np.where(pos < NOT_LISTED, -pos, -np.inf)))
    result["results"]["rrf_no_learning"] = metrics(ranks_from_scores(val, rrf_scores(val, cols)))

    # Learned fusion
    learners = [LogRegFusion]
    try:
        import lightgbm  # noqa: F401
        learners.append(LightGBMFusion)
    except ImportError:
        print("lightgbm not installed: skipping it")
    fitted = {}
    for L in learners:
        model = L().fit(train)
        fitted[L.name] = model
        result["results"][f"fusion:{L.name}"] = metrics(ranks_from_scores(val, model.score(val)))

    # Learning curve: does it keep improving with more training patients?
    best = fitted.get(LightGBMFusion.name) and LightGBMFusion or LogRegFusion
    order = random.Random(42).sample(range(len(train["groups"])), len(train["groups"]))
    curve = {}
    for n in cfg["learning_curve"]:
        if n > len(order):
            continue
        m = best().fit(_subset(train, order[:n]))
        r = metrics(ranks_from_scores(val, m.score(val)), n_boot=200)
        curve[n] = {"top1": r["top1"]["value"], "top10": r["top10"]["value"]}
    result["learning_curve"] = {"model": best.name, "val_by_train_size": curve}

    if LightGBMFusion.name in fitted:
        gains = fitted[LightGBMFusion.name].model.booster_.feature_importance("gain")
        top = sorted(zip(cols, gains), key=lambda x: -x[1])[:10]
        result["most_useful_columns"] = {c: round(float(g), 1) for c, g in top}

    (RESULTS / "fusion_val.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["split", "ranker", "llm", "fusion", "ablation", "timing"])
    parser.add_argument("--split", default="val", choices=SPLITS, help="which sample to evaluate on")
    parser.add_argument("--model", default="openai/gpt-6-luna", help="for `llm`: provider/model")
    parser.add_argument("--limit", type=int, default=None, help="for `llm`: only the first N cases")
    args = parser.parse_args()
    if args.command == "split":
        print(json.dumps(make_splits(), indent=2))
    elif args.command == "ranker":
        print(json.dumps(eval_ranker(args.split), indent=2))
    elif args.command == "llm":
        print(json.dumps(eval_llm(args.model, args.split, args.limit), indent=2))
    elif args.command == "fusion":
        print(json.dumps(eval_fusion(), indent=2))
    elif args.command == "ablation":
        print(json.dumps(eval_ablation(), indent=2))
    elif args.command == "timing":
        print(json.dumps(eval_timing(args.limit or 10), indent=2))


if __name__ == "__main__":
    main()
