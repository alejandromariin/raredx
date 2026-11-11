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
    llms = cfg["llm"]["ensemble"] + [cfg["llm"]["contamination_model"]]
    cutoffs = {m["model"]: str(m["cutoff"]) for m in llms if m.get("cutoff")}
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
    from .llm import Overloaded, RateLimited
    skipped, in_a_row = 0, 0
    for i, c in enumerate(cases, 1):
        try:
            r = ens.diagnose(c, model)
            in_a_row = 0
        except RateLimited as err:
            print(f"\nStopped at case {i}/{len(cases)}: {err}\n"
                  "The cases already done are cached. Run the same command again later "
                  "(free daily quotas reset at midnight Pacific time) and it will continue.")
            raise SystemExit(1)
        except Overloaded:
            # the provider is overloaded: skip this case for now (it is not cached, so the next
            # run retries it); stop if it keeps happening
            skipped += 1
            in_a_row += 1
            print(f"  {i}/{len(cases)} skipped: provider overloaded", flush=True)
            if in_a_row >= 3:
                print(f"\nStopped at case {i}/{len(cases)}: the provider is overloaded. The cases already "
                      "done are cached. Run the same command again in a while and it will continue.")
                raise SystemExit(1)
            continue
        ranks.append(Ensemble.rank_of(c["disease_id"], r))
        how.update(x["how"] for x in r["candidates"])
        usd += r["usd"]; cached += r["cached"]
        print(f"  {i}/{len(cases)} rank={ranks[-1]} cost so far={usd:.4f} $", flush=True)
    if skipped:
        print(f"\n{skipped} cases were skipped because the provider was overloaded. "
              "Run the same command again to complete them; results are saved only when all cases are done.")
        raise SystemExit(1)
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

# =============================================================================
# CONTAMINATION (RE6): did the LLM memorise the published cases?
# An LLM may have read a paper during training. Then it "remembers" the answer
# instead of reasoning. Test: compare its accuracy on cases published BEFORE its
# training cutoff (it may have read them) and AFTER (it cannot have).
# Fair comparison:
#   - same diseases in both groups (pairs: one older and one newer case of the
#     same disease), so the newer group is not just "rarer, newer diseases";
#   - the ontology ranker (no cutoff, with leakage guard) runs on the same cases
#     as a difficulty check: if the ranker also drops, the newer cases are simply
#     harder; if only the LLM drops, that points to memorisation.
# Example: LLM 30% before vs 15% after, ranker 20% vs 19% -> suspicious.
# Runs on all cases (not only the samples): this model is not part of the fusion,
# so nothing leaks into training or test.
# =============================================================================

def contamination_pairs(max_per_disease: int = 3, seed: int = 42) -> list[tuple[dict, dict]]:
    """(older case, newer case) pairs of the same disease, around the model's cutoff."""
    cut = str(load_settings()["llm"]["contamination_model"]["cutoff"])
    dates = load_pub_dates()
    rng = random.Random(seed)
    older, newer = defaultdict(list), defaultdict(list)
    for c in load_cases():
        (newer if dates[c["pmid"]] > cut else older)[c["disease_id"]].append(c)

    def spread(cs: list[dict], n: int) -> list[dict]:
        """n cases, from as many different papers as possible."""
        rng.shuffle(cs)
        by_paper = defaultdict(list)
        for c in cs:
            by_paper[c["pmid"]].append(c)
        out = []
        while len(out) < n and any(by_paper.values()):
            for pmid in list(by_paper):
                if by_paper[pmid] and len(out) < n:
                    out.append(by_paper[pmid].pop())
        return out

    pairs = []
    for disease in sorted(set(older) & set(newer)):
        n = min(max_per_disease, len(older[disease]), len(newer[disease]))
        pairs += list(zip(spread(older[disease], n), spread(newer[disease], n)))
    return pairs


def eval_contamination(limit: int | None = None) -> dict:
    import numpy as np
    from .ensemble import Ensemble
    from .llm import RateLimited
    from .ranker import OntologyRanker
    model_cfg = load_settings()["llm"]["contamination_model"]
    model = f"{model_cfg['provider']}/{model_cfg['model']}"
    pairs = contamination_pairs()
    if limit:
        pairs = random.Random(42).sample(pairs, min(limit, len(pairs)))
    ranker = OntologyRanker()
    ens = Ensemble(onto=ranker.onto)

    ranks = {"llm": {"before": [], "after": []}, "ranker": {"before": [], "after": []}}
    for i, pair in enumerate(pairs, 1):
        for when, c in zip(("before", "after"), pair):
            try:
                r = ens.diagnose(c, model)
            except RateLimited as err:
                print(f"\nStopped at pair {i}/{len(pairs)}: {err}\n"
                      "Done calls are cached. Run the same command again later and it will continue.")
                raise SystemExit(1)
            ranks["llm"][when].append(Ensemble.rank_of(c["disease_id"], r))
            ranks["ranker"][when].append(ranker.rank_of(
                c["disease_id"], c["hpo_present"], c["hpo_excluded"], exclude_ref=c["pmid"]))
        print(f"  {i}/{len(pairs)} llm before={ranks['llm']['before'][-1]} after={ranks['llm']['after'][-1]}", flush=True)

    def hit10(rs):
        return np.array([r is not None and r <= 10 for r in rs], dtype=float)

    # Paired bootstrap of the drop (before - after) in top-10, resampling pairs
    rng = np.random.default_rng(42)
    idx = [rng.integers(0, len(pairs), len(pairs)) for _ in range(1000)]
    drop = {}
    for src in ("llm", "ranker"):
        d = hit10(ranks[src]["before"]) - hit10(ranks[src]["after"])
        lo, hi = np.percentile([d[j].mean() for j in idx], [2.5, 97.5])
        drop[src] = {"value": round(float(d.mean()), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)]}
    dd = (hit10(ranks["llm"]["before"]) - hit10(ranks["llm"]["after"])) \
        - (hit10(ranks["ranker"]["before"]) - hit10(ranks["ranker"]["after"]))
    lo, hi = np.percentile([dd[j].mean() for j in idx], [2.5, 97.5])

    result = {
        "model": model, "cutoff": str(model_cfg["cutoff"]),
        "pairs": len(pairs), "diseases": len({a["disease_id"] for a, _ in pairs}),
        "metrics": {src: {w: metrics(ranks[src][w]) for w in ("before", "after")} for src in ranks},
        "top10_drop_before_minus_after": drop,
        # > 0 means the LLM loses more than the ranker on newer cases: a sign of memorisation
        "llm_drop_minus_ranker_drop": {"value": round(float(dd.mean()), 3),
                                       "ci95": [round(float(lo), 3), round(float(hi), 3)]},
    }
    (RESULTS / "contamination.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result

# =============================================================================
# EXTRACTION (RE10): round trip. We have no real clinical notes with known HPO codes,
# so we make them: an LLM writes a note from a published case, and the extractor
# turns it back into codes. Example:
#   case:      Seizure, Hypertelorism, Arachnodactyly; absent: Hearing impairment
#   note:      "...recurrent fits since age two, widely spaced eyes, very long, thin fingers..."
#   extracted: Seizure, Hypertelorism, Arachnodactyly  -> 3 of 3 recovered
# What matters most is the last check: does the ranker still find the disease when it
# uses the extracted codes instead of the original ones?
# Caveat: the same LLM family writes and reads the notes, so real notes may be harder.
# =============================================================================

def eval_extraction(split: str = "val", limit: int = 100) -> dict:
    from .phenotype import PhenotypeExtractor
    from .ranker import OntologyRanker
    ranker = OntologyRanker()
    onto = ranker.onto
    ext = PhenotypeExtractor(onto=onto)
    pool = sample_cases_of(split)
    cases = random.Random(42).sample(pool, min(limit, len(pool)))

    def near(a: str, b: str, steps: int = 2) -> bool:
        """a and b are the same term, or one is a parent / grandparent of the other."""
        def up(t):
            level, seen = {t}, {t}
            for _ in range(steps):
                level = {p for x in level for p in onto._parents.get(x, ())}
                seen |= level
            return seen
        return a == b or a in up(b) or b in up(a)

    tp = fp = fn = close = ex_tp = ex_n = unmatched = n_feat = 0
    ranks_gold, ranks_ext, usd, secs, examples = [], [], 0.0, [], []
    for i, c in enumerate(cases, 1):
        note, absent = ext.narrate(c)
        out = ext.extract(note)
        gold = {onto.canonical(t) for t in c["hpo_present"]}
        got = {onto.canonical(t) for t in out["hpo_present"]}
        tp += len(gold & got); fp += len(got - gold); fn += len(gold - got)
        close += sum(any(near(g, e) for e in got) for g in gold)
        ex_gold = {onto.canonical(t) for t in absent}
        ex_tp += len(ex_gold & {onto.canonical(t) for t in out["hpo_excluded"]}); ex_n += len(ex_gold)
        unmatched += sum(f["how"] == "unmatched" for f in out["features"]); n_feat += len(out["features"])
        ranks_gold.append(ranker.rank_of(c["disease_id"], c["hpo_present"], absent, exclude_ref=c["pmid"]))
        ranks_ext.append(ranker.rank_of(c["disease_id"], out["hpo_present"], out["hpo_excluded"], exclude_ref=c["pmid"]))
        usd += out["usd"]; secs.append(out["seconds"])
        if len(examples) < 3:
            name = lambda t: onto.names.get(t, t)
            examples.append({"case_id": c["case_id"], "note": note,
                             "original": [name(t) for t in sorted(gold)],
                             "extracted": [f"{'+' if f['present'] else '-'} {f['term']} -> {f['hpo_name']}" for f in out["features"]]})
        print(f"  {i}/{len(cases)} recovered {len(gold & got)}/{len(gold)} | rank original={ranks_gold[-1]} extracted={ranks_ext[-1]}", flush=True)

    result = {
        "split": split, "cases": len(cases),
        "present_exact": {"precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3)},
        # original symptom recovered exactly or as a parent / child / grandparent term
        "present_recall_close": round(close / max(tp + fn, 1), 3),
        "absent_recall_exact": round(ex_tp / max(ex_n, 1), 3),
        "features_unmatched": f"{unmatched}/{n_feat}",
        "ranker_with_original_codes": metrics(ranks_gold),
        "ranker_with_extracted_codes": metrics(ranks_ext),
        "seconds_per_note": round(float(sum(secs) / max(len(secs), 1)), 2),
        "usd_this_run": round(usd, 4),
    }
    (RESULTS / f"extraction_{split}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (RESULTS / f"extraction_examples_{split}.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result

def train_fusion() -> dict:
    """Train the fusion on the training sample and save it to models/fusion.pkl (used by the app).
    The validation and test samples are not used, so they stay clean for evaluation."""
    from .ensemble import Ensemble
    from .fusion import FeatureBuilder, LightGBMFusion, LogRegFusion, save_fusion
    from .ranker import OntologyRanker
    cfg = load_settings()["fusion"]
    ranker = OntologyRanker()
    fb = FeatureBuilder(ranker, Ensemble(onto=ranker.onto), cfg["models"], cfg["ranker_top"])
    print("building the training table...", flush=True)
    train = fb.dataset(sample_cases_of("train"))
    try:
        import lightgbm  # noqa: F401
        learner = LightGBMFusion
    except ImportError:
        print("lightgbm not installed: saving the logistic regression instead")
        learner = LogRegFusion
    path = save_fusion(learner().fit(train), fb.columns, cfg["models"], len(train["groups"]))
    return {"saved": str(path.relative_to(ROOT)), "learner": learner.name, "llm_models": cfg["models"],
            "train_cases": len(train["groups"])}

# =============================================================================
# FINAL EVALUATION ON TEST (RE5, RE6): the 500 test cases were never used to build
# or tune anything. The fusion evaluated is the one the app uses (models/fusion.pkl).
# Paired comparison: every method is scored on the SAME patients, so we compare them
# patient by patient. Example: fusion right and Luna wrong on case 12 -> +1 for the
# fusion. Bootstrap over patients gives a 95 % interval for the difference; if it
# does not include 0, the difference is unlikely to be chance.
# =============================================================================

def _paired(a: list, b: list, seed: int = 42, n_boot: int = 2000) -> dict:
    """Difference a - b in top-1, top-10 and MRR, patient by patient, with 95 % CI."""
    import numpy as np
    ra = np.array([x if x is not None else np.inf for x in a], dtype=float)
    rb = np.array([x if x is not None else np.inf for x in b], dtype=float)
    per_case = {"top1": (ra <= 1).astype(float) - (rb <= 1), "top10": (ra <= 10).astype(float) - (rb <= 10),
                "mrr": 1 / ra - 1 / rb}
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(ra), len(ra)) for _ in range(n_boot)]
    out = {}
    for k, d in per_case.items():
        boots = np.array([d[i].mean() for i in idx])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        # two-sided bootstrap p-value: how often the resampled difference crosses 0
        p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
        out[k] = {"difference": round(float(d.mean()), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
                  # with 2,000 resamples the smallest p we can tell apart is about 0.001
                  "p_value": round(float(p), 3) if p >= 0.001 else "<0.001"}
    return out


def eval_test() -> dict:
    import numpy as np
    from .ensemble import Ensemble
    from .fusion import NOT_LISTED, FeatureBuilder, LogRegFusion, load_fusion, ranks_from_scores, rrf_scores
    from .ranker import OntologyRanker
    saved = load_fusion()
    models = saved["llm_models"]
    ranker = OntologyRanker()
    fb = FeatureBuilder(ranker, Ensemble(onto=ranker.onto), models, load_settings()["fusion"]["ranker_top"])
    if fb.columns != saved["columns"]:
        raise RuntimeError("models/fusion.pkl does not match the current columns: run `make train` first")
    cols = fb.columns
    print("building tables (test)...", flush=True)
    test = fb.dataset(sample_cases_of("test"))

    ranks = {"ranker_alone": ranks_from_scores(test, test["X"][:, cols.index("ranker_score")])}
    for m in models:
        pos = test["X"][:, cols.index(f"{m}:pos")]
        ranks[f"llm_alone:{m}"] = ranks_from_scores(test, np.where(pos < NOT_LISTED, -pos, -np.inf))
    ranks["rrf_no_learning"] = ranks_from_scores(test, rrf_scores(test, cols))
    print("building tables (train) for the logistic regression reference...", flush=True)
    train = fb.dataset(sample_cases_of("train"))
    ranks["fusion:logistic_regression"] = ranks_from_scores(test, LogRegFusion().fit(train).score(test))
    final = f"fusion:{saved['learner']} (models/fusion.pkl)"
    ranks[final] = ranks_from_scores(test, saved["model"].score(test))

    covered = sum(g in names for g, names in zip(test["gold"], test["candidates"]))
    best_single = max((k for k in ranks if k.startswith(("ranker_alone", "llm_alone"))),
                      key=lambda k: np.mean([1 / r if r else 0 for r in ranks[k]]))
    result = {
        "test_cases": len(test["groups"]), "llm_models": models, "fusion_trained_on": f"{saved['train_cases']} training cases",
        "test_correct_among_candidates": round(covered / len(test["groups"]), 3),
        "results": {k: metrics(v) for k, v in ranks.items()},
        "best_single_source": best_single,
        "paired_fusion_vs": {k: _paired(ranks[final], v) for k, v in ranks.items() if k != final},
    }
    (RESULTS / "test.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def eval_all() -> None:
    """Every evaluation, in order, from the response caches (RE8: one command).
    Runs without new LLM calls when the caches are complete; `timing` is left out
    because it makes real calls on purpose."""
    cfg = load_settings()
    steps = [("ranker val", lambda: eval_ranker("val")), ("ranker test", lambda: eval_ranker("test"))]
    for m in cfg["fusion"]["models"]:
        for split in SPLITS:
            steps.append((f"llm {m} {split}", lambda m=m, split=split: eval_llm(m, split)))
    steps += [("fusion (validation)", eval_fusion), ("ablation", eval_ablation), ("final test", eval_test),
              ("contamination", eval_contamination), ("extraction", lambda: eval_extraction("val", 100))]
    for name, run in steps:
        print(f"\n=== {name} ===", flush=True)
        run()
    print("\nAll results written to results/.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["split", "ranker", "llm", "fusion", "ablation", "timing", "contamination", "extraction", "train-fusion", "test", "all"])
    parser.add_argument("--split", default="val", choices=SPLITS, help="which sample to evaluate on")
    parser.add_argument("--model", default="openai/gpt-6-luna", help="for `llm`: provider/model")
    parser.add_argument("--limit", type=int, default=None, help="for `llm`/`contamination`: a random subset of N cases/pairs")
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
    elif args.command == "contamination":
        print(json.dumps(eval_contamination(args.limit), indent=2))
    elif args.command == "test":
        print(json.dumps(eval_test(), indent=2))
    elif args.command == "all":
        eval_all()
    elif args.command == "train-fusion":
        print(json.dumps(train_fusion(), indent=2))
    elif args.command == "extraction":
        print(json.dumps(eval_extraction(args.split, args.limit or 100), indent=2))


if __name__ == "__main__":
    main()
