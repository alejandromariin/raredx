"""Step 5: learned fusion. Combines what each source says into ONE final ranking.

For each patient:
  1. CANDIDATES: the ranker's top 50 diseases + every disease proposed by any LLM.
  2. ONE ROW PER CANDIDATE with what each source says about it, e.g. for Marfan:
       ranker position 2, ranker score 4.6, Luna position 1, proposed by 1 of 1 LLMs,
       prevalence 1-5 / 10 000 ...
     and the label: 1 if it is the correct diagnosis, 0 if not.
  3. A MODEL learns from the training patients how much to trust each column, and
     gives every candidate a score. Candidates are sorted by that score.

Models: logistic regression (simple reference) and LightGBM with a ranking objective
(lambdarank: it optimises putting the correct disease as high as possible per patient).
Reference without learning: RRF (reciprocal rank fusion), score = sum of 1 / (60 + position).
"""
from __future__ import annotations

import datetime
import math
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"        # trained fusion model, committed so the demo needs no training
NOT_LISTED = 11          # position given to a disease an LLM did not list (its list has 10)

# Orphanet prevalence classes -> approximate value (fraction of the population)
PREVALENCE = {"<1 / 1 000 000": 5e-7, "1-9 / 1 000 000": 5e-6, "1-9 / 100 000": 5e-5,
              "1-5 / 10 000": 3e-4, "6-9 / 10 000": 7.5e-4, ">1 / 1000": 2e-3}


def load_prevalence() -> dict[str, float]:
    """ORPHA code -> log10 of its highest known prevalence (e.g. Marfan -> log10(3e-4) = -3.5).
    Parsing the Orphanet XML takes ~15 s, so the result is cached in data/processed/."""
    import json
    cache = ROOT / "data" / "processed" / "prevalence.json"
    source = RAW / "orphanet_product9_prev.xml"
    if cache.exists() and cache.stat().st_mtime >= source.stat().st_mtime:
        return json.loads(cache.read_text(encoding="utf-8"))
    out = {}
    for _, el in ET.iterparse(RAW / "orphanet_product9_prev.xml"):
        if el.tag == "Disorder" and el.find("OrphaCode") is not None:
            values = [PREVALENCE[c] for c in (p.findtext("PrevalenceClass/Name") for p in el.iter("Prevalence")) if c in PREVALENCE]
            if values:
                out["ORPHA:" + el.findtext("OrphaCode")] = math.log10(max(values))
            el.clear()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


class FeatureBuilder:
    """Turns one patient into a table: one row per candidate disease."""

    def __init__(self, ranker, ensemble, models: list[str], ranker_top: int = 50):
        self.ranker, self.ensemble, self.models, self.top = ranker, ensemble, models, ranker_top
        prev = load_prevalence()
        names = ensemble.names
        self.prevalence = {
            d: max((prev[o] for o in names.omim_to_orpha.get(d, []) if o in prev), default=np.nan)
            for d in ranker.diseases
        }
        self.columns = (
            ["ranker_pos_log", "ranker_score", "ranker_gap", "ranker_rel"]
            + [f"{m}:{k}" for m in models for k in ("pos", "listed", "family")]
            + ["llm_votes", "llm_best_pos", "prevalence_log10", "prevalence_known",
               "disease_annotations_log", "case_n_present", "case_mean_ic", "case_n_excluded"]
        )

    def case_table(self, case: dict, guard: bool = True, llm_answers: dict | None = None) -> tuple[list[str], np.ndarray]:
        """(candidate OMIM codes, feature matrix) for one patient.
        llm_answers: the LLMs' answers when they were already obtained (the pipeline asks
        them in parallel before this step). A model missing from it counts as having
        proposed nothing (e.g. it failed)."""
        r = self.ranker
        scores = r.score(case["hpo_present"], case["hpo_excluded"], exclude_ref=case["pmid"] if guard else None)
        order = np.argsort(-scores, kind="stable")
        position = np.empty(len(scores), dtype=int)
        position[order] = np.arange(1, len(scores) + 1)
        top_score = float(scores[order[0]]) if len(scores) else 0.0

        # what each LLM said: {model: {code: (position, family size)}}
        llm = {}
        for m in self.models:
            said = {}
            answer = llm_answers.get(m) if llm_answers is not None else self.ensemble.diagnose(case, m)
            for c in (answer or {"candidates": []})["candidates"]:
                for code in c["codes"]:
                    if code in r.index and code not in said:
                        said[code] = (c["rank"], len(c["codes"]))
            llm[m] = said

        candidates = [r.diseases[i] for i in order[: self.top]]
        for said in llm.values():
            candidates += [d for d in said if d not in candidates]

        present = [t for t in case["hpo_present"] if t in r.ic]
        case_cols = [len(case["hpo_present"]), float(np.mean([r.ic[t] for t in present])) if present else 0.0, len(case["hpo_excluded"])]

        rows = []
        for d in candidates:
            i = r.index[d]
            s = float(scores[i]) if np.isfinite(scores[i]) else -1.0
            row = [math.log1p(position[i]), s, top_score - s, s / top_score if top_score > 0 else 0.0]
            positions = []
            for m in self.models:
                pos, fam = llm[m].get(d, (NOT_LISTED, 0))
                positions.append(pos)
                row += [pos, int(pos < NOT_LISTED), fam]
            prev = self.prevalence.get(d, np.nan)
            row += [sum(p < NOT_LISTED for p in positions), min(positions, default=NOT_LISTED),
                    prev if np.isfinite(prev) else -7.0, int(np.isfinite(prev)),
                    math.log1p(len(r.terms[d]))] + case_cols
            rows.append(row)
        return candidates, np.array(rows, dtype=float)

    def dataset(self, cases: list[dict], guard: bool = True) -> dict:
        """Stack all patients: X (rows), y (1 = correct disease), groups (rows per patient)."""
        X, y, groups, cands = [], [], [], []
        for c in cases:
            names, table = self.case_table(c, guard)
            X.append(table)
            y += [int(d == c["disease_id"]) for d in names]
            groups.append(len(names))
            cands.append(names)
        return {"X": np.vstack(X), "y": np.array(y), "groups": groups, "candidates": cands,
                "gold": [c["disease_id"] for c in cases]}


# =============================================================================
# MODELS: each returns a score per row (higher = more likely the correct disease)
# =============================================================================

def rrf_scores(ds: dict, columns: list[str], k: int = 60) -> np.ndarray:
    """Reference without learning: sum over sources of 1 / (k + position)."""
    X = ds["X"]
    ranker_pos = np.expm1(X[:, columns.index("ranker_pos_log")])
    score = 1 / (k + ranker_pos)
    for j, col in enumerate(columns):
        if col.endswith(":pos"):
            score = score + np.where(X[:, j] < NOT_LISTED, 1 / (k + X[:, j]), 0.0)
    return score


class LogRegFusion:
    name = "logistic_regression"

    def fit(self, ds: dict):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
        self.model.fit(ds["X"], ds["y"])
        return self

    def score(self, ds: dict) -> np.ndarray:
        return self.model.predict_proba(ds["X"])[:, 1]


class LightGBMFusion:
    name = "lightgbm_lambdarank"

    def fit(self, ds: dict):
        import lightgbm as lgb
        self.model = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05,
                                    num_leaves=15, min_child_samples=20, random_state=42, verbose=-1)
        self.model.fit(ds["X"], ds["y"], group=ds["groups"])
        return self

    def score(self, ds: dict) -> np.ndarray:
        return self.model.predict(ds["X"])


def save_fusion(model, columns: list[str], llm_models: list[str], train_cases: int, path: Path | None = None) -> Path:
    """Save the trained fusion with what is needed to use it correctly later."""
    path = path or MODELS / "fusion.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump({"model": model, "learner": model.name, "columns": columns, "llm_models": llm_models,
                     "train_cases": train_cases, "created": datetime.date.today().isoformat()}, fh)
    return path


def load_fusion(path: Path | None = None) -> dict:
    path = path or MODELS / "fusion.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: train it with `make train`")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def ranks_from_scores(ds: dict, scores: np.ndarray) -> list[float | None]:
    """Position of the correct disease per patient (tie-aware; None if not a candidate)."""
    out, start = [], 0
    for size, names, gold in zip(ds["groups"], ds["candidates"], ds["gold"]):
        s = scores[start:start + size]
        start += size
        if gold not in names:
            out.append(None)
            continue
        t = s[names.index(gold)]
        out.append(float((s > t).sum()) + ((s == t).sum() + 1) / 2)
    return out
