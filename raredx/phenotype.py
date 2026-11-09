"""Step 2: from a free-text clinical note to HPO codes (symptoms the ranker understands).

Two steps, with an example. Note: "4-year-old boy with seizures since age 2, eyes set
far apart and very long fingers. Hearing is normal."
  1. EXTRACT (LLM): the LLM lists each symptom as a short standard clinical term and
     says whether the patient has it or explicitly does not:
       "Seizure" (present)  "Hypertelorism" (present)  "Arachnodactyly" (present)
       "Hearing impairment" (absent, from "Hearing is normal")
  2. MATCH (no LLM): each term is looked up among the ~18,000 HPO symptom names and
     their synonyms:
       - exact match first: "Seizure" is an HPO name -> HP:0001250
       - otherwise by meaning, with embeddings: "very long fingers" is closest to
         "Arachnodactyly" (HP:0001166). The 5 closest terms are kept as alternatives,
         so the doctor can pick another one in the interface.
     Terms below `min_score` similarity stay unmatched and are shown for review.

The HPO index (embedding of every HPO name and synonym) is built once, about 0.01 $,
and saved in data/processed/.

Command line (needs Bifrost running):
  python -m raredx.phenotype "4-year-old boy with seizures and very long fingers."
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import yaml

from .llm import LLMClient
from .ranker import PHENOTYPIC_ABNORMALITY, Ontology

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

SYSTEM = (
    "You are a clinical geneticist who codes patients' phenotypic features with the "
    "Human Phenotype Ontology (HPO)."
)

INSTRUCTIONS = """Extract every phenotypic feature of the patient described in the clinical note below.
Rules:
- One feature per item: split combined findings ("short and obese" -> two items).
- Write each feature as a short standard clinical term in English, as named in the HPO
  (e.g. "Global developmental delay", "Hypertelorism", "Elevated circulating creatine kinase concentration"),
  even when the note uses lay words or another language.
- present=false for features the note explicitly says the patient does NOT have
  (e.g. "no seizures" -> "Seizure", present=false; "hearing is normal" -> "Hearing impairment", present=false).
- Ignore features of relatives, the diagnosis itself, genetic results and treatments.
- quote: the exact words of the note the feature comes from.
Answer ONLY with JSON: {"features": [{"term": "...", "present": true, "quote": "..."}]}

Clinical note:
"""

# Used only to test the extraction (round trip): a published case -> a realistic note
NARRATE = """Write a short clinical note (one paragraph, 80 to 200 words) about this patient, as a
referring doctor would write to a clinical geneticist. Describe every feature listed, in natural
clinical prose: paraphrase them and describe some in plain descriptive words instead of the exact
names. Mention the absent features as negative findings. Do not name any diagnosis, syndrome or gene.

{facts}"""


def _cfg() -> dict:
    return yaml.safe_load((ROOT / "settings.yaml").read_text(encoding="utf-8"))


class HpoIndex:
    """Every HPO symptom name and exact synonym, with its embedding."""

    def __init__(self, onto: Ontology, client: LLMClient, model: str, dimensions: int | None):
        slug = model.replace("/", "_") + (f"_{dimensions}" if dimensions else "")
        path = PROCESSED / f"hpo_index_{slug}.npz"
        if path.exists():
            data = np.load(path)
            # str(): numpy loads them as np.str_, which prints badly
            self.strings = [str(x) for x in data["strings"]]
            self.terms = [str(x) for x in data["terms"]]
            self.vectors = data["vectors"]
        else:
            self.strings, self.terms = [], []
            for term, name in onto.names.items():
                if term in onto.obsolete or PHENOTYPIC_ABNORMALITY not in onto.ancestors(term) or term == PHENOTYPIC_ABNORMALITY:
                    continue
                for s in dict.fromkeys([name] + onto.synonyms.get(term, [])):   # unique, label first
                    self.strings.append(s)
                    self.terms.append(term)
            print(f"building the HPO index: {len(self.strings)} names and synonyms (one time)...", flush=True)
            self.vectors = client.embed(model, self.strings, dimensions, use_cache=False).astype(np.float16)
            PROCESSED.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, strings=np.array(self.strings), terms=np.array(self.terms), vectors=self.vectors)
        self.exact = {}
        for s, t in zip(self.strings, self.terms):
            self.exact.setdefault(s.lower(), t)
        self._matrix = self.vectors.astype(np.float32)

    def nearest(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        """The k most similar HPO terms (each term once, by its best name or synonym)."""
        sims = self._matrix @ vector
        best: dict[str, float] = {}
        for i in np.argsort(-sims)[: k * 8]:
            t = self.terms[i]
            if t not in best:
                best[t] = float(sims[i])
            if len(best) == k:
                break
        return list(best.items())


def parse_features(text: str) -> list[dict]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for f in data.get("features", []) if isinstance(data, dict) else []:
        if isinstance(f, dict) and str(f.get("term", "")).strip():
            out.append({"term": str(f["term"]).strip(), "present": f.get("present") is not False,
                        "quote": str(f.get("quote") or "")})
    return out


class PhenotypeExtractor:
    def __init__(self, client: LLMClient | None = None, onto: Ontology | None = None):
        cfg = _cfg()
        self.client = client or LLMClient()
        self.onto = onto or Ontology()
        u, e = cfg["llm"]["utility"], cfg["llm"]["embedding"]
        self.model = f"{u['provider']}/{u['model']}"
        self.emb_model, self.dimensions = f"{e['provider']}/{e['model']}", e.get("dimensions")
        self.min_score = cfg["phenotype"]["min_score"]
        self.k = cfg["phenotype"]["alternatives"]
        self._index: HpoIndex | None = None

    @property
    def index(self) -> HpoIndex:
        if self._index is None:     # built (or loaded) on first use
            self._index = HpoIndex(self.onto, self.client, self.emb_model, self.dimensions)
        return self._index

    def match(self, features: list[dict]) -> list[dict]:
        """Add the HPO code to each extracted feature (exact name first, then embeddings)."""
        idx = self.index
        todo = [f for f in features if f["term"].lower() not in idx.exact]
        vectors = self.client.embed(self.emb_model, [f["term"] for f in todo], self.dimensions) if todo else []
        vec_of = {id(f): v for f, v in zip(todo, vectors)}
        name = lambda t: self.onto.names.get(t, t)
        for f in features:
            if f["term"].lower() in idx.exact:
                t = idx.exact[f["term"].lower()]
                f.update(hpo=t, hpo_name=name(t), score=1.0, how="exact", alternatives=[])
                continue
            near = idx.nearest(vec_of[id(f)], self.k)
            alts = [{"hpo": t, "hpo_name": name(t), "score": round(s, 3)} for t, s in near]
            top_t, top_s = near[0]
            ok = top_s >= self.min_score
            f.update(hpo=top_t if ok else None, hpo_name=name(top_t) if ok else None,
                     score=round(top_s, 3), how="embedding" if ok else "unmatched", alternatives=alts)
        return features

    def extract(self, note: str) -> dict:
        """Clinical note -> features with HPO codes, plus the present / excluded code lists."""
        started = time.perf_counter()
        r = self.client.ask(self.model, [{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": INSTRUCTIONS + note.strip()}])
        features = self.match(parse_features(r["text"]))
        present = list(dict.fromkeys(f["hpo"] for f in features if f["hpo"] and f["present"]))
        # a term both present and absent (contradiction): keep it as present
        excluded = list(dict.fromkeys(f["hpo"] for f in features if f["hpo"] and not f["present"] and f["hpo"] not in present))
        return {"features": features, "hpo_present": present, "hpo_excluded": excluded,
                "usd": r["usd"], "seconds": round(time.perf_counter() - started, 2)}

    def narrate(self, case: dict, max_absent: int = 10) -> tuple[str, list[str]]:
        """Round-trip test only: write a clinical note from a published case.
        Returns the note and the absent terms actually used (at most max_absent)."""
        from .ensemble import _age
        name = lambda t: self.onto.names.get(self.onto.canonical(t), t)
        absent = case.get("hpo_excluded", [])[:max_absent]
        who = ", ".join(x for x in [(case.get("sex") or "").lower(), _age(case.get("age"))]
                        if x and x != "unknown_sex")
        facts = (f"Patient: {who or 'not stated'}\nPresent features:\n" + "\n".join(f"- {name(t)}" for t in case["hpo_present"])
                 + ("\nAbsent features:\n" + "\n".join(f"- {name(t)}" for t in absent) if absent else ""))
        r = self.client.ask(self.model, [{"role": "user", "content": NARRATE.format(facts=facts)}])
        return r["text"].strip(), absent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("note", help="clinical note in free text (in quotes)")
    args = parser.parse_args()
    out = PhenotypeExtractor().extract(args.note)
    for f in out["features"]:
        sign = "+" if f["present"] else "-"
        print(f"{sign} {f['term']:<40} -> {f['hpo'] or '?':<11} {f['hpo_name'] or '(unmatched)'}  [{f['how']}, {f['score']}]")
    print(f"\npresent: {out['hpo_present']}\nexcluded: {out['hpo_excluded']}\n{out['seconds']} s, {out['usd']:.5f} $")


if __name__ == "__main__":
    main()
