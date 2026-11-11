"""Step 6: explain each top candidate and suggest the most useful next finding.

Everything here except the one-sentence summary comes from the ontology files, so it
is instant, free and cannot invent anything. For each of the top 5 candidates:
  - FITS:     patient findings the disease has in its HPO annotations (the same term,
              or a closely related one, e.g. "Long fingers" ~ "Arachnodactyly")
  - AGAINST:  findings the patient does NOT have but the disease usually has
              (e.g. absent "Intellectual disability", seen in 90 % of cases)
  - TYPICAL:  frequent findings of the disease the note does not mention (worth checking)
  - GENES:    genes linked to the disease (to confirm it with a genetic test)
  - SUMMARY:  one or two sentences by an LLM, written ONLY from the facts above

NEXT FINDINGS (information gain): among the findings of the top candidates, the ones
that best separate them. Example: if Marfan and Loeys-Dietz are on top, "Ectopia
lentis" (90 % in Marfan, rare in Loeys-Dietz) splits them; "Tall stature" (both) does not.
For each finding f: how much the uncertainty over the candidates (entropy) is expected
to drop once we know whether the patient has f.

Sources: phenotype.hpoa (annotations and their frequencies) and genes_to_disease.txt (HPO).
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# HPO frequency terms -> approximate fraction of patients with the finding
FREQ_TERMS = {"HP:0040280": 1.0,    # obligate
              "HP:0040281": 0.9,    # very frequent (80-99 %)
              "HP:0040282": 0.55,   # frequent (30-79 %)
              "HP:0040283": 0.17,   # occasional (5-29 %)
              "HP:0040284": 0.025,  # very rare (1-4 %)
              "HP:0040285": 0.0}    # excluded
UNKNOWN = 0.5      # annotated, frequency not given
NOT_EXPECTED = 0.02  # not annotated at all: rarely seen
MIN_IC = 2.0       # skip very general terms (e.g. "Abnormality of the nervous system")


def _freq(value: str) -> float | None:
    """'HP:0040281' -> 0.9, '3/4' -> 0.75, '40%' -> 0.4, '' -> None."""
    value = value.strip()
    if not value:
        return None
    if value in FREQ_TERMS:
        return FREQ_TERMS[value]
    m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", value)
    if m:
        return int(m.group(1)) / int(m.group(2)) if int(m.group(2)) else None
    m = re.fullmatch(r"([\d.]+)\s*%", value)
    return float(m.group(1)) / 100 if m else None


def load_frequencies(path: Path = RAW / "phenotype.hpoa"):
    """disease -> {term: frequency or None}, and disease -> terms it explicitly does NOT have."""
    values = defaultdict(lambda: defaultdict(list))
    negated = defaultdict(set)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(("#", "database_id")):
                continue
            f = line.rstrip("\n").split("\t")
            if not f[0].startswith("OMIM:") or f[10] != "P":
                continue
            if f[2] == "NOT":
                negated[f[0]].add(f[3])
            else:
                values[f[0]][f[3]].append(_freq(f[7]))
    freqs = {}
    for d, terms in values.items():
        freqs[d] = {}
        for t, vs in terms.items():
            known = [v for v in vs if v is not None]
            freqs[d][t] = float(np.mean(known)) if known else None
    return freqs, negated


def load_genes(path: Path = RAW / "genes_to_disease.txt") -> dict[str, list[str]]:
    """disease -> gene symbols with a Mendelian association, e.g. OMIM:154700 -> ['FBN1']."""
    out = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 4 and f[2] == "MENDELIAN" and f[1] not in out[f[3]]:
                out[f[3]].append(f[1])
    return dict(out)


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


SUMMARY_SYSTEM = "You are a clinical geneticist writing brief, careful notes for a colleague."
SUMMARY_INSTRUCTIONS = """For each candidate disease below, write one or two sentences (at most 45 words)
on how well it fits this patient. Use ONLY the facts listed for that candidate: do not add
findings, frequencies or claims that are not listed, and do not state a diagnosis.
Answer ONLY with JSON: {"notes": [{"disease": "<id>", "text": "<sentences>"}]}

"""


class Explainer:
    def __init__(self, ranker, client=None, model: str | None = None, top: int = 5, n_questions: int = 3):
        self.ranker, self.onto, self.ic = ranker, ranker.onto, ranker.ic
        self.client, self.model = client, model
        self.top, self.n_questions = top, n_questions
        self.freqs, self.negated = load_frequencies()
        self.genes = load_genes()
        self.name = lambda t: self.onto.names.get(t, t)

    # ------------------------------------------------------------------ helpers
    def _related(self, a: str, b: str) -> bool:
        """a and b are the same term or one is an ancestor of the other."""
        return a == b or a in self.onto.ancestors(b) or b in self.onto.ancestors(a)

    def expected(self, disease: str, term: str) -> float:
        """Fraction of patients with `disease` expected to show `term` (or a more specific form of it)."""
        if term in self.negated.get(disease, ()):
            return 0.0
        found = [UNKNOWN if f is None else f for a, f in self.freqs.get(disease, {}).items()
                 if term in self.onto.ancestors(a)]
        return max(found) if found else NOT_EXPECTED

    # ------------------------------------------------------------------ per candidate
    def candidate(self, disease: str, present: list[str], excluded: list[str]) -> dict:
        annotated = self.freqs.get(disease, {})
        closure = set()
        for a in annotated:
            closure |= self.onto.ancestors(a)
        fits, partial, unexplained = [], [], []
        for t in present:
            if t in annotated:
                fits.append({"hpo": t, "name": self.name(t)})
                continue
            rel = [a for a in annotated if self._related(t, a)]
            if rel:
                a = max(rel, key=lambda x: self.ic.get(x, 0))
                fits.append({"hpo": t, "name": self.name(t), "as": self.name(a)})
                continue
            shared = self.onto.ancestors(t) & closure
            best = max(shared, key=lambda x: self.ic.get(x, 0), default=None)
            if best and self.ic.get(best, 0) >= 3.5:
                partial.append({"hpo": t, "name": self.name(t), "shared": self.name(best)})
            else:
                unexplained.append({"hpo": t, "name": self.name(t)})
        against = []
        for t in excluded:
            q = self.expected(disease, t)
            if q >= 0.3 and q != NOT_EXPECTED:
                against.append({"hpo": t, "name": self.name(t), "frequency": round(q, 2)})
        mentioned = present + excluded
        typical = sorted(
            ({"hpo": a, "name": self.name(a), "frequency": round(f, 2)} for a, f in annotated.items()
             if f is not None and f >= 0.8 and self.ic.get(a, 0) >= MIN_IC
             and not any(self._related(a, m) for m in mentioned)),
            key=lambda x: (-x["frequency"], -self.ic.get(x["hpo"], 0)))[:5]
        return {"fits": fits, "partial": partial, "unexplained": unexplained, "against": against,
                "typical": typical, "genes": self.genes.get(disease, [])[:6]}

    # ------------------------------------------------------------------ next findings
    def next_findings(self, diseases: list[str], scores: list[float], present: list[str], excluded: list[str]) -> list[dict]:
        s = np.array(scores, dtype=float)
        p = np.exp(s - s.max())
        p /= p.sum()                                     # relative support of each candidate
        h0 = _entropy(p)
        mentioned = present + excluded
        features = {a for d in diseases for a in self.freqs.get(d, {})
                    if self.ic.get(a, 0) >= MIN_IC and not any(self._related(a, m) for m in mentioned)}
        scored = []
        for f in features:
            q = np.clip([self.expected(d, f) for d in diseases], 0.02, 0.98)
            p_yes = float((p * q).sum())
            post_yes, post_no = p * q / p_yes, p * (1 - q) / (1 - p_yes)
            gain = h0 - p_yes * _entropy(post_yes) - (1 - p_yes) * _entropy(post_no)
            scored.append((gain, f, q))
        scored.sort(key=lambda x: -x[0])
        chosen = []
        for gain, f, q in scored:
            if any(self._related(f, c["hpo"]) for c in chosen):   # no near-duplicates
                continue
            chosen.append({"hpo": f, "name": self.name(f), "gain": round(gain, 3),
                           "expected": {d: round(float(x), 2) for d, x in zip(diseases, q)}})
            if len(chosen) == self.n_questions:
                break
        return chosen

    # ------------------------------------------------------------------ all of it
    def explain(self, ranking: list[dict], present: list[str], excluded: list[str]) -> dict:
        top = ranking[: self.top]
        return {"explanations": {r["disease"]: self.candidate(r["disease"], present, excluded) for r in top},
                "next_findings": self.next_findings([r["disease"] for r in top], [r["score"] for r in top], present, excluded)}

    def summarize(self, ranking: list[dict], explanations: dict, present: list[str], excluded: list[str]) -> dict:
        """One LLM call for the top candidates. Returns {disease: text} ({} if it fails)."""
        lines = ["Patient findings: " + "; ".join(self.name(t) for t in present)]
        if excluded:
            lines.append("Absent: " + "; ".join(self.name(t) for t in excluded))
        for r in ranking[: self.top]:
            e = explanations[r["disease"]]
            pct = lambda xs: ", ".join("%s (%d%%)" % (x["name"], round(x["frequency"] * 100)) for x in xs) or "none"
            names = lambda xs, key=None: ", ".join(x["name"] + (f" ({x[key]})" if key and x.get(key) else "") for x in xs) or "none"
            lines += ["", f"Candidate {r['disease']}: {r['name']}",
                      f"- patient findings it has: {names(e['fits'])}",
                      f"- related findings: {names(e['partial'])}",
                      f"- patient findings it does not explain: {names(e['unexplained'])}",
                      f"- absent in patient but frequent in this disease: {pct(e['against'])}",
                      f"- frequent in this disease, not mentioned: {pct(e['typical'])}"]
        reply = self.client.ask(self.model, [{"role": "system", "content": SUMMARY_SYSTEM},
                                             {"role": "user", "content": SUMMARY_INSTRUCTIONS + "\n".join(lines)}])
        match = re.search(r"\{.*\}", reply["text"], re.S)
        try:
            notes = json.loads(match.group(0))["notes"] if match else []
        except (json.JSONDecodeError, KeyError, TypeError):
            notes = []
        # Match each note to a candidate. LLMs often drop the prefix ("312750" for "OMIM:312750");
        # when no id matches, notes are taken in the order the candidates were listed.
        top = [r["disease"] for r in ranking[: self.top]]
        by_number = {d.split(":")[-1]: d for d in top}
        out = {}
        for i, n in enumerate(n for n in notes if isinstance(n, dict) and n.get("text")):
            raw = str(n.get("disease", "")).strip()
            d = raw if raw in top else by_number.get(re.sub(r"\D", "", raw))
            if d is None and len(notes) == len(top):
                d = top[i]
            if d and d not in out:
                out[d] = str(n["text"]).strip()
        return {"summaries": out, "usd": reply.get("usd", 0.0)}
