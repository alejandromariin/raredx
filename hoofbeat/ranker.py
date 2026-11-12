"""Step 4a: ontology ranker. No LLM involved.

Compares a patient's symptoms (HPO codes) with the symptoms of each disease in
phenotype.hpoa and ranks all ~8,500 OMIM diseases from most to least similar.

How the similarity works, with an example:
  - Every symptom has an "information content" (IC): how rare it is among diseases.
    "Seizure" appears in thousands of diseases -> low IC (matching it says little).
    "Ectopia lentis" appears in very few -> high IC (matching it says a lot).
  - Symptoms are related through the HPO tree. A patient with "Long fingers" partly
    matches a disease annotated with "Arachnodactyly" through their shared parent.
  - For each patient symptom we take the most informative symptom it shares with the
    disease (itself or a common ancestor) and add up its IC (Resnik, best match).
    Final score = average over the patient's symptoms, minus a penalty for symptoms
    the patient explicitly does NOT have but the disease usually has.

Anti-cheating guard (leakage guard 1): when evaluating a case, annotations whose
reference is the case's own paper are ignored, otherwise the ranker would be
"reading the answer" from the same publication.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PHENOTYPIC_ABNORMALITY = "HP:0000118"


def _hp_id(iri: str) -> str:
    return iri.rsplit("/", 1)[-1].replace("_", ":")


class Ontology:
    """The HPO tree: term names and, for each term, all its ancestors (itself included)."""

    def __init__(self, path: Path = RAW / "hp.json"):
        graph = json.loads(path.read_text(encoding="utf-8"))["graphs"][0]
        self.names: dict[str, str] = {}
        self.alt_ids: dict[str, str] = {}
        self.synonyms: dict[str, list[str]] = {}    # exact synonyms, e.g. Hypertelorism -> "Wide-set eyes"
        self.obsolete: set[str] = set()
        for node in graph["nodes"]:
            if "HP_" not in node["id"]:
                continue
            term = _hp_id(node["id"])
            self.names[term] = node.get("lbl", "")
            meta = node.get("meta", {})
            if meta.get("deprecated"):
                self.obsolete.add(term)
            self.synonyms[term] = [x["val"] for x in meta.get("synonyms", []) if x.get("pred") == "hasExactSynonym"]
            for item in node.get("meta", {}).get("basicPropertyValues", []):
                if item.get("pred", "").endswith("hasAlternativeId"):
                    self.alt_ids[item["val"]] = term
        parents = defaultdict(set)
        for edge in graph["edges"]:
            if edge["pred"] == "is_a" and "HP_" in edge["sub"] and "HP_" in edge["obj"]:
                parents[_hp_id(edge["sub"])].add(_hp_id(edge["obj"]))
        self._parents = parents
        self._anc: dict[str, frozenset] = {}

    def canonical(self, term: str) -> str:
        return self.alt_ids.get(term, term)

    def ancestors(self, term: str) -> frozenset:
        """The term plus every term above it in the tree."""
        term = self.canonical(term)
        if term not in self._anc:
            out, stack = {term}, [term]
            while stack:
                for p in self._parents.get(stack.pop(), ()):
                    if p not in out:
                        out.add(p)
                        stack.append(p)
            self._anc[term] = frozenset(out)
        return self._anc[term]


def load_annotations(path: Path = RAW / "phenotype.hpoa") -> list[tuple[str, str, str, frozenset]]:
    """(disease_id, disease_name, hpo_id, references) for OMIM phenotype annotations."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(("#", "database_id")):
                continue
            f = line.rstrip("\n").split("\t")
            if f[0].startswith("OMIM:") and f[10] == "P" and f[2] != "NOT":
                out.append((f[0], f[1], f[3], frozenset(r.strip() for r in f[4].split(";"))))
    return out


class OntologyRanker:
    def __init__(self, ontology: Ontology | None = None, excluded_penalty: float = 0.5):
        self.onto = ontology or Ontology()
        self.excluded_penalty = excluded_penalty
        annotations = load_annotations()

        self.disease_names: dict[str, str] = {}
        self.terms: dict[str, set] = defaultdict(set)              # disease -> annotated terms
        self.term_refs: dict[tuple, frozenset] = {}                 # (disease, term) -> references
        self.diseases_by_ref: dict[str, set] = defaultdict(set)    # PMID -> diseases it annotates
        for dis, name, term, refs in annotations:
            term = self.onto.canonical(term)
            self.disease_names[dis] = name
            self.terms[dis].add(term)
            self.term_refs[(dis, term)] = self.term_refs.get((dis, term), frozenset()) | refs
            for r in refs:
                self.diseases_by_ref[r].add(dis)

        self.diseases = sorted(self.terms)
        self.index = {d: i for i, d in enumerate(self.diseases)}
        n = len(self.diseases)

        # closure = every term annotated to the disease plus all their ancestors
        holders = defaultdict(list)                                 # term -> diseases whose closure has it
        for d in self.diseases:
            for t in self._closure(self.terms[d]):
                holders[t].append(self.index[d])
        self.holders = {t: np.array(v, dtype=np.int32) for t, v in holders.items()}
        self.ic = {t: -math.log(len(v) / n) for t, v in self.holders.items()}

    def _closure(self, terms) -> set:
        out = set()
        for t in terms:
            out |= self.onto.ancestors(t)
        return out

    def _term_vector(self, term: str) -> np.ndarray:
        """For one patient symptom: best IC it shares with each disease (Resnik best match)."""
        vec = np.zeros(len(self.diseases))
        for a in sorted(self.onto.ancestors(term), key=lambda x: self.ic.get(x, 0.0)):
            if a in self.holders:
                vec[self.holders[a]] = self.ic[a]           # higher IC overwrites lower
        return vec

    def _term_score_single(self, term: str, closure: set) -> float:
        return max((self.ic.get(a, 0.0) for a in self.onto.ancestors(term) if a in closure), default=0.0)

    def score(self, present: list[str], excluded: list[str] = (), exclude_ref: str | None = None) -> np.ndarray:
        present = [t for t in present if self.onto.canonical(t) in self.onto.names]
        if not present:
            return np.zeros(len(self.diseases))
        scores = np.mean([self._term_vector(t) for t in present], axis=0)
        for t in excluded:
            t = self.onto.canonical(t)
            if t in self.holders and self.excluded_penalty:
                scores[self.holders[t]] -= self.excluded_penalty * self.ic[t] / len(present)

        # Leakage guard 1: recompute the diseases annotated by the case's own paper
        if exclude_ref:
            for d in self.diseases_by_ref.get(exclude_ref, ()):
                kept = {t for t in self.terms[d] if self.term_refs[(d, t)] - {exclude_ref}}
                closure = self._closure(kept)
                s = np.mean([self._term_score_single(t, closure) for t in present])
                for t in excluded:
                    t = self.onto.canonical(t)
                    if t in closure and t in self.ic:
                        s -= self.excluded_penalty * self.ic[t] / len(present)
                scores[self.index[d]] = s if kept else -np.inf
        return scores

    def rank(self, present, excluded=(), exclude_ref=None, top: int | None = None) -> list[tuple[str, float]]:
        scores = self.score(present, excluded, exclude_ref)
        order = np.argsort(-scores, kind="stable")
        if top:
            order = order[:top]
        return [(self.diseases[i], float(scores[i])) for i in order]

    def rank_of(self, disease: str, present, excluded=(), exclude_ref=None) -> float | None:
        """1-based position of `disease` in the ranking (None if the disease is unknown).
        Ties count fairly: if the disease ties with 4 others for 1st place, its position
        is 3 (the average of 1..5), not 1."""
        if disease not in self.index:
            return None
        scores = self.score(present, excluded, exclude_ref)
        target = scores[self.index[disease]]
        greater, equal = int((scores > target).sum()), int((scores == target).sum())
        return greater + (equal + 1) / 2
