"""Disease names <-> OMIM codes.

LLMs answer with disease NAMES ("Marfan syndrome", "Loeys-Dietz syndrome type 2"),
but to compare with the correct diagnosis and with the ranker we need OMIM CODES
(OMIM:154700). This module builds a dictionary from:
  - phenotype.hpoa: the official OMIM name of each of the 8,465 diseases we rank.
  - Orphanet (product 1): names and synonyms of ~11,600 rare diseases, with their
    OMIM equivalents.

Some names cover a whole family of diseases. Example: Orphanet's "Loeys-Dietz syndrome"
groups OMIM's Loeys-Dietz syndrome 1 to 6, so that name maps to all six codes.

The dictionary is built once and saved to data/processed/disease_names.json.
"""
from __future__ import annotations

import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "processed" / "disease_names.json"

ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10"}
FILLER = {"type", "syndrome", "disease", "disorder", "the", "of", "and", "form"}
# Words that say nothing about WHICH disease it is (used when checking an OMIM number)
GENERIC = FILLER | {"related", "associated", "neurodevelopmental", "deficiency", "autosomal", "dominant",
                    "recessive", "linked", "with", "without", "congenital", "familial", "variant", "due"}
CACHE_VERSION = 2   # bump when the cache format changes, so it is rebuilt


def normalize(name: str) -> str:
    """'Loeys-Dietz Syndrome, Type II' -> 'loeys dietz syndrome type 2'"""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return " ".join(ROMAN.get(w, w) for w in name.split())


def key_words(name: str) -> set[str]:
    """Words that identify a disease: 'Helsmoortel-Van der Aa syndrome' -> {'helsmoortel'}"""
    return {w for w in normalize(name).split() if len(w) >= 4 and w not in GENERIC}


def without_parentheses(name: str) -> str:
    """'Au-Kline syndrome (HNRNPK-related disorder)' -> 'Au-Kline syndrome'"""
    return re.sub(r"\s*\([^)]*\)", "", name).strip()


def loose(name: str) -> str:
    """Looser key without filler words: 'loeys dietz syndrome type 2' -> 'dietz loeys 2'.
    Words are sorted so word order does not matter."""
    return " ".join(sorted(w for w in normalize(name).split() if w not in FILLER))


class DiseaseNames:
    def __init__(self):
        data = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
        if data.get("version") != CACHE_VERSION:
            data = self._build()
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(data), encoding="utf-8")
        self.name_of: dict[str, str] = data["name_of"]                     # OMIM:154700 -> "Marfan syndrome"
        self.omim_to_orpha: dict[str, list[str]] = data["omim_to_orpha"]   # OMIM:154700 -> ["ORPHA:558"]
        self.official = defaultdict(set)                                     # official OMIM name -> its code
        for code, name in self.name_of.items():
            self.official[normalize(name)].add(code)
        self.exact = {k: set(v) for k, v in data["exact"].items()}          # normalized name -> OMIM codes
        self.loose = {k: set(v) for k, v in data["loose"].items()}          # loose name -> OMIM codes
        self.gene_diseases = {g: set(v) for g, v in data["genes"].items()}  # MED13L -> {OMIM:616789}

    @staticmethod
    def _build() -> dict:
        name_of = {}
        with open(RAW / "phenotype.hpoa", encoding="utf-8") as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if f[0].startswith("OMIM:") and len(f) > 10 and f[10] == "P":
                    name_of.setdefault(f[0], f[1])

        exact, loose_ = defaultdict(set), defaultdict(set)

        def add(name: str, codes) -> None:
            exact[normalize(name)].update(codes)
            loose_[loose(name)].update(codes)

        for code, name in name_of.items():
            add(name, [code])

        omim_to_orpha = defaultdict(list)
        for _, el in ET.iterparse(RAW / "orphanet_product1.xml"):
            if el.tag != "Disorder" or el.find("OrphaCode") is None:
                continue
            orpha = "ORPHA:" + el.findtext("OrphaCode")
            codes = [
                "OMIM:" + r.findtext("Reference")
                for r in el.iter("ExternalReference")
                if r.findtext("Source") == "OMIM" and "OMIM:" + r.findtext("Reference") in name_of
            ]
            if codes:
                for n in [el.findtext("Name")] + [s.text for s in el.iter("Synonym") if s.text]:
                    add(n, codes)
                for c in codes:
                    omim_to_orpha[c].append(orpha)
            el.clear()

        genes = defaultdict(set)
        with open(RAW / "genes_to_disease.txt", encoding="utf-8") as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) >= 4 and f[3] in name_of:
                    genes[f[1]].add(f[3])

        return {
            "version": CACHE_VERSION,
            "genes": {g: sorted(v) for g, v in genes.items()},
            "name_of": name_of,
            "omim_to_orpha": dict(omim_to_orpha),
            "exact": {k: sorted(v) for k, v in exact.items()},
            "loose": {k: sorted(v) for k, v in loose_.items()},
        }

    def lookup(self, name: str, omim_hint: str | None = None) -> tuple[set[str], str]:
        """Return (OMIM codes, how it was found). Tried in this order:
          'omim'     the LLM gave an OMIM number that exists and either most of the identifying
                     words of its answer appear in OMIM's name, or it is a disease of the gene it names
          'official' the name is exactly an official OMIM name     ("Marfan syndrome")
          'exact'    exactly a name or synonym from OMIM/Orphanet   ("DiGeorge syndrome")
          'loose'    same words ignoring order, case, "type", "syndrome"... ("Loeys-Dietz syndrome type 2")
          'prefix'   official OMIM names that start with it          ("Homocystinuria" -> "Homocystinuria, B6-...")
          'none'     not found"""
        genes = self._genes_in(name)
        if omim_hint:
            code = "OMIM:" + re.sub(r"\D", "", str(omim_hint))
            if code in self.name_of:
                words = key_words(without_parentheses(name)) or key_words(name)
                shared = words & key_words(self.name_of[code])
                numbers = lambda n: {w for w in normalize(without_parentheses(n)).split() if w.isdigit()}
                same_subtype = not (numbers(name) and numbers(self.name_of[code]) and numbers(name) != numbers(self.name_of[code]))
                if same_subtype and ((words and len(shared) / len(words) >= 0.5) or any(code in self.gene_diseases[g] for g in genes)):
                    return {code}, "omim"
        for variant in (name, without_parentheses(name)):
            found = self._by_name(variant)
            if found[0]:
                return found
        gene_codes = set().union(*(self.gene_diseases[g] for g in genes)) if genes else set()
        if 0 < len(gene_codes) <= 10:
            return gene_codes, "gene"
        return set(), "none"

    def _genes_in(self, name: str) -> list[str]:
        """Gene symbols mentioned in a name: 'KAT6B-related disorder' -> ['KAT6B']"""
        return [t for t in re.split(r"[^A-Za-z0-9]+", name) if len(t) >= 3 and t in self.gene_diseases and t.upper() == t]

    def _by_name(self, name: str) -> tuple[set[str], str]:
        key = normalize(name)
        if key in self.official:
            return set(self.official[key]), "official"
        if key in self.exact:
            return self.exact[key], "exact"
        if (key := loose(name)) in self.loose:
            return self.loose[key], "loose"
        if len(key) >= 6:
            found = {c for k, codes in self.official.items() if k.startswith(key + " ") for c in codes}
            if 0 < len(found) <= 10:
                return found, "prefix"
        return set(), "none"


if __name__ == "__main__":
    import time
    t = time.time(); d = DiseaseNames(); print(f"loaded in {time.time() - t:.1f}s: {len(d.name_of)} OMIM diseases, {len(d.exact)} names")
    tests = [("Marfan syndrome", None), ("Marfan Syndrome, Type I", None), ("Loeys-Dietz syndrome", None),
             ("Loeys-Dietz syndrome type 2", None), ("Homocystinuria", None), ("Dravet syndrome", None),
             ("Rett syndrome", None), ("DiGeorge syndrome", None), ("22q11.2 deletion syndrome", None),
             ("Developmental and epileptic encephalopathy 14", None), ("KCNT1-related epilepsy", None),
             ("Kabuki syndrome", None), ("Ehlers-Danlos syndrome, vascular type", None),
             ("Marfan syndrome", "154700"), ("Some made up syndrome", "154700")]
    for name, hint in tests:
        codes, how = d.lookup(name, hint)
        shown = ", ".join(f"{c} ({d.name_of[c][:35]})" for c in sorted(codes)[:3]) + (f" +{len(codes) - 3} more" if len(codes) > 3 else "")
        print(f"{name!r:50} hint={hint!s:7} -> [{how}] {shown}")
