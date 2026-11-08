"""Step 4b: LLM ensemble. Each LLM proposes its 10 most likely diagnoses for a patient.

For each patient and each model:
  1. PROMPT: the patient's symptoms by readable name (not HPO codes), the symptoms
     explicitly absent, age and sex. The LLM never sees the source paper or the answer.
  2. ANSWER: the LLM replies with JSON, e.g.
       {"diagnoses": [{"name": "Marfan syndrome", "omim": "154700"}, ...]}
  3. CODES: each name is translated to OMIM codes with raredx.diseases, e.g.
       1. "Marfan syndrome" (154700) -> {OMIM:154700}
       2. "Loeys-Dietz syndrome"     -> {OMIM:609192, OMIM:610168, ...}  (a family)
"""
from __future__ import annotations

import json
import re

from .diseases import DiseaseNames
from .llm import LLMClient
from .ranker import Ontology

SYSTEM = (
    "You are an expert clinical geneticist specialised in rare diseases. "
    "Given a patient's phenotypic features, you produce a differential diagnosis of rare "
    "genetic diseases, as specific as possible (name the exact subtype when the features allow it)."
)

INSTRUCTIONS = (
    "List the {k} most likely rare genetic diagnoses for this patient, most likely first.\n"
    'Answer ONLY with JSON in this exact format: {{"diagnoses": [{{"name": "<disease name>", '
    '"omim": "<6-digit OMIM phenotype number, or null if unsure>"}}]}}'
)

MAX_ABSENT = 25  # keep the prompt short: at most this many absent features


def _age(iso: str | None) -> str | None:
    """'P5Y6M' -> '5 years 6 months'; 'P3D' -> '3 days'."""
    if not iso:
        return None
    parts = re.findall(r"(\d+)([YMWD])", iso.split("T")[0])
    units = {"Y": "year", "M": "month", "W": "week", "D": "day"}
    words = [f"{n} {units[u]}{'s' if n != '1' else ''}" for n, u in parts]
    return " ".join(words) or None


def build_prompt(case: dict, onto: Ontology, k: int = 10) -> list[dict]:
    name = lambda t: onto.names.get(onto.canonical(t), t)
    who = [w for w in [(case.get("sex") or "").lower() or None, _age(case.get("age"))] if w and w != "unknown_sex"]
    lines = []
    if who:
        lines.append("Patient: " + ", ".join(who) + (" old" if _age(case.get("age")) else "") + ".")
    lines.append("Present phenotypic features:")
    lines += [f"- {name(t)}" for t in case["hpo_present"]]
    if case.get("hpo_excluded"):
        lines.append("Explicitly absent features:")
        lines += [f"- {name(t)}" for t in case["hpo_excluded"][:MAX_ABSENT]]
    lines += ["", INSTRUCTIONS.format(k=k)]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def parse_answer(text: str) -> list[dict]:
    """Extract [{"name", "omim"}, ...] from the LLM reply, tolerating text around the JSON."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    items = data.get("diagnoses", []) if isinstance(data, dict) else []
    out = []
    for it in items:
        if isinstance(it, dict) and it.get("name"):
            omim = it.get("omim")
            out.append({"name": str(it["name"]).strip(), "omim": None if omim in (None, "", "null") else str(omim)})
    return out


class Ensemble:
    def __init__(self, client: LLMClient | None = None, onto: Ontology | None = None, names: DiseaseNames | None = None):
        self.client = client or LLMClient()
        self.onto = onto or Ontology()
        self.names = names or DiseaseNames()

    def diagnose(self, case: dict, model: str, k: int = 10) -> dict:
        """One model's differential for one patient:
        {"model", "candidates": [{"rank", "name", "omim", "codes", "how"}], "usd", "cached", "raw"}"""
        reply = self.client.ask(model, build_prompt(case, self.onto, k))
        candidates = []
        for rank, item in enumerate(parse_answer(reply["text"])[:k], start=1):
            codes, how = self.names.lookup(item["name"], item["omim"])
            candidates.append({"rank": rank, **item, "codes": sorted(codes), "how": how})
        return {"model": model, "candidates": candidates, "usd": reply["usd"], "cached": reply["cached"], "raw": reply["text"]}

    @staticmethod
    def rank_of(disease: str, result: dict) -> int | None:
        """Position of the correct disease in one model's list (None if it is not there)."""
        for c in result["candidates"]:
            if disease in c["codes"]:
                return c["rank"]
        return None
