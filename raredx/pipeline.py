"""The full flow, as a LangGraph graph. The demo app and the command line use it.

    note --> extract --> review --> ranker ----------+
                         (pause)  \\-> llm (each LLM) --+--> fuse --> explain --> summarize --> result
                                    (all at the same time)

The STATE is the patient's file, shared by every step: each step reads what it needs
and writes its result, e.g. extract writes {"features": [...], "hpo_present": [...]},
fuse reads hpo_present and the LLMs' answers and writes {"ranking": [...]}.

REVIEW is a pause (LangGraph `interrupt`): the graph stops after the extraction and
waits for the doctor to correct the symptoms. It continues when resumed with
    Command(resume={"hpo_present": [...], "hpo_excluded": [...]})
or with Command(resume={}) to accept the symptoms as extracted.

If an LLM fails, the flow goes on without it: the fusion treats it as having proposed
nothing and the result lists it in "failed".

Command line (needs Bifrost running; approves the symptoms without changes):
  python -m raredx.pipeline "3-year-old girl who does not walk yet, frequent seizures..."
  python -m raredx.pipeline --hpo HP:0001250,HP:0000252 --sex female --age 3
"""
from __future__ import annotations

import argparse
import operator
import time
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

import numpy as np
import yaml

from .ensemble import Ensemble
from .explain import Explainer
from .fusion import FeatureBuilder, load_fusion
from .llm import LLMClient
from .phenotype import PhenotypeExtractor
from .ranker import OntologyRanker

ROOT = Path(__file__).resolve().parents[1]
TOP_K = 10    # diseases shown in the result


def load_settings() -> dict:
    return yaml.safe_load((ROOT / "settings.yaml").read_text(encoding="utf-8"))


def _merge(a: dict, b: dict) -> dict:
    """Reducer: several steps running in parallel each add their own key."""
    return {**(a or {}), **(b or {})}


class State(TypedDict, total=False):
    # input
    note: str
    sex: str | None                 # "female" / "male" / None
    age: str | None                 # ISO 8601 duration, e.g. "P3Y" = 3 years
    # extract + review
    features: list[dict]
    hpo_present: list[str]
    hpo_excluded: list[str]
    # candidate sources (written in parallel, hence the reducers)
    llm: Annotated[dict, _merge]          # {model: answer}
    failed: Annotated[list, operator.add]  # e.g. ["gemini/gemini-3.5-flash-lite: rate limit"]
    timings: Annotated[dict, _merge]       # seconds per step
    usd: Annotated[float, operator.add]
    # result
    ranking: list[dict]
    explanations: dict              # {disease: fits / against / typical / genes}
    next_findings: list             # findings that best separate the top candidates
    summaries: dict                 # {disease: one or two sentences by the LLM}


def _node_name(model: str) -> str:
    """'openai/gpt-6-luna' -> 'llm_gpt_6_luna' (LangGraph node names avoid ':' and '/')."""
    return "llm_" + "".join(ch if ch.isalnum() else "_" for ch in model.split("/")[-1])


class Pipeline:
    """Loads everything once (ontology, ranker, HPO index, fusion model) and runs patients."""

    def __init__(self, verbose: bool = False):
        t = time.perf_counter()
        def took(what):
            nonlocal t
            if verbose:
                print(f"  {what}: {time.perf_counter() - t:.1f} s", flush=True)
            t = time.perf_counter()
        client = LLMClient().fail_fast()        # a doctor must not wait minutes for a provider that is down
        self.ranker = OntologyRanker()
        self.onto = self.ranker.onto
        took("ontology + ranker")
        self.ensemble = Ensemble(client=client, onto=self.onto)
        took("disease names")
        self.extractor = PhenotypeExtractor(client=client, onto=self.onto)
        self.extractor.index                      # load the HPO index now, not on the first patient
        took("HPO index")
        saved = load_fusion()
        self.fusion, self.models = saved["model"], saved["llm_models"]
        self.features = FeatureBuilder(self.ranker, self.ensemble, self.models)
        took("fusion model + prevalence")
        cfg = load_settings()["llm"]["utility"]
        self.summary_model = f"{cfg['provider']}/{cfg['model']}"
        self.explainer = Explainer(self.ranker, client, self.summary_model)
        took("explanations (annotation frequencies, genes)")
        if self.features.columns != saved["columns"]:
            raise RuntimeError("models/fusion.pkl was trained with other columns: retrain it with `make train`")
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ steps
    def _case(self, state: State) -> dict:
        return {"hpo_present": state.get("hpo_present", []), "hpo_excluded": state.get("hpo_excluded", []),
                "sex": state.get("sex"), "age": state.get("age")}

    def extract(self, state: State) -> dict:
        if not state.get("note"):            # symptoms given directly as HPO codes: nothing to extract
            return {"features": [], "timings": {"extract": 0.0}}
        t = time.perf_counter()
        out = self.extractor.extract(state["note"])
        return {"features": out["features"], "hpo_present": out["hpo_present"], "hpo_excluded": out["hpo_excluded"],
                "usd": out["usd"], "timings": {"extract": round(time.perf_counter() - t, 2)}}

    def review(self, state: State) -> dict:
        from langgraph.types import interrupt
        decision = interrupt({"features": state.get("features", []),
                              "hpo_present": state.get("hpo_present", []),
                              "hpo_excluded": state.get("hpo_excluded", [])}) or {}
        present = [self.onto.canonical(t) for t in decision.get("hpo_present", state.get("hpo_present", []))]
        excluded = [self.onto.canonical(t) for t in decision.get("hpo_excluded", state.get("hpo_excluded", []))]
        if not present:
            raise ValueError("At least one present symptom is needed")
        return {"hpo_present": present, "hpo_excluded": [t for t in excluded if t not in present]}

    def rank(self, state: State) -> dict:
        t = time.perf_counter()
        self.ranker.score(state["hpo_present"], state.get("hpo_excluded", []))   # ~0.003 s; fuse reuses the ranker
        return {"timings": {"ranker": round(time.perf_counter() - t, 3)}}

    def ask_llm(self, model: str):
        def step(state: State) -> dict:
            t = time.perf_counter()
            try:
                answer = self.ensemble.diagnose(self._case(state), model)
            except Exception as err:          # one LLM failing must not stop the diagnosis
                return {"failed": [f"{model}: {err}"[:300]], "timings": {model: round(time.perf_counter() - t, 2)}}
            return {"llm": {model: answer}, "usd": answer["usd"], "timings": {model: round(time.perf_counter() - t, 2)}}
        return step

    def fuse(self, state: State) -> dict:
        t = time.perf_counter()
        llm = state.get("llm", {})
        candidates, X = self.features.case_table(self._case(state), guard=False, llm_answers=llm)
        scores = self.fusion.score({"X": X})
        cols = self.features.columns
        ranking = []
        for i in np.argsort(-scores, kind="stable")[:TOP_K]:
            d, row = candidates[i], X[i]
            ranking.append({
                "rank": len(ranking) + 1, "disease": d, "name": self.ranker.disease_names.get(d, d),
                "score": round(float(scores[i]), 3),
                # where each source put it (None = not proposed): the first "why" for the doctor
                "ranker_position": int(round(np.expm1(row[cols.index("ranker_pos_log")]))),
                "llm_positions": {m: (int(row[cols.index(f"{m}:pos")]) if m in llm and row[cols.index(f"{m}:listed")] else None)
                                  for m in self.models},
            })
        return {"ranking": ranking, "timings": {"fuse": round(time.perf_counter() - t, 3)}}

    def explain(self, state: State) -> dict:
        t = time.perf_counter()
        out = self.explainer.explain(state["ranking"], state["hpo_present"], state.get("hpo_excluded", []))
        return {**out, "timings": {"explain": round(time.perf_counter() - t, 3)}}

    def summarize(self, state: State) -> dict:
        t = time.perf_counter()
        try:
            out = self.explainer.summarize(state["ranking"], state["explanations"],
                                           state["hpo_present"], state.get("hpo_excluded", []))
        except Exception as err:               # the summary is a bonus: never block the result
            return {"summaries": {}, "failed": [f"{self.summary_model} (summary): {err}"[:300]],
                    "timings": {"summarize": round(time.perf_counter() - t, 2)}}
        return {"summaries": out["summaries"], "usd": out["usd"],
                "timings": {"summarize": round(time.perf_counter() - t, 2)}}

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph
        try:
            from langgraph.checkpoint.memory import InMemorySaver as Saver
        except ImportError:                    # older LangGraph versions
            from langgraph.checkpoint.memory import MemorySaver as Saver
        g = StateGraph(State)
        g.add_node("extract", self.extract)
        g.add_node("review", self.review)
        g.add_node("ranker", self.rank)
        sources = ["ranker"]
        for m in self.models:
            g.add_node(_node_name(m), self.ask_llm(m))
            sources.append(_node_name(m))
        g.add_node("fuse", self.fuse)
        g.add_node("explain", self.explain)
        g.add_node("summarize", self.summarize)
        g.add_edge(START, "extract")
        g.add_edge("extract", "review")
        for s in sources:
            g.add_edge("review", s)          # all sources start at the same time...
        g.add_edge(sources, "fuse")          # ...and fuse waits for all of them
        g.add_edge("fuse", "explain")              # explanations from the ontology (instant)
        g.add_edge("explain", "summarize")         # one LLM call: a sentence per candidate
        g.add_edge("summarize", END)
        # the checkpointer keeps each patient's state while the graph waits for the doctor
        return g.compile(checkpointer=Saver())

    # ------------------------------------------------------------------ helpers for the API / CLI
    def start(self, note: str | None = None, hpo: list[str] | None = None, sex: str | None = None,
              age_years: float | None = None, thread_id: str | None = None, hpo_excluded: list[str] | None = None):
        """Run until the review pause. Yields (thread_id, step name, update) as each step finishes."""
        thread_id = thread_id or uuid.uuid4().hex
        state: State = {"note": note or "", "sex": sex, "age": f"P{int(age_years)}Y" if age_years is not None else None,
                        "llm": {}, "failed": [], "timings": {}, "usd": 0.0}
        if hpo:
            state["hpo_present"] = [self.onto.canonical(t) for t in hpo]
            state["hpo_excluded"] = [self.onto.canonical(t) for t in hpo_excluded or []]
        for update in self.graph.stream(state, {"configurable": {"thread_id": thread_id}}, stream_mode="updates"):
            for step, value in update.items():
                yield thread_id, step, value

    def resume(self, thread_id: str, hpo_present: list[str] | None = None, hpo_excluded: list[str] | None = None):
        """Continue after the doctor's review. Yields (step name, update) as each step finishes."""
        from langgraph.types import Command
        # Never resume with an empty dict: LangGraph would read it as "no answer" and pause again.
        decision = {"accepted": True}
        if hpo_present is not None:
            decision["hpo_present"] = hpo_present
        if hpo_excluded is not None:
            decision["hpo_excluded"] = hpo_excluded
        for update in self.graph.stream(Command(resume=decision), {"configurable": {"thread_id": thread_id}},
                                        stream_mode="updates"):
            for step, value in update.items():
                yield step, value

    def final_state(self, thread_id: str) -> dict:
        return self.graph.get_state({"configurable": {"thread_id": thread_id}}).values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("note", nargs="?", help="clinical note in free text (in quotes)")
    parser.add_argument("--hpo", help="symptoms as HPO codes instead of a note, comma-separated")
    parser.add_argument("--sex", choices=["female", "male"])
    parser.add_argument("--age", type=float, help="age in years")
    args = parser.parse_args()
    if not args.note and not args.hpo:
        parser.error("give a note or --hpo")

    t0 = time.perf_counter()
    print("loading (ontology, ranker, HPO index, fusion model)...", flush=True)
    p = Pipeline(verbose=True)
    print(f"loaded in {time.perf_counter() - t0:.1f} s\n")
    t0 = time.perf_counter()
    hpo = args.hpo.split(",") if args.hpo else None
    thread = None
    for thread, step, _ in p.start(args.note, hpo, args.sex, args.age):
        label = "review: paused, waiting for the doctor" if step == "__interrupt__" else step
        print(f"[{time.perf_counter() - t0:5.1f} s] {label}")
    state = p.final_state(thread)
    for f in state.get("features", []):
        print(f"   {'+' if f['present'] else '-'} {f['term']} -> {f['hpo_name'] or '(unmatched)'}")
    print("   (review: symptoms accepted without changes)")
    for step, _ in p.resume(thread):
        print(f"[{time.perf_counter() - t0:5.1f} s] {step}")
    state = p.final_state(thread)

    short = {m: m.split("/")[-1] for m in p.models}
    print(f"\n{'#':>2}  {'score':>6}  {'ranker':>6}  " + "  ".join(f"{s[:12]:>12}" for s in short.values()) + "  disease")
    for r in state["ranking"]:
        llm = "  ".join(f"{(r['llm_positions'][m] or '-'):>12}" for m in p.models)
        print(f"{r['rank']:>2}  {r['score']:>6.2f}  {r['ranker_position']:>6}  {llm}  {r['name']} ({r['disease']})")
    if state.get("next_findings"):
        print("\nmost useful next findings:", ", ".join(f["name"] for f in state["next_findings"]))
    for d, text in state.get("summaries", {}).items():
        print(f"\n{d}: {text}")
    if state.get("failed"):
        print("\nfailed:", *state["failed"], sep="\n  ")
    print(f"\ntotal {time.perf_counter() - t0:.1f} s, {state.get('usd', 0):.5f} $, steps: {state.get('timings')}")


if __name__ == "__main__":
    main()
