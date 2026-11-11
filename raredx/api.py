"""HTTP API for the interface: the bridge between the LangGraph flow and the web page.

Start it (with Bifrost running):  make api   (= uvicorn raredx.api:app --port 8000)
Interface:                        http://localhost:8000
Interactive docs:                 http://localhost:8000/docs

Calls, in the order the interface makes them:
  1. POST /cases                 {"note": "...", "sex": "female", "age_years": 3}
       -> extracts the symptoms and PAUSES for the doctor's review. Returns
          {"case_id": "...", "features": [...], "hpo_present": [...], "hpo_excluded": [...]}
  2. GET  /hpo/search?q=long fing
       -> to add a missing symptom: [{"hpo": "HP:0001166", "name": "Arachnodactyly", ...}]
  3. POST /cases/{case_id}/resume {"hpo_present": [...], "hpo_excluded": [...]}
       -> continues the flow with the reviewed symptoms. The answer is a STREAM of
          events (Server-Sent Events), one per finished step, so the page can show
          progress instead of waiting in silence:
            event: step     data: {"step": "running", "label": "Asking the ontology ranker, GPT-6 Luna..."}
            event: step     data: {"step": "llm_gpt_6_luna", "label": "GPT-6 Luna answered", "seconds": 8.6}
            event: ranking   data: {"ranking": [...]}                       (as soon as the fusion is done)
            event: explain   data: {"explanations": {...}, "next_findings": [...]}
            event: summaries data: {"summaries": {"OMIM:154700": "..."}}
            event: result    data: {"ranking": [...], "failed": [], "timings": {...}, "usd": 0.0004, ...}
  4. GET  /cases/{case_id}       -> the case as it stands (e.g. to reload the page)
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .pipeline import ROOT, Pipeline, _node_name

P: Pipeline | None = None      # loaded once when the server starts (~5-20 s)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global P
    P = Pipeline(verbose=True)
    yield


app = FastAPI(title="raredx", version="0.1.0", lifespan=lifespan,
              description="Rare-disease differential diagnosis support. Research prototype: not a diagnostic tool.")
# the interface runs on another port during development (e.g. Vite on 5173)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------- request bodies
class NewCase(BaseModel):
    note: str | None = Field(None, description="clinical note in free text")
    hpo: list[str] | None = Field(None, description="symptoms as HPO codes, instead of a note")
    hpo_excluded: list[str] | None = Field(None, description="absent symptoms as HPO codes (with `hpo`)")
    sex: str | None = Field(None, pattern="^(female|male)$")
    age_years: float | None = Field(None, ge=0, le=120)


class Review(BaseModel):
    hpo_present: list[str] | None = Field(None, description="symptoms the patient has, after the doctor's review")
    hpo_excluded: list[str] | None = Field(None, description="symptoms the patient explicitly does not have")


# ---------------------------------------------------------------------- helpers
def _named(codes: list[str]) -> list[dict]:
    """['HP:0001250'] -> [{'hpo': 'HP:0001250', 'name': 'Seizure'}] (the page shows names)."""
    return [{"hpo": c, "name": P.onto.names.get(c, c)} for c in codes]


def _state(case_id: str) -> dict:
    values = P.final_state(case_id)
    if not values:
        raise HTTPException(404, f"case {case_id} not found (cases live in memory and are lost when the server restarts)")
    return values


def _labels() -> dict[str, str]:
    """Readable progress messages per graph step."""
    out = {"ranker": "Ontology ranker done", "fuse": "Final ranking ready",
           "explain": "Explanations ready", "summarize": "Summaries written", "evidence": "Live evidence found"}
    for m in P.models:
        out[_node_name(m)] = f"{m.split('/')[-1]} answered"
    return out


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------- endpoints
@app.get("/health")
def health() -> dict:
    return {"ok": P is not None, "llm_models": P.models if P else [],
            "summary_model": P.summary_model if P else None,
            "evidence": bool(P and P.evidence.available)}


@app.post("/cases")
def new_case(body: NewCase) -> dict:
    """Extract the symptoms from the note and pause for the doctor's review."""
    if not (body.note and body.note.strip()) and not body.hpo:
        raise HTTPException(422, "give a clinical note or a list of HPO codes")
    case_id = None
    for case_id, _, _ in P.start(body.note, body.hpo, body.sex, body.age_years, hpo_excluded=body.hpo_excluded):
        pass                                   # runs extract, then stops at the review pause
    s = _state(case_id)
    return {"case_id": case_id, "features": s.get("features", []),
            "hpo_present": _named(s.get("hpo_present", [])), "hpo_excluded": _named(s.get("hpo_excluded", [])),
            "usd": s.get("usd", 0.0), "timings": s.get("timings", {})}


@app.post("/cases/{case_id}/resume")
def resume(case_id: str, body: Review) -> StreamingResponse:
    """Continue after the review; streams one event per finished step, then the result."""
    s = _state(case_id)
    if s.get("ranking"):
        raise HTTPException(409, "this case is already finished: create a new one to change the symptoms")
    present = body.hpo_present if body.hpo_present is not None else s.get("hpo_present", [])
    if not present:
        raise HTTPException(422, "at least one present symptom is needed")
    unknown = [c for c in present + (body.hpo_excluded or []) if P.onto.canonical(c) not in P.onto.names]
    if unknown:
        raise HTTPException(422, f"unknown HPO codes: {unknown}")
    labels = _labels()

    def events():
        sources = ["ontology ranker"] + [m.split("/")[-1] for m in P.models]
        yield _sse("step", {"step": "running", "label": "Asking " + ", ".join(sources) + "..."})
        try:
            for step, update in P.resume(case_id, body.hpo_present, body.hpo_excluded):
                if step.startswith("__") or step == "review":
                    continue
                update = update or {}
                timings = update.get("timings", {})
                yield _sse("step", {"step": step, "label": labels.get(step, step),
                                    "seconds": next(iter(timings.values()), None)})
                # partial results as soon as they exist, so the page fills in step by step
                if step == "fuse":
                    yield _sse("ranking", {"ranking": update.get("ranking", []), "confidence": update.get("confidence", {})})
                elif step == "explain":
                    yield _sse("explain", {"explanations": update.get("explanations", {}),
                                           "next_findings": update.get("next_findings", [])})
                elif step == "summarize":
                    yield _sse("summaries", {"summaries": update.get("summaries", {})})
                elif step == "evidence":
                    yield _sse("evidence", {"evidence": update.get("evidence", {})})
        except Exception as err:               # the page shows the error instead of hanging
            yield _sse("error", {"message": str(err)[:500]})
            return
        final = P.final_state(case_id)
        yield _sse("result", {"ranking": final.get("ranking", []), "failed": final.get("failed", []),
                              "explanations": final.get("explanations", {}), "next_findings": final.get("next_findings", []),
                              "summaries": final.get("summaries", {}), "evidence": final.get("evidence", {}),
                              "confidence": final.get("confidence", {}),
                              "hpo_present": _named(final.get("hpo_present", [])),
                              "hpo_excluded": _named(final.get("hpo_excluded", [])),
                              "timings": final.get("timings", {}), "usd": round(final.get("usd", 0.0), 5)})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/cases/{case_id}")
def get_case(case_id: str) -> dict:
    s = _state(case_id)
    return {"case_id": case_id, "status": "done" if s.get("ranking") else "waiting_for_review",
            "features": s.get("features", []), "hpo_present": _named(s.get("hpo_present", [])),
            "hpo_excluded": _named(s.get("hpo_excluded", [])), "ranking": s.get("ranking", []),
            "failed": s.get("failed", []), "timings": s.get("timings", {}), "usd": s.get("usd", 0.0)}


@app.get("/hpo/search")
def hpo_search(q: str, limit: int = 10) -> list[dict]:
    """Find HPO symptoms by name or synonym, for the doctor to add one.
    Names that START with the text come first, then names that contain it; shorter first.
    Example: q="long fing" -> Long fingers, ..."""
    q = q.strip().lower()
    if len(q) < 2:
        return []
    idx = P.extractor.index
    hits: dict[str, tuple] = {}
    for text, term in zip(idx.strings, idx.terms):
        t = text.lower()
        pos = t.find(q)
        if pos < 0:
            continue
        key = (0 if pos == 0 else 1, len(t))
        if term not in hits or key < hits[term][0]:
            hits[term] = (key, text)
    best = sorted(hits.items(), key=lambda kv: kv[1][0])[: max(1, min(limit, 50))]
    return [{"hpo": term, "name": P.onto.names.get(term, term), "matched": text} for term, (_, text) in best]


# The interface (frontend/: index.html, styles.css, app.js) at http://localhost:8000/
# Mounted last, so the API routes above take precedence.
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
