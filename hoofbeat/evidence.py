"""Step 7: live evidence from the web with Tavily, for the top candidates.

The ontology files are updated every few months; this finds what is new or practical
right now. For each of the top 3 candidates, three searches (Tavily, ~1 credit each):
  - LITERATURE: case reports and articles (PubMed, PMC) and the clinical summary
    (GeneReviews). First the last year; if nothing, any date.
  - TRIALS: clinical trials (ClinicalTrials.gov).
  - SUPPORT: patient information and organisations (Orphanet, NORD, GARD).
Example result: {"kind": "trials", "title": "Trofinetide for Rett syndrome ...",
                 "url": "https://clinicaltrials.gov/...", "site": "clinicaltrials.gov", "snippet": "..."}

The evidence is only SHOWN to the doctor: it never changes the ranking, so the
evaluation is not affected. Answers are cached like the LLM answers (.cache/llm.sqlite).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from .llm import CACHE_DB, _LockedDB, _load_env

TAVILY_URL = "https://api.tavily.com/search"
SEARCHES = {   # kind -> (sites, results)
    "literature": (["pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov"], 3),
    "trials": (["clinicaltrials.gov"], 2),
    "support": (["orpha.net", "rarediseases.org", "rarediseases.info.nih.gov"], 2),
}
QUERIES = {
    "literature": "{name} case report clinical features",
    "trials": "{name} clinical trial",
    "support": "{name} patient information support organization",
}


def _clean_title(title: str) -> str:
    """'Study Details | NCT03758924 | Study of X | ClinicalTrials.gov' -> 'Study of X (NCT03758924)';
    'MECP2 Disorders - GeneReviews® - NCBI Bookshelf' -> 'MECP2 Disorders · GeneReviews'."""
    m = re.match(r"^Study Details \| (NCT\d+) \|\s*(.*?)\s*\| ClinicalTrials\.gov$", title)
    if m:   # some trial pages have no title: show the trial number only
        return f"{m.group(2)} ({m.group(1)})" if m.group(2) else f"Clinical trial {m.group(1)}"
    title = re.sub(r"\s+-\s+(GeneReviews|StatPearls)®?\s+-\s+NCBI Bookshelf$", r" · \1", title)
    return re.sub(r"\s+[-|]\s+(PubMed|PMC|NCBI)$", "", title)


class Evidence:
    def __init__(self, top: int = 3, max_results: int = 3, timeout: int = 20):
        _load_env()
        self.key = os.environ.get("TAVILY_API_KEY", "").strip()
        self.top, self.max_results, self.timeout = top, max_results, timeout
        self.searches = 0
        CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
        self.db = _LockedDB(CACHE_DB)
        self.db.execute("CREATE TABLE IF NOT EXISTS tavily (key TEXT PRIMARY KEY, response TEXT, ts REAL)")
        self.db.commit()

    @property
    def available(self) -> bool:
        return bool(self.key)

    def search(self, query: str, domains: list[str], time_range: str | None = None, max_results: int | None = None) -> list[dict]:
        body = {"query": query, "search_depth": "basic", "max_results": max_results or self.max_results,
                "include_domains": domains, "include_answer": False}
        if time_range:
            body["time_range"] = time_range
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        row = None if os.environ.get("HOOFBEAT_NO_CACHE") else \
            self.db.execute("SELECT response FROM tavily WHERE key = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
        req = urllib.request.Request(TAVILY_URL, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                results = json.load(resp).get("results", [])
        except urllib.error.HTTPError as err:
            raise RuntimeError(f"Tavily returned {err.code}: {err.read().decode(errors='replace')[:300]}") from None
        self.searches += 1                       # real searches made (Tavily credits)
        self.db.execute("INSERT OR REPLACE INTO tavily VALUES (?, ?, ?)", (key, json.dumps(results), time.time()))
        self.db.commit()
        return results

    def _for(self, name: str, kind: str) -> list[dict]:
        query = QUERIES[kind].format(name=name)
        sites, n = SEARCHES[kind]
        found = self.search(query, sites, time_range="year" if kind == "literature" else None, max_results=n)
        if not found and kind == "literature":          # nothing in the last year: any date
            found = self.search(query, sites, max_results=n)
        out = []
        for r in found:
            # trial pages give boilerplate text instead of a useful summary: keep only the title
            snippet = "" if kind == "trials" else " ".join(str(r.get("content", "")).split())
            out.append({"kind": kind, "title": _clean_title(str(r.get("title", "")).strip()) or r.get("url", ""),
                        "url": r.get("url", ""), "site": urlparse(r.get("url", "")).netloc.removeprefix("www."),
                        "snippet": snippet[:220] + ("…" if len(snippet) > 220 else ""),
                        "date": r.get("published_date")})
        return out

    def gather(self, ranking: list[dict]) -> dict:
        """{disease: [results]} for the top candidates; all searches in parallel (~2 s)."""
        jobs = [(r["disease"], r["name"].split(";")[0], kind) for r in ranking[: self.top] for kind in SEARCHES]
        with ThreadPoolExecutor(max_workers=len(jobs) or 1) as pool:
            found = list(pool.map(lambda j: self._for(j[1], j[2]), jobs))
        out: dict[str, list] = {}
        for (d, _, _), res in zip(jobs, found):
            out.setdefault(d, []).extend(res)
        return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Try the live evidence search for one disease name.")
    parser.add_argument("name", help='e.g. "Rett syndrome"')
    args = parser.parse_args()
    ev = Evidence()
    if not ev.available:
        raise SystemExit("TAVILY_API_KEY is not set in .env")
    t = time.perf_counter()
    res = ev.gather([{"disease": "query", "name": args.name}])["query"]
    for r in res:
        print(f"[{r['kind']}] {r['title']}\n   {r['url']}\n   {r['snippet']}\n")
    print(f"{len(res)} results in {time.perf_counter() - t:.1f} s")


if __name__ == "__main__":
    main()
