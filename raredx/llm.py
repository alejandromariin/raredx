"""LLM client: every call to a language model in the project goes through here.

It does three things:
  1. CALL a model through Bifrost (our gateway on http://localhost:8080).
     Example: ask("openai/gpt-6-luna", [{"role": "user", "content": "Reply only: OK"}])
  2. CACHE: the exact same question to the same model is answered from disk, for free.
     Re-running an evaluation therefore costs nothing for the calls already made.
  3. SPEND LIMIT: every call's cost is recorded; once the total reaches `budget_usd`
     from settings.yaml (50 $), new calls are refused.

Cache and spend live in .cache/llm.sqlite (not committed).
Command line:
  python -m raredx.llm ping     # one tiny test call through Bifrost
  python -m raredx.llm spend    # how much we have spent so far

Embeddings (turning a text into a list of numbers, so similar meanings get similar
numbers) also go through here: embed("openai/text-embedding-3-small", ["short stature"]).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CACHE_DB = ROOT / ".cache" / "llm.sqlite"


class _Rows(list):
    def fetchone(self):
        return self[0] if self else None


class _LockedDB:
    """The SQLite cache, usable from several threads at once (the pipeline asks
    several LLMs in parallel). Each statement runs under a lock."""

    def __init__(self, path: Path):
        self._con = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()

    def execute(self, sql: str, args: tuple = ()) -> _Rows:
        with self._lock:
            return _Rows(self._con.execute(sql, args).fetchall())

    def commit(self) -> None:
        with self._lock:
            self._con.commit()


class BudgetExceeded(RuntimeError):
    pass


class RateLimited(RuntimeError):
    """The provider keeps refusing because of its rate limit (often the free daily quota)."""


def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip())


class LLMClient:
    def __init__(self, settings: dict | None = None, db_path: Path = CACHE_DB):
        _load_env()
        cfg = settings or yaml.safe_load((ROOT / "settings.yaml").read_text(encoding="utf-8"))
        self.budget = float(cfg["budget_usd"])
        self.prices = cfg["llm"]["prices"]
        self.active = set(cfg["llm"]["active_providers"])
        self.reasoning_effort = cfg["llm"].get("reasoning_effort")
        self.url = os.environ.get("BIFROST_URL", "http://localhost:8080").rstrip("/")
        # Free tiers limit requests per minute: wait between calls to the same provider
        self.min_interval = {p: 60.0 / rpm for p, rpm in cfg["llm"].get("requests_per_minute", {}).items()}
        self._last_call: dict[str, float] = {}
        self._pace = threading.Lock()

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = _LockedDB(db_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS spend (ts REAL, model TEXT, tokens_in INT, tokens_out INT, usd REAL)")
        if "seconds" not in [r[1] for r in self.db.execute("PRAGMA table_info(spend)")]:
            self.db.execute("ALTER TABLE spend ADD COLUMN seconds REAL")   # how long each real call took
        self.db.commit()

    # ------------------------------------------------------------------ spend
    def spent(self) -> float:
        return self.db.execute("SELECT COALESCE(SUM(usd), 0) FROM spend").fetchone()[0]

    def _cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        # "openai/gpt-6-luna" -> "gpt-6-luna"; "groq/openai/gpt-oss-120b" -> "openai/gpt-oss-120b"
        price = self.prices.get(model.split("/", 1)[-1])
        if price is None:
            raise KeyError(f"No price for {model} in settings.yaml (llm.prices)")
        return (tokens_in * price[0] + tokens_out * price[1]) / 1_000_000

    # ------------------------------------------------------------------ call
    def ask(self, model: str, messages: list[dict], temperature: float | None = None, use_cache: bool = True, **extra) -> dict:
        """Ask `model` ("provider/model", e.g. "openai/gpt-6-luna"). Returns
        {"text", "tokens_in", "tokens_out", "usd", "cached"}.
        temperature is not sent by default: reasoning models (e.g. GPT-6) only accept
        their default value. Repeatability comes from the cache instead.
        use_cache=False forces a real call (used to measure time); the answer is still saved."""
        provider = model.split("/", 1)[0]
        if provider not in self.active:
            raise RuntimeError(f"Provider '{provider}' is not active (settings.yaml: llm.active_providers)")

        body = {"model": model, "messages": messages, **extra}
        if temperature is not None:
            body["temperature"] = temperature
        if self.reasoning_effort and "reasoning_effort" not in body:
            body["reasoning_effort"] = self.reasoning_effort

        # 2. CACHE: same model + same request = same key
        key = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        row = self.db.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone() if use_cache else None
        if row:
            return {**json.loads(row[0]), "usd": 0.0, "cached": True}

        # 3. SPEND LIMIT
        if self.spent() >= self.budget:
            raise BudgetExceeded(f"Spent {self.spent():.2f} $ of {self.budget:.2f} $; refusing new calls")

        # 1. CALL through Bifrost (retries on rate limits and server errors)
        with self._pace:
            wait = self._last_call.get(provider, 0.0) + self.min_interval.get(provider, 0.0) - time.monotonic()
            self._last_call[provider] = time.monotonic() + max(wait, 0.0)
        if wait > 0:
            time.sleep(wait)
        started = time.perf_counter()
        data = self._post("/v1/chat/completions", body)
        seconds = time.perf_counter() - started
        usage = data.get("usage", {})
        result = {
            "text": data["choices"][0]["message"].get("content") or "",
            "tokens_in": usage.get("prompt_tokens", 0),
            "tokens_out": usage.get("completion_tokens", 0),
        }
        usd = self._cost(model, result["tokens_in"], result["tokens_out"])
        self.db.execute("INSERT INTO spend (ts, model, tokens_in, tokens_out, usd, seconds) VALUES (?, ?, ?, ?, ?, ?)",
                        (time.time(), model, result["tokens_in"], result["tokens_out"], usd, seconds))
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(result)))
        self.db.commit()
        return {**result, "usd": usd, "seconds": seconds, "cached": False}

    def embed(self, model: str, texts: list[str], dimensions: int | None = None,
              use_cache: bool = True, batch: int = 512):
        """Embeddings of `texts` as a float32 matrix (one row per text, rows of length 1).
        Example: embed(m, ["short stature", "small height"]) -> 2 rows that point in
        almost the same direction, so their dot product (similarity) is close to 1.
        use_cache=False for big one-off jobs (the HPO index is saved to its own file)."""
        import base64
        import numpy as np
        provider = model.split("/", 1)[0]
        if provider not in self.active:
            raise RuntimeError(f"Provider '{provider}' is not active (settings.yaml: llm.active_providers)")
        key_of = lambda t: "emb:" + hashlib.sha256(f"{model}|{dimensions}|{t}".encode()).hexdigest()
        out: dict[int, np.ndarray] = {}
        todo = []
        for i, t in enumerate(texts):
            row = self.db.execute("SELECT response FROM cache WHERE key = ?", (key_of(t),)).fetchone() if use_cache else None
            if row:
                out[i] = np.frombuffer(base64.b64decode(row[0]), dtype=np.float32)
            else:
                todo.append(i)
        for start in range(0, len(todo), batch):
            if self.spent() >= self.budget:
                raise BudgetExceeded(f"Spent {self.spent():.2f} $ of {self.budget:.2f} $; refusing new calls")
            idx = todo[start:start + batch]
            body = {"model": model, "input": [texts[i] for i in idx]}
            if dimensions:
                body["dimensions"] = dimensions
            started = time.perf_counter()
            data = self._post("/v1/embeddings", body)
            seconds = time.perf_counter() - started
            tokens = data.get("usage", {}).get("prompt_tokens", 0)
            self.db.execute("INSERT INTO spend (ts, model, tokens_in, tokens_out, usd, seconds) VALUES (?, ?, ?, ?, ?, ?)",
                            (time.time(), model, tokens, 0, self._cost(model, tokens, 0), seconds))
            for item in data["data"]:
                i = idx[item["index"]]
                vec = np.asarray(item["embedding"], dtype=np.float32)
                vec /= np.linalg.norm(vec) or 1.0
                out[i] = vec
                if use_cache:
                    self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)",
                                    (key_of(texts[i]), base64.b64encode(vec.tobytes()).decode()))
            self.db.commit()
            if len(todo) > batch:
                print(f"  embedded {min(start + batch, len(todo))}/{len(todo)}", flush=True)
        return np.vstack([out[i] for i in range(len(texts))]) if texts else np.zeros((0, 0), dtype=np.float32)

    def _post(self, path: str, body: dict, attempts: int = 4) -> dict:
        req = urllib.request.Request(
            self.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        for i in range(attempts):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as err:
                detail = err.read().decode(errors="replace")[:500]
                if err.code == 429:
                    if i < attempts - 1:
                        time.sleep(15 * 2 ** i)  # rate limit: 15 s, 30 s, 60 s
                        continue
                    raise RateLimited(f"{body.get('model')}: rate limit or daily quota reached. {detail}") from None
                if err.code in (500, 502, 503, 504) and i < attempts - 1:
                    time.sleep(2 ** i)  # 1 s, 2 s, 4 s
                    continue
                raise RuntimeError(f"Bifrost returned {err.code}: {detail}") from None
            except urllib.error.URLError as err:
                raise RuntimeError(f"Cannot reach Bifrost at {self.url} (is it running? `make gateway`): {err.reason}") from None
        raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["ping", "spend"])
    parser.add_argument("--model", default="openai/gpt-6-luna", help="provider/model to ping")
    args = parser.parse_args()
    client = LLMClient()
    if args.command == "ping":
        for attempt in ("first call", "same call again"):
            r = client.ask(args.model, [{"role": "user", "content": "Reply only: OK"}])
            print(f"{attempt}: {r['text']!r} | cached={r['cached']} | cost={r['usd']:.6f} $")
    print(f"total spent: {client.spent():.4f} $ of {client.budget:.2f} $")


if __name__ == "__main__":
    main()
