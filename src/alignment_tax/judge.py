"""Identification grading.

Detection is graded by regex (see ``prompts``); only *identification* -- does
the named concept match the injected one -- needs semantic judgement. Two
graders:

* ``LexicalJudge``: alias and substring matching. Free, deterministic, and the
  default so a run never depends on an API key.
* ``ClaudeJudge``: an LLM grader for the cases lexical matching would miss
  ("the sea" for "ocean"). Cached to disk, batched, and always validated
  against hand labels via ``cohens_kappa``.

Whichever grader is used, ``validate_judge`` writes a sample for hand labelling
and reports Cohen's kappa against it. Reporting that agreement is worth more to
the execution-quality score than any extra experimental condition.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .prompts import clean_concept


@dataclass
class Grade:
    correct: bool
    grader: str
    detail: str = ""


class LexicalJudge:
    """Alias/substring/token-overlap matching."""

    name = "lexical"

    def __init__(self, aliases: dict[str, list[str]]):
        self.aliases = aliases

    def grade(self, target: str, response: str | None) -> Grade:
        if not response:
            return Grade(False, self.name, "empty")
        resp = clean_concept(response) or ""
        forms = [target.lower(), *(a.lower() for a in self.aliases.get(target, []))]
        for f in forms:
            if f and (f in resp or resp in f):
                return Grade(True, self.name, f"matched {f!r}")
        # stem-level overlap catches plurals and simple morphology
        resp_tokens = {_stem(t) for t in re.findall(r"[a-z0-9\-]+", resp)}
        for f in forms:
            if {_stem(t) for t in f.split()} & resp_tokens:
                return Grade(True, self.name, f"stem match {f!r}")
        return Grade(False, self.name, f"no match for {target!r} in {resp!r}")


def _stem(tok: str) -> str:
    for suf in ("ing", "ies", "es", "s"):
        if len(tok) > 4 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


class ClaudeJudge:
    """LLM grader over the Anthropic API, with an on-disk cache.

    Falls back to the lexical judge if no API key is configured, so notebooks
    run unattended either way.
    """

    name = "claude"

    def __init__(self, aliases: dict[str, list[str]], cache_path: Path,
                 model: str = "claude-sonnet-5", api_key: str | None = None):
        self.fallback = LexicalJudge(aliases)
        self.cache_path = Path(cache_path)
        self.cache: dict[str, bool] = (
            json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        )
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = None
        if self.api_key:
            try:
                from anthropic import Anthropic

                self.client = Anthropic(api_key=self.api_key)
            except Exception as exc:  # noqa: BLE001
                print(f"[judge] anthropic client unavailable ({exc}); using lexical judge")

    def grade(self, target: str, response: str | None) -> Grade:
        if not response:
            return Grade(False, self.name, "empty")
        if self.client is None:
            return self.fallback.grade(target, response)
        key = f"{target}||{response}"
        if key in self.cache:
            return Grade(self.cache[key], self.name, "cached")
        prompt = (
            "A language model was asked to name a concept that had been injected into its "
            "activations. Decide whether its answer refers to the same concept as the target.\n\n"
            f"Target concept: {target}\n"
            f"Model answer: {response}\n\n"
            "Answer with exactly one word: MATCH or MISMATCH."
        )
        try:
            msg = self.client.messages.create(
                model=self.model, max_tokens=5, messages=[{"role": "user", "content": prompt}]
            )
            verdict = "match" in msg.content[0].text.strip().lower()
        except Exception as exc:  # noqa: BLE001
            return self.fallback.grade(target, response)
        self.cache[key] = verdict
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache))
        return Grade(verdict, self.name, "api")


def make_judge(aliases: dict[str, list[str]], cache_path: Path, use_llm: bool = False) -> LexicalJudge | ClaudeJudge:
    return ClaudeJudge(aliases, cache_path) if use_llm else LexicalJudge(aliases)


# ------------------------------------------------------------ judge validation


def sample_for_hand_labelling(records: list[dict], n: int, out_path: Path, seed: int = 0) -> Path:
    """Write ``n`` identification outputs to a CSV-ish JSONL for hand labelling.

    Fill in the ``human`` field with 1/0 and pass the file to ``cohens_kappa``.
    """
    import random

    rng = random.Random(seed)
    pool = [r for r in records if r.get("response")]
    sample = rng.sample(pool, min(n, len(pool)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for r in sample:
            fh.write(json.dumps({
                "concept": r["concept"], "response": r.get("concept_reported") or r["response"],
                "judge": int(bool(r.get("identified"))), "human": None,
                "lam": r.get("lam"), "condition": r.get("condition"),
            }) + "\n")
    return out_path


def cohens_kappa(labelled_path: Path) -> dict:
    """Cohen's kappa between the judge and hand labels."""
    rows = [json.loads(l) for l in Path(labelled_path).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("human") is not None]
    if not rows:
        return {"n": 0, "kappa": None, "note": "no hand labels filled in yet"}
    a = [int(r["judge"]) for r in rows]
    b = [int(r["human"]) for r in rows]
    n = len(rows)
    agree = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    kappa = (agree - chance) / (1 - chance) if chance < 1 else 1.0
    return {"n": n, "raw_agreement": agree, "chance_agreement": chance, "kappa": kappa}
