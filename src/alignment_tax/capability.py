"""Family C: general capability across lambda, cheapest first.

* CE loss on a held-out Alpaca slice -- one forward pass per example, near free,
  and doubles as the fluency/coherence proxy at high lambda.
* MMLU (500 items) scored by log-probability over the four options -- no
  generation.
* TruthfulQA MC1 -- the benchmark Arditi et al. found most sensitive to
  directional ablation, so our most informative detector.
* GSM8K (100 items) -- requires generation; off by default, enabled only if the
  runtime holds up.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from .model import HookedModel

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _mc_prompt(question: str, choices: Sequence[str]) -> str:
    body = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"{question.strip()}\n{body}\nAnswer:"


def _clip(item: dict) -> tuple[list[str], int]:
    """TruthfulQA MC1 items can carry more options than we have letters; keep the
    gold answer and enough distractors to fill the alphabet."""
    choices, answer = list(item["choices"]), int(item["answer"])
    if len(choices) <= len(LETTERS):
        return choices, answer
    keep = [i for i in range(len(choices)) if i != answer][: len(LETTERS) - 1]
    keep = sorted([*keep, answer])
    return [choices[i] for i in keep], keep.index(answer)


def multiple_choice_accuracy(hm: HookedModel, items: list[dict], verbose: bool = False) -> dict:
    """Score by log P(" <letter>") after the options block. No generation."""
    correct, n, chance = 0, 0, 0.0
    for item in items:
        choices, answer = _clip(item)
        prompt = hm.chat_prompt(_mc_prompt(item["question"], choices))
        completions = [f" {LETTERS[i]}" for i in range(len(choices))]
        scores = hm.sequence_logprob([prompt] * len(completions), completions, length_normalise=False)
        pred = max(range(len(scores)), key=lambda i: scores[i])
        correct += int(pred == answer)
        chance += 1.0 / max(len(choices), 1)
        n += 1
    return {"accuracy": correct / max(n, 1), "n": n, "chance": chance / max(n, 1)}


def ce_loss(hm: HookedModel, texts: list[str]) -> dict:
    return {"ce_loss": hm.cross_entropy(texts), "n": len(texts)}


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def gsm8k_accuracy(hm: HookedModel, items: list[dict], max_new_tokens: int = 256) -> dict:
    prompts = [
        hm.chat_prompt(f"{it['question']}\nSolve step by step, then end with 'Answer: <number>'.")
        for it in items
    ]
    outs = hm.generate(prompts, max_new_tokens=max_new_tokens)
    correct = 0
    for out, it in zip(outs, items):
        nums = _NUM.findall(out.replace(",", ""))
        pred = nums[-1] if nums else None
        gold = it["answer"].replace(",", "")
        correct += int(pred is not None and abs(float(pred) - float(gold)) < 1e-4)
    return {"accuracy": correct / max(len(items), 1), "n": len(items),
            "mean_output_words": sum(len(o.split()) for o in outs) / max(len(outs), 1)}


def run_capability_sweep(
    hm: HookedModel,
    direction,
    lambdas: Sequence[float],
    mmlu: list[dict],
    truthfulqa: list[dict],
    alpaca_texts: list[str],
    out_path: Path,
    gsm8k: list[dict] | None = None,
    gsm8k_max_new_tokens: int = 256,
    verbose: bool = True,
) -> Path:
    """One row of capability numbers per lambda, written incrementally."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out_path.exists():
        existing = {r["lam"]: r for r in json.loads(out_path.read_text())}

    for lam in lambdas:
        if lam in existing:
            continue
        if verbose:
            print(f"[capability] lambda={lam}")
        with hm.ablated(direction, lam):
            row = {"lam": lam}
            row.update(ce_loss(hm, alpaca_texts))
            row["mmlu"] = multiple_choice_accuracy(hm, mmlu)["accuracy"]
            row["truthfulqa_mc1"] = multiple_choice_accuracy(hm, truthfulqa)["accuracy"]
            if gsm8k:
                g = gsm8k_accuracy(hm, gsm8k, gsm8k_max_new_tokens)
                row["gsm8k"] = g["accuracy"]
                row["gsm8k_output_words"] = g["mean_output_words"]
        existing[lam] = row
        out_path.write_text(json.dumps([existing[k] for k in sorted(existing)], indent=2))
        if verbose:
            print(f"           {row}")
    return out_path
