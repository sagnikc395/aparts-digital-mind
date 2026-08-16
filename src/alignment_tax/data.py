"""Dataset loading with offline fallbacks.

Every loader tries HuggingFace first and falls back to a small bundled copy in
``data/fallback`` so the pipeline still runs on a Colab box with no network or
a rate-limited hub. Fallbacks are marked in the returned provenance string and
should never silently stand in for a real run -- ``load_all`` logs which source
each split came from.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .config import DATA_DIR

FALLBACK_DIR = DATA_DIR / "fallback"


@dataclass
class Split:
    name: str
    items: list
    source: str

    def __len__(self) -> int:
        return len(self.items)


def _fallback(name: str) -> list:
    path = FALLBACK_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no fallback bundled for {name!r} at {path}")
    return json.loads(path.read_text())


def _try_hf(fn, name: str) -> Split:
    try:
        items = fn()
        if items:
            return Split(name, items, "huggingface")
    except Exception as exc:  # noqa: BLE001 - any hub/network failure is equivalent here
        print(f"[data] {name}: HF load failed ({type(exc).__name__}: {exc}); using bundled fallback")
    return Split(name, _fallback(name), "fallback")


# --------------------------------------------------------------- harmful sets


def load_advbench(n: int | None = None, seed: int = 0) -> Split:
    def fn():
        from datasets import load_dataset

        ds = load_dataset("walledai/AdvBench", split="train")
        return [r["prompt"] for r in ds]

    return _subsample(_try_hf(fn, "advbench"), n, seed)


def load_harmbench(n: int | None = None, seed: int = 0) -> Split:
    def fn():
        from datasets import load_dataset

        ds = load_dataset("walledai/HarmBench", "standard", split="train")
        return [r["prompt"] for r in ds]

    return _subsample(_try_hf(fn, "harmbench"), n, seed)


def load_jailbreakbench(n: int | None = None, seed: int = 0) -> Split:
    """JBB-Behaviors harmful behaviours, used for the safety family (B)."""

    def fn():
        from datasets import load_dataset

        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        return [{"prompt": r["Goal"], "category": r.get("Category", "")} for r in ds]

    split = _try_hf(fn, "jailbreakbench")
    split.items = [i if isinstance(i, dict) else {"prompt": i, "category": ""} for i in split.items]
    return _subsample(split, n, seed)


# -------------------------------------------------------------- harmless sets


def load_alpaca(n: int | None = None, seed: int = 0) -> Split:
    def fn():
        from datasets import load_dataset

        ds = load_dataset("tatsu-lab/alpaca", split="train")
        return [r["instruction"] for r in ds if not r["input"]]

    return _subsample(_try_hf(fn, "alpaca"), n, seed)


# ----------------------------------------------------------------- capability


def load_mmlu(n: int = 500, seed: int = 0) -> Split:
    def fn():
        from datasets import load_dataset

        ds = load_dataset("cais/mmlu", "all", split="test")
        return [
            {"question": r["question"], "choices": list(r["choices"]), "answer": int(r["answer"]),
             "subject": r.get("subject", "")}
            for r in ds
        ]

    return _subsample(_try_hf(fn, "mmlu"), n, seed)


def load_truthfulqa(n: int = 400, seed: int = 0) -> Split:
    """TruthfulQA MC1 -- Arditi et al. found this the benchmark most sensitive
    to directional ablation, so it is our most informative capability probe."""

    def fn():
        from datasets import load_dataset

        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
        out = []
        for r in ds:
            mc1 = r["mc1_targets"]
            out.append(
                {"question": r["question"], "choices": list(mc1["choices"]),
                 "answer": int(list(mc1["labels"]).index(1))}
            )
        return out

    return _subsample(_try_hf(fn, "truthfulqa"), n, seed)


def load_gsm8k(n: int = 100, seed: int = 0) -> Split:
    def fn():
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [{"question": r["question"], "answer": r["answer"].split("####")[-1].strip()} for r in ds]

    return _subsample(_try_hf(fn, "gsm8k"), n, seed)


# ------------------------------------------------------------------- concepts


def load_concepts(n: int | None = None, seed: int = 0) -> list[dict]:
    """The concept bank used both for injection and as the forced-choice pool.

    Each entry has ``name``, a set of ``prompts`` that instantiate the concept
    (used for the difference-of-means concept vector), and ``aliases`` accepted
    by the identification grader.
    """
    concepts = json.loads((DATA_DIR / "concepts.json").read_text())
    if n is not None and n < len(concepts):
        rng = random.Random(seed)
        concepts = rng.sample(concepts, n)
    return concepts


def load_baseline_corpus() -> list[str]:
    """Generic text used as the contrast set for concept vectors and as the
    calibration set for the residual-norm scale."""
    return json.loads((DATA_DIR / "baseline_corpus.json").read_text())


# --------------------------------------------------------------------- helpers


def _subsample(split: Split, n: int | None, seed: int) -> Split:
    if n is not None and n < len(split.items):
        rng = random.Random(seed)
        split.items = rng.sample(list(split.items), n)
    return split


def direction_splits(cfg, seed: int = 0) -> dict[str, Split]:
    """Train/val splits for refusal-direction extraction (section 3.2).

    Training harmful comes from AdvBench, training harmless from Alpaca, and
    validation uses *held-out* HarmBench + Alpaca so that direction selection is
    never scored on the data that produced it.
    """
    harmful_train = load_advbench(cfg.n_train_harmful, seed)
    harmful_val = load_harmbench(cfg.n_val_harmful, seed + 1)
    alpaca = load_alpaca(cfg.n_train_harmless + cfg.n_val_harmless + cfg.n_kl_prompts, seed)
    items = list(alpaca.items)
    harmless_train = Split("alpaca_train", items[: cfg.n_train_harmless], alpaca.source)
    harmless_val = Split(
        "alpaca_val", items[cfg.n_train_harmless : cfg.n_train_harmless + cfg.n_val_harmless], alpaca.source
    )
    kl_probe = Split("alpaca_kl", items[cfg.n_train_harmless + cfg.n_val_harmless :], alpaca.source)
    return {
        "harmful_train": harmful_train,
        "harmless_train": harmless_train,
        "harmful_val": harmful_val,
        "harmless_val": harmless_val,
        "kl_probe": kl_probe,
    }
