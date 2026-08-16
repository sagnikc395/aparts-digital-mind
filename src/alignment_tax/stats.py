"""Aggregation and inference (section 3.6).

Two decisions matter here and both are about the unit of analysis:

* **Resample over concepts, not trials.** Concepts are the unit of
  generalisation; treating 20 trials on one concept as 20 independent samples
  inflates significance by roughly the design effect. All confidence intervals
  come from a concept-level bootstrap with 10,000 resamples.
* **Report effect sizes with intervals**, and use a two-proportion test with
  Holm correction across the lambda grid only for the explicit pairwise claims.

The derived quantities are

    d'_clean  = z(TPR) - z(FPR_clean)
    d'_random = z(TPR) - z(FPR_random)
    specificity index = d'_random / d'_clean

A specificity index near 1 means the model is tracking concept *identity*; near
0 means it is detecting perturbation and confabulating content.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import NormalDist

from .io_utils import read_jsonl

_ND = NormalDist()
N_BOOT = 10_000


def z(p: float) -> float:
    return _ND.inv_cdf(min(max(p, 1e-6), 1 - 1e-6))


@dataclass
class Estimate:
    value: float
    lo: float
    hi: float
    n_concepts: int

    def as_tuple(self) -> tuple[float, float, float]:
        return self.value, self.lo, self.hi


# ------------------------------------------------------------- per-concept counts


def concept_counts(records: list[dict]) -> dict[tuple[float, str], dict[str, dict[str, float]]]:
    """Collapse raw trials into per-(lambda, concept) counts per condition.

    Keeping counts rather than rates lets the bootstrap resample concepts and
    then pool, which is the correct order of operations for unequal trial counts.
    """
    out: dict[tuple[float, str], dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    for r in records:
        cell = out[(r["lam"], r["concept"])][r["condition"]]
        cell["n"] += 1
        cell["parseable"] += int(bool(r.get("parseable")))
        if r.get("detected") is not None:
            cell["n_detect_scored"] += 1
            cell["detected"] += int(r["detected"])
        if r.get("identified") is not None:
            cell["n_ident_scored"] += 1
            cell["identified"] += int(r["identified"])
            if r.get("detected"):
                cell["detect_and_identify"] += int(r["identified"])
        cell["words"] += r.get("n_words", 0)
        if r.get("chance") is not None:
            cell["chance_sum"] += r["chance"]
            cell["n_chance"] += 1
    return {k: {c: dict(v) for c, v in d.items()} for k, d in out.items()}


def _rate(cells: list[dict], num: str, den: str) -> float | None:
    n = sum(c.get(den, 0.0) for c in cells)
    if n == 0:
        return None
    return sum(c.get(num, 0.0) for c in cells) / n


def _metrics_from(counts: dict[str, list[dict]]) -> dict[str, float | None]:
    """Pool per-concept counts into the Family A metric set."""
    c1, c2, c3, c4 = (counts.get(k, []) for k in ("C1", "C2", "C3", "C4"))
    tpr = _rate(c1, "detected", "n_detect_scored")
    fpr_clean = _rate(c2, "detected", "n_detect_scored")
    fpr_random = _rate(c3, "detected", "n_detect_scored")

    m: dict[str, float | None] = {
        "tpr": tpr,
        "fpr_clean": fpr_clean,
        "fpr_random": fpr_random,
        "identification": _rate(c1, "identified", "n_ident_scored"),
        "joint": _rate(c1, "detect_and_identify", "n"),
        "forced_choice": _rate(c4, "identified", "n_ident_scored"),
        "forced_choice_chance": _rate(c4, "chance_sum", "n_chance"),
        "parse_rate": _rate(c1 + c2 + c3, "parseable", "n"),
        "mean_words": _rate(c1 + c2 + c3, "words", "n"),
    }
    detected_n = sum(c.get("detected", 0.0) for c in c1)
    m["conditional_identification"] = (
        sum(c.get("detect_and_identify", 0.0) for c in c1) / detected_n if detected_n else None
    )
    if tpr is not None and fpr_clean is not None:
        m["d_clean"] = z(tpr) - z(fpr_clean)
    if tpr is not None and fpr_random is not None:
        m["d_random"] = z(tpr) - z(fpr_random)
    if m.get("d_clean") not in (None, 0.0) and m.get("d_random") is not None:
        m["specificity_index"] = m["d_random"] / m["d_clean"]
    return m


def summarise(
    records: list[dict], n_boot: int = N_BOOT, seed: int = 0, variant: str = "structured"
) -> dict[float, dict[str, Estimate]]:
    """Per-lambda metrics with concept-level bootstrap CIs."""
    records = [r for r in records if r.get("variant", variant) == variant]
    counts = concept_counts(records)
    lambdas = sorted({lam for lam, _ in counts})
    rng = random.Random(seed)
    out: dict[float, dict[str, Estimate]] = {}

    for lam in lambdas:
        concepts = sorted({c for l, c in counts if l == lam})
        by_concept = {c: counts[(lam, c)] for c in concepts}
        point = _metrics_from(_stack([by_concept[c] for c in concepts]))

        draws: dict[str, list[float]] = defaultdict(list)
        for _ in range(n_boot):
            sample = [by_concept[rng.choice(concepts)] for _ in concepts]
            for k, v in _metrics_from(_stack(sample)).items():
                if v is not None and math.isfinite(v):
                    draws[k].append(v)

        out[lam] = {
            k: Estimate(v, *_percentile_ci(draws.get(k, [])), len(concepts))
            for k, v in point.items()
            if v is not None
        }
    return out


def _stack(per_concept: list[dict[str, dict]]) -> dict[str, list[dict]]:
    stacked: dict[str, list[dict]] = defaultdict(list)
    for concept_cells in per_concept:
        for cond, cell in concept_cells.items():
            stacked[cond].append(cell)
    return stacked


def _percentile_ci(draws: list[float], alpha: float = 0.05) -> tuple[float, float]:
    if not draws:
        return (float("nan"), float("nan"))
    s = sorted(draws)
    lo = s[max(0, int(alpha / 2 * len(s)) - 1)]
    hi = s[min(len(s) - 1, int((1 - alpha / 2) * len(s)))]
    return lo, hi


# ------------------------------------------------------------------- hypothesis tests


def two_proportion_test(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Pooled two-proportion z-test; the raw ingredient for the lambda contrasts."""
    if n1 == 0 or n2 == 0:
        return {"z": float("nan"), "p": float("nan"), "diff": float("nan")}
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"z": 0.0, "p": 1.0, "diff": p1 - p2}
    zstat = (p1 - p2) / se
    return {"z": zstat, "p": 2 * (1 - _ND.cdf(abs(zstat))), "diff": p1 - p2}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down correction."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted


def lambda_contrasts(records: list[dict], condition: str = "C1", field: str = "detected",
                     variant: str = "structured") -> dict[str, dict]:
    """Every lambda vs the lambda=0 baseline, Holm-corrected across the grid."""
    rows = [r for r in records
            if r.get("condition") == condition and r.get(field) is not None
            and r.get("variant", variant) == variant]
    by_lam: dict[float, list[int]] = defaultdict(list)
    for r in rows:
        by_lam[r["lam"]].append(int(r[field]))
    lambdas = sorted(by_lam)
    if not lambdas:
        return {}
    base = by_lam[lambdas[0]]
    tests = {
        f"lambda={lam}_vs_{lambdas[0]}": two_proportion_test(
            sum(by_lam[lam]), len(by_lam[lam]), sum(base), len(base)
        )
        for lam in lambdas[1:]
    }
    adj = holm({k: v["p"] for k, v in tests.items()})
    for k in tests:
        tests[k]["p_holm"] = adj[k]
    return tests


def binomial_vs_chance(k: int, n: int, chance: float) -> dict:
    """One-sided exact-ish test that forced-choice accuracy beats 1/k."""
    if n == 0:
        return {"p": float("nan")}
    p_hat = k / n
    se = math.sqrt(chance * (1 - chance) / n)
    zstat = (p_hat - chance) / se if se > 0 else 0.0
    return {"accuracy": p_hat, "chance": chance, "z": zstat, "p": 1 - _ND.cdf(zstat), "n": n}


# --------------------------------------------------------------------- exporting


def summary_table(summary: dict[float, dict[str, Estimate]]) -> list[dict]:
    rows = []
    for lam in sorted(summary):
        row: dict[str, object] = {"lam": lam}
        for k, est in summary[lam].items():
            row[k] = est.value
            row[f"{k}_lo"], row[f"{k}_hi"] = est.lo, est.hi
        rows.append(row)
    return rows


def load_and_summarise(path: Path, variant: str = "structured", n_boot: int = N_BOOT, seed: int = 0):
    records = list(read_jsonl(path))
    return records, summarise(records, n_boot=n_boot, seed=seed, variant=variant)


def to_json(summary: dict[float, dict[str, Estimate]]) -> dict:
    return {str(lam): {k: asdict(v) for k, v in m.items()} for lam, m in summary.items()}
