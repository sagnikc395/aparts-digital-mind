"""Join the three metric families into one tidy record per lambda.

``analysis.json`` is the single artifact the figures and the paper read from, so
that no number in the write-up is computed twice by two different code paths.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .io_utils import read_jsonl
from .stats import (
    binomial_vs_chance,
    is_num,
    lambda_contrasts,
    load_and_summarise,
    summary_table,
    to_json,
)


def _num(row: dict, *keys):
    """First key in ``keys`` that holds a real number, else None."""
    for k in keys:
        if is_num(row.get(k)):
            return row[k]
    return None


def safety_by_lambda(path: Path) -> dict[float, dict]:
    rows = defaultdict(list)
    for r in read_jsonl(path):
        rows[r["lam"]].append(r)
    out = {}
    for lam, rs in rows.items():
        n = len(rs)
        judged = [r for r in rs if r.get("safe") is not None]
        out[lam] = {
            "refusal_rate": sum(r["refused"] for r in rs) / n,
            "compliance_proxy": sum(r["complied_proxy"] for r in rs) / n,
            "safety_score": (sum(r["safe"] for r in judged) / len(judged)) if judged else None,
            "mean_words": sum(r["n_words"] for r in rs) / n,
            "n": n,
        }
    return out


def capability_by_lambda(path: Path) -> dict[float, dict]:
    if not Path(path).exists():
        return {}
    return {r["lam"]: {k: v for k, v in r.items() if k != "lam"} for r in json.loads(Path(path).read_text())}


def parse_diagnostics(records: list[dict], variant: str, min_parse_rate: float = 0.5) -> dict:
    """Parse rate per (lambda, condition), and a warning when it is on the floor.

    An injection strong enough to derail generation makes every report
    unparseable, which silently voids TPR, the false-positive rates and both
    d-primes. That failure has to be loud and it has to be visible in the
    artifact, because it is indistinguishable from "no effect" downstream.
    """
    cells: dict[tuple, list[int]] = defaultdict(list)
    for r in records:
        if r.get("variant", variant) != variant or r.get("condition") == "C4":
            continue
        cells[(r["lam"], r["condition"])].append(int(r.get("detected") is not None))

    by_cell = {f"lam={lam}/{cond}": sum(v) / len(v) for (lam, cond), v in sorted(cells.items())}
    injected = [rate for key, rate in by_cell.items() if key.endswith(("C1", "C3"))]
    worst = min(injected) if injected else None
    warnings = []
    if worst is not None and worst < min_parse_rate:
        warnings.append(
            f"parse rate under injection is {worst:.0%} (< {min_parse_rate:.0%}): the injected "
            f"reports are not in the requested format, so TPR / FPR_random / d' are undefined. "
            f"Lower InjectionConfig.alpha (currently the sweep's alpha) or set "
            f"Injection.positions='prompt', then re-run the sweep."
        )
    return {"parse_rate_by_cell": by_cell, "min_injected_parse_rate": worst, "warnings": warnings}


def build_analysis(
    sweep_path: Path,
    safety_path: Path | None,
    capability_path: Path | None,
    out_path: Path,
    variant: str = "structured",
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    records, summary = load_and_summarise(sweep_path, variant=variant, n_boot=n_boot, seed=seed)
    safety = safety_by_lambda(safety_path) if safety_path and Path(safety_path).exists() else {}
    capability = capability_by_lambda(capability_path) if capability_path else {}

    rows = summary_table(summary)
    for row in rows:
        lam = row["lam"]
        row.update({f"safety_{k}": v for k, v in safety.get(lam, {}).items()})
        row.update({f"cap_{k}": v for k, v in capability.get(lam, {}).items()})

    # Forced choice vs the 1/k baseline, per lambda.
    fc = {}
    for lam in sorted({r["lam"] for r in records}):
        c4 = [r for r in records if r["lam"] == lam and r["condition"] == "C4"]
        if c4:
            chance = sum(r.get("chance", 0.1) for r in c4) / len(c4)
            fc[str(lam)] = binomial_vs_chance(sum(bool(r["identified"]) for r in c4), len(c4), chance)

    diagnostics = parse_diagnostics(records, variant)
    for w in diagnostics["warnings"]:
        print(f"[analysis] WARNING: {w}")

    analysis = {
        "variant": variant,
        "n_records": len(records),
        "diagnostics": diagnostics,
        "lambdas": sorted({r["lam"] for r in records}),
        "rows": rows,
        "summary": to_json(summary),
        "safety": {str(k): v for k, v in safety.items()},
        "capability": {str(k): v for k, v in capability.items()},
        "forced_choice_vs_chance": fc,
        "contrasts": {
            "detection_C1": lambda_contrasts(records, "C1", "detected", variant),
            "fpr_random_C3": lambda_contrasts(records, "C3", "detected", variant),
            "identification_C1": lambda_contrasts(records, "C1", "identified", variant),
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(analysis, indent=2))
    return analysis


def frontier_points(analysis: dict) -> list[dict]:
    """(introspection gain, safety, capability loss) triples for Figure 2.

    Introspection gain is d'_random -- the discriminability that survives the
    norm-matched random-direction control -- because that is the quantity we are
    willing to call introspection. Capability loss is the drop in the mean of the
    available capability benchmarks relative to lambda = 0.
    """
    rows = {r["lam"]: r for r in analysis["rows"]}
    lam0 = min(rows) if rows else None
    keys = [k for k in ("cap_mmlu", "cap_truthfulqa_mc1", "cap_gsm8k") if lam0 is not None and k in rows[lam0]]
    pts = []
    for lam in sorted(rows):
        r = rows[lam]
        base = [rows[lam0][k] for k in keys]
        cur = [r.get(k) for k in keys]
        cap_loss = (
            sum(b - c for b, c in zip(base, cur) if c is not None) / len(keys) if keys else 0.0
        )
        # d'_random is the preferred axis; fall back to raw TPR when the
        # random-direction control is undefined (e.g. unparseable reports).
        use_tpr = _num(r, "d_random") is None
        src = ("tpr",) if use_tpr else ("d_random",)
        pts.append({
            "lam": lam,
            "introspection": _num(r, *src),
            "introspection_lo": _num(r, f"{src[0]}_lo"),
            "introspection_hi": _num(r, f"{src[0]}_hi"),
            "introspection_metric": src[0],
            "safety": _num(r, "safety_safety_score", "safety_refusal_rate"),
            "capability_loss": cap_loss,
            "ce_loss": _num(r, "cap_ce_loss"),
        })
    return pts


def exchange_rate(analysis: dict) -> dict:
    """The headline number: safety lost per unit of introspection gained.

    Computed as the slope between the lambda = 0 and lambda = 1 endpoints, plus
    the local slope at each step so that a non-linear frontier is visible.
    """
    all_pts = frontier_points(analysis)
    pts = [p for p in all_pts if p["introspection"] is not None and p["safety"] is not None]
    if len(pts) < 2:
        missing = "introspection" if all(p["introspection"] is None for p in all_pts) else "safety"
        return {"unavailable": f"{missing} is undefined at {len(all_pts) - len(pts)}/{len(all_pts)} "
                               f"lambdas; see analysis['diagnostics']"}
    a, b = pts[0], pts[-1]
    d_int = b["introspection"] - a["introspection"]
    steps = []
    for p, q in zip(pts, pts[1:]):
        di = q["introspection"] - p["introspection"]
        steps.append({
            "from": p["lam"], "to": q["lam"],
            "d_introspection": di,
            "d_safety": q["safety"] - p["safety"],
            "slope": (q["safety"] - p["safety"]) / di if di else None,
        })
    return {
        "endpoint_d_introspection": d_int,
        "endpoint_d_safety": b["safety"] - a["safety"],
        "endpoint_d_capability": b["capability_loss"] - a["capability_loss"],
        "safety_per_introspection": (b["safety"] - a["safety"]) / d_int if d_int else None,
        "steps": steps,
    }
