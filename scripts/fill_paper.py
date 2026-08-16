"""Substitute measured numbers into the paper draft.

``paper/paper.md`` contains placeholders of the form ``{{key}}``. This script
reads ``analysis_structured.json`` from a run directory and fills them in,
writing ``paper/paper_filled.md``. Any placeholder with no matching number is
left visibly as ``[[MISSING: key]]`` rather than being quietly dropped, so an
unfinished results section cannot be mistaken for a finished one.

    python scripts/fill_paper.py results/Qwen3-4B-Instruct-2507
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = re.compile(r"\{\{([a-zA-Z0-9_.@=-]+)\}\}")


def flatten(analysis: dict) -> dict[str, str]:
    """Build the placeholder table.

    Keys look like ``tpr@0``, ``d_random@1.0``, ``safety_refusal_rate@0.5``,
    ``exchange.safety_per_introspection``.
    """
    out: dict[str, str] = {}

    def fmt(v, digits=3):
        if v is None:
            return None
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return str(v)

    for row in analysis.get("rows", []):
        lam = f"{row['lam']:g}"
        for k, v in row.items():
            if k == "lam" or v is None:
                continue
            f = fmt(v)
            if f is not None:
                out[f"{k}@{lam}"] = f

    for lam, metrics in analysis.get("summary", {}).items():
        lam_s = f"{float(lam):g}"
        for k, est in metrics.items():
            out[f"{k}_ci@{lam_s}"] = f"[{est['lo']:.3f}, {est['hi']:.3f}]"

    for k, v in (analysis.get("exchange_rate") or {}).items():
        if isinstance(v, (int, float)):
            out[f"exchange.{k}"] = fmt(v)

    for name, tests in (analysis.get("contrasts") or {}).items():
        for contrast, res in tests.items():
            out[f"p.{name}.{contrast}"] = f"{res['p_holm']:.2g}" if res.get("p_holm") is not None else None

    for lam, fc in (analysis.get("forced_choice_vs_chance") or {}).items():
        lam_s = f"{float(lam):g}"
        out[f"fc_acc@{lam_s}"] = fmt(fc.get("accuracy"))
        out[f"fc_chance@{lam_s}"] = fmt(fc.get("chance"))
        out[f"fc_p@{lam_s}"] = f"{fc['p']:.2g}" if fc.get("p") is not None else None

    out["n_records"] = str(analysis.get("n_records", ""))
    out["lambdas"] = ", ".join(f"{l:g}" for l in analysis.get("lambdas", []))
    return {k: v for k, v in out.items() if v is not None}


def run_metadata(run_dir: Path) -> dict[str, str]:
    """Placeholders that come from the run artifacts rather than the analysis:
    the selected direction, the pilot verdict, the judge agreement, and the
    sweep dimensions."""
    out: dict[str, str] = {}

    rd = run_dir / "refusal_direction.json"
    if rd.exists():
        d = json.loads(rd.read_text())
        sel = d["selected"]
        out.update({
            "direction_layer": str(d["layer"]),
            "direction_position": str(d["position"]),
            "direction_bypass": f"{sel['bypass_score']:+.3f}",
            "direction_induce": f"{sel['induce_score']:+.3f}",
            "direction_kl": f"{sel['kl']:.4f}",
            "direction_note": sel.get("reason") or "all filters passed",
        })

    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        c = json.loads(cfg_path.read_text())
        out.update({
            "model_name": c["model"]["name"],
            "n_concepts": str(c["injection"]["n_concepts"]),
            "n_trials": str(c["injection"]["n_trials"]),
            "alpha": f"{c['injection']['alpha']:g}",
            "injection_layer": str(c["injection"]["layer"]),
            "k_choices": str(c["injection"]["k_choices"]),
        })

    pilot = run_dir / "pilot_decision.json"
    if pilot.exists():
        v = json.loads(pilot.read_text())["verdict"]
        out["pilot_decision"] = v["decision"]
        if v.get("best_cell"):
            b = v["best_cell"]
            out["pilot_best"] = (f"layer {b['layer']}, alpha {b['alpha']:g}, "
                                 f"accuracy {b['accuracy']:.3f} vs chance {b['chance']:.3f}")

    kappa = run_dir / "judge_kappa.json"
    if kappa.exists():
        k = json.loads(kappa.read_text())
        out["judge_kappa"] = f"{k['kappa']:.2f}" if k.get("kappa") is not None else None
    return {k: v for k, v in out.items() if v is not None}


def fill(template: str, table: dict[str, str]) -> tuple[str, list[str]]:
    missing: list[str] = []

    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key in table:
            return table[key]
        missing.append(key)
        return f"[[MISSING: {key}]]"

    return PLACEHOLDER.sub(sub, template), missing


def main(argv: list[str]) -> int:
    run_dir = Path(argv[1]) if len(argv) > 1 else None
    if run_dir is None:
        candidates = sorted((ROOT / "results").rglob("analysis_structured.json"))
        if not candidates:
            print("no analysis_structured.json found; run the analyse stage first")
            return 1
        analysis_path = candidates[-1]
    else:
        analysis_path = run_dir / "analysis_structured.json"

    analysis = json.loads(analysis_path.read_text())
    table = flatten(analysis) | run_metadata(analysis_path.parent)

    src = ROOT / "paper" / "paper.md"
    out = ROOT / "paper" / "paper_filled.md"
    filled, missing = fill(src.read_text(), table)
    out.write_text(filled)

    print(f"filled {out} from {analysis_path}")
    if missing:
        print(f"{len(missing)} unresolved placeholder(s): {sorted(set(missing))[:12]}")
    (ROOT / "paper" / "numbers.json").write_text(json.dumps(table, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
