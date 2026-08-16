"""Command-line entry point.

    python -m alignment_tax.cli pilot        --model Qwen/Qwen3-4B-Instruct-2507
    python -m alignment_tax.cli direction
    python -m alignment_tax.cli sweep        --lambdas 0,0.25,0.5,0.75,1.0
    python -m alignment_tax.cli safety
    python -m alignment_tax.cli capability
    python -m alignment_tax.cli analyse
    python -m alignment_tax.cli figures
    python -m alignment_tax.cli all --smoke

``--smoke`` swaps in a tiny cached model and miniature splits so the whole
pipeline can be exercised end to end in a few minutes on a laptop.
"""

from __future__ import annotations

import argparse
import json

from .config import LAMBDA_OVER, RunConfig, smoke_config
from . import pipeline


def _cfg_from_args(a) -> RunConfig:
    cfg = smoke_config() if a.smoke else (RunConfig.load(a.config) if a.config else RunConfig())
    if a.model:
        cfg.model.name = a.model
    if a.device:
        cfg.model.device = a.device
    if a.batch_size:
        cfg.model.batch_size = a.batch_size
    if a.lambdas:
        cfg.lambdas = tuple(float(x) for x in a.lambdas.split(","))
    if a.over_ablate:
        cfg.lambdas = tuple(sorted(set(cfg.lambdas) | {LAMBDA_OVER}))
    if a.concepts:
        cfg.injection.n_concepts = a.concepts
    if a.trials:
        cfg.injection.n_trials = a.trials
    if a.layer is not None:
        cfg.injection.layer = a.layer
    if a.alpha:
        cfg.injection.alpha = a.alpha
    if a.results_dir:
        from pathlib import Path

        cfg.results_dir = Path(a.results_dir)
    if a.gsm8k:
        cfg.evals.run_gsm8k = True
    return cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="alignment_tax", description="The alignment tax of introspection")
    p.add_argument("stage", choices=["pilot", "direction", "concepts", "sweep", "skeptical", "safety",
                                     "capability", "analyse", "figures", "all", "judge-sample",
                                     "judge-kappa"])
    p.add_argument("--model")
    p.add_argument("--device")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lambdas", help="comma-separated, e.g. 0,0.25,0.5,0.75,1.0")
    p.add_argument("--over-ablate", action="store_true", help="add lambda=1.25")
    p.add_argument("--concepts", type=int)
    p.add_argument("--trials", type=int)
    p.add_argument("--layer", type=int, help="absolute injection layer (fixed after the pilot)")
    p.add_argument("--alpha", type=float)
    p.add_argument("--results-dir")
    p.add_argument("--config", help="path to a saved run_config.json")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--gsm8k", action="store_true")
    p.add_argument("--llm-judge", action="store_true", help="use the Claude identification judge")
    p.add_argument("--guard", action="store_true", help="score safety with a Llama-Guard-class judge")
    p.add_argument("--skip-pilot", action="store_true")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--n-label", type=int, default=100)
    a = p.parse_args(argv)

    cfg = _cfg_from_args(a)
    cfg.save()

    needs_model = a.stage in {"pilot", "direction", "concepts", "sweep", "skeptical", "safety",
                              "capability", "all"}
    hm = pipeline.load_model(cfg) if needs_model else None

    if a.stage == "all":
        pipeline.run_all(cfg, skip_pilot=a.skip_pilot, use_guard=a.guard, use_llm_judge=a.llm_judge)
        return 0
    if a.stage == "direction":
        pipeline.stage_direction(hm, cfg)
        return 0
    if a.stage == "concepts":
        pipeline.stage_concepts(hm, cfg)
        return 0
    if a.stage == "pilot":
        rd = pipeline.stage_direction(hm, cfg)
        bank = pipeline.stage_concepts(hm, cfg)
        pipeline.stage_pilot(hm, cfg, bank, rd.vector)
        return 0
    if a.stage in {"sweep", "skeptical"}:
        rd = pipeline.stage_direction(hm, cfg)
        bank = pipeline.stage_concepts(hm, cfg)
        variant = "structured" if a.stage == "sweep" else "skeptical"
        # The skeptical variant is a robustness check at the endpoints only.
        lambdas = cfg.lambdas if variant == "structured" else (min(cfg.lambdas), max(cfg.lambdas))
        pipeline.stage_sweep(hm, cfg, bank, rd.vector, variant=variant, lambdas=lambdas,
                             use_llm_judge=a.llm_judge)
        return 0
    if a.stage == "safety":
        rd = pipeline.stage_direction(hm, cfg)
        pipeline.stage_safety(hm, cfg, rd.vector, use_guard=a.guard)
        return 0
    if a.stage == "capability":
        rd = pipeline.stage_direction(hm, cfg)
        pipeline.stage_capability(hm, cfg, rd.vector)
        return 0
    if a.stage == "analyse":
        analysis = pipeline.stage_analyse(cfg, n_boot=a.n_boot)
        print(json.dumps(analysis.get("exchange_rate", {}), indent=2))
        return 0
    if a.stage == "figures":
        pipeline.stage_figures(cfg)
        return 0
    if a.stage == "judge-sample":
        from .io_utils import read_jsonl
        from .judge import sample_for_hand_labelling

        recs = [r for r in read_jsonl(cfg.artifact("sweep_structured.jsonl")) if r["condition"] == "C1"]
        out = sample_for_hand_labelling(recs, a.n_label, cfg.artifact("judge_labels.jsonl"), seed=cfg.seed)
        print(f"wrote {out} -- fill in the 'human' field with 1/0 and run judge-kappa")
        return 0
    if a.stage == "judge-kappa":
        from .judge import cohens_kappa

        res = cohens_kappa(cfg.artifact("judge_labels.jsonl"))
        cfg.artifact("judge_kappa.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
