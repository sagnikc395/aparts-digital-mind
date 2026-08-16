"""Stage functions shared by the CLI and the Colab notebooks.

Each stage takes a ``RunConfig``, writes its artifact under ``cfg.run_dir``, and
is safe to re-run: existing artifacts are loaded rather than recomputed unless
``force=True``. That is what makes the notebooks restartable after a session
kill without a separate bookkeeping layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from . import data
from .analysis import build_analysis, exchange_rate
from .capability import run_capability_sweep
from .concepts import ConceptBank, build_concept_bank, dump_bank_summary, resolve_layers
from .config import RunConfig
from .directions import RefusalDirection, extract_refusal_direction
from .introspection import run_pilot, run_sweep
from .judge import make_judge
from .model import HookedModel
from .safety import LlamaGuardJudge, run_safety_sweep


def load_model(cfg: RunConfig) -> HookedModel:
    hm = HookedModel(cfg.model)
    print(f"[model] {cfg.model.name} on {hm.device} ({hm.dtype}), "
          f"{hm.n_layers} layers, d_model={hm.d_model}")
    return hm


# --------------------------------------------------------------------- stages


def stage_direction(hm: HookedModel, cfg: RunConfig, force: bool = False) -> RefusalDirection:
    path = cfg.artifact("refusal_direction.pt")
    if path.exists() and not force:
        rd = RefusalDirection.load(path)
        print(f"[direction] loaded layer={rd.layer} position={rd.position}")
        return rd
    splits = data.direction_splits(cfg.direction, seed=cfg.seed)
    for k, v in splits.items():
        print(f"[data] {k}: n={len(v)} source={v.source}")
    rd = extract_refusal_direction(hm, splits, cfg.direction)
    rd.save(path)
    return rd


def stage_concepts(hm: HookedModel, cfg: RunConfig, force: bool = False) -> ConceptBank:
    path = cfg.artifact("concept_bank.pt")
    if path.exists() and not force:
        bank = ConceptBank.load(path)
        print(f"[concepts] loaded {len(bank.names)} concepts")
        return bank
    concepts = data.load_concepts(cfg.injection.n_concepts, seed=cfg.seed)
    baseline = data.load_baseline_corpus()
    layers = resolve_layers(hm.n_layers, cfg.injection.layer_fracs)
    if cfg.injection.layer is not None:
        layers = sorted(set(layers) | {cfg.injection.layer})
    bank = build_concept_bank(hm, concepts, baseline, layers=layers)
    bank.save(path)
    dump_bank_summary(bank, layers[len(layers) // 2], cfg.artifact("concept_bank_summary.json"))
    return bank


def stage_pilot(hm: HookedModel, cfg: RunConfig, bank: ConceptBank, direction: torch.Tensor | None,
                n_concepts: int = 20, n_trials: int = 5) -> dict:
    """Go/no-go gate. Returns the decision alongside the per-cell accuracies."""
    path = cfg.artifact("pilot.jsonl")
    layers = resolve_layers(hm.n_layers, cfg.injection.layer_fracs)
    run_pilot(hm, bank, direction, cfg.injection, path, layers=layers,
              alphas=cfg.injection.alpha_pilot, lambdas=(0.0, 1.0),
              n_concepts=min(n_concepts, len(bank.names)), n_trials=n_trials, seed=cfg.seed)
    bank.save(cfg.artifact("concept_bank.pt"))
    decision = summarise_pilot(path, k=cfg.injection.k_choices)
    cfg.artifact("pilot_decision.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision["verdict"], indent=2))
    return decision


def summarise_pilot(path: Path, k: int = 10) -> dict:
    """Aggregate the pilot and apply the stated decision rule."""
    from collections import defaultdict

    from .io_utils import read_jsonl
    from .stats import binomial_vs_chance

    cells = defaultdict(list)
    for r in read_jsonl(path):
        cells[(r["protocol"], r["lam"], r["layer"], r["alpha"])].append(int(bool(r["identified"])))

    rows = []
    for (protocol, lam, layer, alpha), vals in sorted(cells.items()):
        chance = 1.0 / k if protocol == "forced_choice" else 0.0
        test = binomial_vs_chance(sum(vals), len(vals), max(chance, 1e-9))
        rows.append({"protocol": protocol, "lam": lam, "layer": layer, "alpha": alpha,
                     "accuracy": sum(vals) / len(vals), "n": len(vals), "chance": chance,
                     "p_vs_chance": test["p"]})

    fc = [r for r in rows if r["protocol"] == "forced_choice"]
    best = max(fc, key=lambda r: r["accuracy"]) if fc else None
    above_chance = bool(best and best["accuracy"] > best["chance"] and best["p_vs_chance"] < 0.05)
    rises = False
    if fc:
        by_lam = defaultdict(list)
        for r in fc:
            by_lam[r["lam"]].append(r["accuracy"])
        if 0.0 in by_lam and 1.0 in by_lam:
            rises = max(by_lam[1.0]) > max(by_lam[0.0])

    verdict = {
        "above_chance": above_chance,
        "rises_under_ablation": rises,
        "best_cell": best,
        "decision": "proceed" if above_chance else "switch model or fall back to the reduced claim",
    }
    return {"rows": rows, "verdict": verdict}


def stage_sweep(hm: HookedModel, cfg: RunConfig, bank: ConceptBank, direction: torch.Tensor | None,
                variant: str = "structured", lambdas=None, use_llm_judge: bool = False) -> Path:
    judge = make_judge(bank.aliases, cfg.artifact("judge_cache.json"), use_llm=use_llm_judge)
    path = cfg.artifact(f"sweep_{variant}.jsonl")
    out = run_sweep(hm, bank, direction, cfg.injection, lambdas or cfg.lambdas, path,
                    judge=judge, variant=variant, seed=cfg.seed)
    bank.save(cfg.artifact("concept_bank.pt"))  # persist any newly calibrated layer scales
    return out


def stage_safety(hm: HookedModel, cfg: RunConfig, direction: torch.Tensor | None,
                 use_guard: bool = False, guard_model: str = "meta-llama/Llama-Guard-3-8B") -> Path:
    behaviours = data.load_jailbreakbench(cfg.evals.n_jailbreak, seed=cfg.seed).items
    guard = LlamaGuardJudge(guard_model) if use_guard else None
    return run_safety_sweep(hm, behaviours, direction, cfg.lambdas, cfg.artifact("safety.jsonl"),
                            max_new_tokens=cfg.evals.safety_max_new_tokens, guard=guard)


def stage_capability(hm: HookedModel, cfg: RunConfig, direction: torch.Tensor | None) -> Path:
    mmlu = data.load_mmlu(cfg.evals.n_mmlu, seed=cfg.seed).items
    tqa = data.load_truthfulqa(cfg.evals.n_truthfulqa, seed=cfg.seed).items
    alpaca = data.load_alpaca(cfg.evals.n_ce_loss, seed=cfg.seed + 7).items
    gsm = data.load_gsm8k(cfg.evals.n_gsm8k, seed=cfg.seed).items if cfg.evals.run_gsm8k else None
    return run_capability_sweep(hm, direction, cfg.lambdas, mmlu, tqa, alpaca,
                                cfg.artifact("capability.json"), gsm8k=gsm,
                                gsm8k_max_new_tokens=cfg.evals.gsm8k_max_new_tokens)


def stage_analyse(cfg: RunConfig, variant: str = "structured", n_boot: int = 10_000) -> dict:
    analysis = build_analysis(
        cfg.artifact(f"sweep_{variant}.jsonl"),
        cfg.artifact("safety.jsonl"),
        cfg.artifact("capability.json"),
        cfg.artifact(f"analysis_{variant}.json"),
        variant=variant,
        n_boot=n_boot,
        seed=cfg.seed,
    )
    analysis["exchange_rate"] = exchange_rate(analysis)
    cfg.artifact(f"analysis_{variant}.json").write_text(json.dumps(analysis, indent=2))
    return analysis


def stage_figures(cfg: RunConfig, variant: str = "structured", fig_dir: Path | None = None) -> list[Path]:
    from .config import FIGURE_DIR
    from .figures import make_all

    return make_all(cfg.artifact(f"analysis_{variant}.json"), fig_dir or FIGURE_DIR)


def run_all(cfg: RunConfig, skip_pilot: bool = False, use_guard: bool = False,
            use_llm_judge: bool = False) -> dict:
    """The whole pipeline, in the order of the timeline in section 6."""
    cfg.save()
    hm = load_model(cfg)
    rd = stage_direction(hm, cfg)
    bank = stage_concepts(hm, cfg)
    if not skip_pilot:
        stage_pilot(hm, cfg, bank, rd.vector)
    stage_sweep(hm, cfg, bank, rd.vector, use_llm_judge=use_llm_judge)
    stage_safety(hm, cfg, rd.vector, use_guard=use_guard)
    stage_capability(hm, cfg, rd.vector)
    analysis = stage_analyse(cfg)
    stage_figures(cfg)
    return analysis
