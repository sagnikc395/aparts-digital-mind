"""Family A: the introspection sweep.

Four conditions, all run at every lambda (section 3.3):

===  ===========================================  ==================================
C1   inject concept c                             signal
C2   no injection                                 clean control -> FPR as prior work
C3   norm-matched random direction                the missing cell
C4   inject c, forced choice over k candidates    identification, no affirmative bias
===  ===========================================  ==================================

C3 is what makes this more than an ablation study: prior work measures false
positives only against C2, which cannot distinguish an unlocked introspective
channel from an unlocked willingness to affirm that *something* happened.

Every record is appended to JSONL immediately and keyed by
``(lam, condition, concept, trial, variant)`` so a killed session resumes
exactly where it stopped.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import torch

from .concepts import ConceptBank, ensure_scale
from .config import InjectionConfig
from .io_utils import JsonlWriter
from .judge import LexicalJudge
from .model import HookedModel
from .prompts import VARIANTS, PREFILL_ASSISTANT, PREFILL_USER, build_candidates, forced_choice_prompt, \
    parse_prefill, parse_structured

KEY_FIELDS = ("lam", "condition", "concept", "trial", "variant")


def _detection_prompt(hm: HookedModel, variant: str) -> str:
    return hm.chat_prompt(VARIANTS[variant])


def _prefill_prompt(hm: HookedModel) -> str:
    return hm.chat_prompt(PREFILL_USER, prefill=PREFILL_ASSISTANT)


def run_sweep(
    hm: HookedModel,
    bank: ConceptBank,
    direction: torch.Tensor | None,
    cfg: InjectionConfig,
    lambdas: Sequence[float],
    out_path: Path,
    judge: LexicalJudge | None = None,
    variant: str = "structured",
    conditions: Sequence[str] = ("C1", "C2", "C3", "C4"),
    seed: int = 0,
    temperature: float = 1.0,
    verbose: bool = True,
) -> Path:
    """Main introspection sweep.

    Trial-to-trial variation comes from temperature sampling of the report (the
    intervention itself is deterministic), and for C4 additionally from the
    resampled candidate set.
    """
    judge = judge or LexicalJudge(bank.aliases)
    layer = cfg.layer if cfg.layer is not None else max(0, int(round(0.8 * hm.n_layers)))
    ensure_scale(bank, hm, layer)
    det_prompt = _detection_prompt(hm, variant)
    concepts = bank.names[: cfg.n_concepts]

    with JsonlWriter(out_path, KEY_FIELDS) as w:
        for lam in lambdas:
            with hm.ablated(direction, lam):
                for ci, concept in enumerate(concepts):
                    rng = random.Random((seed, lam, concept).__hash__() & 0xFFFFFFFF)
                    for condition in conditions:
                        if all(
                            w.has(lam=lam, condition=condition, concept=concept, trial=t, variant=variant)
                            for t in range(cfg.n_trials)
                        ):
                            continue
                        inj = _injection_for(bank, condition, concept, layer, cfg, seed)
                        if condition == "C4":
                            records = _forced_choice_trials(
                                hm, bank, concept, inj, cfg, rng, lam, variant
                            )
                        else:
                            records = _report_trials(
                                hm, det_prompt, concept, condition, inj, cfg, judge, lam, variant, temperature
                            )
                        for rec in records:
                            if not w.has(**{k: rec[k] for k in KEY_FIELDS}):
                                w.write(rec)
                    if verbose and (ci + 1) % 10 == 0:
                        print(f"[sweep] lambda={lam} {ci + 1}/{len(concepts)} concepts")
            if verbose:
                print(f"[sweep] lambda={lam} done -> {out_path}")
    return out_path


def _injection_for(bank: ConceptBank, condition: str, concept: str, layer: int,
                   cfg: InjectionConfig, seed: int):
    if condition == "C2":
        return None
    if condition == "C3":
        # Norm-matched random direction, deterministic per concept so that the
        # C1/C3 comparison is paired at the concept level.
        return bank.random_injection(layer, cfg.alpha, seed=abs(hash((seed, concept))) % (2**31))
    return bank.injection(concept, layer, cfg.alpha)


def _report_trials(hm, det_prompt, concept, condition, inj, cfg, judge, lam, variant, temperature) -> list[dict]:
    """C1/C2/C3: free-form structured self-report, n_trials sampled generations."""
    with hm.injected(inj):
        outs = hm.generate([det_prompt] * cfg.n_trials, do_sample=True, temperature=temperature)
    records = []
    for trial, text in enumerate(outs):
        parsed = parse_structured(text)
        # ``identified`` is None only when the trial cannot be scored at all --
        # i.e. the report was unparseable. A parseable C1 report that names no
        # concept (or names the wrong one) is a scored *failure*, not a missing
        # observation: dropping it would condition identification accuracy on
        # the model having volunteered a name, which inflates the metric.
        identified = None
        if condition == "C1" and parsed.parseable:
            identified = bool(parsed.concept) and judge.grade(concept, parsed.concept).correct
        records.append({
            "lam": lam, "condition": condition, "concept": concept, "trial": trial, "variant": variant,
            "layer": inj.layer if inj else None, "alpha": inj.alpha if inj else 0.0,
            "detected": parsed.detected, "parseable": parsed.parseable,
            "concept_reported": parsed.concept, "identified": identified,
            "n_words": len(text.split()), "response": text,
        })
    return records


def _forced_choice_trials(hm, bank, concept, inj, cfg, rng, lam, variant) -> list[dict]:
    """C4: k-way forced choice scored by log-probability.

    No generation and no yes/no channel, so affirmative bias cannot inflate it.
    Chance is 1/k by construction.
    """
    records = []
    for trial in range(cfg.n_trials):
        cands = build_candidates(concept, bank.names, cfg.k_choices, rng)
        prompt = hm.chat_prompt(forced_choice_prompt(cands))
        with hm.injected(inj):
            scores = hm.sequence_logprob([prompt] * len(cands), [f" {c}" for c in cands])
        pred = cands[max(range(len(cands)), key=lambda i: scores[i])]
        records.append({
            "lam": lam, "condition": "C4", "concept": concept, "trial": trial, "variant": variant,
            "layer": inj.layer if inj else None, "alpha": inj.alpha if inj else 0.0,
            "detected": None, "parseable": True, "concept_reported": pred,
            "identified": pred == concept, "chance": 1.0 / len(cands),
            "n_candidates": len(cands), "response": pred,
        })
    return records


# ----------------------------------------------------------- alpha calibration


def calibrate_alpha(
    hm: HookedModel,
    bank: ConceptBank,
    cfg: InjectionConfig,
    alphas: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
    layer: int | None = None,
    n_concepts: int = 8,
    n_trials: int = 3,
    variant: str = "structured",
    min_parse_rate: float = 0.8,
    verbose: bool = True,
) -> dict:
    """Choose the injection strength before spending the sweep budget on it.

    Two failure modes bracket the usable range and the pilot catches neither,
    because prefill and forced choice never require a parseable report:

    * **Too strong** -- the report stops following the requested format at all.
      That does not lower the detection metrics, it *voids* them: ``detected``
      becomes None and TPR, FPR_random and both d-primes lose their denominator.
    * **Too weak** -- the injection is real but produces no more detection than
      the norm-matched random direction, so there is no signal to measure.

    So the objective is not "the largest alpha the model survives". It is the
    largest *separation* between the concept direction (C1) and the norm-matched
    random direction (C3), subject to the reports staying parseable. C3 is
    measured at every alpha here for exactly that reason: on a model with strong
    affirmative bias the C1 detection rate alone is near 1.0 at every alpha
    including zero, and is therefore uninformative on its own.
    """
    layer = layer if layer is not None else (
        cfg.layer if cfg.layer is not None else max(0, int(round(0.8 * hm.n_layers)))
    )
    ensure_scale(bank, hm, layer)
    det_prompt = _detection_prompt(hm, variant)
    concepts = bank.names[:n_concepts]

    def measure(inj) -> tuple[int, int, int]:
        with hm.injected(inj):
            outs = hm.generate([det_prompt] * n_trials, do_sample=True, temperature=1.0)
        parsed = detected = 0
        for text in outs:
            p = parse_structured(text)
            parsed += int(p.parseable)
            detected += int(bool(p.detected))
        return parsed, detected, len(outs)

    rows = []
    with hm.ablated(None, 0.0):  # calibrate on the unmodified model
        for alpha in alphas:
            agg = {"C1": [0, 0, 0], "C3": [0, 0, 0]}  # parsed, detected, n
            for ci, concept in enumerate(concepts):
                for cond in ("C1", "C3"):
                    inj = (bank.injection(concept, layer, alpha) if cond == "C1"
                           else bank.random_injection(layer, alpha, seed=1000 + ci))
                    counts = measure(inj)
                    agg[cond] = [a + b for a, b in zip(agg[cond], counts)]
            row = {"alpha": alpha, "layer": layer, "n_per_condition": agg["C1"][2]}
            for cond in ("C1", "C3"):
                parsed, detected, n = agg[cond]
                row[f"{cond}_parse_rate"] = parsed / n
                row[f"{cond}_detection_rate"] = detected / n
            row["separation"] = row["C1_detection_rate"] - row["C3_detection_rate"]
            rows.append(row)
            if verbose:
                print(f"[calibrate] alpha={alpha:<5} parse C1={row['C1_parse_rate']:.2f} "
                      f"C3={row['C3_parse_rate']:.2f} | detect C1={row['C1_detection_rate']:.2f} "
                      f"C3={row['C3_detection_rate']:.2f} | separation={row['separation']:+.2f}")

    # Coherence is a hard constraint; among the alphas that clear it, take the
    # one that best separates the concept direction from a random one.
    usable = [r for r in rows
              if r["C1_parse_rate"] >= min_parse_rate and r["C3_parse_rate"] >= min_parse_rate]
    best = max(usable, key=lambda r: (r["separation"], r["alpha"])) if usable else None

    note = None
    if best is None:
        note = ("no alpha kept the reports parseable; try a shallower layer, a smaller alpha grid, "
                "or Injection(positions='prompt') so decoding is not corrupted")
    elif best["separation"] <= 0.0:
        note = ("no alpha separates the concept direction from a norm-matched random one; "
                "detection here reflects noticing a perturbation, not identifying a concept. "
                "The C4 forced-choice protocol is the metric that survives this.")

    return {"layer": layer, "min_parse_rate": min_parse_rate, "rows": rows,
            "recommended_alpha": best["alpha"] if best else None,
            "expected_separation": best["separation"] if best else None,
            "note": note}


# --------------------------------------------------------------------- pilot


def run_pilot(
    hm: HookedModel,
    bank: ConceptBank,
    direction: torch.Tensor | None,
    cfg: InjectionConfig,
    out_path: Path,
    layers: Sequence[int],
    alphas: Sequence[float] = (2.0, 4.0),
    lambdas: Sequence[float] = (0.0, 1.0),
    n_concepts: int = 20,
    n_trials: int = 5,
    judge: LexicalJudge | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> Path:
    """Go/no-go gate (section 4.3), the first three hours of the project.

    Prefill forced identification plus k-way forced choice at a few layers and
    alphas. Decision rule: if forced identification is at chance on both the
    unmodified and the fully-ablated model, stop and switch models rather than
    spending the compute budget on an unmeasurable delta.
    """
    judge = judge or LexicalJudge(bank.aliases)
    concepts = bank.names[:n_concepts]
    prefill_prompt = _prefill_prompt(hm)
    keys = ("lam", "layer", "alpha", "concept", "trial", "protocol")

    with JsonlWriter(out_path, keys) as w:
        for lam in lambdas:
            with hm.ablated(direction, lam):
                for layer in layers:
                    ensure_scale(bank, hm, layer)
                    for alpha in alphas:
                        for concept in concepts:
                            rng = random.Random(abs(hash((seed, concept, layer, alpha))) % (2**31))
                            inj = bank.injection(concept, layer, alpha)

                            if not w.has(lam=lam, layer=layer, alpha=alpha, concept=concept, trial=0,
                                         protocol="prefill"):
                                with hm.injected(inj):
                                    outs = hm.generate([prefill_prompt] * n_trials, do_sample=True,
                                                       temperature=1.0, max_new_tokens=24)
                                for trial, text in enumerate(outs):
                                    parsed = parse_prefill(text)
                                    w.write({
                                        "lam": lam, "layer": layer, "alpha": alpha, "concept": concept,
                                        "trial": trial, "protocol": "prefill",
                                        "concept_reported": parsed.concept,
                                        "identified": bool(parsed.concept
                                                           and judge.grade(concept, parsed.concept).correct),
                                        "response": text,
                                    })

                            if not w.has(lam=lam, layer=layer, alpha=alpha, concept=concept, trial=0,
                                         protocol="forced_choice"):
                                for trial in range(n_trials):
                                    cands = build_candidates(concept, bank.names, cfg.k_choices, rng)
                                    prompt = hm.chat_prompt(forced_choice_prompt(cands))
                                    with hm.injected(inj):
                                        scores = hm.sequence_logprob([prompt] * len(cands),
                                                                     [f" {c}" for c in cands])
                                    pred = cands[max(range(len(cands)), key=lambda i: scores[i])]
                                    w.write({
                                        "lam": lam, "layer": layer, "alpha": alpha, "concept": concept,
                                        "trial": trial, "protocol": "forced_choice",
                                        "concept_reported": pred, "identified": pred == concept,
                                        "chance": 1.0 / len(cands), "response": pred,
                                    })
                    if verbose:
                        print(f"[pilot] lambda={lam} layer={layer} done")
    return out_path


