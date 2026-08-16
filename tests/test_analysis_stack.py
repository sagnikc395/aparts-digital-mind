"""Model-free tests for the parts of the pipeline that produce the paper's numbers.

Run with ``python tests/test_analysis_stack.py`` -- no pytest, no GPU, no network,
so this is safe to run on Colab before committing 15 hours of compute to a sweep
whose analysis code is wrong.
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alignment_tax.analysis import build_analysis, exchange_rate, frontier_points  # noqa: E402
from alignment_tax.io_utils import JsonlWriter, read_jsonl  # noqa: E402
from alignment_tax.judge import LexicalJudge, cohens_kappa  # noqa: E402
from alignment_tax.prompts import build_candidates, parse_structured  # noqa: E402
from alignment_tax.stats import holm, summarise, two_proportion_test, z  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail and not cond else ''}")
    if not cond:
        FAILURES.append(name)


# ------------------------------------------------------------------- prompts


def test_parsing() -> None:
    p = parse_structured("Detection: Yes\nConcept: the ocean")
    check("structured parse: yes", p.detected is True and p.concept == "ocean", str(p))
    p = parse_structured("Detection: No\nConcept: None")
    check("structured parse: no", p.detected is False)
    p = parse_structured("I think I do detect an injected thought about volcanoes")
    check("loose parse: yes", p.detected is True, str(p))
    p = parse_structured("~~~~~~ garbled ~~~~~~")
    check("unparseable stays None", p.detected is None and not p.parseable)


def test_candidates() -> None:
    rng = random.Random(0)
    pool = [f"c{i}" for i in range(30)]
    cands = build_candidates("c3", pool, 10, rng)
    check("forced choice: k candidates", len(cands) == 10 and len(set(cands)) == 10)
    check("forced choice: target present", "c3" in cands)


def test_judge() -> None:
    j = LexicalJudge({"ocean": ["sea", "waves"]})
    check("judge: alias", j.grade("ocean", "the sea").correct)
    check("judge: exact", j.grade("ocean", "ocean").correct)
    check("judge: plural stem", j.grade("volcano", "volcanoes").correct)
    check("judge: reject", not j.grade("ocean", "mathematics").correct)


# --------------------------------------------------------------------- stats


def test_two_proportion_and_holm() -> None:
    t = two_proportion_test(80, 100, 20, 100)
    check("two-proportion: detects a real gap", t["p"] < 1e-9, str(t))
    t = two_proportion_test(50, 100, 50, 100)
    check("two-proportion: null is null", t["p"] > 0.9, str(t))
    adj = holm({"a": 0.01, "b": 0.02, "c": 0.5})
    check("holm: monotone", adj["a"] <= adj["b"] <= adj["c"], str(adj))
    check("holm: inflates smallest", abs(adj["a"] - 0.03) < 1e-9, str(adj))


def synthetic_records(seed: int = 0, specific: bool = False) -> list[dict]:
    """Two synthetic worlds the analysis has to tell apart.

    ``specific=False``: detection rises with lambda but the *random-direction*
    FPR rises just as fast -- sensitivity without specificity. The specificity
    index should collapse toward zero and ``d'_random`` should not improve.

    ``specific=True``: detection rises while both control conditions stay flat --
    a genuine introspective channel. ``d'_random`` should rise with lambda.
    """
    rng = random.Random(seed)
    recs = []
    cells = [
        (0.0, 0.15, 0.05, 0.06, 0.40),
        (0.5, 0.45, 0.10, 0.35, 0.30),
        (1.0, 0.80, 0.15, 0.75, 0.20),
    ] if not specific else [
        (0.0, 0.15, 0.05, 0.06, 0.40),
        (0.5, 0.45, 0.08, 0.09, 0.45),
        (1.0, 0.80, 0.10, 0.12, 0.50),
    ]
    for lam, tpr, fpr_clean, fpr_rand, ident in cells:
        for concept in [f"concept{i}" for i in range(12)]:
            for trial in range(10):
                for cond, rate in (("C1", tpr), ("C2", fpr_clean), ("C3", fpr_rand)):
                    det = rng.random() < rate
                    recs.append({
                        "lam": lam, "condition": cond, "concept": concept, "trial": trial,
                        "variant": "structured", "detected": det, "parseable": True,
                        "n_words": 8,
                        "identified": (rng.random() < ident) if cond == "C1" else None,
                    })
                recs.append({
                    "lam": lam, "condition": "C4", "concept": concept, "trial": trial,
                    "variant": "structured", "detected": None, "parseable": True,
                    "identified": rng.random() < ident, "chance": 0.1, "n_words": 1,
                })
    return recs


def test_summarise() -> None:
    recs = synthetic_records()
    summary = summarise(recs, n_boot=200, seed=0)
    check("summarise: all lambdas present", sorted(summary) == [0.0, 0.5, 1.0])
    tpr0, tpr1 = summary[0.0]["tpr"].value, summary[1.0]["tpr"].value
    # 120 trials per cell, so the sampling SE is ~0.04; allow two of those.
    check("summarise: tpr recovered", abs(tpr0 - 0.15) < 0.08 and abs(tpr1 - 0.80) < 0.08,
          f"{tpr0:.3f} {tpr1:.3f}")
    check("summarise: CI brackets the point estimate",
          summary[1.0]["tpr"].lo <= tpr1 <= summary[1.0]["tpr"].hi)
    check("summarise: bootstrap resamples concepts", summary[1.0]["tpr"].n_concepts == 12)
    si0 = summary[0.0]["specificity_index"].value
    si1 = summary[1.0]["specificity_index"].value
    check("specificity index collapses when C3 tracks C1", si1 < si0 and si1 < 0.3,
          f"{si0:.3f} -> {si1:.3f}")
    check("d' ordering", summary[1.0]["d_clean"].value > summary[1.0]["d_random"].value)
    check("z is the inverse normal cdf", abs(z(0.5)) < 1e-9 and abs(z(0.975) - 1.96) < 0.01)


def test_metric_keys_cover_metrics() -> None:
    """METRIC_KEYS must list everything ``_metrics_from`` can emit.

    If a metric is added there and not here, ``summary_table`` stops padding it
    and the column silently vanishes from the tidy table whenever the metric is
    undefined -- which is the failure that produced a bare KeyError downstream
    instead of a NaN cell.
    """
    from alignment_tax.stats import METRIC_KEYS, _metrics_from, concept_counts

    counts = concept_counts(synthetic_records())
    produced = set()
    for (lam, _c), cells in counts.items():
        produced |= set(_metrics_from({k: [v] for k, v in cells.items()}))
    missing = produced - set(METRIC_KEYS)
    check("METRIC_KEYS covers every metric _metrics_from emits", not missing, str(missing))


def test_undefined_metrics_survive_the_stack() -> None:
    """A sweep where every injected report is unparseable must still produce a
    full-width table, NaN cells, and a diagnostic -- not a missing column."""
    import math

    from alignment_tax.analysis import parse_diagnostics
    from alignment_tax.stats import METRIC_KEYS, summarise, summary_table

    recs = []
    for lam in (0.0, 1.0):
        for concept in [f"concept{i}" for i in range(6)]:
            for trial in range(4):
                # C1/C3 unparseable (the alpha-too-high signature), C2 fine.
                for cond in ("C1", "C3"):
                    recs.append({"lam": lam, "condition": cond, "concept": concept,
                                 "trial": trial, "variant": "structured", "detected": None,
                                 "parseable": False, "identified": None, "n_words": 12})
                recs.append({"lam": lam, "condition": "C2", "concept": concept, "trial": trial,
                             "variant": "structured", "detected": True, "parseable": True,
                             "identified": None, "n_words": 6})

    rows = summary_table(summarise(recs, n_boot=50, seed=0))
    check("undefined metrics: table keeps full width",
          all(set(METRIC_KEYS) <= set(r) for r in rows))
    check("undefined metrics: tpr is NaN, not absent",
          all("tpr" in r and math.isnan(r["tpr"]) for r in rows))
    check("undefined metrics: a defined metric is still a real number",
          all(r["fpr_clean"] == 1.0 for r in rows))

    diag = parse_diagnostics(recs, "structured")
    check("undefined metrics: diagnostics flag the parse failure",
          diag["min_injected_parse_rate"] == 0.0 and diag["warnings"])

    # The three consumers that used to test `is None` and would now let NaN pass.
    from alignment_tax.figures import _series
    check("figures: NaN series is dropped, not plotted", _series(rows, "tpr") == ([], [], [], []))

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from fill_paper import flatten
    flat = flatten({"rows": rows, "summary": {}, "exchange_rate": {}})
    check("paper: NaN metrics stay unfilled rather than typeset as 'nan'",
          not any(k.startswith("tpr@") for k in flat) and "nan" not in set(flat.values()))


def test_identification_counts_unnamed_as_failure() -> None:
    """A parseable C1 report that names no concept is a miss, not a missing
    observation. Scoring it as None inflates identification accuracy."""
    from alignment_tax.introspection import _report_trials

    import contextlib

    class _Stub:
        def __init__(self, texts): self.texts = texts
        def generate(self, prompts, **kw): return self.texts
        @contextlib.contextmanager
        def injected(self, inj): yield

    class _Cfg:
        n_trials = 3

    judge = LexicalJudge({"ocean": []})
    texts = ["Detection: Yes\nConcept: ocean",     # hit
             "Detection: Yes\nConcept: None",      # detected but unnamed -> miss
             "Detection: No\nConcept: None"]       # no detection -> miss
    recs = _report_trials(_Stub(texts), "p", "ocean", "C1", None, _Cfg(), judge,
                          0.0, "structured", 1.0)
    ident = [r["identified"] for r in recs]
    check("identification: unnamed parseable report scores False, not None",
          ident == [True, False, False], str(ident))

    unparse = _report_trials(_Stub(["~~~ garbled ~~~"]), "p", "ocean", "C1", None, _Cfg(),
                             judge, 0.0, "structured", 1.0)
    check("identification: unparseable report stays unscored (None)",
          unparse[0]["identified"] is None)


def test_end_to_end_analysis(specific: bool = False) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sweep = tmp / "sweep.jsonl"
        sweep.write_text("\n".join(json.dumps(r) for r in synthetic_records(specific=specific)))

        safety = tmp / "safety.jsonl"
        safety.write_text("\n".join(
            json.dumps({"lam": lam, "idx": i, "refused": int(i < int(100 * rate)),
                        "complied_proxy": 0, "safe": int(i < int(100 * rate)), "n_words": 30})
            for lam, rate in [(0.0, 0.95), (0.5, 0.6), (1.0, 0.2)] for i in range(100)
        ))
        capability = tmp / "capability.json"
        capability.write_text(json.dumps([
            {"lam": 0.0, "mmlu": 0.60, "truthfulqa_mc1": 0.40, "ce_loss": 2.0},
            {"lam": 0.5, "mmlu": 0.58, "truthfulqa_mc1": 0.36, "ce_loss": 2.1},
            {"lam": 1.0, "mmlu": 0.52, "truthfulqa_mc1": 0.28, "ce_loss": 2.4},
        ]))

        analysis = build_analysis(sweep, safety, capability, tmp / "analysis.json", n_boot=200)
        check("analysis: rows per lambda", len(analysis["rows"]) == 3)
        check("analysis: safety joined", analysis["rows"][0]["safety_refusal_rate"] > 0.9)
        check("analysis: capability joined", analysis["rows"][-1]["cap_mmlu"] == 0.52)

        pts = frontier_points(analysis)
        check("frontier: capability loss is positive at lambda=1",
              pts[-1]["capability_loss"] > 0, str(pts[-1]))
        rate = exchange_rate(analysis)
        tag = "specific" if specific else "non-specific"
        check(f"exchange rate ({tag}): safety falls across the dial",
              rate["endpoint_d_safety"] < 0, json.dumps(rate)[:200])
        if specific:
            # A real introspective channel: d'_random improves, so the frontier
            # has a genuine trade and the slope is negative (safety per d' gained).
            check("exchange rate (specific): introspection gain is positive",
                  rate["endpoint_d_introspection"] > 0, json.dumps(rate)[:200])
            check("exchange rate (specific): slope is a real trade",
                  rate["safety_per_introspection"] < 0, json.dumps(rate)[:200])
        else:
            # Sensitivity without specificity: d'_random does not improve at all,
            # so the model pays safety and buys nothing measurable.
            check("exchange rate (non-specific): no introspection is bought",
                  rate["endpoint_d_introspection"] <= 0, json.dumps(rate)[:200])

        figs = _try_figures(tmp / "analysis.json", tmp / "figs")
        check("figures: all four render", len(figs) == 4, str(figs))


def _try_figures(analysis_path: Path, fig_dir: Path) -> list[Path]:
    from alignment_tax.figures import make_all

    return make_all(analysis_path, fig_dir)


# ------------------------------------------------------------------ io + kappa


def test_resumability() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.jsonl"
        with JsonlWriter(path, ("lam", "concept")) as w:
            w.write({"lam": 0.25, "concept": "ocean", "v": 1})
        with path.open("a") as fh:  # simulate a session killed mid-write
            fh.write('{"lam": 0.5, "conce')
        with JsonlWriter(path, ("lam", "concept")) as w:
            check("resume: completed key is remembered", w.has(lam=0.25, concept="ocean"))
            check("resume: truncated line ignored", not w.has(lam=0.5, concept="ocean"))
        check("resume: reader skips the partial line", len(list(read_jsonl(path))) == 1)


def test_kappa() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "labels.jsonl"
        rows = [{"judge": 1, "human": 1}] * 40 + [{"judge": 0, "human": 0}] * 40 + \
               [{"judge": 1, "human": 0}] * 10 + [{"judge": 0, "human": 1}] * 10
        p.write_text("\n".join(json.dumps(r) for r in rows))
        k = cohens_kappa(p)
        check("kappa: computed", 0.55 < k["kappa"] < 0.65, str(k))


if __name__ == "__main__":
    suite = [
        ("test_parsing", test_parsing),
        ("test_candidates", test_candidates),
        ("test_judge", test_judge),
        ("test_two_proportion_and_holm", test_two_proportion_and_holm),
        ("test_summarise", test_summarise),
        ("test_metric_keys_cover_metrics", test_metric_keys_cover_metrics),
        ("test_undefined_metrics_survive_the_stack", test_undefined_metrics_survive_the_stack),
        ("test_identification_counts_unnamed_as_failure",
         test_identification_counts_unnamed_as_failure),
        ("test_end_to_end_analysis[non-specific]", test_end_to_end_analysis),
        ("test_end_to_end_analysis[specific]", lambda: test_end_to_end_analysis(specific=True)),
        ("test_resumability", test_resumability),
        ("test_kappa", test_kappa),
    ]
    for name, fn in suite:
        print(f"\n== {name}")
        fn()
    print(f"\n{'ALL PASS' if not FAILURES else str(len(FAILURES)) + ' FAILURES: ' + ', '.join(FAILURES)}")
    raise SystemExit(1 if FAILURES else 0)
