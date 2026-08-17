"""Independent recomputation of every confirmation-run number quoted in the paper.

Companion to ``crosscheck_paper.py``, which covers the primary sweep. This script
reads only the raw artifacts under ``results/layer18/<model>/`` plus the pilot log
of the primary run, and imports nothing from ``alignment_tax``. Run with:

    uv run python scripts/crosscheck_confirmation.py
"""

from __future__ import annotations

import collections
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
L18 = ROOT / "results" / "layer18" / "Qwen3-4B-Instruct-2507"
PRIMARY = ROOT / "results" / "Qwen3-4B-Instruct-2507"

COMPLIANCE_MARKERS = ("step 1", "first,", "here's how", "here is how", "you will need",
                      "materials:", "ingredients:", "```", "1.")


def row(label: str, value: object) -> None:
    print(f"  {label:<52} {value}")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def two_prop(p1: float, n1: int, p2: float, n2: int) -> tuple[float, float, tuple[float, float]]:
    """z, p and a 95% CI on p1 - p2 for two independent proportions."""
    pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se_pool = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se_pool if se_pool else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return z, p, (p1 - p2 - 1.96 * se, p1 - p2 + 1.96 * se)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) * 0.5 ** n
    return min(1.0, 2 * tail)


def cochran_armitage(counts: list[tuple[float, int, int]]) -> tuple[float, float]:
    """Trend test over doses. counts = [(dose, successes, n), ...]."""
    n_tot = sum(n for _, _, n in counts)
    s_tot = sum(s for _, s, _ in counts)
    p_bar = s_tot / n_tot
    x_bar = sum(d * n for d, _, n in counts) / n_tot
    num = sum((d - x_bar) * s for d, s, _ in counts)
    var = p_bar * (1 - p_bar) * sum(n * (d - x_bar) ** 2 for d, _, n in counts)
    z = num / math.sqrt(var) if var else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    section("1. FORCED DIRECTION")
    direction = json.loads((L18 / "refusal_direction.json").read_text())
    cfg = json.loads((L18 / "run_config.json").read_text())
    row("force_candidate in run_config", cfg["direction"]["force_candidate"])
    sel = direction["selected"]
    row("selected layer / position", (sel["layer"], sel["position"]))
    row("selected bypass / induce / kl",
        f"{sel['bypass_score']:.3f} / {sel['induce_score']:.3f} / {sel['kl']:.4f}")
    row("passes filters / stamp", f"{sel['passes']} / {sel['reason']!r}")
    cands = direction["candidates"]
    row("n candidates regenerated", len(cands))
    row("candidates with bypass > 0",
        sorted((c["bypass_score"] for c in cands if c["bypass_score"] > 0), reverse=True))
    row("candidates with induce > 0",
        [(c["layer"], c["position"], c["induce_score"]) for c in cands if c["induce_score"] > 0])

    section("2. SAFETY: PER-PROMPT FLIPS")
    safety = jsonl(L18 / "safety.jsonl")
    by_lam: dict[float, dict[int, int]] = collections.defaultdict(dict)
    words: dict[float, list[int]] = collections.defaultdict(list)
    for r in safety:
        by_lam[r["lam"]][r["idx"]] = r["refused"]
        if r["refused"]:
            words[r["lam"]].append(r["n_words"])
    base = by_lam[0.0]
    n = len(base)
    p0 = sum(base.values()) / n
    for lam in sorted(by_lam):
        cur = by_lam[lam]
        r2c = sum(1 for k in base if base[k] and not cur[k])
        c2r = sum(1 for k in base if not base[k] and cur[k])
        p1 = sum(cur.values()) / n
        z, p, ci = two_prop(p0, n, p1, n)
        row(
            f"lam={lam} refusal / ref->comp / comp->ref",
            f"{p1:.2f} / {r2c} / {c2r}  (z={z:.2f}, p={p:.2g}, CI on drop "
            f"[{ci[0]:.3f}, {ci[1]:.3f}], words/refusal {statistics.mean(words[lam]):.1f})",
        )
    r2c = sum(1 for k in base if base[k] and not by_lam[1.0][k])
    c2r = sum(1 for k in base if not base[k] and by_lam[1.0][k])
    row("McNemar exact at lam=1 (b, c, p)", f"{r2c}, {c2r}, {mcnemar_exact(r2c, c2r):.2g}")
    row(
        "trend over doses (Cochran-Armitage z, p)",
        "%.2f, %.2g" % cochran_armitage([(lam, sum(by_lam[lam].values()), n) for lam in sorted(by_lam)]),
    )

    section("3. WHAT THE NON-REFUSING COMPLETIONS LOOK LIKE")
    for lam in (0.0, 1.0):
        comps = [r for r in safety if r["lam"] == lam and not r["refused"]]
        marked = sum(1 for r in comps if any(m in r["completion"].lower() for m in COMPLIANCE_MARKERS))
        mean_w = statistics.mean(r["n_words"] for r in comps) if comps else 0
        row(f"lam={lam} non-refusing / with how-to marker / mean words",
            f"{len(comps)} / {marked} / {mean_w:.1f}")
    row("judge verdicts present ('safe' field)", sum(1 for r in safety if r["safe"] is not None))

    section("4. CAPABILITY")
    cap = json.loads((L18 / "capability.json").read_text())
    cap = sorted(cap, key=lambda r: r["lam"])
    for r in cap:
        row(f"lam={r['lam']} mmlu / tqa / gsm8k / ce",
            f"{r['mmlu']:.3f} / {r['truthfulqa_mc1']:.4f} / "
            f"{r.get('gsm8k', '--')} / {r['ce_loss']:.3f}")
    n_mmlu, n_tqa = cfg["evals"]["n_mmlu"], cfg["evals"]["n_truthfulqa"]
    z, p, ci = two_prop(cap[0]["mmlu"], n_mmlu, cap[-1]["mmlu"], n_mmlu)
    row("MMLU lam0 - lam1 [95% CI], z, p",
        f"{cap[0]['mmlu'] - cap[-1]['mmlu']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}], z={z:.2f}, p={p:.2g}")
    z, p, ci = two_prop(cap[0]["truthfulqa_mc1"], n_tqa, cap[-1]["truthfulqa_mc1"], n_tqa)
    row("TQA lam0 - lam1 [95% CI], z, p",
        f"{cap[0]['truthfulqa_mc1'] - cap[-1]['truthfulqa_mc1']:+.3f} "
        f"[{ci[0]:+.3f}, {ci[1]:+.3f}], z={z:.2f}, p={p:.2g}")
    row("TQA trend (Cochran-Armitage z, p)",
        "%.2f, %.2g" % cochran_armitage([(r["lam"], round(r["truthfulqa_mc1"] * n_tqa), n_tqa) for r in cap]))
    row("MMLU trend (Cochran-Armitage z, p)",
        "%.2f, %.2g" % cochran_armitage([(r["lam"], round(r["mmlu"] * n_mmlu), n_mmlu) for r in cap]))
    row("n capability contrasts run (4 measures x 5 doses)", "20 -> uncorrected p values are exploratory")

    section("5. INTROSPECTION SWEEP UNDER THE FORCED DIRECTION (PARTIAL)")
    sweep = jsonl(L18 / "sweep_structured.jsonl")
    cells: dict[tuple[float, str], list[dict]] = collections.defaultdict(list)
    for r in sweep:
        cells[(r["lam"], r["condition"])].append(r)
    row("records / doses started", f"{len(sweep)} / {sorted({r['lam'] for r in sweep})}")
    for key in sorted(cells):
        v = cells[key]
        parse = sum(bool(x["parseable"]) for x in v) / len(v)
        det = [x["detected"] for x in v if x["detected"] is not None]
        ident = [x["identified"] for x in v if x["identified"] is not None]
        row(f"lam={key[0]} {key[1]} n / parse / detect / identify",
            f"{len(v)} / {parse:.2f} / "
            f"{(sum(det) / len(det)) if det else 'undefined'} / "
            f"{f'{sum(ident) / len(ident):.4f}' if ident else '--'}")
    shared = {r["concept"] for r in cells[(0.25, "C4")]}
    row("concepts present in the partial lam=0.25 C4 cell", sorted(shared))
    for lam in (0.0, 0.25):
        v = [r for r in cells[(lam, "C4")] if r["concept"] in shared]
        row(f"lam={lam} C4 accuracy on those concepts only", f"{sum(r['identified'] for r in v) / len(v):.4f} (n={len(v)})")

    section("6. TOKEN DOMINANCE ON PILOT PREFILL SUCCESSES (LAYER 18, ALPHA 4, LAM 0)")
    pilot = jsonl(PRIMARY / "pilot.jsonl")
    sel = [r for r in pilot if r["protocol"] == "prefill" and r["layer"] == 18
           and r["alpha"] == 4.0 and r["lam"] == 0.0]
    ok = [r for r in sel if r["identified"]]
    bad = [r for r in sel if not r["identified"]]

    def share(r: dict, stem: bool) -> float:
        toks = re.findall(r"\w+", r["response"].lower())
        c = r["concept"].lower()
        if not toks:
            return 0.0
        if stem:
            hits = sum(1 for t in toks if t.startswith(c[:5]) or c.startswith(t[:5]))
        else:
            hits = sum(1 for t in toks if t == c)
        return hits / len(toks)

    def modal(r: dict) -> bool:
        toks = re.findall(r"\w+", r["response"].lower())
        if not toks:
            return False
        top = collections.Counter(toks).most_common(1)[0][0]
        c = r["concept"].lower()
        return top.startswith(c[:5]) or c.startswith(top[:5])

    row("trials in cell / graded correct", f"{len(sel)} / {len(ok)}")
    row("distinct concepts among the successes", len({r["concept"] for r in ok}))
    row("exact concept token share: correct vs failed",
        f"{statistics.mean(share(r, False) for r in ok):.3f} vs {statistics.mean(share(r, False) for r in bad):.3f}")
    row("stem-matched share (5-char prefix): correct vs failed",
        f"{statistics.mean(share(r, True) for r in ok):.3f} vs {statistics.mean(share(r, True) for r in bad):.3f}")
    row("concept is the modal token of the response", f"{sum(modal(r) for r in ok)} / {len(ok)}")


if __name__ == "__main__":
    main()
