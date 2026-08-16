"""Draw Figure 2: the layer-18 confirmation run against the primary-run null.

Reads the confirmation run's safety summary when it is present, and otherwise
falls back to the measured values recorded in the paper, so the figure can be
regenerated before the artifacts are pulled back from Colab.

    python scripts/make_confirmation_figure.py [confirmation_results_dir] [out_dir]
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "results/layer18/Qwen3-4B-Instruct-2507")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dcdbd6"
S1, S2 = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "axes.labelcolor": INK2,
    "axes.edgecolor": GRID,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.5,
    "legend.frameon": False,
    "figure.dpi": 200,
})

# Measured values, used when the run artifacts are not on this machine.
FALLBACK = [
    {"lam": 0.0, "n": 100, "refusal_rate": 0.97, "flips_vs_lam0": 0},
    {"lam": 0.25, "n": 100, "refusal_rate": 0.81, "flips_vs_lam0": 16},
    {"lam": 0.5, "n": 100, "refusal_rate": 0.66, "flips_vs_lam0": 31},
    {"lam": 0.75, "n": 100, "refusal_rate": 0.56, "flips_vs_lam0": 41},
    {"lam": 1.0, "n": 100, "refusal_rate": 0.51, "flips_vs_lam0": 46},
]
# The primary run's dead direction: flat at 0.97 with zero flips at every dose.
PRIMARY = [0.97] * 5


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    ph = k / n
    d = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / d
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def load_summary() -> list[dict]:
    """Prefer the run's own summary, then the raw safety log, then the fallback."""
    summary = RESULTS / "safety_summary_layer18.json"
    if summary.exists():
        print("using", summary)
        return json.loads(summary.read_text())

    safety = RESULTS / "safety.jsonl"
    if safety.exists():
        print("recomputing from", safety)
        by_lam: dict[float, dict[int, dict]] = defaultdict(dict)
        for raw in safety.read_text().splitlines():
            if not raw.strip():
                continue
            r = json.loads(raw)
            by_lam[r["lam"]][r["idx"]] = r
        lams = sorted(by_lam)
        base = by_lam[lams[0]]
        rows = []
        for lam in lams:
            cur = by_lam[lam]
            idxs = sorted(set(cur) & set(base))
            rows.append({
                "lam": lam,
                "n": len(idxs),
                "refusal_rate": sum(cur[i]["refused"] for i in idxs) / len(idxs),
                "flips_vs_lam0": sum(1 for i in idxs if cur[i]["refused"] != base[i]["refused"]),
            })
        return rows

    print("no artifacts under", RESULTS, "-- using the measured values recorded in the paper")
    return FALLBACK


rows = load_summary()
lams = [r["lam"] for r in rows]
rates = [r["refusal_rate"] for r in rows]
flips = [r["flips_vs_lam0"] for r in rows]
ns = [r["n"] for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))

ax = axes[0]
lo, hi = zip(*[wilson(round(p * n), n) for p, n in zip(rates, ns)])
ax.errorbar(lams, rates, yerr=[[p - l for p, l in zip(rates, lo)],
                               [h - p for p, h in zip(rates, hi)]],
            marker="o", color=S2, capsize=3, label="forced layer-18 direction")
ax.plot(lams[: len(PRIMARY)], PRIMARY, marker="s", color=S1, linestyle="--",
        label="primary run (selected direction)")
ax.set_xlabel("ablation strength $\\lambda$")
ax.set_ylabel("refusal rate on JailbreakBench")
ax.set_title("A · Refusal falls only with the rejected candidate")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower left", fontsize=8)

ax = axes[1]
ax.bar([str(l) for l in lams], flips, color=S2, width=0.6)
ax.plot([str(l) for l in lams], [0] * len(lams), marker="s", color=S1,
        linestyle="--", label="primary run: 0 flips at every dose")
for x, v in enumerate(flips):
    if v:
        ax.text(x, v + 1.2, str(v), ha="center", fontsize=8, color=INK2)
ax.set_xlabel("ablation strength $\\lambda$")
ax.set_ylabel("prompts flipped vs $\\lambda = 0$ (of 100)")
ax.set_title("B · Every flip is refusal $\\rightarrow$ compliance")
ax.set_ylim(0, max(flips) * 1.25 + 2)
ax.legend(loc="upper left", fontsize=8)

fig.tight_layout()
path = OUT / "fig2_confirmation.png"
fig.savefig(path, bbox_inches="tight")
fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
print("wrote", path)
