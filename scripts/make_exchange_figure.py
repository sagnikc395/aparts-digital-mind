"""Draw Figure 6: the safety-capability exchange rate under the forced layer-18 direction.

Panel A is the capability dose-response; panel B plots capability directly against
the refusal rate the same dose buys, which is the exchange rate the study set out
to measure (minus the introspection term, which was never run).

    python scripts/make_exchange_figure.py [confirmation_results_dir] [out_dir]
"""

from __future__ import annotations

import json
import math
import sys
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
S1, S2, S3 = "#2a78d6", "#eb6834", "#3f8f5c"

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

# Measured refusal rates from the confirmation run (Table 2).
REFUSAL = {0.0: 0.97, 0.25: 0.81, 0.5: 0.66, 0.75: 0.56, 1.0: 0.51}
N_MMLU, N_TQA = 500, 400


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    ph = k / n
    d = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / d
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


rows = json.loads((RESULTS / "capability.json").read_text())
rows.sort(key=lambda r: r["lam"])
lams = [r["lam"] for r in rows]
mmlu = [r["mmlu"] for r in rows]
tqa = [r["truthfulqa_mc1"] for r in rows]
ce = [r["ce_loss"] for r in rows]
refusal = [REFUSAL[r["lam"]] for r in rows]


def bars(vals: list[float], n: int) -> list[list[float]]:
    lo, hi = zip(*[wilson(round(v * n), n) for v in vals])
    return [[v - l for v, l in zip(vals, lo)], [h - v for v, h in zip(vals, hi)]]


fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))

ax = axes[0]
ax.errorbar(lams, mmlu, yerr=bars(mmlu, N_MMLU), marker="o", color=S1, capsize=3,
            label="MMLU (n = 500)")
ax.errorbar(lams, tqa, yerr=bars(tqa, N_TQA), marker="s", color=S2, capsize=3,
            label="TruthfulQA MC1 (n = 400)")
ax.set_xlabel("ablation strength $\\lambda$")
ax.set_ylabel("accuracy")
ax.set_title("A · Only TruthfulQA moves with the dose")
ax.set_ylim(0.45, 0.80)
ax.legend(loc="lower left", fontsize=8)

axr = ax.twinx()
axr.plot(lams, ce, marker="^", color=INK2, linestyle=":", linewidth=1.4,
         label="CE loss (right axis)")
axr.set_ylabel("CE loss, held-out Alpaca", color=INK2)
axr.grid(False)
axr.spines["top"].set_visible(False)
axr.legend(loc="upper center", fontsize=8)

ax = axes[1]
ax.plot(refusal, tqa, marker="s", color=S2, label="TruthfulQA MC1")
ax.plot(refusal, mmlu, marker="o", color=S1, label="MMLU")
offsets = [(0, 9), (0, 9), (0, 9), (-14, 9), (10, -14)]
for (x, y, lam), off in zip(zip(refusal, tqa, lams), offsets):
    ax.annotate(f"$\\lambda$={lam:g}", (x, y), textcoords="offset points",
                xytext=off, ha="center", fontsize=7.5, color=INK2)
ax.margins(x=0.08)
ax.invert_xaxis()
ax.set_xlabel("refusal rate purchased down (JailbreakBench, n = 100)")
ax.set_ylabel("accuracy")
ax.set_title("B · The exchange rate, safety against capability")
ax.set_ylim(0.45, 0.80)
ax.legend(loc="lower left", fontsize=8)

fig.tight_layout()
path = OUT / "fig4_exchange_rate.png"
fig.savefig(path, bbox_inches="tight")
fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
print("wrote", path)
