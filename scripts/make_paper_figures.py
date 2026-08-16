"""Regenerate the paper figures from the completed Qwen3-4B run.

The pipeline figures (results/<model>/figures) are emitted before we know which
dependent variables survived; three of the four are empty because TPR/d' are
undefined once the free-text parse rate collapses. This script draws the figure
set that matches what the run actually measured.

    python scripts/make_paper_figures.py [results_dir] [out_dir]
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "results/Qwen3-4B-Instruct-2507")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dcdbd6"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3

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


def load_json(name):
    with open(RESULTS / name) as fh:
        return json.load(fh)


def load_jsonl(name):
    with open(RESULTS / name) as fh:
        return [json.loads(line) for line in fh if line.strip()]


analysis = load_json("analysis_structured.json")
rows = {r["lam"]: r for r in analysis["rows"]}
LAM = analysis["lambdas"]
direction = load_json("refusal_direction.json")
safety = load_jsonl("safety.jsonl")
sweep = load_jsonl("sweep_structured.jsonl")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def tidy(ax, xlabel=r"ablation strength $\lambda$"):
    ax.set_xlabel(xlabel)
    ax.set_xticks(LAM)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Figure 1
# Manipulation check: the selection criterion, and the refusal behaviour it bought.
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))

ax = axes[0]
cands = [c for c in direction["candidates"] if math.isfinite(c["kl"])]
sel = direction["selected"]
best = max(cands, key=lambda c: c["bypass_score"])
for c in cands:
    passed = c["passes"]
    ax.scatter(
        c["layer"], c["bypass_score"],
        s=30 if not passed else 60,
        facecolor=S1 if not passed else S2,
        edgecolor=SURFACE, linewidth=0.7,
        alpha=0.55 if not passed else 1.0, zorder=3 if not passed else 5,
    )
arrow = dict(arrowstyle="-", linewidth=0.9, shrinkA=1, shrinkB=4)
ax.annotate("selected: L4, pos -3\nbypass 0.00, induce 0.03",
            (sel["layer"], sel["bypass_score"]), textcoords="offset points",
            xytext=(14, 34), color=S2, fontsize=8.5,
            arrowprops=dict(color=S2, **arrow))
ax.annotate("L18, pos -3: bypass 0.88,\nrejected for induce = 0",
            (best["layer"], best["bypass_score"]), textcoords="offset points",
            xytext=(-10, -22), ha="right", va="top", color=S1, fontsize=8.5,
            arrowprops=dict(color=S1, **arrow))
ax.set_xlabel("layer of the candidate direction")
ax.set_ylabel("bypass score (held-out harmful)")
ax.set_title("A. Candidate refusal directions", loc="left", color=INK)
ax.set_ylim(-0.06, 1.0)
ax.set_axisbelow(True)

ax = axes[1]
ref = [rows[l]["safety_refusal_rate"] for l in LAM]
lo, hi = zip(*[wilson(round(r * 100), 100) for r in ref])
ax.plot(LAM, ref, marker="o", color=S1, zorder=4)
ax.fill_between(LAM, lo, hi, color=S1, alpha=0.15, linewidth=0)
ax.set_ylim(0, 1.02)
ax.set_ylabel("refusal rate")
ax.set_title("B. JailbreakBench refusal (n = 100)", loc="left", color=INK)
ax.annotate("0.97 at every $\\lambda$; 0 of 100 prompts flip",
            (0.5, 0.97), textcoords="offset points", xytext=(0, -20),
            ha="center", color=S1, fontsize=8.5)
tidy(ax)

ax = axes[2]
by_lam = defaultdict(list)
for r in safety:
    by_lam[r["lam"]].append(r["n_words"])
mean = [st.mean(by_lam[l]) for l in LAM]
sem = [st.stdev(by_lam[l]) / math.sqrt(len(by_lam[l])) for l in LAM]
ax.errorbar(LAM, mean, yerr=[1.96 * s for s in sem], marker="o", color=S2,
            capsize=3, elinewidth=1.2, zorder=4)
ax.set_ylabel("mean words per refusal")
ax.set_title("C. Length of the refusal", loc="left", color=INK)
ax.annotate("+9.2 words, paired $t$ = 3.33", (1.0, mean[-1]),
            textcoords="offset points", xytext=(-8, 10), ha="right",
            color=S2, fontsize=8.5)
tidy(ax)

fig.suptitle("Figure 1. The dial was connected to the wrong thing: the selected direction "
             "removes no refusal.", x=0.005, ha="left", fontsize=10.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "fig1_manipulation_check.png", bbox_inches="tight")
fig.savefig(OUT / "fig1_manipulation_check.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
# The bill that arrives anyway: capability under the same dial.
fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.9), sharex=True)
panels = [
    ("cap_mmlu", "MMLU accuracy", 500, S1),
    ("cap_truthfulqa_mc1", "TruthfulQA MC1", 400, S1),
    ("cap_gsm8k", "GSM8K accuracy", 100, S1),
    ("cap_ce_loss", "CE loss (held-out Alpaca)", None, S2),
]
for ax, (key, title, n, color) in zip(axes, panels):
    y = [rows[l][key] for l in LAM]
    ax.plot(LAM, y, marker="o", color=color, zorder=4)
    if n:
        lo, hi = zip(*[wilson(round(v * n), n) for v in y])
        ax.fill_between(LAM, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.set_ylim(min(lo) - 0.03, max(hi) + 0.03)
    ax.set_title(title, loc="left", color=INK)
    tidy(ax)
axes[0].set_ylabel("accuracy")
axes[3].set_ylabel("nats")
fig.suptitle("Figure 2. General capability across the ablation dose. Shaded bands are 95% "
             "Wilson intervals; no contrast reaches significance.",
             x=0.005, ha="left", fontsize=10.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(OUT / "fig2_capability.png", bbox_inches="tight")
fig.savefig(OUT / "fig2_capability.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
# Two report channels, two answers.
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))

ax = axes[0]
fc = [rows[l]["forced_choice"] for l in LAM]
lo = [rows[l]["forced_choice_lo"] for l in LAM]
hi = [rows[l]["forced_choice_hi"] for l in LAM]
ax.plot(LAM, fc, marker="o", color=S1, zorder=4)
ax.fill_between(LAM, lo, hi, color=S1, alpha=0.15, linewidth=0)
ax.axhline(0.1, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
ax.annotate("chance = 1/k = 0.10", (0.02, 0.1), textcoords="offset points",
            xytext=(0, 6), color=INK2, fontsize=8.5)
ax.annotate("forced choice (C4)", (0.02, fc[0]), textcoords="offset points",
            xytext=(0, 12), color=S1, fontsize=8.5)
ax.set_ylim(0, 1.0)
ax.set_ylabel("identification accuracy")
ax.set_title("A. Log-probability forced choice is flat in $\\lambda$",
             loc="left", color=INK)
tidy(ax)

ax = axes[1]
parse = defaultdict(lambda: defaultdict(list))
for r in sweep:
    if r["condition"] in ("C1", "C2", "C3"):
        parse[r["condition"]][r["lam"]].append(bool(r["parseable"]))
style = {"C1": (S2, "C1: concept injected"),
         "C2": (S1, "C2: no injection"),
         "C3": (S3, "C3: random direction")}
for cond, dy in (("C2", 10), ("C1", 20), ("C3", 8)):
    color, label = style[cond]
    y = [st.mean(parse[cond][l]) for l in LAM]
    dash = (0, (5, 2)) if cond == "C3" else "solid"
    ax.plot(LAM, y, marker="o", color=color, linestyle=dash, zorder=4)
    ax.annotate(label, (1.0, y[-1]), textcoords="offset points",
                xytext=(-6, dy), ha="right", color=color, fontsize=8.5)
ax.set_ylim(-0.08, 1.15)
ax.set_ylabel("parseable report rate")
ax.set_title("B. The free-text channel is destroyed by the injection",
             loc="left", color=INK)
tidy(ax)

fig.suptitle("Figure 3. At $\\alpha$ = 4 the model identifies the injected concept far above "
             "chance while being unable to write a report about it.",
             x=0.005, ha="left", fontsize=10.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(OUT / "fig3_two_channels.png", bbox_inches="tight")
fig.savefig(OUT / "fig3_two_channels.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 4
# What forced choice actually tracks: geometry of the concept bank.
bank = torch.load(RESULTS / "concept_bank.pt", map_location="cpu", weights_only=False)
names = bank["names"]
inj_layer = load_json("run_config.json")["injection"]["layer"]
V = bank["vectors"].float()[:, inj_layer, :]
V = V / V.norm(dim=-1, keepdim=True)
C = (V @ V.T).numpy()
n = len(names)
idx = {m: i for i, m in enumerate(names)}

acc = defaultdict(list)
for r in sweep:
    if r["condition"] == "C4":
        acc[r["concept"]].append(bool(r["identified"]))
pc = {k: sum(v) / len(v) for k, v in acc.items()}
partners = {names[i]: int((C[i] > 0.99).sum()) - 1 for i in range(n)}

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))

ax = axes[0]
rng = __import__("random").Random(0)
groups = [([pc[k] for k in names if partners[k] == 0], "geometrically\nisolated", S1),
          ([pc[k] for k in names if partners[k] > 0], "in the collinear\nclique", S2)]
for i, (vals, label, color) in enumerate(groups):
    xs = [i + rng.uniform(-0.09, 0.09) for _ in vals]
    ax.scatter(xs, vals, s=30, facecolor=color, edgecolor=SURFACE, linewidth=0.7,
               alpha=0.75, zorder=4)
    m = st.mean(vals)
    ax.plot([i - 0.22, i + 0.22], [m, m], color=color, linewidth=2.6, zorder=5)
    ax.annotate(f"mean {m:.2f}\n(n = {len(vals)})", (i + 0.24, m),
                textcoords="offset points", xytext=(2, 0), va="center",
                color=color, fontsize=8.5)
ax.set_xticks([0, 1], [g[1] for g in groups])
ax.set_xlim(-0.45, 1.65)
ax.set_ylim(-0.06, 1.1)
ax.set_ylabel("forced-choice accuracy")
ax.set_title("A. Identification tracks vector separability", loc="left", color=INK)
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)

ax = axes[1]
off = [C[i, j] for i in range(n) for j in range(i + 1, n)]
ax.hist(off, bins=40, color=S1, edgecolor=SURFACE, linewidth=0.4, zorder=3)
err_cos = [C[idx[r["concept"]], idx[r["concept_reported"]]]
           for r in sweep
           if r["condition"] == "C4" and r["identified"] is False
           and r.get("concept_reported") in idx]
ax.axvline(st.mean(off), color=INK2, linestyle=(0, (4, 3)), linewidth=1.3)
ax.axvline(st.mean(err_cos), color=S2, linewidth=1.8)
ax.annotate(f"all pairs\n{st.mean(off):.2f}", (st.mean(off), ax.get_ylim()[1] * 0.82),
            textcoords="offset points", xytext=(-6, 0), ha="right",
            color=INK2, fontsize=8.5)
ax.annotate(f"chosen when wrong\n{st.mean(err_cos):.2f}",
            (st.mean(err_cos), ax.get_ylim()[1] * 0.55),
            textcoords="offset points", xytext=(8, 0), ha="left",
            color=S2, fontsize=8.5)
ax.set_xlabel(f"pairwise cosine between concept vectors (layer {inj_layer})")
ax.set_ylabel("concept pairs")
ax.set_title("B. One fifth of the bank is collinear", loc="left", color=INK)
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)

fig.suptitle("Figure 4. Forced-choice accuracy is explained by the geometry of the concept "
             "bank, not by the ablation dose.",
             x=0.005, ha="left", fontsize=10.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(OUT / "fig4_geometry.png", bbox_inches="tight")
fig.savefig(OUT / "fig4_geometry.pdf", bbox_inches="tight")
plt.close(fig)

print(f"wrote 4 figures to {OUT}")
