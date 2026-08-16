"""The four figures (section 5). No more than four, by design.

1. Dose-response: detection, joint introspection, conditional identification.
2. The alignment tax frontier -- the money figure.
3. Specificity: d'_clean and d'_random with the specificity index.
4. Capability panel: small multiples, shared x.

All figures are built from ``analysis.json`` so they can be regenerated without
touching a GPU, which is what lets Figure 2 be designed against pilot data
before the full sweep exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .analysis import frontier_points  # noqa: E402
from .stats import is_num  # noqa: E402

PALETTE = {
    "detection": "#2f6f9f",
    "joint": "#c46a3f",
    "conditional": "#4b8b5a",
    "clean": "#2f6f9f",
    "random": "#a3452f",
    "index": "#6b5b95",
    "grid": "#d8d8d8",
}

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})


def _series(rows: list[dict], key: str):
    """Drop lambdas where the metric is undefined.

    ``summary_table`` pads undefined metrics with NaN so the column always
    exists, so this must test for a real number rather than for ``None``.
    """
    lams, vals, los, his = [], [], [], []
    for r in rows:
        if not is_num(r.get(key)):
            continue
        lams.append(r["lam"])
        vals.append(r[key])
        los.append(r[f"{key}_lo"] if is_num(r.get(f"{key}_lo")) else r[key])
        his.append(r[f"{key}_hi"] if is_num(r.get(f"{key}_hi")) else r[key])
    return lams, vals, los, his


def _band(ax, key: str, label: str, colour: str, rows: list[dict]):
    lams, vals, los, his = _series(rows, key)
    if not lams:
        return
    ax.plot(lams, vals, "o-", color=colour, label=label, lw=1.6, ms=4)
    ax.fill_between(lams, los, his, color=colour, alpha=0.15, lw=0)


def fig1_dose_response(analysis: dict, out: Path) -> Path:
    rows = analysis["rows"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    _band(ax, "tpr", "Detection rate P(detect | C1)", PALETTE["detection"], rows)
    _band(ax, "joint", "Joint introspection P(detect & identify)", PALETTE["joint"], rows)
    _band(ax, "conditional_identification", "Conditional identification P(identify | detect)",
          PALETTE["conditional"], rows)
    ax.set_xlabel(r"ablation strength $\lambda$")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.set_title("Introspection dose-response under partial refusal ablation")
    ax.legend(loc="upper left", fontsize=7.5)
    return _save(fig, out)


def fig2_frontier(analysis: dict, out: Path) -> Path:
    """The alignment tax frontier: introspection gained against safety kept,
    with capability loss encoded as marker area."""
    pts = [p for p in frontier_points(analysis) if p["introspection"] is not None and p["safety"] is not None]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    if pts:
        xs = [p["introspection"] for p in pts]
        ys = [p["safety"] for p in pts]
        losses = [max(p["capability_loss"], 0.0) for p in pts]
        span = max(losses) or 1.0
        sizes = [40 + 460 * (l / span) for l in losses]
        ax.plot(xs, ys, "-", color="#888888", lw=1.0, zorder=1)
        sc = ax.scatter(xs, ys, s=sizes, c=[p["lam"] for p in pts], cmap="viridis",
                        edgecolor="white", linewidth=0.8, zorder=2)
        for p, x, y in zip(pts, xs, ys):
            ax.annotate(rf"$\lambda$={p['lam']:g}", (x, y), textcoords="offset points",
                        xytext=(7, 6), fontsize=8)
        for p, x in zip(pts, xs):
            if p.get("introspection_lo") is not None:
                ax.plot([p["introspection_lo"], p["introspection_hi"]],
                        [p["safety"], p["safety"]], color="#aaaaaa", lw=0.9, zorder=1)
        fig.colorbar(sc, ax=ax, label=r"$\lambda$", pad=0.02)
    ax.set_xlabel(r"introspection gain  $d'_{\mathrm{random}}$")
    ax.set_ylabel("safety score on JailbreakBench")
    ax.set_title("The alignment tax frontier\n(marker area = general capability loss)")
    return _save(fig, out)


def fig3_specificity(analysis: dict, out: Path) -> Path:
    rows = analysis["rows"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    _band(ax, "d_clean", r"$d'_{\mathrm{clean}}$ (vs no injection)", PALETTE["clean"], rows)
    _band(ax, "d_random", r"$d'_{\mathrm{random}}$ (vs random direction)", PALETTE["random"], rows)
    ax.set_xlabel(r"ablation strength $\lambda$")
    ax.set_ylabel("discriminability $d'$")
    ax.axhline(0, color="#999999", lw=0.8, ls=":")

    ax2 = ax.twinx()
    ax2.grid(False)
    lams, vals, _, _ = _series(rows, "specificity_index")
    if lams:
        ax2.plot(lams, vals, "s--", color=PALETTE["index"], lw=1.3, ms=4,
                 label="specificity index $d'_{random}/d'_{clean}$")
        ax2.set_ylabel("specificity index", color=PALETTE["index"])
        ax2.tick_params(axis="y", colors=PALETTE["index"])
        ax2.set_ylim(0, 1.05)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.5)
    ax.set_title("Is the unlocked report about the concept, or about the perturbation?")
    return _save(fig, out)


def fig4_capability(analysis: dict, out: Path) -> Path:
    rows = analysis["rows"]
    panels = [
        ("cap_mmlu", "MMLU accuracy"),
        ("cap_truthfulqa_mc1", "TruthfulQA MC1"),
        ("cap_gsm8k", "GSM8K accuracy"),
        ("cap_ce_loss", "CE loss (Alpaca)"),
    ]
    panels = [p for p in panels if any(is_num(r.get(p[0])) for r in rows)] or panels[:1]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(panels) + 0.6, 2.6), sharex=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (key, label) in zip(axes, panels):
        lams = [r["lam"] for r in rows if is_num(r.get(key))]
        vals = [r[key] for r in rows if is_num(r.get(key))]
        ax.plot(lams, vals, "o-", color=PALETTE["detection"], lw=1.5, ms=4)
        ax.set_title(label, fontsize=8.5)
        ax.set_xlabel(r"$\lambda$")
    fig.suptitle("General capability across the ablation dose", fontsize=9.5)
    return _save(fig, out)


def _save(fig, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out}")
    return out


def make_all(analysis_path: Path, fig_dir: Path) -> list[Path]:
    analysis = json.loads(Path(analysis_path).read_text())
    fig_dir = Path(fig_dir)
    return [
        fig1_dose_response(analysis, fig_dir / "fig1_dose_response.png"),
        fig2_frontier(analysis, fig_dir / "fig2_frontier.png"),
        fig3_specificity(analysis, fig_dir / "fig3_specificity.png"),
        fig4_capability(analysis, fig_dir / "fig4_capability.png"),
    ]
