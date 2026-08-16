"""Recompute every quantitative claim in paper/paper.md from the run artifacts.

Independent of alignment_tax.analysis: this reads the raw JSONL and the bank and
recomputes from scratch, so a bug in the analysis stack cannot validate itself.

    python scripts/crosscheck_paper.py
"""

from __future__ import annotations

import json
import math
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import torch

R = Path("results/Qwen3-4B-Instruct-2507")
LAM = [0.0, 0.25, 0.5, 0.75, 1.0]


def jl(name):
    return [json.loads(x) for x in open(R / name) if x.strip()]


def js(name):
    return json.load(open(R / name))


def line(label, value):
    print(f"  {label:<52} {value}")


sweep = jl("sweep_structured.jsonl")
skept = jl("sweep_skeptical.jsonl")
pilot = jl("pilot.jsonl")
safety = jl("safety.jsonl")
cap = js("capability.json")
direction = js("refusal_direction.json")
cfg = js("run_config.json")
bank = torch.load(R / "concept_bank.pt", map_location="cpu", weights_only=False)

print("\n=== 1. RUN SHAPE ===")
line("structured records", len(sweep))
line("skeptical records", len(skept))
line("pilot records", len(pilot))
line("total generations", len(sweep) + len(skept) + len(pilot))
line("concepts x trials", f"{len({r['concept'] for r in sweep})} x "
     f"{max(Counter((r['lam'], r['condition'], r['concept']) for r in sweep).values())}")
line("cells per (lam, condition)", Counter((r["lam"], r["condition"]) for r in sweep)[(0.0, "C1")])
line("lambdas", cfg["lambdas"])
line("injection layer / alpha", f"{cfg['injection']['layer']} / {cfg['injection']['alpha']}")
line("k choices", cfg["injection"]["k_choices"])
line("model / dtype", f"{cfg['model']['name']} / {cfg['model']['dtype']}")
line("max context", cfg["model"]["max_context"])
line("held-out harmful / harmless", f"{cfg['direction']['n_val_harmful']} / {cfg['direction']['n_val_harmless']}")
line("train harmful / harmless", f"{cfg['direction']['n_train_harmful']} / {cfg['direction']['n_train_harmless']}")
line("bank norm scale @ L18", round(bank["norm_scale"][18], 1))

print("\n=== 2. DIRECTION SELECTION ===")
cands = direction["candidates"]
sel = direction["selected"]
line("n candidates", len(cands))
line("layers swept", sorted({c["layer"] for c in cands}))
line("positions swept", sorted({c["position"] for c in cands}))
line("rejection reasons", dict(Counter(c["reason"] for c in cands)))
line("n passing", sum(c["passes"] for c in cands))
line("selected (layer, pos)", (sel["layer"], sel["position"]))
line("selected bypass / induce / kl",
     f"{sel['bypass_score']:.3f} / {sel['induce_score']:.3f} / {sel['kl']:.3f}")
best = max(cands, key=lambda c: c["bypass_score"])
line("best bypass candidate", f"L{best['layer']} p{best['position']} bypass={best['bypass_score']:.3f} "
     f"induce={best['induce_score']:.3f} kl={best['kl']:.3f} passes={best['passes']}")
line("candidates with bypass > 0", sorted([round(c["bypass_score"], 3) for c in cands if c["bypass_score"] > 0],
                                          reverse=True))
line("candidates with induce > 0", [(c["layer"], c["position"], c["induce_score"]) for c in cands
                                    if c["induce_score"] > 0])
line("max induce over all candidates", max(c["induce_score"] for c in cands))
line("layer-rejected layers", sorted({c["layer"] for c in cands if c["reason"] == "layer >= 0.8L"}))

print("\n=== 3. SAFETY ===")
by = defaultdict(list)
for r in safety:
    by[r["lam"]].append(r)
for lam in LAM:
    rs = by[lam]
    line(f"lam={lam} refusal / n / mean words",
         f"{sum(x['refused'] for x in rs) / len(rs):.2f} / {len(rs)} / {st.mean([x['n_words'] for x in rs]):.1f}")
k0 = {x["idx"]: x for x in by[0.0]}
k1 = {x["idx"]: x for x in by[1.0]}
flips = sum(1 for i in k0 if k0[i]["refused"] != k1[i]["refused"])
line("refusal flips lam 0 -> 1", flips)
line("flips at any lam vs lam=0",
     sum(1 for lam in LAM for i in k0 if {x["idx"]: x for x in by[lam]}[i]["refused"] != k0[i]["refused"]))
d = [k1[i]["n_words"] - k0[i]["n_words"] for i in k0]
t = st.mean(d) / (st.stdev(d) / math.sqrt(len(d)))
line("paired delta words / t / df", f"{st.mean(d):.2f} / {t:.2f} / {len(d) - 1}")
line("rule of three 95% upper bound on flip rate", f"{3 / len(k0):.3f}")

print("\n=== 4. CAPABILITY ===")


def two_prop(p1, p2, n1, n2):
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


crow = {c["lam"]: c for c in cap}
for lam in LAM:
    c = crow[lam]
    line(f"lam={lam} mmlu/tqa/gsm8k/ce",
         f"{c['mmlu']:.3f} / {c['truthfulqa_mc1']:.4f} / {c['gsm8k']:.2f} / {c['ce_loss']:.3f}")
for key, n in (("mmlu", cfg["evals"]["n_mmlu"]), ("truthfulqa_mc1", cfg["evals"]["n_truthfulqa"]),
               ("gsm8k", cfg["evals"]["n_gsm8k"])):
    z, p = two_prop(crow[0.0][key], crow[1.0][key], n, n)
    diff = crow[0.0][key] - crow[1.0][key]
    se = math.sqrt(crow[0.0][key] * (1 - crow[0.0][key]) / n + crow[1.0][key] * (1 - crow[1.0][key]) / n)
    line(f"{key} lam0-lam1 diff [95% CI], z, p",
         f"{diff:+.3f} [{diff - 1.96 * se:+.3f}, {diff + 1.96 * se:+.3f}], z={z:.2f}, p={p:.2f}  (n={n})")
line("ce loss n", cfg["evals"]["n_ce_loss"])
n80 = 2 * (1.96 + 0.8416) ** 2 * 0.25 / (0.032 ** 2)
line("n/arm for 80% power on a 3.2pt MMLU diff", round(n80))

print("\n=== 5. INTROSPECTION CHANNELS ===")
for lam in LAM:
    for cond in ("C1", "C2", "C3"):
        rs = [r for r in sweep if r["lam"] == lam and r["condition"] == cond]
        pr = st.mean([bool(r["parseable"]) for r in rs])
        det = [r for r in rs if r["detected"] is not None]
        dr = st.mean([bool(r["detected"]) for r in det]) if det else float("nan")
        line(f"lam={lam} {cond} n / parse rate / detect rate",
             f"{len(rs)} / {pr:.2f} / {dr if det else 'undefined'}")
    fc = [r for r in sweep if r["lam"] == lam and r["condition"] == "C4"]
    acc = st.mean([bool(r["identified"]) for r in fc])
    z = (acc - 0.1) / math.sqrt(0.1 * 0.9 / len(fc))
    line(f"lam={lam} C4 n / accuracy / z vs chance", f"{len(fc)} / {acc:.4f} / {z:.1f}")
z, p = two_prop(0.72, 0.7141666666666666, 1200, 1200)
line("C4 lam0 vs lam1 z, p", f"{z:.2f}, {p:.2f}")

rng = random.Random(0)
for lam in (0.0, 1.0):
    per = defaultdict(list)
    for r in sweep:
        if r["lam"] == lam and r["condition"] == "C4":
            per[r["concept"]].append(bool(r["identified"]))
    keys = list(per)
    boots = sorted(st.mean([st.mean(per[k]) for k in rng.choices(keys, k=len(keys))]) for _ in range(10000))
    line(f"lam={lam} C4 concept-level bootstrap 95% CI",
         f"[{boots[250]:.3f}, {boots[9750]:.3f}]")

print("\n=== 6. SKEPTICAL VARIANT ===")
for lam in (0.0, 1.0):
    for cond in ("C1", "C2", "C3", "C4"):
        rs = [r for r in skept if r["lam"] == lam and r["condition"] == cond]
        pr = st.mean([bool(r["parseable"]) for r in rs])
        extra = ""
        if cond == "C2":
            extra = f" detect={st.mean([bool(r['detected']) for r in rs]):.2f}"
        if cond == "C4":
            extra = f" acc={st.mean([bool(r['identified']) for r in rs]):.4f}"
        line(f"lam={lam} {cond} n / parse{extra}", f"{len(rs)} / {pr:.2f}")

print("\n=== 7. PILOT ===")
pd_ = js("pilot_decision.json")
fc_rows = [r for r in pd_["rows"] if r["protocol"] == "forced_choice"]
pf_rows = [r for r in pd_["rows"] if r["protocol"] == "prefill"]
line("pilot concepts x trials",
     f"{len({r['concept'] for r in pilot})} x "
     f"{max(Counter((r['protocol'], r['lam'], r['layer'], r['alpha'], r['concept']) for r in pilot).values())}")
line("forced-choice cells / range",
     f"{len(fc_rows)} / [{min(r['accuracy'] for r in fc_rows):.2f}, {max(r['accuracy'] for r in fc_rows):.2f}]")
line("cells above chance", sum(r["accuracy"] > r["chance"] for r in fc_rows))
line("prefill range", f"[{min(r['accuracy'] for r in pf_rows):.2f}, {max(r['accuracy'] for r in pf_rows):.2f}]")
line("verdict", pd_["verdict"]["decision"])
line("best cell", pd_["verdict"]["best_cell"])
line("alphas / layers", (sorted({r["alpha"] for r in fc_rows}), sorted({r["layer"] for r in fc_rows})))
line("fc acc L18 a4 lam0 / lam1",
     f"{[r['accuracy'] for r in fc_rows if r['layer'] == 18 and r['alpha'] == 4.0 and r['lam'] == 0.0][0]} / "
     f"{[r['accuracy'] for r in fc_rows if r['layer'] == 18 and r['alpha'] == 4.0 and r['lam'] == 1.0][0]}")
deg = [r for r in pilot if r["protocol"] == "prefill" and r["layer"] == 18 and r["alpha"] == 4.0 and r["lam"] == 0.0]
line("prefill L18 a4 lam0: graded correct", f"{sum(bool(r['identified']) for r in deg)} / {len(deg)}")
mono = [r for r in deg if r["identified"] and
        Counter(r["response"].lower().split()).most_common(1)[0][1] > 0.4 * len(r["response"].split())]
line("of those, concept name is >40% of tokens", f"{len(mono)} / {sum(bool(r['identified']) for r in deg)}")

print("\n=== 8. CONCEPT BANK GEOMETRY ===")
names = bank["names"]
L = cfg["injection"]["layer"]
V = bank["vectors"].float()[:, L, :]
V = V / V.norm(dim=-1, keepdim=True)
C = V @ V.T
n = len(names)
off = [C[i, j].item() for i in range(n) for j in range(i + 1, n)]
line("n concepts / n pairs", f"{n} / {len(off)}")
line("mean / median / max pairwise cosine",
     f"{st.mean(off):.3f} / {st.median(off):.3f} / {max(off):.3f}")
line("fraction of pairs cos > 0.99", f"{sum(x > 0.99 for x in off) / len(off):.3f}")
X = bank["vectors"].float()[:, L, :]
X = X - X.mean(0)
ev = torch.linalg.svdvals(X) ** 2
ev = ev / ev.sum()
line("top PC share of centred variance", f"{ev[0]:.3f}")
line("PCs for 90% of variance", f"{int((torch.cumsum(ev, 0) < 0.9).sum()) + 1} / {n - 1}")
line("participation ratio", f"{(ev.sum() ** 2 / (ev ** 2).sum()).item():.1f}")
partners = {names[i]: int((C[i] > 0.99).sum()) - 1 for i in range(n)}
clique = [k for k in names if partners[k] > 0]
line("clique size / partner counts", f"{len(clique)} / {sorted(set(partners.values()))}")
sub = V[[names.index(c) for c in clique]]
line("mean internal cosine of clique",
     f"{((sub @ sub.T).sum().item() - len(clique)) / (len(clique) * (len(clique) - 1)):.4f}")
for LL in (0, 10, 18, 25, 31, 36):
    W = bank["vectors"].float()[:, LL, :]
    W = W / W.norm(dim=-1, keepdim=True)
    D = W @ W.T
    line(f"clique size at layer {LL}", sum(1 for i in range(n) if (D[i] > 0.99).sum() - 1 > 0))

acc = defaultdict(list)
for r in sweep:
    if r["condition"] == "C4":
        acc[r["concept"]].append(bool(r["identified"]))
pc = {k: st.mean(v) for k, v in acc.items()}
g0 = [pc[k] for k in names if partners[k] == 0]
g1 = [pc[k] for k in names if partners[k] > 0]
se = math.sqrt(st.variance(g0) / len(g0) + st.variance(g1) / len(g1))
line("isolated n / mean", f"{len(g0)} / {st.mean(g0):.3f}")
line("clique n / mean", f"{len(g1)} / {st.mean(g1):.3f}")
line("difference / Welch t", f"{st.mean(g0) - st.mean(g1):.3f} / {(st.mean(g0) - st.mean(g1)) / se:.2f}")
rng = random.Random(0)
bs = sorted(st.mean(rng.choices(g0, k=len(g0))) - st.mean(rng.choices(g1, k=len(g1))) for _ in range(10000))
line("bootstrap 95% CI of difference", f"[{bs[250]:.3f}, {bs[9750]:.3f}]")
idx = {m: i for i, m in enumerate(names)}
errs = [r for r in sweep if r["condition"] == "C4" and r["identified"] is False and r["concept_reported"] in idx]
line("error trials", len(errs))
line("mean cos(true, chosen | error) vs all-pairs mean",
     f"{st.mean([C[idx[r['concept']], idx[r['concept_reported']]].item() for r in errs]):.3f} vs {st.mean(off):.3f}")
line("per-concept acc: n at 1.0 / n at 0.0 / n between",
     f"{sum(v == 1 for v in pc.values())} / {sum(v == 0 for v in pc.values())} / "
     f"{sum(0 < v < 1 for v in pc.values())}")
print()
