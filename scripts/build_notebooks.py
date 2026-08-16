"""Generate the Colab runner notebooks in ``notebooks/``.

The notebooks are thin: every non-trivial line lives in ``src/alignment_tax``
and is version-controlled and testable. Keeping them generated from this script
means a change to the pipeline is a one-line edit here rather than hand-patching
JSON blobs.

    python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

CLONE = '''\
#@title Clone the repo and install dependencies { display-mode: "form" }
# Colab: paste a GitHub PAT with repo:read scope. It is used only for the clone
# and is not written to disk.
import os, subprocess, sys, getpass, pathlib

REPO   = "sagnikc395/apart-mind-digital-mind"  #@param {type:"string"}
BRANCH = "main"                                 #@param {type:"string"}
WORKDIR = "/content"

if pathlib.Path("/content").exists():
    token = os.environ.get("GITHUB_TOKEN") or getpass.getpass("GitHub token (blank if public): ")
    url = f"https://{token}@github.com/{REPO}.git" if token else f"https://github.com/{REPO}.git"
    dest = pathlib.Path(WORKDIR) / REPO.split("/")[-1]
    if dest.exists():
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
    else:
        subprocess.run(["git", "clone", "--branch", BRANCH, "--depth", "1", url, str(dest)], check=True)
    os.chdir(dest)
    subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                    "torch", "transformers>=4.44", "accelerate", "datasets", "matplotlib"], check=True)
else:
    os.chdir(pathlib.Path.cwd())  # already inside the repo, e.g. running locally

sys.path.insert(0, str(pathlib.Path.cwd() / "src"))
print("cwd:", os.getcwd())
'''

DRIVE = '''\
#@title Persist results to Drive (survives a session kill)
import os, pathlib

RESULTS = "/content/drive/MyDrive/alignment_tax_results"  #@param {type:"string"}
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as exc:
    RESULTS = str(pathlib.Path.cwd() / "results")
    print("no Drive; writing to", RESULTS, f"({exc})")
os.environ["ALIGNMENT_TAX_RESULTS"] = RESULTS
pathlib.Path(RESULTS).mkdir(parents=True, exist_ok=True)
print("results ->", RESULTS)
'''

CONFIG = '''\
#@title Run configuration
from pathlib import Path
import os
from alignment_tax.config import RunConfig, LAMBDA_GRID

cfg = RunConfig()
cfg.model.name = "Qwen/Qwen3-4B-Instruct-2507"  #@param {type:"string"}
cfg.model.batch_size = 16                        #@param {type:"integer"}
cfg.lambdas = LAMBDA_GRID                        # 0, 0.25, 0.5, 0.75, 1.0
cfg.results_dir = Path(os.environ.get("ALIGNMENT_TAX_RESULTS", "results"))
cfg.save()
print(cfg.run_dir)
'''


def nb(cells: list[tuple[str, str]]) -> dict:
    out = []
    for kind, src in cells:
        if kind == "md":
            out.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})
        else:
            out.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                        "source": src.splitlines(keepends=True)})
    return {
        "cells": out,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "L4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "00_setup_and_pilot.ipynb": [
        ("md", """# 00 · Setup and the go/no-go pilot

Hours 0-3 of the plan. This notebook extracts the refusal direction, builds the
concept bank, and runs the pilot that decides whether the effect is measurable
at this scale at all.

**Decision rule.** If forced-choice identification is at chance both on the
unmodified model and at full ablation, stop and switch models (or fall back to
the reduced claim). Do not spend more than three hours here.
"""),
        ("code", CLONE),
        ("code", DRIVE),
        ("code", """#@title Build the bundled data files (concepts, baseline corpus, offline fallbacks)
!python -m alignment_tax.build_data"""),
        ("code", CONFIG),
        ("code", """#@title Load the model
from alignment_tax import pipeline

hm = pipeline.load_model(cfg)"""),
        ("md", """## Refusal direction

Difference-in-means over harmful (AdvBench) minus harmless (Alpaca) at every
(post-instruction position, layer) pair, then selected on *held-out* HarmBench +
Alpaca by bypass score, induce score > 0, KL < 0.1 on harmless inputs, and
layer < 0.8L. The selected layer and position are reported in the paper."""),
        ("code", """rd = pipeline.stage_direction(hm, cfg)
print("selected layer:", rd.layer, "position:", rd.position)
print("bypass:", round(rd.scores.bypass_score, 3),
      "induce:", round(rd.scores.induce_score, 3),
      "KL:", round(rd.scores.kl, 4))"""),
        ("md", """## Concept bank

Difference-of-means concept vectors against a generic baseline corpus, unit
normalised, plus the mean residual-stream norm at each candidate injection layer
(that norm is what makes the injection strength `alpha` comparable across layers
and model families)."""),
        ("code", """bank = pipeline.stage_concepts(hm, cfg)
print(len(bank.names), "concepts; norm scale:", bank.norm_scale)"""),
        ("md", """## Pilot

20 concepts, alpha in {2, 4}, three layers spanning 0.5-0.85 of depth, at
lambda in {0, 1}, using prefill forced identification and k-way forced choice."""),
        ("code", """decision = pipeline.stage_pilot(hm, cfg, bank, rd.vector)
import pandas as pd
pd.DataFrame(decision["rows"]).sort_values("accuracy", ascending=False).head(12)"""),
        ("code", """#@title Fix the injection layer and alpha for the main sweep
best = decision["verdict"]["best_cell"]
if best:
    cfg.injection.layer = int(best["layer"])
    cfg.injection.alpha = float(best["alpha"])
    cfg.save()
print("locked layer:", cfg.injection.layer, "alpha:", cfg.injection.alpha)
print("decision:", decision["verdict"]["decision"])"""),
    ],
    "01_main_sweep.ipynb": [
        ("md", """# 01 · Main introspection sweep

Hours 9-17. Conditions C1-C4 at every lambda, structured prompt variant.

Every generation is appended to JSONL as it is produced and keyed by
`(lam, condition, concept, trial, variant)`, so re-running this cell after a
session kill resumes exactly where it stopped."""),
        ("code", CLONE),
        ("code", DRIVE),
        ("code", """#@title Reload the locked configuration from the pilot
from pathlib import Path
import os
from alignment_tax.config import RunConfig
from alignment_tax import pipeline

results = Path(os.environ.get("ALIGNMENT_TAX_RESULTS", "results"))
cfg = RunConfig.load(next(results.rglob("run_config.json")))
cfg.results_dir = results
print("layer:", cfg.injection.layer, "alpha:", cfg.injection.alpha, "lambdas:", cfg.lambdas)

hm = pipeline.load_model(cfg)
rd = pipeline.stage_direction(hm, cfg)
bank = pipeline.stage_concepts(hm, cfg)"""),
        ("code", """#@title Main sweep (resumable -- just re-run this cell if the session dies)
path = pipeline.stage_sweep(hm, cfg, bank, rd.vector, variant="structured")
print(path)"""),
        ("code", """#@title Robustness: skeptical prompt variant at the lambda endpoints only
pipeline.stage_sweep(hm, cfg, bank, rd.vector, variant="skeptical",
                     lambdas=(min(cfg.lambdas), max(cfg.lambdas)))"""),
        ("code", """#@title Quick look at where things stand
from alignment_tax.stats import load_and_summarise
import pandas as pd

records, summary = load_and_summarise(cfg.artifact("sweep_structured.jsonl"), n_boot=500)
pd.DataFrame([{"lam": lam, **{k: round(v.value, 3) for k, v in m.items()}} for lam, m in summary.items()])"""),
    ],
    "02_safety_and_capability.ipynb": [
        ("md", """# 02 · Safety and capability across lambda

Hours 17-21. Family B (100 JailbreakBench behaviours, refusal heuristic plus an
optional Llama-Guard-class judge) and Family C (CE loss, MMLU, TruthfulQA MC1,
optionally GSM8K).

No harmful completion is ever reproduced in the paper: raw generations stay in
the local JSONL, and only aggregate scores are exported."""),
        ("code", CLONE),
        ("code", DRIVE),
        ("code", """from pathlib import Path
import os
from alignment_tax.config import RunConfig
from alignment_tax import pipeline

results = Path(os.environ.get("ALIGNMENT_TAX_RESULTS", "results"))
cfg = RunConfig.load(next(results.rglob("run_config.json")))
cfg.results_dir = results
hm = pipeline.load_model(cfg)
rd = pipeline.stage_direction(hm, cfg)"""),
        ("code", """#@title Safety sweep (500 generations at the default grid)
USE_GUARD = False  #@param {type:"boolean"}
pipeline.stage_safety(hm, cfg, rd.vector, use_guard=USE_GUARD)"""),
        ("code", """#@title Capability sweep
cfg.evals.run_gsm8k = False  #@param {type:"boolean"}
pipeline.stage_capability(hm, cfg, rd.vector)"""),
        ("code", """import json
print(json.dumps(json.loads(cfg.artifact("capability.json").read_text()), indent=2))"""),
    ],
    "03_analysis_and_figures.ipynb": [
        ("md", """# 03 · Analysis and figures

Hours 24-30. Concept-level bootstrap (10,000 resamples), the specificity index,
the exchange rate, and the four figures. Runs on CPU -- no GPU needed."""),
        ("code", CLONE),
        ("code", DRIVE),
        ("code", """from pathlib import Path
import os, json
from alignment_tax.config import RunConfig
from alignment_tax import pipeline

results = Path(os.environ.get("ALIGNMENT_TAX_RESULTS", "results"))
cfg = RunConfig.load(next(results.rglob("run_config.json")))
cfg.results_dir = results

analysis = pipeline.stage_analyse(cfg, n_boot=10_000)
print(json.dumps(analysis["exchange_rate"], indent=2))"""),
        ("code", """import pandas as pd
pd.DataFrame(analysis["rows"])[[
    "lam", "tpr", "fpr_clean", "fpr_random", "identification",
    "conditional_identification", "d_clean", "d_random", "specificity_index",
    "safety_refusal_rate", "cap_mmlu", "cap_truthfulqa_mc1", "cap_ce_loss",
]].round(3)"""),
        ("code", """#@title Figures 1-4
paths = pipeline.stage_figures(cfg)
from IPython.display import Image, display
for p in paths:
    display(Image(str(p)))"""),
        ("md", """## Judge validation

Hand-label 100 identification outputs and report Cohen's kappa against the
grader. Run `judge-sample`, fill in the `human` field, then `judge-kappa`."""),
        ("code", """!python -m alignment_tax.cli judge-sample --results-dir $ALIGNMENT_TAX_RESULTS
# ... fill in the 'human' field in judge_labels.jsonl, then:
!python -m alignment_tax.cli judge-kappa --results-dir $ALIGNMENT_TAX_RESULTS"""),
        ("code", """#@title Statistical contrasts (two-proportion tests, Holm-corrected)
print(json.dumps(analysis["contrasts"]["detection_C1"], indent=2))
print(json.dumps(analysis["forced_choice_vs_chance"], indent=2))"""),
    ],
}


def build() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        (NB_DIR / name).write_text(json.dumps(nb(cells), indent=1))
        print("wrote", NB_DIR / name)


if __name__ == "__main__":
    build()
