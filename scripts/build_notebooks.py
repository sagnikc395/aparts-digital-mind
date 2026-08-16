"""Generate the single Colab runner notebook in ``notebooks/``.

The notebook is thin: every non-trivial line lives in ``src/alignment_tax``
and is version-controlled and testable. Keeping it generated from this script
means a change to the pipeline is a one-line edit here rather than hand-patching
a JSON blob.

    python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_NAME = "alignment_tax_full_run.ipynb"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("md", text.strip("\n")))


def code(body: str, title: str | None = None) -> None:
    src = body.strip("\n")
    if title:
        src = f"#@title {title}\n{src}"
    CELLS.append(("code", src))


# --------------------------------------------------------------------- cells

md("""
# The alignment tax of introspection — full run

One notebook for the whole experiment: setup, refusal-direction extraction, the
concept bank, the go/no-go pilot, the main introspection sweep, safety and
capability across lambda, analysis, figures, and a final export step that
collects **every** artifact — checkpoints, JSONL logs, analysis, plots, paper
figures — under `results/`.

Run it top to bottom. Every stage is resumable: artifacts already on disk are
loaded rather than recomputed, and the sweeps append to JSONL keyed by
`(lam, condition, concept, trial, variant)`, so re-running a cell after a
session kill picks up exactly where it stopped.

**Pilot decision rule.** If forced-choice identification is at chance both on the
unmodified model and at full ablation, stop and switch models (or fall back to
the reduced claim). Do not spend more than three hours on the pilot.
""")

md("## 0 · Environment")

code(
    '''
# Tokens come from Colab secrets (key icon in the left sidebar): add GITHUB_TOKEN
# (repo:read, only needed while the repo is private) and HF_TOKEN (read, needed
# for gated models), and toggle "Notebook access" on for both. Outside Colab we
# fall back to the environment, then to an interactive prompt. Neither token is
# written to disk.
import os, subprocess, sys, getpass, pathlib

REPO   = "sagnikc395/apart-mind-digital-mind"  #@param {type:"string"}
BRANCH = "main"                                 #@param {type:"string"}
WORKDIR = "/content"

try:
    from google.colab import userdata as _colab_secrets
except Exception:
    _colab_secrets = None


def get_secret(name: str, prompt: str) -> str:
    """Colab secret -> environment -> interactive prompt. Blank means 'skip'."""
    if _colab_secrets is not None:
        try:
            value = _colab_secrets.get(name)
        except Exception as exc:            # not set, or notebook access is off
            print(f"{name}: no Colab secret ({type(exc).__name__})")
        else:
            if value:
                os.environ[name] = value
                print(f"{name}: from Colab secrets")
                return value
    value = os.environ.get(name)
    if value:
        print(f"{name}: from environment")
        return value
    value = getpass.getpass(prompt).strip()
    if value:
        os.environ[name] = value
    return value


GITHUB_TOKEN = get_secret("GITHUB_TOKEN", "GitHub token (blank if public): ")
HF_TOKEN = get_secret("HF_TOKEN", "Hugging Face token (blank if not gated): ")
if HF_TOKEN:
    # the two names transformers/huggingface_hub actually read
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["HF_TOKEN"] = HF_TOKEN

if pathlib.Path("/content").exists():
    token = GITHUB_TOKEN
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

REPO_ROOT = pathlib.Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))
print("cwd:", os.getcwd())

if HF_TOKEN:
    try:
        from huggingface_hub import HfApi
        print("hugging face:", HfApi().whoami(token=HF_TOKEN)["name"])
    except Exception as exc:
        print("hugging face token present but whoami failed:", exc)
''',
    'Clone the repo and install dependencies { display-mode: "form" }',
)

code(
    '''
# Everything the run produces lands under RESULTS. On Colab that is a Drive
# folder so the run survives a session kill; the final export step mirrors it
# back into the repo's results/ directory either way.
import os, pathlib

RESULTS = "/content/drive/MyDrive/alignment_tax_results"  #@param {type:"string"}
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as exc:
    RESULTS = str(REPO_ROOT / "results")
    print("no Drive; writing to", RESULTS, f"({exc})")
os.environ["ALIGNMENT_TAX_RESULTS"] = RESULTS
RESULTS = pathlib.Path(RESULTS)
RESULTS.mkdir(parents=True, exist_ok=True)
LOCAL_RESULTS = REPO_ROOT / "results"   # export target inside the repo
print("results ->", RESULTS)
''',
    "Results directory (Drive when available, else repo results/)",
)

code(
    """
!python -m alignment_tax.build_data
""",
    "Build the bundled data files (concepts, baseline corpus, offline fallbacks)",
)

code(
    '''
from pathlib import Path
from alignment_tax.config import RunConfig, LAMBDA_GRID
from alignment_tax import pipeline

cfg = RunConfig()
cfg.model.name = "Qwen/Qwen3-4B-Instruct-2507"  #@param {type:"string"}
cfg.model.batch_size = 16                        #@param {type:"integer"}
cfg.lambdas = LAMBDA_GRID                        # 0, 0.25, 0.5, 0.75, 1.0
cfg.results_dir = RESULTS
cfg.save()
print("run dir:", cfg.run_dir)
''',
    "Run configuration",
)

code(
    """
hm = pipeline.load_model(cfg)
""",
    "Load the model",
)

md("""
## 1 · Refusal direction

Difference-in-means over harmful (AdvBench) minus harmless (Alpaca) at every
(post-instruction position, layer) pair, then selected on *held-out* HarmBench +
Alpaca by bypass score, induce score > 0, KL < 0.1 on harmless inputs, and
layer < 0.8L. The selected layer and position are reported in the paper.
Checkpoint: `refusal_direction.pt`.
""")

code('''
rd = pipeline.stage_direction(hm, cfg)
print("selected layer:", rd.layer, "position:", rd.position)
print("bypass:", round(rd.scores.bypass_score, 3),
      "induce:", round(rd.scores.induce_score, 3),
      "KL:", round(rd.scores.kl, 4))
''')

md("""
## 2 · Concept bank

Difference-of-means concept vectors against a generic baseline corpus, unit
normalised, plus the mean residual-stream norm at each candidate injection layer
(that norm is what makes the injection strength `alpha` comparable across layers
and model families). Checkpoint: `concept_bank.pt`.
""")

code('''
bank = pipeline.stage_concepts(hm, cfg)
print(len(bank.names), "concepts; norm scale:", bank.norm_scale)
''')

md("""
## 3 · Pilot (go / no-go)

20 concepts, alpha in {2, 4}, three layers spanning 0.5-0.85 of depth, at
lambda in {0, 1}, using prefill forced identification and k-way forced choice.
""")

code('''
decision = pipeline.stage_pilot(hm, cfg, bank, rd.vector)
import pandas as pd
pd.DataFrame(decision["rows"]).sort_values("accuracy", ascending=False).head(12)
''')

code(
    '''
best = decision["verdict"]["best_cell"]
if best:
    cfg.injection.layer = int(best["layer"])
    cfg.injection.alpha = float(best["alpha"])
    cfg.save()
print("locked layer:", cfg.injection.layer, "alpha:", cfg.injection.alpha)
print("decision:", decision["verdict"]["decision"])
''',
    "Fix the injection layer and alpha for the main sweep",
)

md("""
## 4 · Main introspection sweep

Conditions C1-C4 at every lambda, structured prompt variant, plus the skeptical
prompt variant at the lambda endpoints as a robustness check.
""")

code(
    '''
sweep_path = pipeline.stage_sweep(hm, cfg, bank, rd.vector, variant="structured")
print(sweep_path)
''',
    "Main sweep (resumable -- just re-run this cell if the session dies)",
)

code(
    '''
pipeline.stage_sweep(hm, cfg, bank, rd.vector, variant="skeptical",
                     lambdas=(min(cfg.lambdas), max(cfg.lambdas)))
''',
    "Robustness: skeptical prompt variant at the lambda endpoints only",
)

code(
    '''
from alignment_tax.stats import load_and_summarise
import pandas as pd

records, summary = load_and_summarise(cfg.artifact("sweep_structured.jsonl"), n_boot=500)
pd.DataFrame([{"lam": lam, **{k: round(v.value, 3) for k, v in m.items()}} for lam, m in summary.items()])
''',
    "Quick look at where things stand",
)

md("""
## 5 · Safety and capability across lambda

Family B (100 JailbreakBench behaviours, refusal heuristic plus an optional
Llama-Guard-class judge) and Family C (CE loss, MMLU, TruthfulQA MC1, optionally
GSM8K).

No harmful completion is ever reproduced in the paper: raw generations stay in
the JSONL under `results/`, and only aggregate scores are exported.
""")

code(
    '''
USE_GUARD = False  #@param {type:"boolean"}
pipeline.stage_safety(hm, cfg, rd.vector, use_guard=USE_GUARD)
''',
    "Safety sweep (500 generations at the default grid)",
)

code(
    '''
cfg.evals.run_gsm8k = False  #@param {type:"boolean"}
pipeline.stage_capability(hm, cfg, rd.vector)
''',
    "Capability sweep",
)

code('''
import json
print(json.dumps(json.loads(cfg.artifact("capability.json").read_text()), indent=2))
''')

md("""
## 6 · Analysis and figures

Concept-level bootstrap (10,000 resamples), the specificity index, the exchange
rate, and the four figures. This section is CPU-only -- if the GPU session died
after the sweeps, it still runs.
""")

code('''
import json
analysis = pipeline.stage_analyse(cfg, n_boot=10_000)
print(json.dumps(analysis["exchange_rate"], indent=2))
''')

code('''
import pandas as pd
pd.DataFrame(analysis["rows"])[[
    "lam", "tpr", "fpr_clean", "fpr_random", "identification",
    "conditional_identification", "d_clean", "d_random", "specificity_index",
    "safety_refusal_rate", "cap_mmlu", "cap_truthfulqa_mc1", "cap_ce_loss",
]].round(3)
''')

code(
    '''
# Figures are written into the results tree first (so they travel with the run)
# and mirrored into paper/figures for the write-up.
import shutil
from pathlib import Path

fig_dir = Path(cfg.run_dir) / "figures"
paths = pipeline.stage_figures(cfg, fig_dir=fig_dir)

paper_figs = REPO_ROOT / "paper" / "figures"
paper_figs.mkdir(parents=True, exist_ok=True)
for p in Path(fig_dir).glob("fig*.*"):
    shutil.copy2(p, paper_figs / p.name)

from IPython.display import Image, display
for p in paths:
    display(Image(str(p)))
''',
    "Figures 1-4",
)

code(
    '''
print(json.dumps(analysis["contrasts"]["detection_C1"], indent=2))
print(json.dumps(analysis["forced_choice_vs_chance"], indent=2))
''',
    "Statistical contrasts (two-proportion tests, Holm-corrected)",
)

md("""
### Judge validation

Hand-label 100 identification outputs and report Cohen's kappa against the
grader. Run `judge-sample`, fill in the `human` field in `judge_labels.jsonl`,
then run `judge-kappa`.
""")

code("""
!python -m alignment_tax.cli judge-sample --results-dir $ALIGNMENT_TAX_RESULTS
# ... fill in the 'human' field in judge_labels.jsonl, then:
!python -m alignment_tax.cli judge-kappa --results-dir $ALIGNMENT_TAX_RESULTS
""")

md("""
## 7 · Export everything to `results/`

Collects the whole run -- config, checkpoints (`refusal_direction.pt`,
`concept_bank.pt`), every JSONL log, the analysis JSON, and the plots -- into
the repo's `results/` directory, writes a `MANIFEST.json` listing each file with
its size and sha256, and zips the run for download or attachment.

If the run wrote to Drive, this mirrors it back into the repo; if it already
wrote to `results/`, the copy is a no-op and only the manifest and zip are new.
""")

code(
    '''
import hashlib, json, shutil
from pathlib import Path

src_run = Path(cfg.run_dir)
dst_run = LOCAL_RESULTS / cfg.model.short_name
dst_run.mkdir(parents=True, exist_ok=True)

if src_run.resolve() != dst_run.resolve():
    shutil.copytree(src_run, dst_run, dirs_exist_ok=True)
    print("mirrored", src_run, "->", dst_run)

def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

files = sorted(p for p in dst_run.rglob("*") if p.is_file() and p.name != "MANIFEST.json")
manifest = {
    "model": cfg.model.name,
    "lambdas": list(cfg.lambdas),
    "injection_layer": cfg.injection.layer,
    "injection_alpha": cfg.injection.alpha,
    "direction_layer": rd.layer if "rd" in dir() else None,
    "exchange_rate": analysis.get("exchange_rate") if "analysis" in dir() else None,
    "files": [
        {"path": str(p.relative_to(dst_run)), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in files
    ],
}
(dst_run / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

archive = shutil.make_archive(str(LOCAL_RESULTS / f"{cfg.model.short_name}_run"), "zip",
                              root_dir=dst_run.parent, base_dir=dst_run.name)

total = sum(f["bytes"] for f in manifest["files"])
for f in manifest["files"]:
    print(f"{f['bytes']:>12,}  {f['path']}")
print(f"\\n{len(files)} files, {total / 1e6:.1f} MB in {dst_run}")
print("archive:", archive)
''',
    "Collect run artifacts, checkpoints, plots into results/",
)

code(
    '''
# Optional: commit the exported results back to the repo. Off by default --
# checkpoints and JSONL logs can be large, so check the manifest sizes first.
PUSH_TO_GIT = False  #@param {type:"boolean"}
COMMIT_MESSAGE = "results: full alignment-tax run"  #@param {type:"string"}

if PUSH_TO_GIT:
    import subprocess
    rel = str(dst_run.relative_to(REPO_ROOT))
    subprocess.run(["git", "-C", str(REPO_ROOT), "add", "-f", rel, "paper/figures"], check=True)
    subprocess.run(["git", "-C", str(REPO_ROOT), "commit", "-m", COMMIT_MESSAGE], check=False)
    subprocess.run(["git", "-C", str(REPO_ROOT), "push"], check=False)
else:
    print("PUSH_TO_GIT is off; results are on disk at", dst_run)
''',
    "Optional: commit results/ back to the repository",
)


# --------------------------------------------------------------------- build


def notebook(cells: list[tuple[str, str]]) -> dict:
    out = []
    for kind, src in cells:
        lines = src.splitlines(keepends=True)
        if kind == "md":
            out.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            out.append({"cell_type": "code", "execution_count": None, "metadata": {},
                        "outputs": [], "source": lines})
    return {
        "cells": out,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    path = NB_DIR / NB_NAME
    path.write_text(json.dumps(notebook(CELLS), indent=1) + "\n")
    print("wrote", path, f"({len(CELLS)} cells)")


if __name__ == "__main__":
    build()
