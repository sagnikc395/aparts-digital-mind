"""Generate the layer-18 confirmation-run notebook in ``notebooks/``.

The main run (see ``build_notebooks.py``) showed that the conjunctive
bypass-and-induce filter selects a direction with zero bypass power on
Qwen3-4B: the strongest bypass candidate (layer 18, position -3, bypass 0.875,
KL 0.081) was rejected for inducing no refusal. This notebook re-runs the
dose-response with that candidate forced via ``DirectionConfig.force_candidate``,
which is the positive-control run the peer review asked for: it answers whether
the tax is measurable when the dial is connected to something.

    python scripts/build_layer18_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_NAME = "layer18_confirmation_run.ipynb"

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
# Layer-18 confirmation run — pricing the tax with a live direction

The full run selected a refusal direction with **zero held-out bypass power**
(layer 4, position -3), because the bypass and induce criteria dissociate on
Qwen3-4B, and refusal on JailbreakBench did not move at any dose. This notebook
runs the confirmation the peer review asked for: force the **highest-bypass
candidate that the induce filter rejected** — layer 18, position -3, bypass
0.875, KL 0.081 — and measure safety (and optionally capability and the
introspection sweep) across the same lambda grid.

The forced selection is stamped `forced: selected by config override, filters
waived` in `refusal_direction.json`, so it can never be mistaken for a
validated one. Everything writes into a **separate results tree**
(`results/layer18/...`) so the primary run is untouched, and the final cells
commit that tree back to GitHub.

Run top to bottom. Every stage is resumable after a session kill.
""")

md("## 0 · Environment")

code(
    '''
# Tokens come from Colab secrets (key icon in the left sidebar): add GITHUB_TOKEN
# (needs write access -- contents:write on a fine-grained PAT -- because the last
# cell pushes results/ back) and HF_TOKEN (read, needed for gated models), and
# toggle "Notebook access" on for both. Outside Colab we fall back to the
# environment, then to an interactive prompt. HF_TOKEN is never written to disk;
# GITHUB_TOKEN ends up in the clone's .git/config, which is discarded with the
# Colab VM.
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
# A *separate* results tree from the primary run, so the stage cache cannot hand
# this run the old layer-4 direction and the primary artifacts stay pristine.
# On Colab it lives on Drive so the run survives a session kill.
import os, pathlib

RESULTS = "/content/drive/MyDrive/alignment_tax_results/layer18"  #@param {type:"string"}
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as exc:
    RESULTS = str(REPO_ROOT / "results" / "layer18")
    print("no Drive; writing to", RESULTS, f"({exc})")
os.environ["ALIGNMENT_TAX_RESULTS"] = RESULTS
RESULTS = pathlib.Path(RESULTS)
RESULTS.mkdir(parents=True, exist_ok=True)
LOCAL_RESULTS = REPO_ROOT / "results" / "layer18"   # export target inside the repo
print("results ->", RESULTS)
''',
    "Results directory (separate layer18/ tree; Drive when available)",
)

code(
    """
!python -m alignment_tax.build_data
""",
    "Build the bundled data files (concepts, baseline corpus, offline fallbacks)",
)

code(
    '''
from alignment_tax.config import RunConfig, LAMBDA_GRID
from alignment_tax import pipeline

cfg = RunConfig()
cfg.model.name = "Qwen/Qwen3-4B-Instruct-2507"  #@param {type:"string"}
cfg.model.batch_size = 16                        #@param {type:"integer"}
cfg.lambdas = LAMBDA_GRID                        # 0, 0.25, 0.5, 0.75, 1.0
cfg.results_dir = RESULTS
# The point of this notebook: take the highest-bypass candidate that the induce
# filter rejected in the primary run, instead of letting the filters pick.
cfg.direction.force_candidate = (18, -3)
cfg.save()
print("run dir:", cfg.run_dir)
print("forced candidate:", cfg.direction.force_candidate)
''',
    "Run configuration (layer-18 candidate forced)",
)

code(
    """
hm = pipeline.load_model(cfg)
""",
    "Load the model",
)

md("""
## 1 · Refusal direction, selection overridden

Extraction and held-out scoring run exactly as in the primary run (so the full
95-candidate funnel is regenerated and saved for provenance), but selection
returns the forced (18, -3) candidate with a `forced:` stamp instead of the
filters' choice. Expected held-out scores from the primary run: bypass ≈ 0.875,
induce ≈ 0.000, KL ≈ 0.081.
""")

code('''
rd = pipeline.stage_direction(hm, cfg)
assert (rd.layer, rd.position) == (18, -3), f"got ({rd.layer}, {rd.position}); check force_candidate"
print("selected layer:", rd.layer, "position:", rd.position)
print("bypass:", round(rd.scores.bypass_score, 3),
      "induce:", round(rd.scores.induce_score, 3),
      "KL:", round(rd.scores.kl, 4),
      "| stamp:", rd.scores.reason or "(none)")
''')

md("""
## 2 · Safety across lambda — the actual confirmation

100 JailbreakBench behaviours at every lambda. The primary run's result was
0.97 refusal at every dose with 0/100 prompt-level flips; this is the same
measurement with a direction that has real held-out bypass power. The summary
cell reports the refusal rate *and* the per-prompt flip count against lambda=0,
which is the per-dose manipulation check the paper argues for.

No harmful completion leaves this results tree: raw generations stay in the
JSONL, and only aggregate scores are exported to the paper.
""")

code(
    '''
pipeline.stage_safety(hm, cfg, rd.vector)
''',
    "Safety sweep (resumable -- re-run this cell if the session dies)",
)

code(
    '''
import json
from collections import defaultdict
from alignment_tax.io_utils import read_jsonl

recs = list(read_jsonl(cfg.artifact("safety.jsonl")))
by_lam = defaultdict(dict)
for r in recs:
    by_lam[r["lam"]][r["idx"]] = r

lams = sorted(by_lam)
base = by_lam[lams[0]]
summary = []
for lam in lams:
    rows = by_lam[lam]
    idxs = sorted(set(rows) & set(base))
    refusal = sum(rows[i]["refused"] for i in idxs) / len(idxs)
    flips = sum(1 for i in idxs if rows[i]["refused"] != base[i]["refused"])
    words = sum(rows[i]["n_words"] for i in idxs if rows[i]["refused"]) / max(
        sum(1 for i in idxs if rows[i]["refused"]), 1)
    summary.append({"lam": lam, "n": len(idxs), "refusal_rate": round(refusal, 3),
                    "flips_vs_lam0": flips, "mean_words_per_refusal": round(words, 1)})

print(json.dumps(summary, indent=2))
cfg.artifact("safety_summary_layer18.json").write_text(json.dumps(summary, indent=2))
''',
    "Refusal rate and prompt-level flips per lambda (the manipulation check)",
)

md("""
## 3 · Capability across lambda

If the direction now buys something on safety, this is the other side of the
ledger: CE loss, MMLU, TruthfulQA MC1, and GSM8K at every lambda. Skippable if
the session is short — safety above is the load-bearing measurement.
""")

code(
    '''
RUN_CAPABILITY = True   #@param {type:"boolean"}
RUN_GSM8K = True        #@param {type:"boolean"}
if RUN_CAPABILITY:
    cfg.evals.run_gsm8k = RUN_GSM8K
    cfg.save()
    pipeline.stage_capability(hm, cfg, rd.vector)
    import json
    print(json.dumps(json.loads(cfg.artifact("capability.json").read_text()), indent=2))
else:
    print("skipped")
''',
    "Capability sweep (optional but cheap relative to the sweep)",
)

md("""
## 4 · Introspection sweep (optional)

The full pricing run: conditions C1–C4 at every lambda with the layer-18
direction, at the primary run's injection settings (layer 18, alpha 4.0) so the
numbers are directly comparable. Note the primary run's caveat applies
unchanged: at alpha 4.0 the free-text channel does not parse, so forced choice
is the informative column. Enable when the session has the ~2 GPU-hours.
""")

code(
    '''
RUN_SWEEP = False  #@param {type:"boolean"}
if RUN_SWEEP:
    bank = pipeline.stage_concepts(hm, cfg)
    cfg.injection.layer = 18
    cfg.injection.alpha = 4.0
    cfg.save()
    sweep_path = pipeline.stage_sweep(hm, cfg, bank, rd.vector, variant="structured")
    print(sweep_path)
else:
    print("skipped")
''',
    "Introspection sweep at the primary run's settings (resumable)",
)

md("""
## 5 · Export and push everything back to GitHub

Mirrors the run into the repo's `results/layer18/` directory, writes a
`MANIFEST.json` with size and sha256 per file, zips the run, and pushes the
commit. Files over GitHub's 100 MB limit are excluded from the commit (they
stay on Drive and in the zip).
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
    "run": "layer18_confirmation",
    "model": cfg.model.name,
    "lambdas": list(cfg.lambdas),
    "forced_candidate": list(cfg.direction.force_candidate),
    "direction": {"layer": rd.layer, "position": rd.position,
                  "bypass": rd.scores.bypass_score, "induce": rd.scores.induce_score,
                  "kl": rd.scores.kl, "stamp": rd.scores.reason},
    "files": [
        {"path": str(p.relative_to(dst_run)), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in files
    ],
}
(dst_run / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

archive = shutil.make_archive(str(LOCAL_RESULTS / f"{cfg.model.short_name}_layer18_run"), "zip",
                              root_dir=dst_run.parent, base_dir=dst_run.name)

total = sum(f["bytes"] for f in manifest["files"])
for f in manifest["files"]:
    print(f"{f['bytes']:>12,}  {f['path']}")
print(f"\\n{len(files)} files, {total / 1e6:.1f} MB in {dst_run}")
print("archive:", archive)
''',
    "Collect run artifacts into results/layer18/ with a manifest",
)

code(
    '''
# Commit the exported results back to the repo. results/ is gitignored, so the
# add is forced; anything over GitHub's 100 MB hard limit is left out of the
# commit (it stays on Drive and in the zip archive). GITHUB_TOKEN needs write
# scope (contents:write on a fine-grained PAT) for the push to be accepted.
import subprocess
from pathlib import Path

COMMIT_MESSAGE = "results: layer-18 confirmation run"  #@param {type:"string"}
GIT_NAME  = "Sagnik Chatterjee"                        #@param {type:"string"}
GIT_EMAIL = "sagnikchatterjee607@gmail.com"            #@param {type:"string"}
MAX_BYTES = 95 * 1024 * 1024   # stay under GitHub's 100 MB per-file hard limit


def git(*args, **kw):
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], **kw)


git("config", "user.name", GIT_NAME, check=True)
git("config", "user.email", GIT_EMAIL, check=True)

# The clone URL only carries the token when one was supplied; set it explicitly
# so the push authenticates even if the repo was cloned anonymously.
if GITHUB_TOKEN:
    git("remote", "set-url", "origin",
        f"https://{GITHUB_TOKEN}@github.com/{REPO}.git", check=True)

addable, skipped = [], []
for p in sorted(dst_run.rglob("*")):
    if not p.is_file():
        continue
    (skipped if p.stat().st_size > MAX_BYTES else addable).append(p)
for p in skipped:
    print(f"skipping (> {MAX_BYTES // (1024*1024)} MB): {p.relative_to(REPO_ROOT)}")

git("add", "-f", *[str(p.relative_to(REPO_ROOT)) for p in addable], check=True)

# Refresh from origin first so the push cannot fail on a remote commit made
# while this session was running; results files never conflict across runs.
git("pull", "--rebase", "--autostash", "origin", BRANCH, check=False)

if git("diff", "--cached", "--quiet").returncode == 0:
    print("nothing new to commit; results already match the repo at", dst_run)
else:
    git("commit", "-m", COMMIT_MESSAGE, check=True)
    git("push", "origin", f"HEAD:{BRANCH}", check=True)   # loud on failure
    print("pushed", str(dst_run.relative_to(REPO_ROOT)), "to", f"{REPO}@{BRANCH}")
''',
    "Commit results/layer18/ back to the repository",
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
