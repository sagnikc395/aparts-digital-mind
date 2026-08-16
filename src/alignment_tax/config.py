"""Central configuration for the alignment-tax-of-introspection experiment.

Every stage of the pipeline reads its knobs from here so that a single
``--config`` override (or environment variable) reproduces a whole run.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = Path(os.environ.get("ALIGNMENT_TAX_RESULTS", REPO_ROOT / "results"))
FIGURE_DIR = REPO_ROOT / "paper" / "figures"

#: Dose-response grid for partial directional ablation (section 3.1 of the plan).
LAMBDA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
#: Optional over-ablation point; enabled with ``--over-ablate``.
LAMBDA_OVER: float = 1.25


@dataclass
class ModelConfig:
    """Which checkpoint to run and how to place it.

    bf16 only: section 4.1 of the plan rules out FP8/4-bit because block
    quantisation noise lands on exactly the signal path we inject into.
    """

    name: str = "Qwen/Qwen3-4B-Instruct-2507"
    dtype: str = "bfloat16"
    device: str = "auto"  # "auto" -> cuda, else mps, else cpu
    max_context: int = 8192
    batch_size: int = 16
    max_new_tokens: int = 48
    trust_remote_code: bool = False

    @property
    def short_name(self) -> str:
        return self.name.split("/")[-1]


@dataclass
class DirectionConfig:
    """Refusal-direction extraction, following Arditi et al. (2024)."""

    n_train_harmful: int = 128
    n_train_harmless: int = 128
    n_val_harmful: int = 32
    n_val_harmless: int = 32
    #: Post-instruction token positions to scan, counted from the end of the prompt.
    positions: tuple[int, ...] = (-1, -2, -3, -4, -5)
    #: Only consider layers below this fraction of depth (Arditi's l < 0.8L).
    max_layer_frac: float = 0.8
    #: Candidate directions must not perturb harmless behaviour more than this.
    max_kl: float = 0.1
    n_kl_prompts: int = 32
    val_max_new_tokens: int = 32
    #: Score every ``layer_stride``-th layer. The full scan is ~180 cells on a
    #: 36-layer model and each cell costs two generation passes; a stride of 2
    #: halves that at negligible cost, since neighbouring layers give nearly
    #: identical difference-in-means directions.
    layer_stride: int = 2
    #: Strength of the added direction in the induce test, as a multiple of the
    #: mean residual-stream norm at the source layer.
    induce_alpha: float = 1.0
    #: If no candidate passes every filter, fall back to the best KL-passing
    #: candidate by bypass score and record that the induce filter was waived.
    #: Always reported; never silent.
    allow_relaxed_selection: bool = True
    #: Override held-out selection with a specific (layer, position) candidate,
    #: e.g. (18, -3) for the confirmation run with the highest-bypass candidate
    #: that the induce filter rejected. The candidate is still scored and the
    #: override is stamped into the saved JSON as unvalidated; never silent.
    force_candidate: tuple[int, int] | None = None


@dataclass
class InjectionConfig:
    """Concept-injection protocol (section 3.3)."""

    #: Injection strength as a multiple of the mean residual-stream norm at the
    #: injection layer -- this normalisation is stated explicitly in the paper.
    #:
    #: Under *this* normalisation alpha=1 injects a vector as large as the whole
    #: residual stream, so the usable range is well below 1 and the coherence
    #: cliff is sharp. Measured on Qwen3-0.6B at layer 22: alpha=0.25 keeps 83%
    #: of reports parseable, alpha=0.5 keeps 67%, and alpha>=1.0 keeps none --
    #: at which point detection is not merely low, it is undefined.
    #:
    #: This default is only a starting point: the cliff moves with model and
    #: layer, so ``pipeline.stage_calibrate`` measures it per run and overwrites
    #: this value. Do not raise it without re-running that stage.
    alpha: float = 0.25
    alpha_pilot: tuple[float, ...] = (0.25, 0.5)
    #: Layer sweep expressed as fractions of depth; fixed after the pilot.
    layer_fracs: tuple[float, ...] = (0.5, 0.7, 0.85)
    layer: int | None = None  # resolved absolute layer once the pilot fixes it
    n_concepts: int = 60
    n_trials: int = 20
    #: Forced-choice candidate-set size for condition C4.
    k_choices: int = 10
    #: Concept vectors are read at the final token of each elicitation prompt.
    read_token: int = -1


@dataclass
class EvalConfig:
    """Safety and capability evaluation sizes (families B and C)."""

    n_jailbreak: int = 100
    safety_max_new_tokens: int = 128
    n_mmlu: int = 500
    n_truthfulqa: int = 400
    n_gsm8k: int = 100
    gsm8k_max_new_tokens: int = 256
    n_ce_loss: int = 200
    run_gsm8k: bool = False


@dataclass
class RunConfig:
    """Top-level bundle written next to every results file for provenance."""

    model: ModelConfig = field(default_factory=ModelConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    injection: InjectionConfig = field(default_factory=InjectionConfig)
    evals: EvalConfig = field(default_factory=EvalConfig)
    lambdas: tuple[float, ...] = LAMBDA_GRID
    seed: int = 0
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)

    @property
    def run_dir(self) -> Path:
        return self.results_dir / self.model.short_name

    def artifact(self, *parts: str) -> Path:
        path = self.run_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def to_dict(self) -> dict:
        d = asdict(self)
        d["results_dir"] = str(self.results_dir)
        return d

    def save(self, path: Path | None = None) -> Path:
        path = path or self.artifact("run_config.json")
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        raw = json.loads(Path(path).read_text())
        direction_raw = {**raw["direction"], "positions": tuple(raw["direction"]["positions"])}
        if direction_raw.get("force_candidate") is not None:
            direction_raw["force_candidate"] = tuple(direction_raw["force_candidate"])
        return cls(
            model=ModelConfig(**raw["model"]),
            direction=DirectionConfig(**direction_raw),
            injection=InjectionConfig(
                **{
                    **raw["injection"],
                    "alpha_pilot": tuple(raw["injection"]["alpha_pilot"]),
                    "layer_fracs": tuple(raw["injection"]["layer_fracs"]),
                }
            ),
            evals=EvalConfig(**raw["evals"]),
            lambdas=tuple(raw["lambdas"]),
            seed=raw["seed"],
            results_dir=Path(raw["results_dir"]),
        )


#: A tiny, fully-cached configuration used to exercise the whole pipeline in
#: minutes. Results from it are not scientifically meaningful.
def smoke_config() -> RunConfig:
    cfg = RunConfig()
    cfg.model = ModelConfig(name="Qwen/Qwen3-0.6B", batch_size=8, max_new_tokens=24, max_context=2048)
    cfg.direction = DirectionConfig(
        n_train_harmful=16,
        n_train_harmless=16,
        n_val_harmful=8,
        n_val_harmless=8,
        positions=(-1, -2),
        n_kl_prompts=8,
        val_max_new_tokens=16,
        layer_stride=6,
    )
    cfg.injection = InjectionConfig(n_concepts=6, n_trials=3, layer_fracs=(0.5, 0.7))
    cfg.evals = EvalConfig(n_jailbreak=8, n_mmlu=20, n_truthfulqa=20, n_gsm8k=5, n_ce_loss=16, safety_max_new_tokens=48)
    cfg.lambdas = (0.0, 0.5, 1.0)
    cfg.results_dir = RESULTS_DIR / "smoke"
    return cfg
