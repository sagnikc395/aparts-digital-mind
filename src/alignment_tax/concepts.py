"""Concept vectors and the injection-strength normalisation.

Concept vector for concept c: difference of means between residual-stream
activations on prompts instantiating c and on a generic baseline corpus, read at
the final token (Lindsey, 2026). The vector is normalised to unit norm, and
injection strength alpha is defined as a multiple of the *mean residual-stream
norm at the injection layer*:

    x <- x + alpha * ||x||_mean(layer) * c_hat

Stating this normalisation explicitly is a small methods contribution: prior
work leaves the scale implicit, which makes alpha incomparable across layers and
across model families.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .model import HookedModel, Injection


@dataclass
class ConceptBank:
    """Unit-norm concept vectors at every layer, plus per-layer norm scales."""

    names: list[str]
    aliases: dict[str, list[str]]
    vectors: torch.Tensor  # [n_concepts, n_layers + 1, d_model], unit norm
    norm_scale: dict[int, float]  # layer -> mean residual-stream norm

    def index(self, name: str) -> int:
        return self.names.index(name)

    def vector(self, name: str, layer: int) -> torch.Tensor:
        return self.vectors[self.index(name), layer]

    def injection(self, name: str, layer: int, alpha: float) -> Injection:
        return Injection(
            vector=self.vector(name, layer),
            layer=layer,
            alpha=alpha,
            norm_scale=self.norm_scale[layer],
        )

    def random_injection(self, layer: int, alpha: float, seed: int) -> Injection:
        """Norm-matched random direction: condition C3.

        Same layer, same alpha, same scale -- only the direction is meaningless.
        This is the control that separates "detects *this concept*" from
        "detects *that something happened*".
        """
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(self.vectors.shape[-1], generator=g)
        return Injection(vector=v / v.norm(), layer=layer, alpha=alpha, norm_scale=self.norm_scale[layer])

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"names": self.names, "aliases": self.aliases, "vectors": self.vectors,
             "norm_scale": self.norm_scale},
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "ConceptBank":
        b = torch.load(path, map_location="cpu", weights_only=False)
        return cls(b["names"], b["aliases"], b["vectors"], {int(k): v for k, v in b["norm_scale"].items()})


def build_concept_bank(
    hm: HookedModel,
    concepts: list[dict],
    baseline_corpus: list[str],
    layers: list[int] | None = None,
    verbose: bool = True,
) -> ConceptBank:
    """Compute concept vectors at every layer and the per-layer norm scale.

    Concept prompts and baseline prompts are both run raw (no chat template): we
    want the representation of the *content*, not of an instruction to discuss it.
    """
    baseline = hm.hidden_states(baseline_corpus, positions=(-1,)).mean(dim=0)[:, 0, :]  # [L+1, d]

    vectors = []
    for i, c in enumerate(concepts):
        acts = hm.hidden_states(c["prompts"], positions=(-1,)).mean(dim=0)[:, 0, :]
        v = acts - baseline
        vectors.append(v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        if verbose and (i + 1) % 10 == 0:
            print(f"[concepts] {i + 1}/{len(concepts)}")
    V = torch.stack(vectors)  # [n, L+1, d]

    layers = layers if layers is not None else list(range(V.shape[1]))
    norm_scale = {int(l): hm.mean_resid_norm(baseline_corpus, l) for l in layers}
    if verbose:
        print(f"[concepts] built {V.shape[0]} vectors; norm scale at layers {sorted(norm_scale)[:5]}...")

    return ConceptBank(
        names=[c["name"] for c in concepts],
        aliases={c["name"]: c.get("aliases", []) for c in concepts},
        vectors=V,
        norm_scale=norm_scale,
    )


def ensure_scale(bank: ConceptBank, hm: HookedModel, layer: int,
                 corpus: list[str] | None = None) -> float:
    """Compute and cache the residual-norm scale for a layer the bank was not
    built with -- e.g. when the pilot fixes an injection layer outside the
    original sweep."""
    if layer not in bank.norm_scale:
        from .data import load_baseline_corpus

        bank.norm_scale[layer] = hm.mean_resid_norm(corpus or load_baseline_corpus(), layer)
    return bank.norm_scale[layer]


def cosine_confusability(bank: ConceptBank, layer: int) -> torch.Tensor:
    """Pairwise cosine similarity of the concept vectors at ``layer``.

    Reported as a sanity check: if the bank is highly collinear, low
    identification accuracy is a property of the stimuli, not of the model.
    """
    V = bank.vectors[:, layer, :]
    V = V / V.norm(dim=-1, keepdim=True)
    return V @ V.T


def resolve_layers(n_layers: int, fracs: tuple[float, ...]) -> list[int]:
    """Fractions of depth -> absolute layer indices.

    For a 36-layer Qwen3 this puts the sweep at roughly layers 18, 25 and 31,
    centred on the 0.8-depth optimum reported for Qwen3-235B.
    """
    return sorted({max(0, min(n_layers - 1, int(round(f * n_layers)))) for f in fracs})


def dump_bank_summary(bank: ConceptBank, layer: int, path: Path) -> Path:
    sim = cosine_confusability(bank, layer)
    off = sim - torch.eye(sim.shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "n_concepts": len(bank.names),
                "layer": layer,
                "mean_abs_cosine": float(off.abs().mean()),
                "max_abs_cosine": float(off.abs().max()),
                "norm_scale": bank.norm_scale.get(layer),
                "concepts": bank.names,
            },
            indent=2,
        )
    )
    return path
