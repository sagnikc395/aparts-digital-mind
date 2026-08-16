"""The alignment tax of introspection.

Partial refusal-direction ablation as a continuous dial, with introspection
gain, safety loss and capability loss measured on one dose-response axis.

Typical use from a notebook::

    from alignment_tax.config import RunConfig
    from alignment_tax import pipeline

    cfg = RunConfig()
    hm = pipeline.load_model(cfg)
    rd = pipeline.stage_direction(hm, cfg)
    bank = pipeline.stage_concepts(hm, cfg)
    pipeline.stage_pilot(hm, cfg, bank, rd.vector)
"""

from .config import LAMBDA_GRID, RunConfig, smoke_config

__all__ = ["LAMBDA_GRID", "RunConfig", "smoke_config", "__version__"]
__version__ = "0.1.0"
