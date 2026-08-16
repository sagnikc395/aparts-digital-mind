"""Model wrapper: residual-stream read/write hooks, batched generation, scoring.

We deliberately use raw HuggingFace forward hooks rather than TransformerLens
(section 4.3): we need exactly one capability -- read and write access to the
residual stream at a given layer -- and hooks give it without a second weight
conversion or the doubled host RAM at load time.

Two interventions compose here:

* **Partial directional ablation** (the independent variable)

      x <- x - lambda * r_hat (r_hat^T x)

  applied to the residual stream at *every* layer and *every* position.
  ``lambda = 0`` is the untouched instruct model; ``lambda = 1`` reproduces the
  standard binary ablation of Arditi et al. (2024).

* **Concept injection** (the probe)

      x <- x + alpha * ||x||_mean * c_hat

  applied at one layer, where ``||x||_mean`` is the mean residual-stream norm at
  that layer measured on a baseline corpus, so ``alpha`` is comparable across
  layers and model families.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig


def resolve_device(spec: str = "auto") -> str:
    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if device == "cpu":
        # bf16 matmuls on CPU are slow and sometimes unimplemented.
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


@dataclass
class Injection:
    """A concept vector to add to the residual stream at one layer."""

    vector: torch.Tensor  # unit norm, shape [d_model]
    layer: int
    alpha: float
    norm_scale: float = 1.0  # mean residual-stream norm at ``layer``
    positions: str = "all"  # "all" or "prompt"

    @property
    def delta(self) -> torch.Tensor:
        return self.vector * (self.alpha * self.norm_scale)


class HookedModel:
    """A causal LM with residual-stream hooks for ablation and injection."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.dtype = resolve_dtype(cfg.dtype, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.name, trust_remote_code=cfg.trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding keeps the post-instruction token at a fixed offset from
        # the end of every sequence in a batch, which matters for both direction
        # extraction and generation.
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.name, dtype=self.dtype, trust_remote_code=cfg.trust_remote_code
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        self.layers = self._decoder_layers()
        self.n_layers = len(self.layers)
        self.d_model = int(self.model.config.hidden_size)

        self._ablation: tuple[torch.Tensor, float] | None = None
        self._injection: Injection | None = None
        self._prompt_len: int | None = None
        self._handles: list = []
        self._install_hooks()

    # ------------------------------------------------------------------ setup

    def _decoder_layers(self):
        for path in ("model.layers", "model.model.layers", "transformer.h", "model.decoder.layers"):
            obj = self.model
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        raise RuntimeError(f"Could not locate decoder layers on {self.cfg.name}")

    def _install_hooks(self) -> None:
        """One pre-hook per layer (resid_pre) plus one post-hook on the last
        layer (final resid_post), so the ablation covers the whole stream."""
        for idx, layer in enumerate(self.layers):
            self._handles.append(
                layer.register_forward_pre_hook(self._make_pre_hook(idx), with_kwargs=True)
            )
        self._handles.append(
            self.layers[-1].register_forward_hook(self._final_hook, with_kwargs=True)
        )

    def _make_pre_hook(self, idx: int):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None:
                return None
            new = self._apply(hidden, layer_idx=idx)
            if new is hidden:
                return None
            if "hidden_states" in kwargs:
                kwargs["hidden_states"] = new
                return args, kwargs
            return (new, *args[1:]), kwargs

        return hook

    def _final_hook(self, module, args, kwargs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        new = self._apply(hidden, layer_idx=None)  # ablation only
        if new is hidden:
            return None
        return (new, *output[1:]) if isinstance(output, tuple) else new

    def _apply(self, hidden: torch.Tensor, layer_idx: int | None) -> torch.Tensor:
        out = hidden
        if self._injection is not None and layer_idx == self._injection.layer:
            delta = self._injection.delta.to(out.device, out.dtype)
            if self._injection.positions == "prompt" and self._prompt_len is not None:
                if out.shape[1] >= self._prompt_len:  # skip cached decode steps
                    out = out.clone()
                    out[:, : self._prompt_len, :] += delta
            else:
                out = out + delta
        if self._ablation is not None:
            r, lam = self._ablation
            if lam != 0.0:
                r = r.to(out.device, out.dtype)
                proj = torch.einsum("bsd,d->bs", out, r).unsqueeze(-1) * r
                out = out - lam * proj
        return out

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ------------------------------------------------------------- intervention

    @contextlib.contextmanager
    def ablated(self, direction: torch.Tensor | None, lam: float) -> Iterator[None]:
        """Scale-`lam` directional ablation, active for the duration of the block."""
        prev = self._ablation
        if direction is None or lam == 0.0:
            self._ablation = None
        else:
            r = direction.to(torch.float32)
            self._ablation = (r / r.norm(), float(lam))
        try:
            yield
        finally:
            self._ablation = prev

    @contextlib.contextmanager
    def injected(self, injection: Injection | None) -> Iterator[None]:
        prev = self._injection
        self._injection = injection
        try:
            yield
        finally:
            self._injection = prev

    # -------------------------------------------------------------- tokenising

    def chat_prompt(self, user: str, prefill: str | None = None, system: str | None = None) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:  # tokenizers without the Qwen3 thinking switch
            text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if prefill:
            text = text + prefill
        return text

    def encode(self, prompts: Sequence[str]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_context,
            add_special_tokens=False,
        )
        return {k: v.to(self.device) for k, v in batch.items()}

    # ------------------------------------------------------------------ compute

    @torch.no_grad()
    def hidden_states(self, prompts: Sequence[str], positions: Sequence[int] = (-1,)) -> torch.Tensor:
        """Residual stream at selected positions.

        Returns ``[n_prompts, n_layers + 1, n_positions, d_model]`` in float32.
        Index ``l`` of the layer axis is the input to decoder layer ``l``
        (``resid_pre``); the last index is the final residual stream.
        Left padding means negative positions are the true final tokens.
        """
        out = []
        for start in range(0, len(prompts), self.cfg.batch_size):
            chunk = list(prompts[start : start + self.cfg.batch_size])
            batch = self.encode(chunk)
            self._prompt_len = batch["input_ids"].shape[1]
            res = self.model(**batch, output_hidden_states=True, use_cache=False)
            hs = torch.stack(res.hidden_states, dim=1)  # [b, L+1, s, d]
            idx = torch.tensor(positions, device=hs.device)
            out.append(hs.index_select(2, idx % hs.shape[2]).float().cpu())
            self._prompt_len = None
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def mean_resid_norm(self, prompts: Sequence[str], layer: int) -> float:
        """Mean per-token L2 norm of the residual stream at ``layer``.

        This is the scale that makes injection strength ``alpha`` comparable
        across layers and models.
        """
        totals, count = 0.0, 0
        for start in range(0, len(prompts), self.cfg.batch_size):
            chunk = list(prompts[start : start + self.cfg.batch_size])
            batch = self.encode(chunk)
            res = self.model(**batch, output_hidden_states=True, use_cache=False)
            hs = res.hidden_states[layer].float()
            mask = batch["attention_mask"].bool()
            norms = hs.norm(dim=-1)[mask]
            totals += norms.sum().item()
            count += norms.numel()
        return totals / max(count, 1)

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], max_new_tokens: int | None = None, do_sample: bool = False,
                 temperature: float = 1.0) -> list[str]:
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens
        outputs: list[str] = []
        for start in range(0, len(prompts), self.cfg.batch_size):
            chunk = list(prompts[start : start + self.cfg.batch_size])
            batch = self.encode(chunk)
            self._prompt_len = batch["input_ids"].shape[1]
            gen = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=0.95 if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new = gen[:, batch["input_ids"].shape[1] :]
            outputs.extend(self.tokenizer.batch_decode(new, skip_special_tokens=True))
            self._prompt_len = None
        return outputs

    @torch.no_grad()
    def sequence_logprob(self, prompts: Sequence[str], completions: Sequence[str],
                         length_normalise: bool = True) -> list[float]:
        """Log P(completion | prompt), used for all multiple-choice scoring so
        that MMLU / TruthfulQA / forced choice need no generation at all."""
        scores: list[float] = []
        for start in range(0, len(prompts), self.cfg.batch_size):
            p_chunk = list(prompts[start : start + self.cfg.batch_size])
            c_chunk = list(completions[start : start + self.cfg.batch_size])
            full = [p + c for p, c in zip(p_chunk, c_chunk)]
            batch = self.encode(full)
            ids, mask = batch["input_ids"], batch["attention_mask"]
            logits = self.model(**batch, use_cache=False).logits.float()
            logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
            target = ids[:, 1:]
            token_lp = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            for i, (p, c) in enumerate(zip(p_chunk, c_chunk)):
                n_comp = len(self.tokenizer(c, add_special_tokens=False)["input_ids"])
                n_comp = max(n_comp, 1)
                valid = mask[i, 1:].bool()
                lp = token_lp[i][valid][-n_comp:]
                scores.append((lp.mean() if length_normalise else lp.sum()).item())
        return scores

    @torch.no_grad()
    def cross_entropy(self, texts: Sequence[str]) -> float:
        """Mean next-token CE loss -- the cheap fluency/coherence proxy logged at
        every lambda (risk table, section 7)."""
        total, ntok = 0.0, 0
        for start in range(0, len(texts), self.cfg.batch_size):
            chunk = list(texts[start : start + self.cfg.batch_size])
            batch = self.encode(chunk)
            ids, mask = batch["input_ids"], batch["attention_mask"]
            logits = self.model(**batch, use_cache=False).logits.float()
            lp = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = ids[:, 1:]
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            valid = mask[:, 1:].bool()
            total += nll[valid].sum().item()
            ntok += int(valid.sum().item())
        return total / max(ntok, 1)

    @torch.no_grad()
    def next_token_logprobs(self, prompts: Sequence[str]) -> torch.Tensor:
        """Full next-token log-prob distribution, for the KL selection criterion."""
        out = []
        for start in range(0, len(prompts), self.cfg.batch_size):
            batch = self.encode(list(prompts[start : start + self.cfg.batch_size]))
            logits = self.model(**batch, use_cache=False).logits[:, -1].float()
            out.append(torch.log_softmax(logits, dim=-1).cpu())
        return torch.cat(out, dim=0)
