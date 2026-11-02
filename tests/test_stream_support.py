"""Shared helpers for the P2b streaming tests: a tiny random Beam and a stub weight source serving its checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from transformers import BeamConfig
from transformers.models.beam.modeling_beam import BeamForCausalLM

from ouroboros.harness.shards import Episode
from ouroboros.harness.stream import FP32_KEYS

WINDOW = 16


def tiny_config(**overrides) -> BeamConfig:
    """7 layers mixing dense and sparse MLPs, sliding RoPE and full NoPE attention.

    Routing keeps Beam's shape: top-8 experts within the best 4 of 8 groups, here of 32 experts.
    """
    kw = dict(vocab_size=512, hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=7,
              num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=32, num_experts_per_tok=8,
              n_group=8, topk_group=4, sliding_window=WINDOW, max_position_embeddings=4096,
              layer_types=["sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
                           "full_attention", "sliding_attention", "full_attention"],
              mlp_layer_types=["dense", "dense", "sparse", "sparse", "sparse", "sparse", "sparse"])
    kw.update(overrides)
    return BeamConfig(**kw)


def tiny_model(config: BeamConfig, seed: int = 0) -> BeamForCausalLM:
    """FP32 reference with spread-out router scores, a nonzero correction bias and non-trivial norms."""
    torch.manual_seed(seed)
    model = BeamForCausalLM._from_config(config, attn_implementation="sdpa", experts_implementation="eager").eval()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            elif name.endswith("gate_bias"):
                p.copy_(0.5 * torch.randn(p.shape, generator=g))
            elif name.endswith("mlp.gate.weight"):
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
        for name, b in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.copy_(0.05 * torch.randn(b.shape, generator=g))
    return model


def bf16_model(model: BeamForCausalLM) -> BeamForCausalLM:
    """The same weights in BF16 with the router kept in FP32, as transformers' from_pretrained loads Beam."""
    out = BeamForCausalLM._from_config(model.config, dtype=torch.bfloat16, attn_implementation="sdpa",
                                       experts_implementation="eager").eval()
    for name, t in [*out.named_parameters(), *out.named_buffers()]:
        if any(key in name for key in FP32_KEYS):
            t.data = t.data.float()
    out.load_state_dict(model.state_dict())
    return out


class StubSource:
    """A WeightSource over the checkpoint save_pretrained writes: per-expert names, router in FP32."""

    def __init__(self, model: BeamForCausalLM, ckpt_dir: Path):
        model.save_pretrained(ckpt_dir)
        self.tensors = load_file(str(Path(ckpt_dir) / "model.safetensors"))
        self.loads: list[int] = []

    @staticmethod
    def _cast(name: str, t: torch.Tensor, device, dtype) -> torch.Tensor:
        keep = name.endswith(("mlp.gate.weight", "mlp.gate.e_score_correction_bias"))
        return t.to(device=device, dtype=torch.float32 if keep else dtype, copy=True)

    def load_layer(self, i: int, device, dtype) -> dict[str, torch.Tensor]:
        self.loads.append(i)
        p = f"model.layers.{i}."
        return {k[len(p):]: self._cast(k, v, device, dtype) for k, v in self.tensors.items() if k.startswith(p)}

    def load_globals(self, device, dtype) -> dict[str, torch.Tensor]:
        return {k.removeprefix("model."): v.to(device=device, dtype=dtype, copy=True)
                for k, v in self.tensors.items() if not k.startswith("model.layers.")}


@torch.no_grad()
def reference(model: BeamForCausalLM, ids: np.ndarray) -> dict:
    """One episode through BeamForCausalLM: each layer's output, each router's (logits, ids), and the logits."""
    layers, routers, hooks = {}, {}, []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(lambda m, a, o, i=i: layers.__setitem__(i, o[0].float())))
        if hasattr(layer.mlp, "gate"):
            hooks.append(layer.mlp.gate.register_forward_hook(
                lambda m, a, o, i=i: routers.__setitem__(i, (o[0].float(), o[2]))))
    try:
        logits = model(torch.from_numpy(ids).long()[None]).logits[0].float()
    finally:
        for h in hooks:
            h.remove()
    return {"layers": layers, "routers": routers, "logits": logits}


def make_episodes(vocab: int = 512, seed: int = 0) -> list[Episode]:
    """Three episodes of different lengths, two of them longer than the sliding window, with anchors."""
    rng = np.random.default_rng(seed)

    def ep(key, n, **anchors):
        gen = np.zeros(n, bool)
        gen[n // 3:] = True
        return Episode(key, rng.integers(0, vocab, n), gen, **anchors)

    return [ep("a", 53, f=10, window_end=30, d=40), ep("b", 9, s=6), ep("c", 120, f=70, window_end=95, s=110)]


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    """Tiny models gain nothing from threads, and threads stall badly when other jobs share the CPU."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    config = tiny_config()
    model = tiny_model(config)
    return config, model, StubSource(model, tmp_path_factory.mktemp("ckpt"))
