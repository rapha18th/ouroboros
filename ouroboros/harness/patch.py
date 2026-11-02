"""Router-bias patches and layer-prefix reuse for Stage B.

A patch is a [moe_layers, experts] FP32 delta added to each MoE layer's e_score_correction_bias before
selection. The bias only picks experts; mixing weights still come from the unbiased sigmoid scores.

Layers below the first patched layer run stock, so their output is computed once per shard, saved at the
input of that layer, and every later patch restarts from there. A saved prefix is one shard of hidden states
(up to 40 GB), so the cache keeps a single file per shard and skips layers too shallow to repay the disk.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ouroboros.harness.capture import Capture, host_tensor, to_numpy

if TYPE_CHECKING:
    from ouroboros.harness.stream import Layout, Streamer

PATCH_KEY = "delta"
# A 3.9M-token shard costs about 20 s of A100 compute per layer, about 57 s to read its 40 GB prefix back from
# local NVMe and about 105 s to write it once. From layer 8 on, a prefix repays its write within two patches.
PREFIX_MIN_LAYER = 8


def save_patch(path: str | Path, delta: torch.Tensor) -> None:
    save_file({PATCH_KEY: delta.detach().to(torch.float32).contiguous().cpu()}, str(path))


def load_patch(path: str | Path, n_moe: int, n_experts: int) -> torch.Tensor:
    delta = load_file(str(path))[PATCH_KEY]
    if delta.shape != (n_moe, n_experts) or delta.dtype != torch.float32:
        raise ValueError(f"{path}: expected FP32 [{n_moe}, {n_experts}], got {delta.dtype} {list(delta.shape)}")
    if not torch.isfinite(delta).all():
        raise ValueError(f"{path}: patch has non-finite values")
    return delta


def first_patched_layer(delta: torch.Tensor | None, moe_layers: list[int]) -> int | None:
    """Decoder index of the first MoE layer with a nonzero delta row, or None for a stock run."""
    if delta is None:
        return None
    rows = torch.nonzero(delta.abs().sum(dim=1)).flatten().tolist()
    return moe_layers[rows[0]] if rows else None


@contextmanager
def patched(layer: torch.nn.Module, delta_row: torch.Tensor | None):
    """Add delta_row to the layer's correction bias in FP32; restore the stock bias bit for bit on exit."""
    if delta_row is None or not delta_row.any():
        yield
        return
    bias = layer.mlp.gate.e_score_correction_bias
    stock = bias.clone()
    bias.add_(delta_row.to(device=bias.device, dtype=torch.float32))
    try:
        yield
    finally:
        bias.copy_(stock)


class PrefixCache:
    """Stock hidden states at the input of one decoder layer, tied to one shard's tokens, weights and kernels.

    `identity` names everything that changes the stock hidden states (layout, weights, attention backend).
    Only layers at or above `min_layer` are saved, and saving a layer removes any other one.
    """

    def __init__(self, root: str | Path, identity: dict, dtype: torch.dtype, min_layer: int = PREFIX_MIN_LAYER):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype
        self.min_layer = min_layer
        meta = {**identity, "dtype": str(dtype)}
        stamp = self.root / "prefix.json"
        if stamp.exists():
            if json.loads(stamp.read_text(encoding="utf-8")) != meta:
                raise ValueError(f"{self.root} holds prefixes for other tokens, weights, kernels or dtype")
        else:
            stamp.write_text(json.dumps(meta), encoding="utf-8")

    def _path(self, layer: int) -> Path:
        return self.root / f"input_{layer:02d}.npy"

    def layers(self) -> list[int]:
        return sorted(int(p.stem.split("_")[1]) for p in self.root.glob("input_*.npy"))

    def nearest(self, layer: int) -> int | None:
        """The deepest saved layer input at or below `layer`."""
        below = [k for k in self.layers() if k <= layer]
        return below[-1] if below else None

    def save(self, layer: int, hidden: torch.Tensor) -> None:
        tmp = self.root / f"partial_{layer:02d}.npy"
        np.save(tmp, to_numpy(hidden))
        os.replace(tmp, self._path(layer))
        for other in self.layers():
            if other != layer:
                self._path(other).unlink()

    def load(self, layer: int, pin: bool = False) -> torch.Tensor:
        saved = np.load(self._path(layer), mmap_mode="r")
        out = host_tensor(saved.shape, self.dtype, pin)
        to_numpy(out)[...] = saved  # writes through the shared host buffer
        return out


def run_patched(streamer: Streamer, layout: Layout, delta: torch.Tensor | None, cache: PrefixCache,
                capture: Capture | None = None) -> tuple[torch.Tensor, int]:
    """Run a forward that starts from the deepest saved stock prefix at or below the first patched layer.

    Without a usable prefix it runs from the embeddings and saves the input of the first patched layer, when
    that layer is at least cache.min_layer. A stock run (no nonzero delta) reuses a prefix but never saves one.
    Returns the last-layer hidden states and the first layer this call ran (its capture starts there).
    """
    n = streamer.config.num_hidden_layers
    first = first_patched_layer(delta, streamer.moe_layers)
    start = cache.nearest(n if first is None else first)
    if start is not None:
        hidden = cache.load(start, pin=streamer.device.type == "cuda")
        streamer.run_layers(hidden, layout, start, n, delta=delta, capture=capture)
        return hidden, start
    hidden = streamer.embed(layout)
    split = 0
    if first is not None and first >= cache.min_layer:
        streamer.run_layers(hidden, layout, 0, first, capture=capture)
        cache.save(first, hidden)
        split = first
    streamer.run_layers(hidden, layout, split, n, delta=delta, capture=capture)
    return hidden, 0
