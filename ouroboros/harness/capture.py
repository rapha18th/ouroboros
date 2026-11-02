"""Per-shard capture, as the pre-registration fixes it, laid out for about 12.5M tokens in all.

Every token and MoE layer: the selected expert ids (uint8) and their sigmoid scores (fp16), layer-major
[L, T, k], about 1.2 KB per token. Window rows (token ranges per episode): all expert scores (fp16) [L, W, E]
and the hidden states output by the tap layers [taps, W, H]. Target tokens: next-token NLL, top-k log-probs
and the log-probs of candidate ids, from lm_head at those positions only. Every token's id (int32) and gen_mask
(bool) ride along, so a window can be narrowed to the tokens Beam wrote.

A shard is written into "<dir>.partial" as .npy memmaps and renamed to "<dir>" when complete, so a finished
directory (holding index.json) is always whole. BF16 arrays are stored as their raw uint16 bits.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ouroboros.harness.weights import host_buffer

INDEX = "index.json"
_STORAGE = {torch.bfloat16: np.uint16, torch.float16: np.float16, torch.float32: np.float32}


def host_tensor(shape: tuple[int, ...], dtype: torch.dtype, pin: bool = False) -> torch.Tensor:
    """An uninitialised host tensor, page-locked at its exact size when `pin` is set and CUDA is available.

    torch's own pin_memory rounds every block up to a power of two and caches it until exit, so a 39.9 GB
    shard of hidden states would lock 64 GiB.
    """
    itemsize = torch.empty((), dtype=dtype).element_size()
    return host_buffer(int(np.prod(shape)) * itemsize, pin).view(dtype).view(shape)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """Host array; a CPU tensor shares its memory. BF16 becomes its raw uint16 bits since numpy has no bfloat16."""
    t = t.detach().cpu()
    return t.view(torch.int16).numpy().view(np.uint16) if t.dtype == torch.bfloat16 else t.numpy()


def bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def topk_set_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of token-layers whose selected expert sets match, for id arrays [..., k]."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    same = (np.sort(a, axis=-1) == np.sort(b, axis=-1)).all(axis=-1)
    return float(same.mean()) if same.size else 1.0


@dataclass(frozen=True)
class CapturePlan:
    """What to keep beyond per-token routing, in shard token coordinates."""

    windows: np.ndarray  # [W] int64 token index of each window row, strictly increasing
    targets: np.ndarray  # [N] int64 token t scored from the hidden state at t - 1, strictly increasing
    candidates: np.ndarray  # [C] int64 token ids scored at every target
    tap_layers: tuple[int, ...] = (4, 16, 28, 40, 46)
    top_k: int = 16

    def __post_init__(self):
        for name in ("windows", "targets"):
            v = getattr(self, name)
            if len(v) > 1 and not (np.diff(v) > 0).all():
                raise ValueError(f"{name} must be strictly increasing")


class Capture:
    """Receives router outputs and layer outputs from the streamer and writes them to disk."""

    def __init__(self, out_dir: str | Path, n_tokens: int, plan: CapturePlan, *, n_moe: int, n_experts: int,
                 experts_per_token: int, hidden_size: int, hidden_dtype: torch.dtype):
        if n_experts > 256:
            raise ValueError("expert ids are stored as uint8")
        self.final = Path(out_dir)
        self.work = self.final.with_name(self.final.name + ".partial")
        shutil.rmtree(self.work, ignore_errors=True)
        self.work.mkdir(parents=True)
        self.plan = plan
        self.hidden_dtype = hidden_dtype
        self.taps = {layer: t for t, layer in enumerate(plan.tap_layers)}
        w = len(plan.windows)
        self.arrays: dict[str, np.ndarray] = {}
        self._array("route_ids", (n_moe, n_tokens, experts_per_token), np.uint8)
        self._array("route_scores", (n_moe, n_tokens, experts_per_token), np.float16)
        self._array("window_scores", (n_moe, w, n_experts), np.float16)
        self._array("tap_hidden", (len(plan.tap_layers), w, hidden_size), _STORAGE[hidden_dtype])
        self.layers_run: set[int] = set()

    def _array(self, name: str, shape: tuple[int, ...], dtype) -> None:
        self.arrays[name] = np.lib.format.open_memmap(self.work / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)

    def _rows(self, lo: int, hi: int) -> tuple[int, int, np.ndarray]:
        """Window rows inside shard tokens [lo, hi), and their offsets from lo."""
        r0, r1 = np.searchsorted(self.plan.windows, [lo, hi])
        return int(r0), int(r1), self.plan.windows[r0:r1] - lo

    def router(self, j: int, lo: int, hi: int, logits: torch.Tensor, ids: torch.Tensor) -> None:
        """MoE layer j on shard tokens [lo, hi): router logits [n, E] (FP32) and selected ids [n, k]."""
        scores = torch.sigmoid(logits)  # the router's own selection scores, before the correction bias
        self.arrays["route_ids"][j, lo:hi] = ids.to(torch.uint8).cpu().numpy()
        self.arrays["route_scores"][j, lo:hi] = scores.gather(1, ids).half().cpu().numpy()
        r0, r1, local = self._rows(lo, hi)
        if r1 > r0:
            index = torch.from_numpy(local).to(scores.device)
            self.arrays["window_scores"][j, r0:r1] = scores[index].half().cpu().numpy()

    def layer_output(self, i: int, lo: int, hi: int, out: torch.Tensor) -> None:
        """Decoder layer i output [n, H] for shard tokens [lo, hi)."""
        self.layers_run.add(i)
        t = self.taps.get(i)
        if t is None:
            return
        r0, r1, local = self._rows(lo, hi)
        if r1 > r0:
            self.arrays["tap_hidden"][t, r0:r1] = to_numpy(out[torch.from_numpy(local).to(out.device)])

    def finalize(self, arrays: dict[str, np.ndarray], index: dict) -> Path:
        """Write per-target arrays and index.json, then move the shard into place."""
        arrays = {**arrays, "windows": self.plan.windows, "targets": self.plan.targets,
                  "candidates": self.plan.candidates}
        for name, a in arrays.items():
            np.save(self.work / f"{name}.npy", a)
        shapes = {name: [list(a.shape), str(a.dtype)] for name, a in {**self.arrays, **arrays}.items()}
        for name in list(self.arrays):
            self.arrays.pop(name).flush()
        gc.collect()  # every memmap must close before the directory can move on Windows
        index = {
            **index,
            "arrays": shapes,
            "hidden_dtype": str(self.hidden_dtype).removeprefix("torch."),
            "tap_layers": list(self.plan.tap_layers),
            "top_k_logprobs": self.plan.top_k,
            "layers_run": sorted(self.layers_run),
        }
        (self.work / INDEX).write_text(json.dumps(index, indent=1), encoding="utf-8")
        if self.final.exists():
            shutil.rmtree(self.final)
        os.replace(self.work, self.final)
        return self.final


def is_complete(path: str | Path) -> bool:
    return (Path(path) / INDEX).exists()


@dataclass(frozen=True)
class ShardCapture:
    """A finished shard, read lazily through memmaps."""

    path: Path
    index: dict

    def array(self, name: str) -> np.ndarray:
        return np.load(self.path / f"{name}.npy", mmap_mode="r")

    def tap_hidden(self) -> np.ndarray:
        """Tap-layer hidden states [taps, W, H] as float32, read fully into memory."""
        raw = self.array("tap_hidden")
        return bf16_bits_to_float32(raw) if self.index["hidden_dtype"] == "bfloat16" else np.asarray(raw, np.float32)


def load(path: str | Path) -> ShardCapture:
    path = Path(path)
    return ShardCapture(path, json.loads((path / INDEX).read_text(encoding="utf-8")))
