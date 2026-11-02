"""Layer-major streaming forward for Beam over packed episodes.

Every decoder layer is a transformers BeamDecoderLayer. One layer at a time is built from a weight source,
run over every token of a shard in chunks of whole episodes, then freed. Hidden states wait in host memory
between layers. Attention never crosses an episode boundary: each chunk carries packed position ids and
cu_seqlens, run by flash_attention_2 varlen when it is installed on CUDA, else by transformers' SDPA per episode.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BeamConfig
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import Concatenate, MergeModulelist
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import AttentionInterface
from transformers.models.beam.modeling_beam import (
    BeamDecoderLayer,
    BeamPreTrainedModel,
    BeamRMSNorm,
    BeamRotaryEmbedding,
    compute_attention_scaling,
)
from transformers.utils import is_flash_attn_2_available

from ouroboros.harness.capture import Capture, host_tensor
from ouroboros.harness.patch import patched

PACKED_SDPA = "ouroboros_packed_sdpa"
log = logging.getLogger(__name__)
FP32_KEYS = tuple(BeamPreTrainedModel._keep_in_fp32_modules_strict)


class WeightSource(Protocol):
    """Dequantized checkpoint tensors: per layer keyed relative to "model.layers.{i}.", globals relative to "model."."""

    def load_layer(self, i: int, device: str | torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]: ...

    def load_globals(self, device: str | torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]: ...


@dataclass(frozen=True)
class Layout:
    """Episodes packed end to end into one token stream."""

    token_ids: np.ndarray  # [T] int64
    offsets: np.ndarray  # [E + 1] int64; episode e spans offsets[e]:offsets[e + 1]

    @classmethod
    def pack(cls, episodes: list[np.ndarray]) -> Layout:
        lengths = np.array([len(e) for e in episodes], dtype=np.int64)
        if (lengths < 1).any():
            raise ValueError("every episode needs at least one token")
        ids = np.concatenate(episodes).astype(np.int64) if episodes else np.zeros(0, np.int64)
        return cls(ids, np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64))

    @property
    def n_tokens(self) -> int:
        return int(self.offsets[-1])

    @functools.cached_property
    def positions(self) -> np.ndarray:
        """Position of every token within its own episode."""
        lengths = np.diff(self.offsets)
        return np.arange(self.n_tokens, dtype=np.int64) - np.repeat(self.offsets[:-1], lengths)

    def chunks(self, max_tokens: int) -> list[tuple[int, int]]:
        """Consecutive whole episodes [a, b) of at most max_tokens; a longer episode forms a chunk alone."""
        out, a = [], 0
        for b in range(1, len(self.offsets)):
            if b - 1 > a and self.offsets[b] - self.offsets[a] > max_tokens:
                out.append((a, b - 1))
                a = b - 1
        if len(self.offsets) > 1:
            out.append((a, len(self.offsets) - 1))
        return out

    def fingerprint(self) -> str:
        h = hashlib.sha256(self.offsets.astype("<i8").tobytes())
        h.update(self.token_ids.astype("<i8").tobytes())
        return h.hexdigest()


def moe_layers(config: BeamConfig) -> list[int]:
    return [i for i, kind in enumerate(config.mlp_layer_types) if kind == "sparse"]


@functools.cache
def expert_fusions() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """transformers' checkpoint conversion for Beam, read as (fused parameter, per-expert source patterns).

    Each fused parameter stacks experts on dim 0 (MergeModulelist) and, with several sources, joins them on dim 1
    (Concatenate): gate_up_proj[e] = cat(gate_proj[e], up_proj[e]).
    """
    plan = []
    for conv in get_checkpoint_conversion_mapping("beam"):
        ops = [(type(op), getattr(op, "dim", None)) for op in getattr(conv, "operations", ())]
        expected = [(MergeModulelist, 0)] + ([(Concatenate, 1)] if len(conv.source_patterns) > 1 else [])
        if ops != expected or len(conv.target_patterns) != 1 or any(p.count("*") != 1 for p in conv.source_patterns):
            raise NotImplementedError(f"unsupported Beam weight conversion: {conv}")
        plan.append((conv.target_patterns[0], tuple(conv.source_patterns)))
    return tuple(plan)


def _copy(dst: torch.Tensor, src: torch.Tensor, name: str) -> None:
    if dst.shape != src.shape:
        raise ValueError(f"{name}: checkpoint shape {tuple(src.shape)} != parameter shape {tuple(dst.shape)}")
    dst.copy_(src)


@torch.no_grad()
def build_layer(config: BeamConfig, i: int, tensors: dict[str, torch.Tensor], device: torch.device,
                dtype: torch.dtype) -> BeamDecoderLayer:
    """A transformers BeamDecoderLayer holding checkpoint tensors (layer-relative names).

    Consumes `tensors`, popping each one as it is copied so device memory is released early.
    The router weight and correction bias stay FP32, as transformers keeps them.
    """
    with torch.device("meta"):
        layer = BeamDecoderLayer(config, i).to(dtype)
    for name, t in [*layer.named_parameters(), *layer.named_buffers()]:
        if any(key in name for key in FP32_KEYS):
            t.data = t.data.float()
    layer = layer.to_empty(device=device).eval().requires_grad_(False)

    state = dict(layer.named_parameters()) | dict(layer.named_buffers())
    for target_name, patterns in expert_fusions():
        target = state.pop(target_name, None)
        if target is None:
            continue
        for e in range(target.shape[0]):
            row = 0
            for pattern in patterns:
                name = pattern.replace("*", str(e))
                src = tensors.pop(name)
                _copy(target[e, row:row + src.shape[0]], src, name)
                row += src.shape[0]
            if row != target.shape[1]:
                raise ValueError(f"layer {i} {target_name}[{e}]: filled {row} of {target.shape[1]} rows")
    for name, t in state.items():
        _copy(t, tensors.pop(name), name)
    if tensors:
        raise ValueError(f"layer {i}: unused checkpoint tensors {sorted(tensors)[:8]}")
    return layer


def _sliding_blocks(module, q, k, v, window: int, scaling: float | None) -> torch.Tensor:
    """Causal attention over the last `window` keys, as blocks of `window` queries over their 2 * window keys."""
    n = q.shape[2]
    nb = -(-n // window)
    pad = nb * window - n
    qb = F.pad(q, (0, 0, 0, pad)).unflatten(2, (nb, window)).transpose(1, 2).flatten(0, 1)

    def windows(x: torch.Tensor) -> torch.Tensor:
        # Key window of block b covers positions [(b - 1) * window, (b + 1) * window).
        x = F.pad(x, (0, 0, window, pad)).unfold(2, 2 * window, window)  # [1, kv, nb, D, 2w]
        return x.permute(0, 2, 1, 4, 3).flatten(0, 1).contiguous()

    i = torch.arange(window, device=q.device)[:, None]
    j = torch.arange(2 * window, device=q.device)[None, :]
    mask = ((j > i) & (j <= i + window)).expand(nb, 1, window, 2 * window).clone()
    mask[0, :, :, :window] = False  # block 0 has no earlier keys
    out, _ = sdpa_attention_forward(module, qb, windows(k), windows(v), mask, scaling=scaling)
    return out.reshape(1, nb * window, *out.shape[2:])[:, :n]


def packed_sdpa_attention(module, query, key, value, attention_mask, dropout: float = 0.0,
                          scaling: float | None = None, sliding_window: int | None = None,
                          cu_seq_lens_q: torch.Tensor | None = None, **kwargs) -> tuple[torch.Tensor, None]:
    """transformers' SDPA attention, run separately on each episode of a packed batch-1 sequence."""
    if attention_mask is not None or cu_seq_lens_q is None:
        raise ValueError("packed attention takes cu_seq_lens_q and no attention mask")
    bounds = cu_seq_lens_q.tolist()
    outs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        q, k, v = query[:, :, a:b], key[:, :, a:b], value[:, :, a:b]
        if sliding_window is None or b - a <= sliding_window:
            outs.append(sdpa_attention_forward(module, q, k, v, None, dropout=dropout, scaling=scaling)[0])
        else:
            outs.append(_sliding_blocks(module, q, k, v, sliding_window, scaling))
    return torch.cat(outs, dim=1), None


AttentionInterface.register(PACKED_SDPA, packed_sdpa_attention)


def attention_backend(attn: str, device: torch.device) -> str:
    """'auto' picks flash_attention_2 varlen on CUDA when flash-attn is installed, else per-episode SDPA."""
    if attn == "auto":
        attn = "flash" if device.type == "cuda" and is_flash_attn_2_available() else "sdpa"
    return {"flash": "flash_attention_2", "sdpa": PACKED_SDPA}[attn]


class Transfers:
    """Chunk copies between host and device. On CUDA they run on a side stream and overlap compute.

    upload() of chunk k + 1 runs while chunk k computes, and download() of chunk k while chunk k + 1 computes.
    Chunks of one layer cover disjoint host rows, and finish() waits for every copy before the host reads them.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.side = torch.cuda.Stream(device) if device.type == "cuda" else None

    def upload(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        if self.side is None:
            return h.to(self.device), None
        with torch.cuda.stream(self.side):
            x = h.to(self.device, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(self.side)
        return x, ready

    def take(self, pending: tuple[torch.Tensor, torch.cuda.Event | None]) -> torch.Tensor:
        x, ready = pending
        if ready is not None:
            compute = torch.cuda.current_stream(self.device)
            compute.wait_event(ready)
            x.record_stream(compute)  # x was allocated on the side stream
        return x

    def download(self, out: torch.Tensor, dst: torch.Tensor) -> None:
        if self.side is None:
            dst.copy_(out)
            return
        computed = torch.cuda.Event()
        computed.record(torch.cuda.current_stream(self.device))
        self.side.wait_event(computed)
        with torch.cuda.stream(self.side):
            dst.copy_(out, non_blocking=True)
        out.record_stream(self.side)  # keeps the allocator from reusing out before the copy lands

    def finish(self) -> None:
        if self.side is not None:
            self.side.synchronize()


class Streamer:
    """Runs Beam layer by layer over a packed shard whose hidden states live in host memory."""

    def __init__(self, config: BeamConfig, source: WeightSource, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.bfloat16, chunk_tokens: int = 65536, attn: str = "auto"):
        self.device = torch.device(device)
        self.dtype = dtype
        self.source = source
        self.chunk_tokens = chunk_tokens
        self.config = copy.deepcopy(config)
        self.config._attn_implementation = attention_backend(attn, self.device)
        if attn == "auto" and self.device.type == "cuda" and self.config._attn_implementation == PACKED_SDPA:
            log.warning("flash-attn is not installed, so attention falls back to per-episode SDPA (about twice the "
                        "work on sliding layers). Install the flash-attn v2.8.3 release wheel built for torch 2.9, "
                        "or pass --attn sdpa to choose this path.")
        self.config._experts_implementation = "eager"
        self.rotary = BeamRotaryEmbedding(self.config, device=self.device)
        self.moe_layers = moe_layers(self.config)
        self.moe_index = {layer: j for j, layer in enumerate(self.moe_layers)}

    @functools.cached_property
    def globals(self) -> dict[str, torch.Tensor]:
        return {k.removeprefix("model."): v for k, v in self.source.load_globals("cpu", self.dtype).items()}

    def _host(self, shape: tuple[int, ...]) -> torch.Tensor:
        return host_tensor(shape, self.dtype, pin=self.device.type == "cuda")

    def free(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def embed(self, layout: Layout) -> torch.Tensor:
        """Input embeddings for every token, in host memory."""
        weight = self.globals["embed_tokens.weight"].to(self.device, self.dtype)
        hidden = self._host((layout.n_tokens, self.config.hidden_size))
        ids = torch.from_numpy(layout.token_ids)
        for lo in range(0, layout.n_tokens, self.chunk_tokens):
            hi = min(lo + self.chunk_tokens, layout.n_tokens)
            hidden[lo:hi].copy_(F.embedding(ids[lo:hi].to(self.device), weight))
        del weight
        self.free()
        return hidden

    def weights(self, start: int, stop: int) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
        """Layer tensors in order; a source with stream() prefetches layer i + 1 while layer i computes."""
        if hasattr(self.source, "stream"):
            yield from self.source.stream(range(start, stop), self.device, self.dtype)
        else:
            for i in range(start, stop):
                yield i, self.source.load_layer(i, self.device, self.dtype)

    def build(self, i: int, tensors: dict[str, torch.Tensor]) -> BeamDecoderLayer:
        return build_layer(self.config, i, tensors, self.device, self.dtype)

    @torch.no_grad()
    def run_layers(self, hidden: torch.Tensor, layout: Layout, start: int = 0, stop: int | None = None,
                   delta: torch.Tensor | None = None, capture: Capture | None = None) -> None:
        """Run decoder layers [start, stop) over every token, updating `hidden` in place."""
        stop = self.config.num_hidden_layers if stop is None else stop
        clock = time.perf_counter()
        for i, tensors in self.weights(start, stop):
            loaded = time.perf_counter()
            layer = self.build(i, tensors)
            self.apply(layer, i, hidden, layout, delta, capture)
            del layer, tensors
            self.free()
            done = time.perf_counter()
            log.info("layer %d: weights %.1f s, compute %.1f s, %d tokens", i, loaded - clock, done - loaded,
                     layout.n_tokens)
            clock = done

    @torch.no_grad()
    def apply(self, layer: BeamDecoderLayer, i: int, hidden: torch.Tensor, layout: Layout,
              delta: torch.Tensor | None = None, capture: Capture | None = None) -> None:
        """Run one built layer over every chunk of the shard, updating `hidden` in place."""
        j = self.moe_index.get(i)
        span = [0, 0]
        hook = None
        if capture is not None and j is not None:
            hook = layer.mlp.gate.register_forward_hook(
                lambda module, args, out: capture.router(j, span[0], span[1], out[0], out[2]))
        chunks = layout.chunks(self.chunk_tokens)
        rows = [(int(layout.offsets[a]), int(layout.offsets[b])) for a, b in chunks]
        io = Transfers(self.device)
        try:
            with patched(layer, None if delta is None or j is None else delta[j]):
                pending = io.upload(hidden[slice(*rows[0])]) if chunks else None
                for k, ((a, b), (lo, hi)) in enumerate(zip(chunks, rows)):
                    x = io.take(pending)
                    if k + 1 < len(chunks):
                        pending = io.upload(hidden[slice(*rows[k + 1])])
                    span[:] = lo, hi
                    out = self._forward_chunk(layer, i, x, layout, a, b)
                    if capture is not None:
                        capture.layer_output(i, lo, hi, out)
                    io.download(out, hidden[lo:hi])
        finally:
            io.finish()
            if hook is not None:
                hook.remove()

    def _forward_chunk(self, layer: BeamDecoderLayer, i: int, x: torch.Tensor, layout: Layout,
                       a: int, b: int) -> torch.Tensor:
        """Decoder layer i on the chunk of episodes [a, b), whose hidden states x [n, H] are on the device."""
        lo, hi = int(layout.offsets[a]), int(layout.offsets[b])
        x = x.unsqueeze(0)
        pos = torch.from_numpy(layout.positions[lo:hi]).to(self.device).unsqueeze(0)
        cu = torch.from_numpy(layout.offsets[a:b + 1] - lo).to(self.device, torch.int32)
        longest = int(np.diff(layout.offsets[a:b + 1]).max())
        # As in BeamModel.forward: RoPE layers get cos/sin, NoPE layers get the position-dependent query scale.
        layer_type = self.config.layer_types[i]
        if self.config.rope_parameters[layer_type] is None:
            pe, scaling = None, compute_attention_scaling(pos)
        else:
            pe, scaling = self.rotary(x, pos, layer_type), None
        out = layer(x, position_embeddings=pe, attention_mask=None, position_ids=pos, attention_scaling=scaling,
                    cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=longest, max_length_k=longest)
        return out[0]

    @functools.cached_property
    def _head(self) -> tuple[BeamRMSNorm, torch.Tensor]:
        norm = BeamRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps).to(self.device, self.dtype)
        with torch.no_grad():
            norm.weight.copy_(self.globals["norm.weight"])
        return norm, self.globals["lm_head.weight"].to(self.device, self.dtype)

    @torch.no_grad()
    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """FP32 softcapped logits from last-layer hidden states on the device, as BeamForCausalLM computes them."""
        norm, lm_head = self._head
        logits = F.linear(norm(h), lm_head).float()
        cap = self.config.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits

    @torch.no_grad()
    def head(self, hidden: torch.Tensor, token_ids: np.ndarray, targets: np.ndarray, candidates: np.ndarray,
             top_k: int, batch: int = 1024) -> dict[str, np.ndarray]:
        """Next-token log-probs at target tokens t, read from the last-layer hidden state at t - 1.

        lm_head runs only at those positions. Returns nll [N], top_ids/top_logprobs [N, top_k] and
        candidate_logprobs [N, C].
        """
        n = len(targets)
        out = {
            "nll": np.empty(n, np.float32),
            "top_ids": np.empty((n, top_k), np.int32),
            "top_logprobs": np.empty((n, top_k), np.float32),
            "candidate_logprobs": np.empty((n, len(candidates)), np.float32),
        }
        cand = torch.as_tensor(candidates, dtype=torch.long, device=self.device)
        for s in range(0, n, batch):
            t = targets[s:s + batch]
            logp = torch.log_softmax(self.logits(hidden[torch.from_numpy(t - 1)].to(self.device)), dim=-1)
            tgt = torch.from_numpy(token_ids[t]).to(self.device)
            values, ids = logp.topk(top_k, dim=-1)
            out["nll"][s:s + len(t)] = (-logp.gather(1, tgt[:, None])[:, 0]).cpu().numpy()
            out["top_ids"][s:s + len(t)] = ids.cpu().numpy()
            out["top_logprobs"][s:s + len(t)] = values.cpu().numpy()
            out["candidate_logprobs"][s:s + len(t)] = logp[:, cand].cpu().numpy()
        return out
