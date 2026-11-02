"""Patch banks: many named router-bias patches in one safetensors file, each with its provenance.

A patch is an FP32 [n_moe, n_experts] delta added to each MoE layer's e_score_correction_bias before selection.
patched() adds it in FP32, so what runs is the effective delta fl(bias + delta) - bias. Beam's bias sits at 18 to
28, where FP32 spacing is 1.9e-6, so L2 is measured and matched on the effective delta. Every nonzero entry is at
least stageb.FLOOR. Surgeon and CEM patches are whole multiples of stageb.UNIT, listed as (decoder layer, expert, u).

A bank holds one tensor per patch, keyed by name. Its safetensors metadata is {"format": "pt", "ouroboros": JSON},
where the JSON carries the exploratory label, the shape, the stock bias hash and one record per patch. The same
patches and records always give the same bytes, so a rebuilt bank keeps its sha256.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save

from ouroboros import stageb

BANK_FORMAT = "ouroboros-patch-bank/1"
NAME = re.compile(r"^[a-z0-9][a-z0-9_]{0,47}$")
RESERVED = ("stock", "twin")
ARMS = ("beam", "cem", "steermoe", "instruction", "random", "jitter", "permutation", "dose", "route_match",
        "zero_shot")
PROVENANCE = ("arm", "round", "slot", "seed", "parent", "built_by", "inputs")
BIAS_KEY = "mlp.gate.e_score_correction_bias"
log = logging.getLogger(__name__)


def _f32(delta) -> np.ndarray:
    """A float32 host array of a tensor or array."""
    if isinstance(delta, torch.Tensor):
        return delta.detach().to(device="cpu", dtype=torch.float32).numpy()
    return np.asarray(delta, dtype=np.float32)


def _tensor(delta) -> torch.Tensor:
    """An FP32, C-contiguous CPU tensor that owns its memory."""
    return torch.from_numpy(np.array(_f32(delta), dtype=np.float32, order="C"))


def patch_sha256(delta) -> str:
    return hashlib.sha256(np.ascontiguousarray(_f32(delta), "<f4").tobytes()).hexdigest()


def check_floor(delta, floor: float = stageb.FLOOR) -> None:
    """Every nonzero entry must be at least `floor` in magnitude; smaller ones vanish into FP32 rounding at the bias."""
    a = np.abs(_f32(delta))
    small = (a > 0) & (a < floor)
    if small.any():
        raise ValueError(f"{int(small.sum())} nonzero entries below the {floor:g} floor; the smallest is "
                         f"{float(a[small].min()):.3g}")


def stock_bias(source, moe_layers: Sequence[int]) -> torch.Tensor:
    """[n_moe, E] FP32 stock correction bias on the CPU, read before any layer streams."""
    names = [f"model.layers.{i}.{BIAS_KEY}" for i in moe_layers]
    if hasattr(source, "load_tensors"):
        got = source.load_tensors(names, "cpu", torch.float32)
        rows = [got[n] for n in names]
    else:
        rows = [source.load_layer(i, "cpu", torch.float32)[BIAS_KEY] for i in moe_layers]
    return torch.stack([r.to(torch.float32) for r in rows]).contiguous()


def effective_delta(bias, delta) -> torch.Tensor:
    """(bias + delta) - bias in FP32 on the CPU: the change patched() makes to the bias, bit for bit."""
    b, d = _tensor(bias), _tensor(delta)
    if b.shape != d.shape:
        raise ValueError(f"delta {list(d.shape)} does not match the bias {list(b.shape)}")
    return (b + d) - b


def effective_l2(bias, delta) -> float:
    return float(torch.linalg.vector_norm(effective_delta(bias, delta).double()))


def effective_l2_per_layer(bias, delta) -> list[float]:
    return [float(x) for x in torch.linalg.vector_norm(effective_delta(bias, delta).double(), dim=1)]


def scale_to_effective_l2(delta, bias, target: float, rel_tol: float = 1e-6) -> torch.Tensor:
    """delta times one scalar, bisected until its effective L2 is within rel_tol of `target`.

    Effective L2 is a step function of the scale: each entry moves in whole FP32 steps of the bias. When those steps
    are coarser than rel_tol (equal-magnitude entries on Beam's bias), the closest reachable scale is returned and
    the miss is logged. ValueError when the scaled delta breaks the floor.
    """
    d = _tensor(delta).double()
    if not target > 0 or not d.any():
        raise ValueError("scaling needs a nonzero delta and a positive target")
    cache: dict[float, float] = {}

    def l2(s: float) -> float:
        if s not in cache:
            cache[s] = effective_l2(bias, (d * s).float())
        return cache[s]

    lo = hi = target / float(torch.linalg.vector_norm(d))
    for _ in range(64):
        if l2(lo) <= target:
            break
        lo /= 2
    for _ in range(64):
        if l2(hi) >= target:
            break
        hi *= 2
    best = min((lo, hi), key=lambda s: abs(l2(s) - target))
    for _ in range(200):
        if abs(l2(best) - target) <= rel_tol * target:
            break
        mid = (lo + hi) / 2
        if mid in (lo, hi):
            break
        if abs(l2(mid) - target) < abs(l2(best) - target):
            best = mid
        lo, hi = (mid, hi) if l2(mid) < target else (lo, mid)
    out = (d * best).float()
    check_floor(out)
    miss = abs(l2(best) - target) / target
    if miss > rel_tol:
        log.warning("effective L2 %.6g misses the target %.6g by %.2g relative: the FP32 steps of the bias are "
                    "coarser than %.0g here", l2(best), target, miss, rel_tol)
    return out


def units_to_delta(entries: Iterable[tuple[int, int, int]], moe_layers: Sequence[int], n_experts: int,
                   unit: float = stageb.UNIT) -> torch.Tensor:
    """FP32 delta from (decoder layer, expert, units) entries."""
    index = {int(layer): j for j, layer in enumerate(moe_layers)}
    delta = torch.zeros(len(moe_layers), n_experts, dtype=torch.float32)
    seen: set[tuple[int, int]] = set()
    for layer, expert, u in entries:
        if u != int(u) or layer != int(layer) or expert != int(expert):
            raise ValueError(f"entry ({layer}, {expert}, {u}) is not whole numbers")
        layer, expert, u = int(layer), int(expert), int(u)
        if layer not in index:
            raise ValueError(f"layer {layer} has no router; MoE layers run {min(index)} to {max(index)}")
        if not 0 <= expert < n_experts:
            raise ValueError(f"expert {expert} is outside [0, {n_experts})")
        if u == 0:
            raise ValueError(f"entry ({layer}, {expert}) has zero units")
        if (layer, expert) in seen:
            raise ValueError(f"entry ({layer}, {expert}) appears twice")
        seen.add((layer, expert))
        delta[index[layer], expert] = u * unit
    return delta


def delta_to_units(delta, moe_layers: Sequence[int], unit: float = stageb.UNIT) -> list[tuple[int, int, int]]:
    """(decoder layer, expert, units) for each nonzero entry, in layer then expert order."""
    a = _f32(delta).astype(np.float64)
    out = []
    for j, e in zip(*np.nonzero(a)):
        u = a[j, e] / unit
        r = int(round(u))
        if r == 0 or abs(u - r) > 1e-3:
            raise ValueError(f"entry ({moe_layers[j]}, {e}) = {a[j, e]:.6g} is not a whole number of {unit:g} units")
        out.append((int(moe_layers[j]), int(e), r))
    return out


def _check_name(name: str) -> None:
    if not NAME.match(name) or name in RESERVED:
        raise ValueError(f"patch name {name!r} must match {NAME.pattern} and be neither stock nor twin")


def _record(name: str, delta: torch.Tensor, bias: torch.Tensor, meta: dict, moe_layers: Sequence[int]) -> dict:
    if not isinstance(meta, Mapping) or not isinstance(meta.get("provenance"), Mapping):
        raise ValueError(f"{name}: meta needs a provenance record")
    given = dict(meta["provenance"])
    provenance = {k: given.pop(k, {} if k == "inputs" else None) for k in PROVENANCE} | given
    if provenance["arm"] not in ARMS:
        raise ValueError(f"{name}: arm {provenance['arm']!r} is not one of {ARMS}")
    units = meta.get("units")
    if units is not None:
        units = [[int(x) for x in entry] for entry in units]
        if not torch.equal(units_to_delta(units, moe_layers, delta.shape[1]), delta):
            raise ValueError(f"{name}: its units do not give its delta")
    return {
        "sha256": patch_sha256(delta),
        "exploratory": True,
        "provenance": provenance,
        "nonzero": int((delta != 0).sum()),
        "nominal_l2": float(torch.linalg.vector_norm(delta.double())),
        "effective_l2": effective_l2(bias, delta),
        "effective_l2_per_layer": effective_l2_per_layer(bias, delta),
        "units": units,
        "footprint": meta.get("footprint"),
        "hypothesis": meta.get("hypothesis"),
    }


def save_patch_bank(path, patches: Mapping[str, Any], meta: Mapping[str, dict], bias,
                    moe_layers: Sequence[int]) -> dict:
    """Write a bank; returns its metadata JSON. Records keep the given provenance, units, footprint and hypothesis."""
    bias = _tensor(bias)
    if bias.dim() != 2 or bias.shape[0] != len(moe_layers):
        raise ValueError(f"bias {list(bias.shape)} does not match {len(moe_layers)} MoE layers")
    unknown = set(meta) - set(patches)
    if unknown:
        raise ValueError(f"meta for patches not in the bank: {sorted(unknown)}")
    tensors, records = {}, {}
    for name, delta in patches.items():
        _check_name(name)
        d = _tensor(delta)
        if d.shape != bias.shape:
            raise ValueError(f"{name}: shape {list(d.shape)}, expected {list(bias.shape)}")
        if not torch.isfinite(d).all():
            raise ValueError(f"{name}: non-finite entries")
        try:
            check_floor(d)
        except ValueError as e:
            raise ValueError(f"{name}: {e}") from None
        records[name] = _record(name, d, bias, meta.get(name), moe_layers)
        tensors[name] = d
    head = {"format": BANK_FORMAT, **stageb.LABEL, "n_moe": int(bias.shape[0]), "n_experts": int(bias.shape[1]),
            "moe_layers": [int(x) for x in moe_layers], "unit": stageb.UNIT, "bias_sha256": patch_sha256(bias),
            "patches": records}
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_bytes(_canonical(save(tensors, metadata={"format": "pt", "ouroboros": json.dumps(head)})))
    os.replace(tmp, path)
    return head


def _canonical(data: bytes) -> bytes:
    """A safetensors file with its header keys in a fixed order.

    safetensors writes its metadata map in hash order, which changes from process to process, so the same bank
    would get another sha256 on every save. The tensor bytes and their offsets stay as safetensors laid them out.
    """
    n = int.from_bytes(data[:8], "little")
    header = json.loads(data[8:8 + n])
    meta = header.pop("__metadata__")
    head = json.dumps({"__metadata__": {k: meta[k] for k in sorted(meta)}, **{k: header[k] for k in sorted(header)}},
                      separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)  # safetensors pads its header to 8 bytes with spaces
    return len(head).to_bytes(8, "little") + head + data[8 + n:]


def load_patch_bank(path, n_moe: int, n_experts: int, bias=None) -> tuple[dict[str, torch.Tensor], dict]:
    """A bank's patches and metadata JSON, every patch checked; ValueError on any failure."""
    with safe_open(str(path), framework="pt") as f:
        raw = f.metadata() or {}
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    if "ouroboros" not in raw:
        raise ValueError(f"{path} carries no ouroboros metadata, so it is no patch bank")
    head = json.loads(raw["ouroboros"])
    if head.get("format") != BANK_FORMAT:
        raise ValueError(f"{path}: format {head.get('format')!r}, expected {BANK_FORMAT}")
    if head.get("exploratory") is not True:
        raise ValueError(f"{path}: the exploratory label is missing")
    if (head.get("n_moe"), head.get("n_experts")) != (n_moe, n_experts):
        raise ValueError(f"{path} holds [{head.get('n_moe')}, {head.get('n_experts')}] patches, "
                         f"the model takes [{n_moe}, {n_experts}]")
    records = head.get("patches", {})
    if set(records) != set(tensors):
        raise ValueError(f"{path}: tensors {sorted(tensors)} and records {sorted(records)} differ")
    for name, t in tensors.items():
        _check_name(name)
        if t.dtype != torch.float32 or tuple(t.shape) != (n_moe, n_experts):
            raise ValueError(f"{path}: {name} is {t.dtype} {list(t.shape)}, expected FP32 [{n_moe}, {n_experts}]")
        if not torch.isfinite(t).all():
            raise ValueError(f"{path}: {name} has non-finite entries")
        try:
            check_floor(t)
        except ValueError as e:
            raise ValueError(f"{path}: {name}: {e}") from None
        if records[name].get("exploratory") is not True:
            raise ValueError(f"{path}: {name} lacks the exploratory label")
        if records[name].get("sha256") != patch_sha256(t):
            raise ValueError(f"{path}: {name} does not match its recorded sha256")
    if bias is not None and head.get("bias_sha256") != patch_sha256(bias):
        raise ValueError(f"{path} was built on stock bias {head.get('bias_sha256')}, this bias is "
                         f"{patch_sha256(bias)}")
    return tensors, head


def merge_banks(path, parts: Sequence[tuple[Mapping[str, Any], dict]], bias, moe_layers: Sequence[int]) -> dict:
    """One bank holding the patches of several (patches, metadata) parts; a name may appear once."""
    patches: dict[str, Any] = {}
    meta: dict[str, dict] = {}
    for tensors, head in parts:
        records = head.get("patches", head)
        for name, t in tensors.items():
            if name in patches:
                raise ValueError(f"patch {name} appears in two banks")
            record = records[name]
            if record.get("sha256") not in (None, patch_sha256(t)):
                raise ValueError(f"patch {name} does not match its recorded sha256")
            patches[name], meta[name] = t, record
    return save_patch_bank(path, patches, meta, bias, moe_layers)
