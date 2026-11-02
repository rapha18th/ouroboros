"""Checks on what the surgeon submits: candidate patches and lookups, with every problem listed in one message.

An entry (l, e, u) adds u units of 1e-4 to the correction bias of expert e in decoder layer l. Nothing here rescales
or re-centres a candidate: a submission is taken as written or refused, and the refusal goes back to Beam as the
tool result, listing every problem at once. round.py runs the same per-candidate checks on each CEM draw and on every
Beam file that reaches the VM, so both sides refuse the same patches.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ouroboros import stageb

if TYPE_CHECKING:
    from ouroboros.surgeon.footprint import WindowSample

LOOKUP_TOOLS = ("expert_stats", "footprint")
MAX_NOTE_WORDS = 100
MAX_STATS_EXPERTS = 16
MAX_LABEL = 48
MAX_HYPOTHESIS = 600


class Invalid(ValueError):
    """A refusal worded for the surgeon. `problems` lists each one; `codes` names their kinds for callers."""

    def __init__(self, problems: Sequence[str], codes: Iterable[str] = ()):
        self.problems = list(problems)
        self.codes = frozenset(codes)
        super().__init__("Refused. Fix every problem below and call again.\n" +
                         "\n".join(f"- {p}" for p in self.problems))


@dataclass(frozen=True)
class Limits:
    n_candidates: int
    budget_units: float
    footprint_cap: float | None
    moe_layers: tuple[int, ...]
    n_experts: int
    max_abs_u: int = stageb.MAX_ABS_UNITS
    max_entries: int = stageb.MAX_ENTRIES
    max_rel_loss: float = 0.02
    lookups: int = stageb.LOOKUPS
    unit: float = stageb.UNIT


@dataclass(frozen=True)
class Candidate:
    name: str  # assigned: beam_r03a, beam_r03b in submission order
    label: str  # Beam's own short label
    hypothesis: str
    entries: tuple[tuple[int, int, int], ...]
    l2_units: float
    sha256: str
    delta: np.ndarray = field(compare=False, repr=False)  # FP32 [n_moe, n_experts]
    footprint: dict | None = None  # {"share", "per_layer", "sample_sha256"} when a window sample was given


def patch_sha256(delta) -> str:
    """sha256 over the little-endian FP32 bytes, as bank.patch_sha256 computes it."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(delta), "<f4").tobytes()).hexdigest()


def units_delta(entries: Iterable[tuple[int, int, int]], moe_layers: Sequence[int], n_experts: int,
                unit: float = stageb.UNIT) -> np.ndarray:
    """FP32 [n_moe, n_experts] delta from (decoder layer, expert, units) entries already checked."""
    row = {layer: j for j, layer in enumerate(moe_layers)}
    delta = np.zeros((len(moe_layers), n_experts), np.float32)
    for layer, expert, u in entries:
        delta[row[layer], expert] = np.float32(u * unit)
    return delta


def l2_units(entries: Iterable[tuple[int, int, int]]) -> float:
    return float(np.sqrt(sum(u * u for _, _, u in entries)))


def parse_arguments(args: dict | str) -> dict:
    """Tool-call arguments as a dict; Invalid when they are not a JSON object."""
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as e:
            raise Invalid([f"the arguments are not valid JSON: {e.msg} at position {e.pos}"], {"json"}) from None
    if not isinstance(args, dict):
        raise Invalid(["the arguments must be a JSON object"], {"json"})
    return args


def _int(x: Any) -> int | None:
    """An integer value, or None for anything else (bools, fractions, strings)."""
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return None


def _layers_text(moe_layers: Sequence[int]) -> str:
    lo, hi = min(moe_layers), max(moe_layers)
    return f"{lo} to {hi}" if list(moe_layers) == list(range(lo, hi + 1)) else ", ".join(map(str, moe_layers))


def _entries(raw: Any, limits: Limits, tag: str, problems: list[str], codes: set[str]) -> list[tuple[int, int, int]]:
    """Per-entry checks: shape, MoE layer, expert range, unit range, repeats and the entry count."""
    if not isinstance(raw, list) or not raw:
        problems.append(f"{tag}: entries must be a list of 1 to {limits.max_entries} objects {{l, e, u}}")
        codes.add("entries")
        return []
    if len(raw) > limits.max_entries:
        problems.append(f"{tag}: {len(raw)} entries; at most {limits.max_entries} are allowed")
        codes.add("entries")
    out, seen = [], set()
    for k, x in enumerate(raw):
        vals = [_int(x.get(name)) for name in ("l", "e", "u")] if isinstance(x, dict) else [None] * 3
        if None in vals or (isinstance(x, dict) and set(x) - {"l", "e", "u"}):
            problems.append(f"{tag}: entry {k + 1} {json.dumps(x)[:80]} needs integer l, e and u and nothing else")
            codes.add("entries")
            continue
        layer, expert, u = vals
        where = f"L{layer}:{expert}"
        if layer not in limits.moe_layers:
            problems.append(f"{tag}: layer {layer} is not an MoE layer (MoE decoder layers are "
                            f"{_layers_text(limits.moe_layers)})")
            codes.add("layer")
        if not 0 <= expert < limits.n_experts:
            problems.append(f"{tag}: expert {expert} is outside 0 to {limits.n_experts - 1}")
            codes.add("expert")
        if u == 0 or abs(u) > limits.max_abs_u:
            problems.append(f"{tag}: u = {u} at {where} must be nonzero and within -{limits.max_abs_u} to "
                            f"{limits.max_abs_u}")
            codes.add("units")
        if (layer, expert) in seen:
            problems.append(f"{tag}: {where} appears more than once")
            codes.add("repeat")
        seen.add((layer, expert))
        out.append((layer, expert, u))
    return out


def _norm(entries: list[tuple[int, int, int]], limits: Limits, tag: str, problems: list[str], codes: set[str]) -> None:
    norm = l2_units(entries)
    if norm > limits.budget_units + 1e-9:
        problems.append(f"{tag}: norm {norm:.1f} units exceeds this round's budget of {limits.budget_units:g} units")
        codes.add("norm")


def _physics(entries: list[tuple[int, int, int]], limits: Limits, tag: str, *, bias: np.ndarray,
             sample: WindowSample | None, problems: list[str], codes: set[str],
             seen: Mapping[str, str]) -> tuple[np.ndarray, str, dict | None] | None:
    """Norm, FP32 rounding at the stock bias, simulated footprint and duplicates of one well-formed entry list. Every
    check runs, so one refusal names every problem."""
    n0 = len(problems)
    _norm(entries, limits, tag, problems, codes)
    delta = units_delta(entries, limits.moe_layers, limits.n_experts, limits.unit)
    stock = np.asarray(bias, np.float32)
    if stock.shape != delta.shape:
        raise ValueError(f"bias shape {stock.shape} differs from the patch shape {delta.shape}")
    nz = delta != 0
    effective = (stock + delta) - stock  # what reaches selection in FP32
    loss = np.abs(effective[nz] - delta[nz]) / np.abs(delta[nz])
    if loss.size and loss.max() > limits.max_rel_loss:
        j, e = np.argwhere(nz)[int(loss.argmax())]
        problems.append(f"{tag}: the entry at L{limits.moe_layers[j]}:{e} loses {loss.max():.1%} to FP32 rounding "
                        f"at the stock bias (at most {limits.max_rel_loss:.0%})")
        codes.add("rounding")
    sha = patch_sha256(delta)
    if sha in seen:
        twin = seen[sha]
        problems.append(f"{tag}: identical to {twin}" +
                        ("" if twin.startswith("candidate ") else ", which is already scored"))
        codes.add("duplicate")
    fp = None
    if sample is not None:
        from ouroboros.surgeon import footprint as footprint_mod  # numpy only; imported where it is needed

        got = footprint_mod.footprint(sample, delta)
        fp = {"share": float(got["share"]), "per_layer": [float(x) for x in got["per_layer"]],
              "sample_sha256": sample.sha256}
        cap = limits.footprint_cap
        if cap is not None and fp["share"] > cap + 1e-12:
            problems.append(f"{tag}: simulated footprint {fp['share']:.4f} exceeds this round's cap of {cap:.4f}")
            codes.add("footprint")
    if len(problems) > n0:
        return None
    return delta, sha, fp


def validate_entries(entries: Sequence[tuple[int, int, int]], limits: Limits, *, bias: np.ndarray,
                     sample: WindowSample | None, seen: Mapping[str, str], name: str) -> Candidate:
    """The per-candidate checks on one entry list (round.py: CEM draws and Beam files on the VM)."""
    problems: list[str] = []
    codes: set[str] = set()
    raw = [{"l": int(l), "e": int(e), "u": int(u)} for l, e, u in entries]
    parsed = _entries(raw, limits, name, problems, codes)
    got = None if problems else _physics(parsed, limits, name, bias=bias, sample=sample, problems=problems,
                                         codes=codes, seen=seen)
    if got is None:
        raise Invalid(problems, codes)
    delta, sha, fp = got
    return Candidate(name, "", "", tuple(parsed), l2_units(parsed), sha, delta, fp)


def validate_submission(args: dict | str, limits: Limits, *, bias: np.ndarray, sample: WindowSample | None,
                        seen: Mapping[str, str], names: Sequence[str]) -> tuple[str, list[Candidate]]:
    """submit_patches arguments as (notes, candidates named from `names`), or Invalid listing every problem."""
    args = parse_arguments(args)
    problems: list[str] = []
    codes: set[str] = set()
    extra = set(args) - {"notes", "candidates"}
    if extra:
        problems.append(f"unknown fields {sorted(extra)}; submit_patches takes notes and candidates")
        codes.add("args")
    notes = args.get("notes", "")
    if not isinstance(notes, str):
        problems.append("notes must be a string")
        codes.add("notes")
        notes = ""
    elif len(notes.split()) > MAX_NOTE_WORDS:
        problems.append(f"notes hold {len(notes.split())} words; at most {MAX_NOTE_WORDS} are allowed")
        codes.add("notes")
    cands = args.get("candidates")
    if not isinstance(cands, list):
        problems.append(f"candidates must be a list of exactly {limits.n_candidates}")
        raise Invalid(problems, codes | {"count"})
    if len(cands) != limits.n_candidates:
        problems.append(f"submit exactly {limits.n_candidates} candidates this round; you sent {len(cands)}")
        codes.add("count")
    if len(names) < min(len(cands), limits.n_candidates):
        raise ValueError(f"{len(names)} names for {len(cands)} candidates")
    out: list[Candidate] = []
    mine = dict(seen)
    for i, c in enumerate(cands):
        label = c.get("name") if isinstance(c, dict) else None
        tag = f"candidate {i + 1}" + (f" ({str(label)[:MAX_LABEL]})" if label else "")
        if not isinstance(c, dict):
            problems.append(f"{tag}: must be an object {{name, hypothesis, entries}}")
            codes.add("args")
            continue
        unknown = set(c) - {"name", "hypothesis", "entries"}
        if unknown:
            problems.append(f"{tag}: unknown fields {sorted(unknown)}")
            codes.add("args")
        hypothesis = c.get("hypothesis")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            problems.append(f"{tag}: needs a one-sentence hypothesis")
            codes.add("hypothesis")
            hypothesis = ""
        n0 = len(problems)
        entries = _entries(c.get("entries"), limits, tag, problems, codes)
        if len(problems) > n0 or not entries:
            if entries:  # the norm needs only the integers, so it is named beside the entry problems
                _norm(entries, limits, tag, problems, codes)
            continue
        got = _physics(entries, limits, tag, bias=bias, sample=sample, problems=problems, codes=codes, seen=mine)
        if got is None or i >= limits.n_candidates:
            continue
        delta, sha, fp = got
        mine[sha] = f"candidate {i + 1} of this submission"
        out.append(Candidate(names[i], str(label or "")[:MAX_LABEL], hypothesis.strip()[:MAX_HYPOTHESIS],
                             tuple(entries), l2_units(entries), sha, delta, fp))
    if problems:
        raise Invalid(problems, codes)
    return notes.strip(), out


def validate_lookup(tool: str, args: dict | str, used: int, limits: Limits) -> dict:
    """A lookup's arguments, normalised: {"tool", "l", "experts"} or {"tool", "entries"}. Invalid when refused."""
    if tool not in LOOKUP_TOOLS:
        raise Invalid([f"unknown tool {tool!r}; the tools are expert_stats, footprint and submit_patches"], {"tool"})
    if used >= limits.lookups:
        raise Invalid([f"lookup limit reached: you have used {used} of {limits.lookups} lookups this round. "
                       "Call submit_patches now"], {"lookups"})
    args = parse_arguments(args)
    problems: list[str] = []
    codes: set[str] = set()
    if tool == "expert_stats":
        extra = set(args) - {"l", "experts"}
        if extra:
            problems.append(f"unknown fields {sorted(extra)}; expert_stats takes l and experts")
            codes.add("args")
        layer = _int(args.get("l"))
        if layer is None or layer not in limits.moe_layers:
            problems.append(f"l = {args.get('l')!r} must be an MoE decoder layer ({_layers_text(limits.moe_layers)})")
            codes.add("layer")
        experts = args.get("experts")
        ids = [_int(x) for x in experts] if isinstance(experts, list) else []
        if not ids or len(ids) > MAX_STATS_EXPERTS or None in ids:
            problems.append(f"experts must be a list of 1 to {MAX_STATS_EXPERTS} integers")
            codes.add("expert")
        elif any(not 0 <= e < limits.n_experts for e in ids) or len(set(ids)) != len(ids):
            problems.append(f"experts must be distinct and within 0 to {limits.n_experts - 1}")
            codes.add("expert")
        if problems:
            raise Invalid(problems, codes)
        return {"tool": tool, "l": layer, "experts": ids}
    extra = set(args) - {"entries"}
    if extra:
        problems.append(f"unknown fields {sorted(extra)}; footprint takes entries")
        codes.add("args")
    entries = _entries(args.get("entries"), limits, "footprint", problems, codes)
    if problems:
        raise Invalid(problems, codes)
    return {"tool": tool, "entries": entries}
