"""Stage A inputs: anchors.csv, the rendered parquets, capture shards and window positions.

Windows hold episode-local positions of the tokens Beam wrote (gen_mask True):
- primary: f < p <= window_end
- f_any: f_any < p <= window_end_any
- h3_scall: the last 512 with p < s_call
- h3_s: the last 512 with p < s
- h3_prefinish (exploratory): the h3_scall positions before first_finish, the <|start|> of the first finish call
  that precedes the accepted one (s_call when there is none)

A shard token is start + p. Its window row is searchsorted(windows, t), and windows[row] == t must hold. route_ids
cover every token; tap_hidden covers the window rows only. The module stays free of torch, so it copies the two
helpers it needs from the harness.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

WINDOW_MAX = 512
WINDOWS = ("primary", "f_any", "h3_scall", "h3_s", "h3_prefinish")
N_EXPERTS = 256
INDEX = "index.json"
ANCHOR_INTS = ("n_tokens", "f", "f_any", "window_end", "window_end_any", "d", "s", "s_call")
INDEX_ANCHORS = ("f", "d", "s", "window_end", "s_call")  # the anchors a shard's index.json repeats


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bf16_to_float32(bits: np.ndarray) -> np.ndarray:
    """BF16 stored as raw uint16 bits, widened to float32 exactly."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _opt_int(x: str | None) -> int | None:
    return None if x is None or x == "" else int(float(x))


def load_anchors(path: str | Path) -> dict[str, dict]:
    """anchors.csv rows by episode_id, anchor columns as int or None and label as int."""
    rows: dict[str, dict] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            for c in ANCHOR_INTS:
                r[c] = _opt_int(r.get(c))
            r["label"] = int(r["label"])
            if r["episode_id"] in rows:
                raise ValueError(f"duplicate episode_id {r['episode_id']} in {path}")
            rows[r["episode_id"]] = r
    return rows


def load_meta(paths: Sequence[str | Path]) -> dict[str, dict]:
    """Per episode_id: source, task_id, condition, effort and label from the rendered parquets."""
    out: dict[str, dict] = {}
    for path in paths:
        table = pq.read_table(path, columns=["episode_id", "source", "task_id", "condition", "effort", "label"])
        for row in table.to_pylist():
            if row["episode_id"] in out:
                raise ValueError(f"duplicate episode_id {row['episode_id']} in {path}")
            out[row["episode_id"]] = row
    return out


def _lists(table, name: str, dtype) -> list[np.ndarray]:
    col = table.column(name).combine_chunks()
    values = col.flatten().to_numpy(zero_copy_only=False).astype(dtype)
    offsets = col.offsets.to_numpy()
    offsets = offsets - offsets[0]
    return [values[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def load_tokens(paths: Sequence[str | Path]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per episode_id: (token_ids int32, gen_mask bool) from the rendered parquets."""
    out = {}
    for path in paths:
        table = pq.read_table(path, columns=["episode_id", "token_ids", "gen_mask"])
        keys = table.column("episode_id").to_pylist()
        for k, ids, gen in zip(keys, _lists(table, "token_ids", np.int32), _lists(table, "gen_mask", bool)):
            out[k] = (ids, gen)
    return out


def window_bounds(kind: str, a: dict) -> tuple[int, int] | None:
    """(lo, hi) such that the window is the Beam-written p with lo < p <= hi, for the f-anchored windows."""
    lo, hi = {"primary": ("f", "window_end"), "f_any": ("f_any", "window_end_any")}[kind]
    if a[lo] is None or a[hi] is None:
        return None
    return a[lo], a[hi]


def first_finish(token_ids: np.ndarray, s_call: int, start_id: int, header_stops: Sequence[int]) -> int:
    """The <|start|> of the first earlier message whose role header (the ids up to the first id in header_stops,
    <|channel|> or <|message|>) equals that of the accepted finish call at s_call; s_call when there is none."""
    ids = np.asarray(token_ids)
    if not 0 <= s_call < len(ids) or ids[s_call] != start_id:
        raise ValueError(f"s_call {s_call} is no <|start|>")
    stops = np.flatnonzero(np.isin(ids, np.asarray(header_stops)))

    def header(p: int) -> np.ndarray:
        j = np.searchsorted(stops, p, side="right")
        return ids[p + 1:stops[j]] if j < len(stops) else ids[p + 1:]

    accepted = header(s_call)
    for p in np.flatnonzero(ids[:s_call] == start_id):
        if np.array_equal(header(int(p)), accepted):
            return int(p)
    return int(s_call)


def window_positions(gen_mask: np.ndarray, kind: str, a: dict) -> np.ndarray | None:
    """Episode-local positions of the window `kind`, or None when its anchors are missing. h3_prefinish reads
    a["first_finish"] (first_finish above)."""
    gen = np.asarray(gen_mask, dtype=bool)
    if kind in ("h3_scall", "h3_s", "h3_prefinish"):
        end = a["s" if kind == "h3_s" else "s_call"]
        if end is None:
            return None
        if not 0 < end <= len(gen):
            raise ValueError(f"{kind} end {end} outside the episode of {len(gen)} tokens")
        pos = np.flatnonzero(gen[:end])[-WINDOW_MAX:]
        if kind == "h3_prefinish":
            if a.get("first_finish") is None:
                raise ValueError("h3_prefinish needs first_finish")
            pos = pos[pos < a["first_finish"]]
    elif kind in ("primary", "f_any"):
        bounds = window_bounds(kind, a)
        if bounds is None:
            return None
        lo, hi = bounds
        if not 0 <= lo < hi < len(gen):
            raise ValueError(f"{kind} window ({lo}, {hi}] outside the episode of {len(gen)} tokens")
        pos = lo + 1 + np.flatnonzero(gen[lo + 1:hi + 1])
    else:
        raise ValueError(f"unknown window {kind}")
    if not 0 < len(pos) <= WINDOW_MAX:
        raise ValueError(f"{kind} window holds {len(pos)} Beam-written tokens; want 1 to {WINDOW_MAX}")
    return pos.astype(np.int64)


def window_rows(windows: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """Window row of each shard token; raises when a token has no row."""
    tokens = np.asarray(tokens, dtype=np.int64)
    rows = np.searchsorted(windows, tokens)
    inside = rows < len(windows)
    hit = np.zeros(len(tokens), bool)
    hit[inside] = windows[rows[inside]] == tokens[inside]
    if not hit.all():
        raise ValueError(f"shard token {int(tokens[~hit][0])} has no window row ({int((~hit).sum())} missing)")
    return rows


def covered(windows: np.ndarray, tokens: np.ndarray) -> bool:
    """True when every token has a window row."""
    rows = np.searchsorted(windows, tokens)
    if (rows >= len(windows)).any():
        return False
    return bool((windows[rows] == tokens).all())


def routing_features(route_ids: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """Frequency of each expert among the selected ids over the tokens, index j * 256 + e for MoE layer j.

    route_ids is [L, T, k]; each layer's frequencies sum to k.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    if len(tokens) == 0 or (np.diff(tokens) <= 0).any():
        raise ValueError("tokens must be non-empty and strictly increasing")
    lo, hi = int(tokens[0]), int(tokens[-1]) + 1
    block = np.asarray(route_ids[:, lo:hi, :])[:, tokens - lo, :]
    n_moe = block.shape[0]
    flat = (block.astype(np.int64) + N_EXPERTS * np.arange(n_moe)[:, None, None]).ravel()
    return np.bincount(flat, minlength=n_moe * N_EXPERTS) / len(tokens)


def hidden_features(tap_hidden: np.ndarray, windows: np.ndarray, tokens: np.ndarray, bf16: bool = True) -> np.ndarray:
    """Per tap layer, the mean hidden state over the tokens' window rows, layers concatenated: [taps * H]."""
    rows = window_rows(windows, tokens)
    lo, hi = int(rows[0]), int(rows[-1]) + 1
    block = np.asarray(tap_hidden[:, lo:hi, :])[:, rows - lo, :]
    x = bf16_to_float32(block) if bf16 else block.astype(np.float32)
    return x.mean(axis=1, dtype=np.float64).ravel()


@dataclass(frozen=True)
class Shard:
    """A finished capture shard, read through memmaps."""

    path: Path
    index: dict

    @classmethod
    def open(cls, path: str | Path) -> Shard:
        path = Path(path)
        return cls(path, json.loads((path / INDEX).read_text(encoding="utf-8")))

    @property
    def number(self) -> int:
        return int(self.index["shard"]) if "shard" in self.index else int(self.path.name.split("_")[-1])

    def has(self, name: str) -> bool:
        return (self.path / f"{name}.npy").exists()

    def array(self, name: str) -> np.ndarray:
        return np.load(self.path / f"{name}.npy", mmap_mode="r")

    def fingerprint(self) -> dict:
        """The index fields that identify what the shard holds."""
        keep = ("shard", "weights", "fingerprint", "n_tokens", "attn_implementation", "hidden_dtype", "tap_layers",
                "truncate", "capture_windows", "patch")
        out = {k: self.index.get(k) for k in keep}
        out["n_episodes"] = len(self.index["episodes"])
        out["moe_layers"] = [min(self.index["moe_layers"]), max(self.index["moe_layers"])]
        return out


def shard_dirs(capture: str | Path, shard: int | None = None) -> list[Path]:
    """Finished shard directories under `capture`: shard_{K:03d}, or all of them."""
    capture = Path(capture)
    if shard is not None:
        path = capture / f"shard_{shard:03d}"
        if not (path / INDEX).exists():
            raise FileNotFoundError(f"{path} holds no {INDEX}")
        return [path]
    dirs = sorted(p for p in capture.glob("shard_*") if (p / INDEX).exists())
    if not dirs:
        raise FileNotFoundError(f"no finished shard under {capture}")
    return dirs
