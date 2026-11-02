"""First-order route-change simulator for router-bias patches, on a sample of captured window rows.

A patch adds a delta to e_score_correction_bias. At the first patched layer the hidden state is still stock, so the
router's unbiased scores are the stock scores, and Beam's grouped top-k over scores + (bias + delta) gives the
patched selection exactly, up to the fp16 rounding of the stored scores. Deeper layers see a changed hidden state;
the simulator keeps their stock scores (first order).

- footprint(): per MoE layer, the share of sample tokens whose top-k set changes; `share` is their mean over every
  MoE layer, patched or not.
- dose_curve(): one expert at one layer moved by each of several doses: the share of tokens where it enters the
  top-k (of those where it was absent) and where it leaves (of those where it was present).
- dose_shares(): the same for every expert of a layer at once, through the group rules instead of a full top-k per
  expert. It equals dose_curve() up to exact ties.

The sample (maps/window_sample.npz, written by analysis.stage_b_maps fetch) holds Beam-written window rows of search
tasks only. This module imports numpy only, so the laptop controller loads it without torch.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UNIT = 1e-4  # one surgeon unit of router bias (stageb.UNIT; repeated so this module stays numpy only)
SAMPLE_FORMAT = "ouroboros-window-sample/1"


@dataclass(frozen=True)
class WindowSample:
    scores: np.ndarray  # [n_moe, N, E] float32, unbiased sigmoid scores
    ids: np.ndarray  # [n_moe, N, k] uint8, captured top-k at the same tokens
    bias: np.ndarray  # [n_moe, E] float32
    moe_layers: tuple[int, ...]
    k: int
    n_group: int
    topk_group: int
    sha256: str

    @property
    def n_tokens(self) -> int:
        return int(self.scores.shape[1])

    def index(self, layer: int) -> int:
        """MoE index of decoder layer `layer`."""
        if layer not in self.moe_layers:
            raise ValueError(f"decoder layer {layer} is not an MoE layer "
                             f"({self.moe_layers[0]} to {self.moe_layers[-1]})")
        return self.moe_layers.index(layer)


def save_sample(path: str | Path, *, scores: np.ndarray, ids: np.ndarray, bias: np.ndarray,
                moe_layers: Sequence[int], k: int, n_group: int, topk_group: int, keys: Sequence[str],
                episode: np.ndarray, position: np.ndarray, label: dict, extra: dict | None = None) -> Path:
    """Write a window sample npz. `label` is stageb.LABEL; `keys` are the episodes the rows come from."""
    scores = np.asarray(scores)
    n_moe, n, e = scores.shape
    if np.asarray(ids).shape != (n_moe, n, k) or np.asarray(bias).shape != (n_moe, e):
        raise ValueError(f"sample shapes disagree: scores {scores.shape}, ids {np.shape(ids)}, bias {np.shape(bias)}")
    if len(moe_layers) != n_moe or len(episode) != n or len(position) != n:
        raise ValueError("moe_layers, episode and position must match the sample")
    keys = [str(x) for x in keys]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, exploratory=np.array(bool(label["exploratory"])), reason=np.array(label["reason"]),
             format=np.array(SAMPLE_FORMAT), moe_layers=np.asarray(moe_layers, np.int64),
             keys_sha256=np.array(keys_sha256(keys)), scores=scores.astype(np.float16),
             ids=np.asarray(ids, np.uint8), bias=np.asarray(bias, np.float32), keys=np.array(keys),
             episode=np.asarray(episode, np.int32), position=np.asarray(position, np.int32),
             n_group=np.array(int(n_group)), topk_group=np.array(int(topk_group)), k=np.array(int(k)),
             **(extra or {}))
    tmp.replace(path)
    return path


def keys_sha256(keys: Sequence[str]) -> str:
    """sha256 of the sorted, newline-joined episode keys."""
    return hashlib.sha256("\n".join(sorted(str(k) for k in keys)).encode("utf-8")).hexdigest()


def load_sample(path: str | Path) -> WindowSample:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path) as z:
        if not bool(z["exploratory"]):
            raise ValueError(f"{path} is not labelled exploratory")
        scores = z["scores"].astype(np.float32)
        ids = np.asarray(z["ids"], np.uint8)
        bias = np.asarray(z["bias"], np.float32)
        layers = tuple(int(x) for x in z["moe_layers"])
        k, n_group, topk_group = int(z["k"]), int(z["n_group"]), int(z["topk_group"])
    if scores.shape[0] != len(layers) or ids.shape[:2] != scores.shape[:2] or ids.shape[2] != k:
        raise ValueError(f"{path}: scores {scores.shape}, ids {ids.shape} and {len(layers)} MoE layers disagree")
    if bias.shape != (scores.shape[0], scores.shape[2]) or scores.shape[2] % n_group:
        raise ValueError(f"{path}: bias {bias.shape} does not fit scores {scores.shape} in {n_group} groups")
    return WindowSample(scores, ids, bias, layers, k, n_group, topk_group, digest)


def sample_from_capture(shard_dir: str | Path, bias: np.ndarray, *, k: int, n_group: int, topk_group: int,
                        keys: Sequence[str] | None = None) -> dict:
    """save_sample() arguments from a finished capture shard: every Beam-written window row of `keys` (default all).

    Tests and tiny inputs use it; the real sample comes from ranged reads in stage_b_maps.fetch.
    """
    shard_dir = Path(shard_dir)
    index = json.loads((shard_dir / "index.json").read_text(encoding="utf-8"))
    ws = np.load(shard_dir / "window_scores.npy", mmap_mode="r")
    rid = np.load(shard_dir / "route_ids.npy", mmap_mode="r")
    gen = np.load(shard_dir / "gen_mask.npy").astype(bool)
    want = None if keys is None else set(keys)
    rows, toks, episode, position, used = [], [], [], [], []
    for ep in index["episodes"]:
        if want is not None and ep["key"] not in want:
            continue
        r0, r1 = ep["window_rows"]
        if r1 <= r0:
            continue
        local = np.concatenate([np.arange(a, b) for a, b in ep["windows"]])
        keep = gen[ep["start"] + local]
        if not keep.any():
            continue
        e = len(used)
        used.append(ep["key"])
        rows.append(np.arange(r0, r1)[keep])
        toks.append(ep["start"] + local[keep])
        episode.append(np.full(int(keep.sum()), e))
        position.append(local[keep])
    rows, toks = np.concatenate(rows), np.concatenate(toks)
    return {"scores": np.asarray(ws[:, rows]), "ids": np.asarray(rid[:, toks]), "bias": np.asarray(bias, np.float32),
            "moe_layers": index["moe_layers"], "k": k, "n_group": n_group, "topk_group": topk_group, "keys": used,
            "episode": np.concatenate(episode), "position": np.concatenate(position)}


# Selection


def _biased(scores: np.ndarray, bias: np.ndarray, delta: np.ndarray | None = None) -> np.ndarray:
    """scores + fl(bias + delta) in FP32, as patched() leaves the bias and the router adds it."""
    b = np.asarray(bias, np.float32)
    if delta is not None:
        b = b + np.asarray(delta, np.float32)
    return np.asarray(scores, np.float32) + b


def _kept(s: np.ndarray, n_group: int, topk_group: int) -> np.ndarray:
    """[N, E] bool: experts inside the best topk_group groups, a group scored by the sum of its best two."""
    n, e = s.shape
    g = np.sort(s.reshape(n, n_group, e // n_group), axis=-1)[..., -2:].sum(-1)
    keep = np.argsort(-g, axis=-1, kind="stable")[:, :topk_group]
    mask = np.zeros((n, n_group), bool)
    np.put_along_axis(mask, keep, True, axis=1)
    return np.repeat(mask, e // n_group, axis=1)


def select(scores: np.ndarray, bias: np.ndarray, k: int, n_group: int, topk_group: int) -> np.ndarray:
    """Beam's grouped top-k on [N, E] unbiased scores plus bias [E]: ids [N, k], best biased score first.

    Ties go to the lower expert id (a stable sort); the router's own topk sees FP32 scores, where ties are rare.
    """
    s = _biased(scores, bias)
    sm = np.where(_kept(s, n_group, topk_group), s, -np.inf)
    return np.argsort(-sm, axis=-1, kind="stable")[:, :k]


def _selected(s: np.ndarray, k: int, n_group: int, topk_group: int) -> np.ndarray:
    """[N, E] bool mask of the grouped top-k over biased scores s."""
    sm = np.where(_kept(s, n_group, topk_group), s, -np.inf)
    ids = np.argsort(-sm, axis=-1, kind="stable")[:, :k]
    mask = np.zeros(s.shape, bool)
    np.put_along_axis(mask, ids, True, axis=1)
    return mask


_STOCK: dict[tuple[str, int], np.ndarray] = {}


def _stock(sample: WindowSample, j: int) -> np.ndarray:
    key = (sample.sha256, j)
    if key not in _STOCK:
        _STOCK[key] = _selected(_biased(sample.scores[j], sample.bias[j]), sample.k, sample.n_group,
                                sample.topk_group)
    return _STOCK[key]


def stock_agreement(sample: WindowSample) -> np.ndarray:
    """[n_moe] share of sample tokens whose recomputed stock top-k set equals the captured one."""
    out = np.empty(len(sample.moe_layers))
    for j in range(len(sample.moe_layers)):
        captured = np.zeros_like(_stock(sample, j))
        np.put_along_axis(captured, sample.ids[j].astype(np.int64), True, axis=1)
        out[j] = (captured == _stock(sample, j)).all(axis=1).mean()
    return out


def selection_margin(sample: WindowSample, j: int) -> np.ndarray:
    """[N] biased score of the k-th selected expert minus the best excluded one inside the kept groups."""
    s = _biased(sample.scores[j], sample.bias[j])
    desc = -np.sort(-np.where(_kept(s, sample.n_group, sample.topk_group), s, -np.inf), axis=-1)
    return desc[:, sample.k - 1] - desc[:, sample.k]


def footprint(sample: WindowSample, delta) -> dict:
    """{"share", "per_layer"}: per MoE layer the share of sample tokens whose top-k set the patch changes."""
    delta = np.asarray(delta.detach().cpu().numpy() if hasattr(delta, "detach") else delta, np.float32)
    if delta.shape != sample.bias.shape:
        raise ValueError(f"delta shape {delta.shape}, sample bias {sample.bias.shape}")
    per_layer = np.zeros(len(sample.moe_layers))
    for j in np.flatnonzero((delta != 0).any(axis=1)):
        patched = _selected(_biased(sample.scores[j], sample.bias[j], delta[j]), sample.k, sample.n_group,
                            sample.topk_group)
        per_layer[j] = (patched != _stock(sample, j)).any(axis=1).mean()
    return {"share": float(per_layer.mean()), "per_layer": [float(x) for x in per_layer]}


def dose_curve(sample: WindowSample, layer: int, expert: int, units: Sequence[int], unit: float = UNIT) -> dict:
    """One expert of decoder layer `layer` moved by u * unit for each u: shares added and removed per dose.

    added: of the tokens where the expert was absent, the share where it enters. removed: of the tokens where it
    was present, the share where it leaves. changed: the share of all tokens whose top-k set changes.
    """
    j = sample.index(layer)
    if not 0 <= expert < sample.bias.shape[1]:
        raise ValueError(f"expert {expert} outside 0 to {sample.bias.shape[1] - 1}")
    stock = _stock(sample, j)
    present = stock[:, expert]
    added, removed, changed = [], [], []
    for u in units:
        delta = np.zeros(sample.bias.shape[1], np.float32)
        delta[expert] = np.float32(u * unit)
        mask = _selected(_biased(sample.scores[j], sample.bias[j], delta), sample.k, sample.n_group,
                         sample.topk_group)
        now = mask[:, expert]
        added.append(float((now & ~present).sum() / max(int((~present).sum()), 1)))
        removed.append(float((~now & present).sum() / max(int(present.sum()), 1)))
        changed.append(float((mask != stock).any(axis=1).mean()))
    return {"layer": int(layer), "expert": int(expert), "units": [int(u) for u in units], "added": added,
            "removed": removed, "changed": changed, "present": float(present.mean()), "n_tokens": sample.n_tokens}


def dose_shares(sample: WindowSample, j: int, units: Sequence[int], unit: float = UNIT) -> tuple[np.ndarray, ...]:
    """Every expert of MoE index j moved alone by u * unit: (added [E, U], removed [E, U], present [E]).

    Moving one expert changes only its own score and its group's score, so its membership follows from the stock
    state: the kept groups, the 8th and 9th scores among them and each group's top three. Exact up to ties.
    """
    s = _biased(sample.scores[j], sample.bias[j])
    n, e = s.shape
    k, ng, tg = sample.k, sample.n_group, sample.topk_group
    size = e // ng
    group = np.arange(e) // size
    grouped = s.reshape(n, ng, size)
    top3 = -np.sort(-grouped, axis=-1)[..., :3]  # [N, G, 3]
    g = top3[..., 0] + top3[..., 1]  # each group's top-2 sum, as the router adds them
    order = np.argsort(-g, axis=-1, kind="stable")
    kept_g = np.zeros((n, ng), bool)
    np.put_along_axis(kept_g, order[:, :tg], True, axis=1)
    g_sorted = np.take_along_axis(g, order, axis=1)
    g_last_kept, g_first_out = g_sorted[:, tg - 1], g_sorted[:, tg]
    last_kept = order[:, tg - 1]
    kept = np.repeat(kept_g, size, axis=1)
    stock = _stock(sample, j)
    in_kept = np.where(kept, s, -np.inf)
    desc = -np.sort(-in_kept, axis=-1)
    t_k, t_next = desc[:, k - 1], desc[:, k]  # the k-th selected score and the best excluded one in kept groups
    rank = np.argsort(np.argsort(-grouped, axis=-1, kind="stable"), axis=-1, kind="stable").reshape(n, e)
    m1, m2, m3 = (top3[:, :, i][:, group] for i in range(3))  # [N, E]: the expert's group's top three
    bias = np.asarray(sample.bias[j], np.float32)
    present = stock.mean(axis=0)
    added = np.zeros((e, len(units)))
    removed = np.zeros((e, len(units)))
    for c, u in enumerate(units):
        x = np.asarray(sample.scores[j], np.float32) + (bias + np.float32(u * unit))  # the moved score, per expert
        if u > 0:
            g_new = np.where(rank == 0, x + m2, m1 + np.maximum(m2, x))  # rank 1: m1 + x; rank >= 2: m1 + max
            enters = ~kept & (g_new > g_last_kept[:, None])
            now = (kept & (x > t_k[:, None]))
            tok, ex = np.nonzero(enters & ~stock)
            if len(tok):
                # The moved group replaces the last kept group: count the experts that still beat x there.
                xs = x[tok, ex]
                above_kept = (in_kept[tok] > xs[:, None]).sum(axis=1)
                lost = grouped[tok, last_kept[tok]]
                above_lost = (lost > xs[:, None]).sum(axis=1)
                above_own = (grouped[tok, group[ex]] > xs[:, None]).sum(axis=1)
                ok = above_kept - above_lost + above_own < k
                now[tok, ex] |= ok
            gain = now & ~stock
            added[:, c] = gain.sum(axis=0) / np.maximum((~stock).sum(axis=0), 1)
        elif u < 0:
            g_new = np.where(rank == 0, m2 + np.maximum(x, m3),
                             np.where(rank == 1, m1 + np.maximum(x, m3), g[:, group]))
            # A kept group leaves when its new score falls below the best group outside.
            leaves = kept & (g_new < g_first_out[:, None])
            gone = stock & (leaves | (x < t_next[:, None]))
            removed[:, c] = gone.sum(axis=0) / np.maximum(stock.sum(axis=0), 1)
    return added, removed, present
