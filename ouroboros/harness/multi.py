"""One weight stream that scores the stock model and many router-bias patches.

A layout group is one episode list in one order: search_dev, a CvD set, the held-out tamper set, the guard, or the
guard in a second order. Each group runs its own zero-patch stock trunk. A variant is one patch on one group. It
shares the trunk up to its fork layer, its first patched MoE layer, where the trunk's hidden state is copied into a
host buffer of its own. Each decoder layer is read and built once per pass, and every call goes through
Streamer.apply and patched() unchanged, so a variant computes exactly what a run.py pass with its patch computes.

head_pass scores every variant's targets beside its group's stock: nll, top-k, candidate log-probs, top-1 agreement
and exact full-vocabulary KL in both directions. It scores run.py's own targets first, in run.py's batches, so the
stock rows can match a run.py pass bit for bit on the GPU as well.

plan_pass refuses a pass whose hidden states would not fit in host memory, before any weight is read.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ouroboros import stageb
from ouroboros.harness import bank, sets, shards
from ouroboros.harness.capture import CapturePlan, host_tensor
from ouroboros.harness.patch import first_patched_layer
from ouroboros.harness.stream import Layout, Streamer

log = logging.getLogger(__name__)

# Host memory allocated after the plan: two pinned prefetch buffers sized for a 10.18 GB MoE layer, and the globals
# (embeddings and lm_head at vocab 151,936). RESERVE_GB covers the process, the OS and the page cache.
PREFETCH_GB, GLOBALS_GB, RESERVE_GB, HOST_GB = 20.5, 3.1, 25.0, 120.0
STOCK, TWIN = "stock", "twin"
KL_ROWS = 256  # rows per float64 KL block; a full-vocabulary float64 row costs 1.2 MB


class PlanRefused(ValueError):
    """A pass whose hidden states would not fit in host memory."""


@dataclass
class LayoutGroup:
    """One episode list in one order, packed once, with the targets every variant of the group scores."""

    name: str
    episodes: list[shards.Episode]
    layout: Layout
    plan: CapturePlan  # windows empty, tap_layers (), the episodes' targets, candidates, top_k
    base_rows: np.ndarray  # rows of plan.targets that base_targets() selects (head block A)
    reference: str | None = None  # group whose stock this group's stock is compared with
    meta: dict = field(default_factory=dict)  # set name, sha256s, reorder seed

    @classmethod
    def build(cls, name: str, episodes, candidates: Sequence[int], top_k: int = 16, reference: str | None = None,
              meta: dict | None = None) -> LayoutGroup:
        episodes = list(episodes)
        keys = [e.key for e in episodes]
        if len(set(keys)) != len(keys):
            raise ValueError(f"group {name} lists an episode twice")
        layout, plan = shards.build(episodes, candidates, (), top_k, with_windows=False)
        base = [start + getattr(e, "base_targets", e.targets)() for e, start in zip(episodes, layout.offsets[:-1])]
        base = np.concatenate(base).astype(np.int64) if base else np.zeros(0, np.int64)
        rows = np.searchsorted(plan.targets, base)
        if len(base) and (rows.max() >= len(plan.targets) or not np.array_equal(plan.targets[rows], base)):
            raise ValueError(f"group {name}: run.py's targets are not a subset of the group's targets")
        return cls(name, episodes, layout, plan, rows.astype(np.int64), reference, dict(meta or {}))

    def hidden_gb(self, hidden_size: int, dtype_bytes: int = 2) -> float:
        return self.layout.n_tokens * hidden_size * dtype_bytes / 1e9


@dataclass(frozen=True)
class Variant:
    """One patch on one group; stock has no delta and runs on the group's trunk."""

    name: str  # "stock", "twin" or a bank name
    group: str
    delta: torch.Tensor | None = field(repr=False, compare=False)  # FP32 [n_moe, E]; None for stock
    sha256: str | None
    fork_layer: int | None  # first decoder layer run on the variant's own hidden state; None for stock
    twin: bool = False

    @property
    def key(self) -> str:
        return f"{self.group}/{self.name}"

    @property
    def is_stock(self) -> bool:
        return self.delta is None

    @classmethod
    def stock(cls, group: str) -> Variant:
        return cls(STOCK, group, None, None, None)

    @classmethod
    def make_twin(cls, group: str, moe_layers: Sequence[int], n_experts: int,
                  layer: int = stageb.TWIN_LAYER) -> Variant:
        """A zero delta forked at the first MoE layer at or above `layer`: it must equal stock bit for bit."""
        fork = next((m for m in moe_layers if m >= layer), None)
        if fork is None:
            raise ValueError(f"no MoE layer at or above decoder layer {layer}")
        delta = torch.zeros(len(moe_layers), n_experts, dtype=torch.float32)
        return cls(TWIN, group, delta, bank.patch_sha256(delta), fork, twin=True)

    @classmethod
    def patch(cls, name: str, group: str, delta, moe_layers: Sequence[int]) -> Variant:
        if name in (STOCK, TWIN):
            raise ValueError(f"patch name {name} is reserved")
        d = bank._tensor(delta)
        if d.dim() != 2 or d.shape[0] != len(moe_layers):
            raise ValueError(f"{name}: delta {list(d.shape)} does not match {len(moe_layers)} MoE layers")
        fork = first_patched_layer(d, list(moe_layers))
        if fork is None:
            raise ValueError(f"{name}: an all-zero delta is the twin; --twin adds it")
        return cls(name, group, d, bank.patch_sha256(d), fork)


def _check(groups: Sequence[LayoutGroup], variants: Sequence[Variant]) -> None:
    seen: set[str] = set()
    for g in groups:
        if g.name in seen:
            raise ValueError(f"group {g.name} appears twice")
        if g.reference is not None and g.reference not in seen:
            raise ValueError(f"group {g.name} references {g.reference}, which must come earlier")
        seen.add(g.name)
    keys = [v.key for v in variants]
    if len(set(keys)) != len(keys):
        raise ValueError("a variant appears twice in one group")
    unknown = {v.group for v in variants} - seen
    if unknown:
        raise ValueError(f"variants name unknown groups {sorted(unknown)}")


def _mem_available_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        log.info("no /proc/meminfo here, so the MemAvailable rule is skipped")
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / 1e9
    return None


def plan_pass(groups: Sequence[LayoutGroup], variants: Sequence[Variant], *, hidden_size: int,
              host_gb: float = HOST_GB, mem_available_gb: float | None = None, dtype_bytes: int = 2) -> dict:
    """Host memory for every hidden-state buffer: one stock trunk per group plus one per patched variant.

    Refuses when that exceeds host_gb, or when it plus the prefetch buffers and globals allocated later exceeds
    MemAvailable minus RESERVE_GB. The reserve also covers what the plan leaves out: the head outputs (about 157
    bytes per target row and variant, 0.5 GB for e2), the weight chunks staged in /dev/shm (workers x chunk size,
    1 to 3 GB), and under routes="ids" the route ids (n_moe x T x k bytes per variant, 4.4 GB for e2).
    """
    _check(groups, variants)
    per_group, total = {}, 0.0
    for g in groups:
        n = sum(1 for v in variants if v.group == g.name and not v.is_stock)
        gb = g.hidden_gb(hidden_size, dtype_bytes) * (1 + n)
        per_group[g.name] = {"n_tokens": g.layout.n_tokens, "n_variants": n, "hidden_gb": gb}
        total += gb
    fixed = PREFETCH_GB + GLOBALS_GB
    if mem_available_gb is None:
        mem_available_gb = _mem_available_gb()
    plan = {"hidden_gb": total, "host_gb": host_gb, "mem_available_gb": mem_available_gb, "fixed_gb": fixed,
            "per_group": per_group}
    if total > host_gb:
        raise PlanRefused(f"hidden states need {total:.1f} GB, over the {host_gb:g} GB host budget")
    if mem_available_gb is not None and total + fixed > mem_available_gb - RESERVE_GB:
        raise PlanRefused(f"hidden states ({total:.1f} GB) plus prefetch buffers and globals ({fixed:.1f} GB) "
                          f"exceed MemAvailable {mem_available_gb:.1f} GB minus the {RESERVE_GB:g} GB reserve")
    return plan


class RouteDiff:
    """Capture for one MoE layer: counts tokens whose top-k expert set differs from a comparator's.

    `record` receives the sorted top-k ids on the device (the stock trunk), `against` holds the comparator's sorted ids
    in this group's token order, and `ids_out` receives the unsorted ids as run.py's capture stores them.
    """

    def __init__(self, record: torch.Tensor | None = None, against: torch.Tensor | None = None,
                 ids_out: np.ndarray | None = None):
        self.record = record
        self.against = against
        self.ids_out = ids_out
        self.changed = 0

    def router(self, j: int, lo: int, hi: int, logits: torch.Tensor, ids: torch.Tensor) -> None:
        if self.ids_out is not None:
            self.ids_out[lo:hi] = ids.to(torch.uint8).cpu().numpy()
        chosen = ids.sort(dim=-1).values.to(torch.uint8)
        if self.record is not None:
            self.record[lo:hi] = chosen
        if self.against is not None:
            self.changed += int((chosen != self.against[lo:hi]).any(dim=-1).sum())

    def layer_output(self, i: int, lo: int, hi: int, out: torch.Tensor) -> None:
        pass


@dataclass
class PassResult:
    hidden: dict[str, torch.Tensor]  # last-layer hidden states by variant key (stock included)
    seconds: dict[str, np.ndarray]  # [num_hidden_layers] seconds per variant key and decoder layer
    route_changed: dict[str, np.ndarray]  # [n_moe] int64 per variant key
    route_ids: dict[str, np.ndarray]  # [n_moe, T, k] uint8 per variant key; only with routes="ids"


def _stock_key(group: str) -> str:
    return f"{group}/{STOCK}"


@torch.no_grad()
def run_pass(streamer: Streamer, groups: Sequence[LayoutGroup], variants: Sequence[Variant], *,
             routes: str = "none", eta_after: int = 3) -> PassResult:
    """Every group's stock trunk and every variant, through each decoder layer read once."""
    if routes not in ("none", "ids"):
        raise ValueError(f"routes must be none or ids, got {routes!r}")
    _check(groups, variants)
    by_name = {g.name: g for g in groups}
    patched = {g.name: [v for v in variants if v.group == g.name and not v.is_stock] for g in groups}
    n_layers = streamer.config.num_hidden_layers
    n_moe, k = len(streamer.moe_layers), streamer.config.num_experts_per_tok
    device, pin = streamer.device, streamer.device.type == "cuda"
    keys = [key for g in groups for key in (_stock_key(g.name), *(v.key for v in patched[g.name]))]
    seconds = {key: np.zeros(n_layers) for key in keys}
    changed = {key: np.zeros(n_moe, np.int64) for key in keys}
    route_ids = {}
    if routes == "ids":
        route_ids = {key: np.zeros((n_moe, by_name[key.rsplit("/", 1)[0]].layout.n_tokens, k), np.uint8)
                     for key in keys}
    aligned = {g.name: torch.from_numpy(sets.align_tokens(by_name[g.reference].episodes, by_name[g.reference].layout,
                                                          g.episodes, g.layout)).to(device)
               for g in groups if g.reference is not None}
    trunk = {g.name: streamer.embed(g.layout) for g in groups}
    hidden: dict[str, torch.Tensor] = {}
    n_tokens = sum(g.layout.n_tokens for g in groups)
    moe_weights: list[float] = []
    moe_stock: dict[str, list[float]] = {g.name: [] for g in groups}
    clock = time.perf_counter()
    for i, tensors in streamer.weights(0, n_layers):
        loaded = time.perf_counter()
        layer = streamer.build(i, tensors)
        j = streamer.moe_index.get(i)
        stock_ids: dict[str, torch.Tensor] = {}
        t_stock = t_variants = 0.0
        active = 0
        for g in groups:
            skey = _stock_key(g.name)
            for v in patched[g.name]:
                if v.fork_layer == i:  # the trunk still holds the input of layer i
                    hidden[v.key] = host_tensor(tuple(trunk[g.name].shape), streamer.dtype, pin=pin)
                    hidden[v.key].copy_(trunk[g.name])
            cap = against = None
            if j is not None:
                if g.reference is not None:
                    against = stock_ids[g.reference][aligned[g.name]]
                record = torch.empty((g.layout.n_tokens, k), dtype=torch.uint8, device=device)
                cap = RouteDiff(record, against, route_ids[skey][j] if routes == "ids" else None)
            t0 = time.perf_counter()
            streamer.apply(layer, i, trunk[g.name], g.layout, None, cap)
            seconds[skey][i] = time.perf_counter() - t0
            t_stock += seconds[skey][i]
            if j is not None:
                stock_ids[g.name] = cap.record
                changed[skey][j] = cap.changed
                moe_stock[g.name].append(seconds[skey][i])
                comparator = cap.record if against is None else against
            for v in patched[g.name]:
                if v.fork_layer > i:  # still on the trunk
                    if j is not None:
                        changed[v.key][j] = changed[skey][j]
                        if routes == "ids":
                            route_ids[v.key][j] = route_ids[skey][j]
                    continue
                vcap = None
                if j is not None:
                    vcap = RouteDiff(None, comparator, route_ids[v.key][j] if routes == "ids" else None)
                t0 = time.perf_counter()
                streamer.apply(layer, i, hidden[v.key], g.layout, v.delta, vcap)
                seconds[v.key][i] = time.perf_counter() - t0
                t_variants += seconds[v.key][i]
                active += 1
                if vcap is not None:
                    changed[v.key][j] = vcap.changed
        del layer, tensors, stock_ids
        streamer.free()
        done = time.perf_counter()
        log.info("layer %d: weights %.1f s, stock %.1f s, variants %.1f s (%d active), %d tokens", i, loaded - clock,
                 t_stock, t_variants, active, n_tokens)
        if j is not None:
            moe_weights.append(loaded - clock)
            if len(moe_weights) == eta_after:
                log.info("eta: %.0f min", _eta(i, streamer, groups, patched, moe_weights, moe_stock) / 60)
        clock = done
    hidden.update({_stock_key(g.name): trunk[g.name] for g in groups})
    return PassResult(hidden, seconds, changed, route_ids)


def _eta(i: int, streamer: Streamer, groups: Sequence[LayoutGroup], patched: Mapping[str, list[Variant]],
         moe_weights: Sequence[float], moe_stock: Mapping[str, Sequence[float]]) -> float:
    """Seconds left after layer i: per remaining layer, the mean weight time plus each group's mean stock time once
    for the trunk and once per variant forked by then."""
    w = float(np.mean(moe_weights))
    left = 0.0
    for layer in range(i + 1, streamer.config.num_hidden_layers):
        left += w
        for g in groups:
            runs = 1 + sum(1 for v in patched[g.name] if v.fork_layer <= layer)
            left += runs * float(np.mean(moe_stock[g.name]))
    return left


def _empty(n: int, top_k: int, n_candidates: int) -> dict[str, np.ndarray]:
    return {
        "nll": np.zeros(n, np.float32),
        "top_ids": np.zeros((n, top_k), np.int32),
        "top_logprobs": np.zeros((n, top_k), np.float32),
        "candidate_logprobs": np.zeros((n, n_candidates), np.float32),
        "kl_stock_v": np.zeros(n, np.float32),
        "kl_v_stock": np.zeros(n, np.float32),
        "top1_agree": np.ones(n, bool),
    }


def _logp(streamer: Streamer, hidden: torch.Tensor, t: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probs of the tokens at t, read from the hidden states at t - 1, as Streamer.head computes them, and the
    argmax of the logits.

    BF16 logits tie often at the top, and topk breaks ties in no stated order, so top-1 is the first maximal logit.
    """
    logits = streamer.logits(hidden[torch.from_numpy(t - 1)].to(streamer.device))
    top1 = logits.argmax(dim=-1)
    return torch.log_softmax(logits, dim=-1), top1


def _score(out: dict[str, np.ndarray], rows: np.ndarray, logp: torch.Tensor, tgt: torch.Tensor, cand: torch.Tensor,
           top_k: int) -> None:
    """Fill nll, top-k and candidate log-probs exactly as Streamer.head does."""
    values, ids = logp.topk(top_k, dim=-1)
    out["nll"][rows] = (-logp.gather(1, tgt[:, None])[:, 0]).cpu().numpy()
    out["top_ids"][rows] = ids.cpu().numpy()
    out["top_logprobs"][rows] = values.cpu().numpy()
    out["candidate_logprobs"][rows] = logp[:, cand].cpu().numpy()


def _compare(out: dict[str, np.ndarray], rows: np.ndarray, logp: torch.Tensor, top1: torch.Tensor,
             stock: torch.Tensor, stock_top1: torch.Tensor) -> None:
    """Top-1 agreement and both KLs against the comparator stock, summed in float64 on the device.

    The FP32 log-probs are renormalized in float64 first. Each row's FP32 log-sum-exp over 151,936 terms is off by
    up to about 1e-5, and that shift would land in the KL whole.
    """
    out["top1_agree"][rows] = (top1 == stock_top1).cpu().numpy()
    kl_sv, kl_vs = [], []
    for a in range(0, len(rows), KL_ROWS):
        s, v = stock[a:a + KL_ROWS].double(), logp[a:a + KL_ROWS].double()
        s, v = s - s.logsumexp(-1, keepdim=True), v - v.logsumexp(-1, keepdim=True)
        d = s - v
        kl_sv.append((s.exp() * d).sum(-1))
        kl_vs.append(-(v.exp() * d).sum(-1))
        del s, v, d
    if kl_sv:
        out["kl_stock_v"][rows] = torch.cat(kl_sv).float().cpu().numpy()
        out["kl_v_stock"][rows] = torch.cat(kl_vs).float().cpu().numpy()


@torch.no_grad()
def head_pass(streamer: Streamer, groups: Sequence[LayoutGroup], variants: Sequence[Variant],
              hidden: Mapping[str, torch.Tensor], *, batch: int = 1024) -> dict[str, dict[str, np.ndarray]]:
    """Per variant key: nll, top_ids, top_logprobs, candidate_logprobs, kl_stock_v, kl_v_stock and top1_agree.

    Block A scores base_rows in batches of `batch`, as Streamer.head batches run.py's targets; block B scores the
    remaining rows after it. Each batch computes the stock log-softmax once. A group with a reference compares every
    variant, its stock included, with the reference stock's log-probs at the aligned rows.
    """
    _check(groups, variants)
    by_name = {g.name: g for g in groups}
    device = streamer.device
    out: dict[str, dict[str, np.ndarray]] = {}
    for g in groups:
        skey = _stock_key(g.name)
        patched = [v for v in variants if v.group == g.name and not v.is_stock]
        top_k, n = g.plan.top_k, len(g.plan.targets)
        cand = torch.as_tensor(g.plan.candidates, dtype=torch.long, device=device)
        res = {key: _empty(n, top_k, len(g.plan.candidates)) for key in (skey, *(v.key for v in patched))}
        ref = None
        if g.reference is not None:
            r = by_name[g.reference]
            align = sets.align_targets(r.episodes, r.layout, r.plan, g.episodes, g.layout, g.plan)
            ref = (hidden[_stock_key(r.name)], r.plan.targets[align])
        rest = np.setdiff1d(np.arange(n, dtype=np.int64), g.base_rows, assume_unique=True)
        done, clock = 0, time.perf_counter()
        for block in (g.base_rows, rest):
            for s in range(0, len(block), batch):
                rows = block[s:s + batch]
                t = g.plan.targets[rows]
                tgt = torch.from_numpy(g.layout.token_ids[t]).to(device)
                stock, stock_top1 = _logp(streamer, hidden[skey], t)
                _score(res[skey], rows, stock, tgt, cand, top_k)
                if ref is None:
                    comp, comp_top1 = stock, stock_top1
                else:
                    comp, comp_top1 = _logp(streamer, ref[0], ref[1][rows])
                    _compare(res[skey], rows, stock, stock_top1, comp, comp_top1)
                for v in patched:
                    logp, top1 = _logp(streamer, hidden[v.key], t)
                    _score(res[v.key], rows, logp, tgt, cand, top_k)
                    _compare(res[v.key], rows, logp, top1, comp, comp_top1)
                    del logp
                del stock, comp
                done += len(rows)
                if time.perf_counter() - clock > 60:
                    log.info("head %s: %d of %d targets, %d variants", g.name, done, n, len(patched) + 1)
                    clock = time.perf_counter()
        out.update(res)
        log.info("head %s: %d targets, %d variants", g.name, n, len(patched) + 1)
    return out
