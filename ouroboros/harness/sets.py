"""Stage B episode lists: extra target spans, a token cap, selection, reordering, set files and the vouched hashes.

A Stage B set is a rendered parquet plus an optional set file (format ouroboros-set/1) that lists episodes by key in
layout order, each with its target spans. StageBEpisode wraps shards.Episode: its targets add every span position to
the tokens run.py scores, and its truncation keeps every span. shards.py stays as the gate vouched for it, so a
StageBEpisode with no spans scores and truncates exactly as a run.py episode does.

A layout group can hold another group's episodes in a second order (the reordered guard). align_tokens and
align_targets map its tokens and target rows onto the first group, so its stock can be compared with the first stock.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np

from ouroboros import stageb
from ouroboros.harness import shards
from ouroboros.harness.capture import CapturePlan
from ouroboros.harness.stream import Layout

SET_FORMAT = "ouroboros-set/1"
HARNESS = Path(__file__).resolve().parent


@dataclass(frozen=True)
class StageBEpisode(shards.Episode):
    """An episode with extra target spans and the token cap applied to it."""

    spans: tuple[tuple[str, int, int], ...] = ()  # (name, a, b), episode-local, b exclusive
    cap: int | None = None  # the cap capped() applied, for the record

    def base_targets(self) -> np.ndarray:
        """Exactly the tokens run.py scores for this episode."""
        return shards.Episode.targets(self)

    def targets(self) -> np.ndarray:
        """run.py's targets plus every span position inside the episode; never the first token."""
        n = len(self.token_ids)
        want = np.zeros(n, bool)
        want[self.base_targets()] = True
        for _, a, b in self.spans:
            want[a:min(b, n)] = True
        want[:1] = False
        return np.flatnonzero(want).astype(np.int64)

    def last_needed(self) -> int | None:
        """End of the last anchor-derived span or extra span, so truncation keeps every span."""
        base = shards.Episode.last_needed(self)
        ends = [b for _, _, b in self.spans] + ([] if base is None else [base])
        return max(ends) if ends else None

    def capped(self, n: int) -> StageBEpisode:
        """Cut an episode with no anchor and no span to n tokens; any other episode is returned as it is."""
        if n < 1:
            raise ValueError(f"{self.key}: a cap needs at least one token, got {n}")
        if self.spans or any(getattr(self, a) is not None for a in shards.ANCHORS) or len(self.token_ids) <= n:
            return self
        return replace(self, token_ids=self.token_ids[:n], gen_mask=self.gen_mask[:n], cap=n)

    def span(self, name: str) -> tuple[int, int] | None:
        for span_name, a, b in self.spans:
            if span_name == name:
                return a, b
        return None


def wrap(episode: shards.Episode, spans: Mapping[str, Sequence[int]] | None = None,
         cap: int | None = None) -> StageBEpisode:
    """A StageBEpisode with the episode's fields and the given spans (a StageBEpisode keeps its own when None)."""
    base = {f.name: getattr(episode, f.name) for f in fields(shards.Episode)}
    if spans is None and isinstance(episode, StageBEpisode):
        out = episode.spans
    else:
        n = len(episode.token_ids)
        out = []
        for name, span in (spans or {}).items():
            if span is None:
                continue
            a, b = (int(x) for x in span)
            if not 0 <= a < b <= n:
                raise ValueError(f"{episode.key}: span {name} [{a}, {b}) lies outside its {n} tokens")
            out.append((str(name), a, b))
        out = tuple(out)
    if cap is None and isinstance(episode, StageBEpisode):
        cap = episode.cap
    return StageBEpisode(**base, spans=out, cap=cap)


def select(episodes: Sequence[shards.Episode], keys: Sequence[str]) -> list[shards.Episode]:
    """The episodes named by `keys`, in that order. KeyError on an unknown or repeated key."""
    by_key = {e.key: e for e in episodes}
    seen: set[str] = set()
    out = []
    for key in keys:
        if key in seen:
            raise KeyError(f"episode {key} is listed twice")
        if key not in by_key:
            raise KeyError(f"no episode {key}")
        seen.add(key)
        out.append(by_key[key])
    return out


def permute(episodes: Sequence[shards.Episode], seed: int = stageb.SEED) -> list[shards.Episode]:
    """The same episodes in a seeded order: the packing floor's second layout."""
    return [episodes[i] for i in np.random.default_rng(seed).permutation(len(episodes))]


def _partners(eps_ref: Sequence[shards.Episode], eps: Sequence[shards.Episode]) -> np.ndarray:
    """Index in eps_ref of each episode of eps, matched by key."""
    ref = {e.key: i for i, e in enumerate(eps_ref)}
    keys = [e.key for e in eps]
    if len(ref) != len(eps_ref) or len(set(keys)) != len(keys) or set(keys) != set(ref):
        raise ValueError("the two episode lists differ as sets of keys")
    return np.array([ref[k] for k in keys], dtype=np.int64)


def align_tokens(eps_ref: Sequence[shards.Episode], layout_ref: Layout, eps: Sequence[shards.Episode],
                 layout: Layout) -> np.ndarray:
    """[T] the token of the reference layout that holds each token of `layout`."""
    idx = _partners(eps_ref, eps)
    lengths = np.diff(layout.offsets)
    if not np.array_equal(np.diff(layout_ref.offsets)[idx], lengths):
        raise ValueError("an episode differs in length between the two layouts")
    out = np.repeat(layout_ref.offsets[:-1][idx] - layout.offsets[:-1], lengths) + np.arange(layout.n_tokens)
    out = out.astype(np.int64)
    if not np.array_equal(layout_ref.token_ids[out], layout.token_ids):
        raise ValueError("an episode holds other tokens in the two layouts")
    return out


def align_targets(eps_ref: Sequence[shards.Episode], layout_ref: Layout, plan_ref: CapturePlan,
                  eps: Sequence[shards.Episode], layout: Layout, plan: CapturePlan) -> np.ndarray:
    """[N] the row of the reference targets that scores the same token as each row of `plan.targets`."""
    want = align_tokens(eps_ref, layout_ref, eps, layout)[plan.targets]
    rows = np.searchsorted(plan_ref.targets, want)
    found = rows < len(plan_ref.targets)
    found[found] = plan_ref.targets[rows[found]] == want[found]
    if not found.all():
        raise ValueError(f"{int((~found).sum())} targets have no partner among the reference targets")
    return rows.astype(np.int64)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


@functools.lru_cache(maxsize=4)
def _rendered(path: str, sha256: str) -> tuple[shards.Episode, ...]:
    """A rendered parquet, cached by path and content: tripwire.parquet serves two held-out sets."""
    return tuple(shards.load_rendered([path]))


def load_set(parquet: str | Path, episodes_file: str | Path | None,
             name: str) -> tuple[list[StageBEpisode], dict]:
    """A set's episodes in layout order, wrapped with their spans, and where they came from.

    Without a set file every parquet row is kept, in order, with no spans. A set file must be labelled, carry
    format ouroboros-set/1 and name this parquet's sha256; ValueError otherwise.
    """
    parquet = Path(parquet)
    digest = sha256_file(parquet)
    rendered = _rendered(str(parquet.resolve()), digest)
    info = {"name": name, "parquet": str(parquet), "parquet_sha256": digest, "episodes_file": None,
            "episodes_sha256": None, "records": None}
    if episodes_file is None:
        return [wrap(e) for e in rendered], info
    episodes_file = Path(episodes_file)
    spec = json.loads(episodes_file.read_text(encoding="utf-8"))
    if spec.get("format") != SET_FORMAT:
        raise ValueError(f"{episodes_file}: format {spec.get('format')!r}, expected {SET_FORMAT}")
    if spec.get("exploratory") is not True:
        raise ValueError(f"{episodes_file}: the exploratory label is missing")
    if spec.get("parquet_sha256") != digest:
        raise ValueError(f"{episodes_file} lists episodes of a parquet with sha256 {spec.get('parquet_sha256')}, "
                         f"but {parquet} has {digest}")
    records = list(spec.get("episodes", []))
    if spec.get("n_episodes", len(records)) != len(records):
        raise ValueError(f"{episodes_file}: n_episodes {spec.get('n_episodes')} but {len(records)} records")
    try:
        chosen = select(rendered, [r["key"] for r in records])
    except KeyError as e:
        raise ValueError(f"{episodes_file}: {e.args[0]} in {parquet.name}") from None
    episodes = [wrap(e, r.get("spans")) for e, r in zip(chosen, records)]
    info.update(episodes_file=str(episodes_file), episodes_sha256=sha256_file(episodes_file), records=records)
    return episodes, info


def vouched_hashes(harness_dir: Path | None = None) -> dict[str, str]:
    """sha256 of each file the gate vouched for, as it stands now."""
    harness_dir = HARNESS if harness_dir is None else Path(harness_dir)
    return {name: sha256_file(harness_dir / name) for name in stageb.VOUCHED_SHA256}


def check_vouched() -> dict[str, str]:
    """The vouched hashes; RuntimeError naming every file that differs from stageb.VOUCHED_SHA256."""
    now = {}
    for name in stageb.VOUCHED_SHA256:
        try:
            now[name] = sha256_file(HARNESS / name)
        except FileNotFoundError:
            now[name] = None
    changed = [f"{name} ({now[name] or 'missing'}, vouched {want})" for name, want in stageb.VOUCHED_SHA256.items()
               if now[name] != want]
    if changed:
        raise RuntimeError("vouched files changed since the gate: " + "; ".join(changed))
    return now
