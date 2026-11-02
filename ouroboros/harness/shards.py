"""Rendered episodes into shards under a host-RAM budget, and the capture plan of a shard.

P1 writes one parquet row per episode: episode_id, token_ids, gen_mask, and the anchor token indices f, d, s,
window_end and s_call (null where they do not apply), all positions within the episode. gen_mask marks the tokens
Beam sampled. f is the last token of the fork tool result and window_end the 512th token Beam wrote after it, so the
primary window is f + 1 through window_end, inclusive. s_call is the <|start|> of the finish call that holds s.
Episodes stay whole inside a shard.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ouroboros.harness.capture import CapturePlan
from ouroboros.harness.stream import Layout

TAMPER_SPAN = 16  # Stage B scores the first 16 tokens of the tamper call
PRE_SLOT = 512  # H3 scores the 512 tokens before the status slot
HIDDEN_BYTES = 2  # BF16 hidden states in host memory
ANCHORS = ("f", "d", "s", "window_end", "s_call")


@dataclass(frozen=True)
class Episode:
    key: str
    token_ids: np.ndarray  # [n] int64
    gen_mask: np.ndarray  # [n] bool, True where Beam sampled the token
    f: int | None = None
    d: int | None = None
    s: int | None = None
    window_end: int | None = None
    s_call: int | None = None
    full_length: int | None = None  # length before truncation
    slot_ids: tuple[int, ...] | None = None  # status word scored from s; None scores the single token at s

    def __post_init__(self):
        if len(self.gen_mask) != len(self.token_ids):
            raise ValueError(f"{self.key}: gen_mask and token_ids differ in length")
        if self.full_length is None:
            object.__setattr__(self, "full_length", len(self.token_ids))

    @property
    def slot_len(self) -> int:
        return len(self.slot_ids) if self.slot_ids else 1

    def pre_slot_start(self) -> int:
        """Start of the H3 window: PRE_SLOT tokens, or PRE_SLOT Beam-written tokens, before s and before s_call.

        The capture covers both readings of "the 512 tokens before s", and the same window ending at s_call, which
        leaves out a BLOCKED reason Beam wrote before s; gen_mask picks the Beam-written rows.
        """
        def back(end: int) -> int:
            written = np.flatnonzero(self.gen_mask[:end])
            start = written[-PRE_SLOT] if len(written) >= PRE_SLOT else (written[0] if len(written) else end)
            return min(end - PRE_SLOT, int(start))

        return max(0, min(back(end) for end in (self.s, self.s_call) if end is not None))

    def windows(self) -> list[tuple[int, int]]:
        """Ranges [a, b) captured in full: the fork token f through window_end, and the H3 window before s."""
        n = len(self.token_ids)
        spans = []
        if self.f is not None and self.window_end is not None:
            spans.append((self.f, self.window_end + 1))
        if self.s is not None:
            spans.append((self.pre_slot_start(), self.s))
        merged: list[tuple[int, int]] = []
        for a, b in sorted((max(a, 0), min(b, n)) for a, b in spans):
            if b <= a:
                continue
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        return merged

    def targets(self) -> np.ndarray:
        """Tokens scored by lm_head: every token Beam sampled, the tamper-call span from d, and the status word at s."""
        want = self.gen_mask.astype(bool).copy()
        if self.d is not None:
            want[self.d:self.d + TAMPER_SPAN] = True
        if self.s is not None:
            want[self.s:self.s + self.slot_len] = True
        want[:1] = False  # the first token has no prediction
        return np.flatnonzero(want).astype(np.int64)

    def last_needed(self) -> int | None:
        """End of the last anchor-derived span, or None when the episode has no anchors."""
        ends = [x for x in (None if self.window_end is None else self.window_end + 1,
                            None if self.d is None else self.d + TAMPER_SPAN,
                            None if self.s is None else self.s + self.slot_len) if x is not None]
        return max(ends) if ends else None

    def with_slot_word(self, word: Sequence[int]) -> Episode:
        """Score the status word `word` (token ids) from s, teacher-forced on the stock prefix.

        When Beam wrote another word at s, the episode ends with `word` in its place: those tokens are not
        Beam's, so their gen_mask is False. A word of several tokens (BLOCK, ED) then gets its full log-prob.
        """
        word = tuple(int(t) for t in word)
        if self.s is None:
            return self
        if tuple(self.token_ids[self.s:self.s + len(word)].tolist()) == word:
            return replace(self, slot_ids=word)
        return replace(self, token_ids=np.concatenate([self.token_ids[:self.s], word]).astype(np.int64),
                       gen_mask=np.concatenate([self.gen_mask[:self.s], np.zeros(len(word), bool)]), slot_ids=word)

    def truncated(self, margin: int) -> Episode:
        """Drop tokens after the last needed position plus `margin`; the model is causal, so nothing before changes."""
        last = self.last_needed()
        if last is None or last + margin >= len(self.token_ids):
            return self
        cut = last + margin
        return replace(self, token_ids=self.token_ids[:cut], gen_mask=self.gen_mask[:cut])


def _lists(table, name: str, dtype) -> list[np.ndarray]:
    col = table.column(name).combine_chunks()
    values = col.flatten().to_numpy(zero_copy_only=False).astype(dtype)
    offsets = col.offsets.to_numpy()
    offsets = offsets - offsets[0]
    return [values[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def load_rendered(paths: Sequence[str | Path]) -> list[Episode]:
    """Episodes from P1's parquet files, in file order then row order."""
    episodes = []
    for path in map(Path, paths):
        table = pq.read_table(path)
        n = table.num_rows
        ids = _lists(table, "token_ids", np.int64)
        gen = _lists(table, "gen_mask", bool)
        keys = (table.column("episode_id").to_pylist() if "episode_id" in table.column_names
                else [f"{path.stem}:{r}" for r in range(n)])
        anchors = {a: table.column(a).to_pylist() if a in table.column_names else [None] * n for a in ANCHORS}
        for r in range(n):
            episodes.append(Episode(str(keys[r]), ids[r], gen[r], **{a: _int(anchors[a][r]) for a in ANCHORS}))
    seen = set()
    for e in episodes:
        if e.key in seen:
            raise ValueError(f"duplicate episode key {e.key}")
        seen.add(e.key)
    return episodes


def _int(x) -> int | None:
    return None if x is None else int(x)


def budget_tokens(shard_gb: float, hidden_size: int) -> int:
    """Tokens whose hidden states fit in `shard_gb` gigabytes of host memory."""
    return int(shard_gb * 1e9) // (hidden_size * HIDDEN_BYTES)


def split(lengths: Sequence[int], max_tokens: int) -> list[list[int]]:
    """Consecutive episodes into shards of at most max_tokens; a longer episode gets a shard alone."""
    shards: list[list[int]] = []
    used = 0
    for e, n in enumerate(lengths):
        if not shards or (shards[-1] and used + n > max_tokens):
            shards.append([])
            used = 0
        shards[-1].append(e)
        used += n
    return shards


def build(episodes: Sequence[Episode], candidates: Sequence[int], tap_layers: Sequence[int],
          top_k: int, with_windows: bool = True) -> tuple[Layout, CapturePlan]:
    """Pack episodes and list their window rows (none without `with_windows`) and target tokens in shard coordinates."""
    layout = Layout.pack([e.token_ids for e in episodes])
    starts = layout.offsets[:-1]
    spans = [(start, a, b) for e, start in zip(episodes, starts) for a, b in e.windows()] if with_windows else []
    windows = [start + np.arange(a, b) for start, a, b in spans]
    targets = [start + e.targets() for e, start in zip(episodes, starts)]
    plan = CapturePlan(
        windows=np.concatenate(windows).astype(np.int64) if windows else np.zeros(0, np.int64),
        targets=np.concatenate(targets).astype(np.int64) if targets else np.zeros(0, np.int64),
        candidates=np.asarray(candidates, dtype=np.int64),
        tap_layers=tuple(tap_layers),
        top_k=top_k,
    )
    return layout, plan


def episode_index(episodes: Sequence[Episode], layout: Layout, plan: CapturePlan) -> list[dict]:
    """Per episode: its token range in the shard and its row ranges in the window and target arrays."""
    out = []
    for e, ep in enumerate(episodes):
        lo, hi = int(layout.offsets[e]), int(layout.offsets[e + 1])
        w0, w1 = np.searchsorted(plan.windows, [lo, hi])
        t0, t1 = np.searchsorted(plan.targets, [lo, hi])
        out.append({
            "key": ep.key,
            "start": lo,
            "length": hi - lo,
            "full_length": ep.full_length,
            **{a: getattr(ep, a) for a in ANCHORS},
            "slot_ids": None if ep.slot_ids is None else list(ep.slot_ids),
            "windows": [list(w) for w in ep.windows()] if w1 > w0 else [],  # the spans this shard captured
            "window_rows": [int(w0), int(w1)],
            "target_rows": [int(t0), int(t1)],
        })
    return out
