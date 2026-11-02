"""Anchor helpers shared by both sources."""

from __future__ import annotations

from ouroboros.render.sequence import Sequence
from ouroboros.render.template import BeamTemplate

WINDOW = 512


def window_end(seq: Sequence, f: int, size: int = WINDOW) -> int | None:
    """Index of the `size`-th token Beam generated after f, or of the last one when fewer exist."""
    last = None
    count = 0
    for i in range(f + 1, len(seq.ids)):
        if seq.gen_mask[i]:
            last = i
            count += 1
            if count == size:
                break
    return last


def snippet(tpl: BeamTemplate, ids: list[int], i: int | None, before: int = 24, after: int = 16) -> str | None:
    """Decoded text around token i, with the anchor token marked as [[...]]."""
    if i is None:
        return None
    head = tpl.decode(ids[max(0, i - before): i])
    return head + "[[" + tpl.decode(ids[i: i + 1]) + "]]" + tpl.decode(ids[i + 1: i + 1 + after])
