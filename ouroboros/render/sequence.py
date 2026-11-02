"""One episode -> one token sequence: the final call's prompt plus Beam's final output.

Message and tool-call boundaries come from rendering prefixes of the conversation, never from searching the text,
because Beam's malformed calls sometimes carry Harmony tokens inside argument values.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from ouroboros.render.template import BeamTemplate, Encoded

_STUB = [{"role": "user", "content": "."}]
_END, _CALL = "<|end|>", "<|call|>"


@dataclass
class Turn:
    msg_index: int
    start: int  # index of the first token Beam generated: the generation prompt is ids[:start]
    end: int  # one past the last generated token (its <|call|> or <|return|>)
    call_starts: list[int] = field(default_factory=list)  # the <|start|> opening each tool call


@dataclass
class Sequence:
    ids: list[int]
    gen_mask: list[int]
    text: str
    offsets: list[tuple[int, int]]
    msg_spans: list[tuple[int, int] | None]  # token span per message; None for the merged leading system block
    turns: list[Turn]
    prefix_failures: list[dict]
    gen_prompt_lens: list[int]  # independently rendered generation-prompt length per turn
    dropped_trailing: int  # messages after the last assistant turn, which no call ever sent
    _starts: list[int] = field(init=False, repr=False)

    def __post_init__(self):
        self._starts = [o[0] for o in self.offsets]

    def boundary(self, char: int) -> int:
        """Token index starting exactly at `char`; raises if `char` falls inside a token."""
        i = bisect.bisect_left(self._starts, char)
        if i < len(self.ids) and self._starts[i] != char:
            raise ValueError(f"char {char} is not a token boundary")
        return i


def _leading_instructions(messages: list[dict]) -> int:
    n = 0
    while n < len(messages) and messages[n]["role"] in ("system", "developer"):
        n += 1
    return n


def _call_char_starts(tpl: BeamTemplate, msg: dict, msg_text: str, tools, effort) -> list[int]:
    """Offsets of each tool call's `<|start|>` inside the rendered assistant message."""
    calls = msg.get("tool_calls") or []
    if not calls:
        return []
    head = "<|start|>assistant<|channel|>analysis<|message|>" + (msg.get("reasoning_content") or "") + _END
    if not msg_text.startswith(head):
        raise ValueError("assistant message does not open with its analysis block")
    starts = [len(head)]
    stub = len(tpl.render(_STUB, tools, effort).text)
    for j in range(1, len(calls)):
        part = tpl.render(_STUB + [{**msg, "tool_calls": calls[:j]}], tools, effort).text[stub:]
        start = len(part) - len(_CALL) + len(_END)
        if part[: -len(_CALL)] != msg_text[: start - len(_END)]:
            raise ValueError("partial rendering of a parallel call is not a prefix of the message")
        starts.append(start)
    for j, s in enumerate(starts):
        name = (calls[j].get("function") or calls[j])["name"]
        if not msg_text.startswith("<|start|>assistant to=" + name, s):
            raise ValueError(f"tool call {j} does not start where expected")
    return starts


def build_sequence(tpl: BeamTemplate, messages: list[dict], tools: list[dict], effort: int,
                   check_prefixes: bool = True) -> Sequence:
    last = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=None)
    kept = messages if last is None else messages[: last + 1]
    dropped = len(messages) - len(kept)
    full = tpl.render(kept, tools, effort, add_generation_prompt=last is None)
    enc: Encoded = tpl.encode(full.text)
    seq = Sequence(enc.ids, [0] * len(enc.ids), full.text, enc.offsets, [], [], [], [], dropped)

    lead = _leading_instructions(kept)
    if lead == 0:
        raise ValueError("both harnesses open with a system message; this conversation does not")
    bounds = [len(tpl.render(kept[:i], tools, effort).text) for i in range(lead, len(kept) + 1)]
    seq.msg_spans = [None] * lead + [(seq.boundary(a), seq.boundary(b)) for a, b in zip(bounds[:-1], bounds[1:])]

    assistant_idx = [i for i, m in enumerate(kept) if m["role"] == "assistant"]
    if len(full.gen_spans) != len(assistant_idx):
        raise ValueError("generation spans do not match assistant turns")
    for i, (cs, ce) in zip(assistant_idx, full.gen_spans):
        start, end = seq.boundary(cs), seq.boundary(ce)
        for t in range(start, end):
            seq.gen_mask[t] = 1
        m_start, m_end = bounds[i - lead], bounds[i - lead + 1]
        calls = _call_char_starts(tpl, kept[i], full.text[m_start:m_end], tools, effort)
        seq.turns.append(Turn(i, start, end, [seq.boundary(m_start + c) for c in calls]))

    if check_prefixes:
        _check_prefixes(tpl, seq, kept, tools, effort)
    return seq


def _check_prefixes(tpl: BeamTemplate, seq: Sequence, kept: list[dict], tools, effort) -> None:
    """Every turn's prompt, rendered on its own with a generation prompt, must be a prefix of the sequence."""
    prompts = [tpl.render(kept[: t.msg_index], tools, effort, add_generation_prompt=True).text for t in seq.turns]
    for k, (turn, text, ids) in enumerate(zip(seq.turns, prompts, tpl.encode_many(prompts))):
        seq.gen_prompt_lens.append(len(ids))
        if not seq.text.startswith(text):
            at = next((i for i, (a, b) in enumerate(zip(text, seq.text)) if a != b), min(len(text), len(seq.text)))
            seq.prefix_failures.append({"turn": k, "msg_index": turn.msg_index, "reason": "text differs",
                                        "char": at, "rendered": text[at: at + 80], "sequence": seq.text[at: at + 80]})
        elif ids != seq.ids[: len(ids)]:
            seq.prefix_failures.append({"turn": k, "msg_index": turn.msg_index, "reason": "tokens differ"})
        elif len(ids) != turn.start:
            reason = f"generation prompt ends at {len(ids)}, generation starts at {turn.start}"
            seq.prefix_failures.append({"turn": k, "msg_index": turn.msg_index, "reason": reason})
