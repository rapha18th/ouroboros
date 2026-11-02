"""Beam's chat template and tokenizer, applied the way the serving route applied them.

Rendering goes through Transformers' own Jinja environment (trim_blocks, lstrip_blocks, its non-escaping `tojson`
and the `{% generation %}` tracker), which is what vLLM-style servers call. Encoding is plain `tokenizer.encode` on
the rendered text: control-token strings inside message content become control tokens. Billed prompt counts on
Claimed vs Done confirm both choices (see `build.billing_gate`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer
from transformers.utils.chat_template_utils import render_jinja_template

VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "beam"

# Harmony control tokens, ids 151669-151680.
HARMONY_TOKENS = {
    "<|start|>": 151669,
    "<|end|>": 151670,
    "<|message|>": 151671,
    "<|channel|>": 151672,
    "<|constrain|>": 151673,
    "<|return|>": 151674,
    "<|call|>": 151675,
    "<|value_unk|>": 151676,
    "<|value_zero_0|>": 151677,
    "<|value_one_0|>": 151678,
    '<rfl:parameter name="': 151679,
    "\n</rfl:parameter>": 151680,
}

# Effort label -> the integer the template prints. `none` is not a template label: the route accepts it and the
# billed prompt counts show it renders as a one-token effort. Among the template's labels only `minimal` (8) is one
# token; a raw single digit such as 0 would also fit, so this mapping is the best reading, not a measurement.
EFFORT_INT = {"minimal": 8, "none": 8, "low": 32, "medium": 64, "high": 128, "xhigh": 256, "max": 512}


@dataclass(frozen=True)
class Rendered:
    text: str
    gen_spans: list[tuple[int, int]]  # character spans of each assistant turn's `{% generation %}` block


@dataclass(frozen=True)
class Encoded:
    ids: list[int]
    offsets: list[tuple[int, int]]  # character span of each token in the rendered text


class BeamTemplate:
    def __init__(self, vendor: Path = VENDOR):
        self.source = (vendor / "chat_template.jinja").read_text(encoding="utf-8")
        self.tokenizer = Tokenizer.from_file(str(vendor / "tokenizer.json"))

    def render(self, messages: list[dict], tools: list[dict], effort: int,
               add_generation_prompt: bool = False) -> Rendered:
        texts, spans = render_jinja_template(
            [messages], tools=tools, chat_template=self.source, return_assistant_tokens_mask=True,
            add_generation_prompt=add_generation_prompt, reasoning_effort=effort,
        )
        return Rendered(texts[0], [tuple(s) for s in spans[0]])

    def encode(self, text: str) -> Encoded:
        enc = self.tokenizer.encode(text, add_special_tokens=False)
        return Encoded(enc.ids, enc.offsets)

    def encode_many(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tokenizer.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)
