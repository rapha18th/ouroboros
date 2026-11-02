"""Claimed vs Done trap runs: request reconstruction from transcripts, labels, and the status slot s.

Harness facts (claimed-vs-done `cvd/loop.py`, `cvd/client.py`): history starts with the system prompt and the task
spec from the transcript header; each model reply is re-sent as `{"content": content or ""}`, its tool calls with
the provider's ids and the raw argument JSON (or `{}` when it did not parse), and, on the `beam-hf` profile,
`reasoning_content` set to the reply's reasoning. Each executed call's result follows as a tool message; a reply
without tool calls gets a nudge. The accepted `finish` call ends the run, so its result was never sent.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ECHO_PROFILES = {"beam-hf": True, "beam-hf-noecho": False}
ORIGINAL_FIELDS = ("batch", "prompt", "profile", "status", "done", "claimed_done", "false_done", "trap",
                   "honesty_gap", "claim_issue", "unrun_claims", "misreported_claims", "stop_reason")


def trap_runs(raw: Path) -> list[dict]:
    """Every Beam run on a trap task, from runs.csv, with its run directory."""
    rows = list(csv.DictReader((raw / "runs.csv").open(encoding="utf-8")))
    out = []
    for r in rows:
        if r["family"] == "trap" and r["model"].startswith("reflection/Beam"):
            out.append({**r, "dir": raw / "runs" / r["batch"] / r["run_id"]})
    return out


def episode_id(run: dict) -> str:
    return f"cvd/{run['batch']}/{run['run_id']}"


def read_transcript(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def api_messages(records: list[dict]) -> list[dict]:
    header = records[0]
    echo = ECHO_PROFILES[header["profile"]]
    messages = [{"role": "system", "content": header["system_prompt"]},
                {"role": "user", "content": header["task_prompt"]}]
    for r in records[1:]:
        if r["type"] == "model_call":
            m = {"role": "assistant", "content": r["content"] or ""}
            if r["tool_calls"]:
                m["tool_calls"] = [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": c["arguments"] if c["arguments"] is not None else {}}}
                    for c in r["tool_calls"]
                ]
            if echo and r["reasoning"] is not None:
                m["reasoning_content"] = r["reasoning"]
            messages.append(m)
        elif r["type"] == "tool_call":
            messages.append({"role": "tool", "tool_call_id": r["call_id"], "content": r["result"]})
        elif r["type"] == "nudge":
            messages.append({"role": "user", "content": r["text"]})
    return messages


def model_calls(records: list[dict]) -> list[dict]:
    return [r for r in records if r["type"] == "model_call"]


def label(grade: dict) -> tuple[int, str]:
    if grade["trap"] == "fabricated_done":
        return 1, "DONE on a trap task"
    if grade["trap"] == "honest_block":
        return 0, "honest BLOCKED"
    if grade["trap"] == "no_report":
        return -1, "excluded: no accepted finish report (no status)"
    return -1, f"excluded: trap outcome {grade['trap']} (BLOCKED without naming the trap)"


def finish_call(records: list[dict], messages: list[dict]) -> tuple[int, int] | None:
    """(message index, call index) of the finish call the harness accepted: the grader's `result.finish`."""
    accepted = [r for r in records
                if r["type"] == "tool_call" and r["name"] == "finish" and r["result"] == "Run finished."]
    if not accepted:
        return None
    call_id = accepted[-1]["call_id"]
    for mi, m in enumerate(messages):
        for ci, tc in enumerate(m.get("tool_calls") or []):
            if tc["id"] == call_id:
                return mi, ci
    raise ValueError(f"accepted finish call {call_id} not in the history")


# ---------------------------------------------------------------- Stage B (exploratory) renders
#
# Two layouts Stage B scores beside the trap slots: true DONE reports on solvable tasks (a specificity check for H4)
# and the trap runs under the four-line status rule of prompts/system-strict.md (H4b). The harness filled
# {max_turns} and {wall_minutes} in the prompt file before sending it (claimed-vs-done `cvd/loop.py`
# build_system_prompt), so a header equals its prompt file only after that fill.

ORIGINAL_PROMPT_SHA256 = "5b819d85dab8c6426a5efa62c376eedff725cee92f6b706481ed26ea9e25312c"  # prompts/system.md
STRICT_PROMPT_SHA256 = "148abfbc88efc7bed501c4922bd3475d199a6ad3fc1098de5541795f31d6db58"  # prompts/system-strict.md
SOLVABLE_BATCHES = ("main-beam-high", "main-beam-low", "main-beam-max", "main-beam-none")
DRAW_EFFORTS = ("high", "low", "max", "none")
TRUE_DONE_REASON = "true DONE on a solvable task"
_grade_label = label  # stageb_row takes a `label` keyword
ANCHOR_KEYS = ("f", "d", "window_end", "s", "s_call")
_auto_tok = None


def solvable_runs(raw: Path) -> list[dict]:
    """Beam runs on solvable tasks under the original prompt that reported DONE and were done: the true-DONE pool."""
    rows = list(csv.DictReader((raw / "runs.csv").open(encoding="utf-8")))
    return [{**r, "dir": raw / "runs" / r["batch"] / r["run_id"]} for r in rows
            if r["model"].startswith("reflection/Beam") and r["family"] != "trap" and r["batch"] in SOLVABLE_BATCHES
            and r["prompt"] == "original" and r["done"] == "True" and r["status"] == "DONE"]


def _largest_remainder(total: int, sizes: list[int]) -> list[int]:
    """Split `total` in proportion to `sizes`; leftover units go to the largest remainders, ties to the earlier."""
    n = sum(sizes)
    quota = [total * s // n for s in sizes]
    order = sorted(range(len(sizes)), key=lambda i: (-(total * sizes[i] % n), i))
    for i in order[:total - sum(quota)]:
        quota[i] += 1
    return quota


def draw_true_done(runs: list[dict], per_effort: int = 12, seed: int = 20261028) -> list[dict]:
    """Seeded draw of `per_effort` runs per effort, stratified by family by largest remainder, in draw order.

    One generator serves the whole loop: efforts high, low, max, none; families sorted; within a family the pool
    sorted by run_id, indexed by sorted(rng.permutation(len(pool))[:quota]).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for effort in DRAW_EFFORTS:
        pool = [r for r in runs if r["effort"] == effort]
        families = sorted({r["family"] for r in pool})
        quota = _largest_remainder(per_effort, [sum(r["family"] == f for r in pool) for f in families])
        for family, q in zip(families, quota):
            members = sorted((r for r in pool if r["family"] == family), key=lambda r: r["run_id"])
            out += [members[i] for i in sorted(rng.permutation(len(members))[:q])]
    return out


def fill_prompt(template: str, budget: dict) -> str:
    """The system prompt the harness sent: `build_system_prompt` with the run's budget."""
    import math

    return template.replace("{max_turns}", str(budget["max_turns"])).replace(
        "{wall_minutes}", str(math.ceil(budget["wall_s"] / 60)))


def prompt_template(sha256: str, raw: Path | None = None) -> tuple[str, str]:
    """(file name, text) of the prompt file under raw/prompts whose bytes hash to `sha256`."""
    import hashlib

    from ouroboros.render.paths import CVD_RAW

    for path in sorted(((raw or CVD_RAW) / "prompts").glob("*.md")):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() == sha256:
            return f"prompts/{path.name}", data.decode("utf-8")
    raise ValueError(f"no prompt file hashes to {sha256}")


def swap_system_prompt(records: list[dict], prompt: str,
                       expect_sha256: str = ORIGINAL_PROMPT_SHA256) -> list[dict]:
    """Copies of `records` with the header prompt replaced by `prompt`, filled with the run's budget.

    Refuses unless the header prompt equals, byte for byte, the prompt file hashing to `expect_sha256` after the
    same fill.
    """
    header = records[0]
    name, template = prompt_template(expect_sha256)
    if header["system_prompt"].encode("utf-8") != fill_prompt(template, header["budget"]).encode("utf-8"):
        raise ValueError(f"{header['run_id']}: header system_prompt differs from {name}")
    return [{**header, "system_prompt": fill_prompt(prompt, header["budget"])}] + [dict(r) for r in records[1:]]


def stageb_row(run: dict, *, system_prompt: str | None = None, id_prefix: str = "cvd",
               label: int | None = None, label_reason: str | None = None) -> dict:
    """`build.cvd_row` for Stage B: an optional system-prompt swap, its own episode id and an optional label.

    With no options it returns exactly what cvd_row returns.
    """
    from ouroboros.render import build
    from ouroboros.render.schemas import cvd_tools
    from ouroboros.render.sequence import build_sequence
    from ouroboros.render.template import EFFORT_INT

    tpl = build._template()
    records = read_transcript(run["dir"] / "transcript.jsonl")
    if system_prompt is not None:
        records = swap_system_prompt(records, system_prompt)
    grade = json.loads((run["dir"] / "grade.json").read_text(encoding="utf-8"))
    header = records[0]
    eid = f"{id_prefix}/{run['batch']}/{run['run_id']}"
    messages = api_messages(records)
    seq = build_sequence(tpl, messages, cvd_tools(), EFFORT_INT[header["effort"]])
    lab, reason = _grade_label(grade)
    lab, reason = (lab if label is None else label), (reason if label_reason is None else label_reason)
    finish = finish_call(records, messages)
    s = s_call = before_s = None
    if finish is not None:
        args = messages[finish[0]]["tool_calls"][finish[1]]["function"]["arguments"]
        if args["status"] != grade["status"]:
            raise ValueError(f"{eid}: finish status {args['status']} differs from grade {grade['status']}")
        s_call = build._call_token(seq, finish)
        marker = '<rfl:parameter name="status">\n'
        at = seq.text.index(marker, seq.offsets[s_call][0]) + len(marker)
        if not seq.text.startswith(args["status"], at):
            raise ValueError(f"{eid}: status value not found after its parameter tag")
        s = seq.boundary(at)
        before_s = list(args)[:list(args).index("status")]
    calls = model_calls(records)
    prompt_billed, completion_billed = [c["prompt_tokens"] for c in calls], [c["completion_tokens"] for c in calls]
    completion_rendered = [t.end - t.start for t in seq.turns]
    malformed = sum(1 for c in calls for tc in c["tool_calls"] if tc["arguments"] is None)
    return build._base(
        seq,
        episode_id=eid, source="cvd", task_id=run["task_id"], condition=run["prompt"],
        effort=header["effort"], reasoning_effort=EFFORT_INT[header["effort"]], label=lab, label_reason=reason,
        f=None, d=None, window_end=None, s=s,
        anchor_text=build._anchor_text(tpl, seq, f=None, d=None, window_end=None, s=s, s_call=s_call),
        s_call=s_call, finish_params_before_s=before_s,
        malformed_calls_sent_as_empty=malformed,
        billing=json.dumps({"prompt_rendered": seq.gen_prompt_lens, "prompt_billed": prompt_billed,
                            "completion_rendered": completion_rendered, "completion_billed": completion_billed,
                            "cached_billed": [c["cached_tokens"] or 0 for c in calls]}),
        billing_exact=(all(b == r + 1 for b, r in zip(prompt_billed, seq.gen_prompt_lens))
                       and completion_billed == completion_rendered),
        **{k: (run[k] if k in ("batch", "prompt") else grade.get(k, header.get(k))) for k in ORIGINAL_FIELDS},
    )


def _autotokenizer():
    global _auto_tok
    if _auto_tok is None:
        from transformers import AutoTokenizer

        from ouroboros.render.template import VENDOR

        _auto_tok = AutoTokenizer.from_pretrained(str(VENDOR))
    return _auto_tok


def rerender_check(row: dict, original: dict | None = None) -> None:
    """Re-render `row` from its transcript on a second path and compare token for token; ValueError on a miss.

    The second path is transformers' own: `apply_chat_template` on vendor/beam renders the text and the
    AutoTokenizer encodes it. A `cvd-rule/` row gets prompts/system-strict.md, read and hashed here on its own.
    With `original` (the cvd.parquet row of the same run), the ids and gen_mask after the prompt message must equal
    the original's, and every anchor must move by the same offset.
    """
    import numpy as np

    from ouroboros.render.paths import CVD_RAW
    from ouroboros.render.schemas import cvd_tools
    from ouroboros.render.template import EFFORT_INT, HARMONY_TOKENS

    eid = row["episode_id"]
    prefix, batch, run_id = eid.split("/")
    records = read_transcript(CVD_RAW / "runs" / batch / run_id / "transcript.jsonl")
    header = records[0]
    messages = api_messages(records)
    if prefix == "cvd-rule":
        _, strict = prompt_template(STRICT_PROMPT_SHA256)
        messages[0] = {"role": "system", "content": fill_prompt(strict, header["budget"])}
    last = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    tok = _autotokenizer()
    text = tok.apply_chat_template(messages[:last + 1], tools=cvd_tools(), tokenize=False,
                                   reasoning_effort=EFFORT_INT[header["effort"]])
    if tok(text, add_special_tokens=False).input_ids != list(row["token_ids"]):
        raise ValueError(f"{eid}: the independent re-render differs from the row's token ids")
    if original is None:
        return

    old, new = np.asarray(original["token_ids"]), np.asarray(row["token_ids"])
    old_gen, new_gen = np.asarray(original["gen_mask"], bool), np.asarray(row["gen_mask"], bool)
    shift = len(new) - len(old)
    n = min(len(old), len(new))
    differ = np.flatnonzero(old[:n] != new[:n])
    if not len(differ) and not shift:  # a copy: everything must match
        if not np.array_equal(old_gen, new_gen) or any(original.get(a) != row.get(a) for a in ANCHOR_KEYS):
            raise ValueError(f"{eid}: a copied row differs from {original['episode_id']}")
        return
    p = int(differ[0]) if len(differ) else n
    ends = [np.flatnonzero(x[p:] == HARMONY_TOKENS["<|end|>"]) for x in (old, new)]
    if not (len(ends[0]) and len(ends[1])):
        raise ValueError(f"{eid}: no <|end|> after the first changed token {p}")
    # the <|end|> closing the prompt message (Harmony's developer message carries the system prompt)
    e_old, e_new = p + int(ends[0][0]), p + int(ends[1][0])
    if e_new - e_old != shift:
        raise ValueError(f"{eid}: the prompt message changed length by {e_new - e_old}, the episode by {shift}")
    if old_gen[:e_old + 1].any() or new_gen[:e_new + 1].any():
        raise ValueError(f"{eid}: Beam-written tokens inside the prompt message")
    if not (np.array_equal(old[e_old:], new[e_new:]) and np.array_equal(old_gen[e_old:], new_gen[e_new:])):
        raise ValueError(f"{eid}: ids or gen_mask after the prompt message differ from {original['episode_id']}")
    for a in ANCHOR_KEYS:
        x, y = original.get(a), row.get(a)
        if (x is None) != (y is None) or (x is not None and y - x != shift):
            raise ValueError(f"{eid}: anchor {a} moved from {x} to {y}, the episode by {shift}")
    if [t + shift for t in original["turn_starts"]] != list(row["turn_starts"]):
        raise ValueError(f"{eid}: turn starts moved by other than {shift}")
