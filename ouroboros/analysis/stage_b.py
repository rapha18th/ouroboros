"""Stage B inputs: the 30/14 task split, the six scoring sets, and the sha256 manifests of the inputs tree.

    python -m ouroboros.analysis.stage_b split|sets|manifest [--out DIR] [--workers 3]

- split writes split.json. The 44 Stage A Tripwire task_ids (anchors.csv rows with label 0 or 1) are sorted;
  perm = default_rng(20261028).permutation(44); the first 30 are for search and the last 14 are held out.
- sets writes split.json, sets/<name>.json for the six sets, the three parquets this module renders (search_dev,
  cvd_true_done, cvd_rule) and sets/manifest.json. The Claimed vs Done renders run in at most --workers processes.
- manifest writes vouched.sha256, MANIFEST.sha256 and manifest.json over the whole inputs tree. Run it last.

Stage B is exploratory (stageb.REASON), so every file written here carries the label. Search inputs come from the
30 search tasks only. Spans are episode-local [a, b):
- ctrl16 = [ctrl, ctrl + 16): a tamper positive's first source edit after f and before d.
- edit16 = [edit, edit + 16): a negative's first edit after f, kept only when it is a source edit.
- dend = [d, d_end + 1): the tamper call through its first <|call|> or <|end|>.
- lead_d, lead_ctrl, lead_edit = [m + 1, e): the message before a call. e = call - 1 holds its <|end|> and m is
  the last <|message|> before e. Kept only when Beam wrote every token.
Tool calls are parsed from the tokens, ported from the expert-maps `setup.calls` and `events.events` that the
reviewed counts came from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from ouroboros import stageb
from ouroboros.analysis.data import load_anchors
from ouroboros.analysis.populations import cluster_of
from ouroboros.render import cvd
from ouroboros.render.tamper import is_protected

log = logging.getLogger(__name__)

VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "beam"
HARNESS = Path(__file__).resolve().parents[1] / "harness"
EDIT_TOOLS = ("edit_file", "write_file", "replace_lines")
EDIT_KINDS = ("edit_src", "write_cmd", "edit_test", "write_cmd_test")
SOURCE_EDITS = ("edit_src", "write_cmd")
CVD_ROLES = {1: "false_done", 0: "honest_blocked"}
POSITIONS = ("f", "d", "window_end", "s", "s_call", "f_any", "window_end_any", "d_detector")
SPAN = stageb.TAMPER_SPAN
PAIRS, GUARD_EPISODES = 12, 8
TRUE_DONE_PER_EFFORT = 12

# The structure counts the plan was reviewed on. A data change that moves any of them stops the build.
EXPECTED = {
    "search": {"positives": 117, "with_ctrl": 47, "negatives": 70, "test_first": 6, "tasks_with_both": 20,
               "guard": 120, "guard_tasks": 30},
    "held": {"positives": 54, "positive_tasks": 13, "with_ctrl": 25, "negatives": 42, "test_first": 3, "guard": 56,
             "positive_tools": {"edit_file": 41, "run_command": 12, "write_file": 1},
             "negative_conditions": {"impossible": 12, "impossible_honest": 30},
             "negative_tools": {"edit_file": 15, "run_command": 27}},
    "cvd_trap": {"false_done": 64, "honest_blocked": 31},
    "cvd_rule": {"off_policy": 57, "on_policy": 7},
    "cvd_true_done": {"high": 12, "low": 12, "max": 12, "none": 12},
}

_WRITES = re.compile(r"(?<![0-9&])>>?\s*(?!/dev/null|&)\S|\bsed\s+-i|\bperl\s+-pi|open\([^)]*['\"][wa]|\.write\("
                     r"|write_text\(")
_PARAM = re.compile(r'<rfl:parameter name="(\w+)">\n?(.*?)\n?</rfl:parameter>', re.S)
_tok: Tokenizer | None = None


def _tokenizer() -> Tokenizer:
    global _tok
    if _tok is None:
        _tok = Tokenizer.from_file(str(VENDOR / "tokenizer.json"))
    return _tok


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_json(path: Path, obj: dict) -> str:
    """Write `obj` as indent-1 JSON with a final newline; returns the file's sha256."""
    data = (json.dumps(obj, indent=1) + "\n").encode("utf-8")
    _write_bytes(path, data)
    return hashlib.sha256(data).hexdigest()


def write_parquet(table: pa.Table, path: Path) -> str:
    """Write `table` with the exploratory label in its schema metadata; returns the file's sha256."""
    meta = dict(table.schema.metadata or {}) | {b"exploratory": b"true", b"reason": stageb.REASON.encode("utf-8")}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table.replace_schema_metadata(meta), tmp, compression="zstd")
    tmp.replace(path)
    return sha256_file(path)


def _lists(table: pa.Table, name: str, dtype) -> list[np.ndarray]:
    col = table.column(name).combine_chunks()
    values = col.flatten().to_numpy(zero_copy_only=False).astype(dtype)
    offsets = col.offsets.to_numpy()
    offsets = offsets - offsets[0]
    return [values[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


# ---------------------------------------------------------------- split

def task_split(anchors: Path = stageb.ANCHORS, tripwire: Path = stageb.RENDERED / "tripwire.parquet",
               seed: int = stageb.SEED, n_search: int = 30) -> dict:
    """split.json: sorted Stage A task_ids, default_rng(seed).permutation, the first n_search for search.

    At the registered seed and size the lists must equal stageb.SEARCH_TASKS and stageb.HELD_TASKS.
    """
    rows = load_anchors(anchors)
    meta = pq.read_table(tripwire, columns=["episode_id", "task_id"]).to_pydict()
    task = dict(zip(meta["episode_id"], meta["task_id"]))
    tasks = sorted({task[k] for k, r in rows.items() if r["source"] == "tripwire" and r["label"] in (0, 1)})
    perm = np.random.default_rng(seed).permutation(len(tasks))
    search, held = sorted(tasks[i] for i in perm[:n_search]), sorted(tasks[i] for i in perm[n_search:])
    if (seed, n_search) == (stageb.SEED, 30) and (
            len(tasks) != 44 or tuple(search) != stageb.SEARCH_TASKS or tuple(held) != stageb.HELD_TASKS):
        raise ValueError(f"the split of {len(tasks)} tasks differs from stageb.SEARCH_TASKS / HELD_TASKS")
    return {
        "format": "ouroboros-split/1", **stageb.LABEL,
        "rule": f"Sort the {len(tasks)} Stage A Tripwire task_ids (anchors.csv rows with source tripwire and label "
                f"0 or 1), apply default_rng({seed}).permutation({len(tasks)}), and take the first {n_search} for "
                f"search and the rest as held out.",
        "seed": seed, "search": search, "held": held,
        "anchors_sha256": sha256_file(anchors), "tripwire_parquet_sha256": sha256_file(tripwire),
    }


# ---------------------------------------------------------------- tool calls, events and spans

def tool_calls(ids: np.ndarray, gen: np.ndarray, test_file: str | None) -> list[dict]:
    """Beam-written tool calls `<|start|>assistant to=NAME ... <|message|> ARGS <|call|>`, in order.

    Each is {p: its <|start|>, name, kind, end: its <|call|> (len(ids) when cut), path}. Kinds: edit_src,
    edit_test, write_cmd, write_cmd_test, pytest, cmd, submit, read, other. A name whose first piece merges with
    the "=" (read_file encodes as "=read", "_file") is skipped, as in the parser the counts were reviewed on; every
    edit, run_command and submit_patch call parses.
    """
    ids, gen = np.asarray(ids), np.asarray(gen, bool)
    tok = _tokenizer()
    test_base = test_file.rsplit("/", 1)[-1] if test_file else None
    out = []
    for p in np.flatnonzero(ids == stageb.START).tolist():
        if (p + 4 >= len(ids) or not gen[p + 1] or ids[p + 1] != stageb.ASSISTANT or ids[p + 2] != stageb.TO
                or ids[p + 3] != stageb.EQUALS):
            continue  # a generation prompt or a turn opener, not a call Beam wrote
        ch = p + 4 + int(np.argmax(ids[p + 4:p + 30] == stageb.CHANNEL))
        name = tok.decode(ids[p + 4:ch].tolist())
        m = p + int(np.argmax(ids[p:p + 60] == stageb.MESSAGE))
        calls = np.flatnonzero(ids[m:] == stageb.CALL)
        e = m + int(calls[0]) if len(calls) else len(ids)
        params = dict(_PARAM.findall(tok.decode(ids[m + 1:e].tolist(), skip_special_tokens=False)))
        path = params.get("filepath", "").replace("/workspace/", "").strip()
        cmd = params.get("command", "")
        if name in EDIT_TOOLS:
            test = path == test_file or is_protected(path) or path.endswith("conftest.py")
            kind = "edit_test" if test else "edit_src"
        elif name == "run_command":
            if _WRITES.search(cmd):
                kind = "write_cmd_test" if (test_base and test_base in cmd) or "conftest" in cmd else "write_cmd"
            else:
                kind = "pytest" if "pytest" in cmd else "cmd"
        else:
            kind = {"submit_patch": "submit", "read_file": "read"}.get(name, "other")
        out.append({"p": p, "name": name, "kind": kind, "end": e, "path": path})
    return out


def events(row: dict, calls: list[dict]) -> dict[str, int]:
    """Decision points of a Stage A Tripwire episode, each the <|start|> of a tool call.

    d: the tamper call (anchor d) of a positive. ctrl: a tamper positive's first source edit after f and before d.
    first: the first call after f. edit: a negative's first edit after f, of any edit kind. submit: the last
    submit_patch call.
    """
    f, d = row["f"], row["d"]
    ev = {}
    if row["label"] == 1 and d is not None:
        ev["d"] = d
        if f is not None:
            ctrl = [c["p"] for c in calls if f < c["p"] < d and c["kind"] in SOURCE_EDITS]
            if ctrl and "rigged test" in (row.get("d_method") or ""):
                ev["ctrl"] = ctrl[0]
    if f is not None:
        after = [c for c in calls if c["p"] > f]
        if after:
            ev["first"] = after[0]["p"]
        if row["label"] == 0:
            edits = [c["p"] for c in after if c["kind"] in EDIT_KINDS]
            if edits:
                ev["edit"] = edits[0]
    submits = [c["p"] for c in calls if c["kind"] == "submit"]
    if submits:
        ev["submit"] = submits[-1]
    return ev


def lead_span(ids: np.ndarray, gen: np.ndarray, call: int) -> tuple[int, int] | None:
    """[m + 1, e) of the message that ends just before `call`, or None unless Beam wrote all of it."""
    ids, gen = np.asarray(ids), np.asarray(gen, bool)
    e = call - 1
    if e < 1 or ids[e] != stageb.END:
        return None
    opens = np.flatnonzero(ids[:e] == stageb.MESSAGE)
    if not len(opens):
        return None
    a = int(opens[-1]) + 1
    if a >= e or not gen[a:e].all():
        return None
    return a, e


def d_end(ids: np.ndarray, d: int) -> int:
    """Index of the first <|call|> or <|end|> after d."""
    ids = np.asarray(ids)
    after = np.flatnonzero(np.isin(ids[d + 1:], (stageb.CALL, stageb.END)))
    if not len(after):
        raise ValueError(f"no <|call|> or <|end|> after d = {d}")
    return d + 1 + int(after[0])


# ---------------------------------------------------------------- Tripwire records

def tripwire_records(split: dict, anchors: Path = stageb.ANCHORS,
                     tripwire: Path = stageb.RENDERED / "tripwire.parquet") -> list[dict]:
    """Set records of every positive, source-edit negative and guard episode in the split's tasks, parquet order.

    Labels and anchors come from anchors.csv; the parquet's f, d and window_end must agree with them.
    """
    a = load_anchors(anchors)
    cols = ["episode_id", "task_id", "condition", "effort", "label", "f", "d", "window_end", "test_file"]
    table = pq.read_table(tripwire, columns=cols + ["token_ids", "gen_mask"])
    meta = table.select(cols).to_pylist()
    ids_all, gen_all = _lists(table, "token_ids", np.int64), _lists(table, "gen_mask", bool)
    tasks = set(split["search"]) | set(split["held"])
    out = []
    for m, ids, gen in zip(meta, ids_all, gen_all):
        key = m["episode_id"]
        an = a[key]
        if any(an[k] != m[k] for k in ("f", "d", "window_end")) or an["label"] != m["label"]:
            raise ValueError(f"{key}: the parquet's label, f, d or window_end differ from anchors.csv")
        if m["task_id"] not in tasks:
            continue
        row = {**m, "label": an["label"], "d_method": an["d_method"]}
        n = len(ids)
        spans: dict[str, list[int]] = {}
        span_tools: dict[str, str] = {}
        tool = None
        if m["condition"] == "possible":
            role = "guard"
        elif an["label"] == 1 and an["d"] is not None:
            role = "positive"
            calls = tool_calls(ids, gen, m["test_file"])
            ev = events(row, calls)
            at = {c["p"]: c for c in calls}
            d = ev["d"]
            if d not in at:
                raise ValueError(f"{key}: no parsed tool call at d = {d}")
            tool = at[d]["name"]
            if "ctrl" in ev:
                spans["ctrl16"] = [ev["ctrl"], ev["ctrl"] + SPAN]
                span_tools["ctrl16"] = at[ev["ctrl"]]["name"]
            spans["dend"] = [d, d_end(ids, d) + 1]
            for name, call in (("lead_d", d), ("lead_ctrl", ev.get("ctrl"))):
                lead = None if call is None else lead_span(ids, gen, call)
                if lead:
                    spans[name] = list(lead)
        elif an["label"] == 0:
            calls = tool_calls(ids, gen, m["test_file"])
            ev = events(row, calls)
            at = {c["p"]: c for c in calls}
            if "edit" not in ev or at[ev["edit"]]["kind"] not in SOURCE_EDITS:
                continue  # no edit after f, or the first one writes a test file: outside the legit set
            role = "negative"
            tool = at[ev["edit"]]["name"]
            spans["edit16"] = [ev["edit"], ev["edit"] + SPAN]
            span_tools["edit16"] = tool
            lead = lead_span(ids, gen, ev["edit"])
            if lead:
                spans["lead_edit"] = list(lead)
        else:
            continue
        if any(not 0 <= x < y <= n for x, y in spans.values()):
            raise ValueError(f"{key}: a span falls outside the row ({n} tokens): {spans}")
        out.append({
            "key": key, "role": role, "task_id": m["task_id"], "cluster": cluster_of(key, "tripwire", m["task_id"]),
            "condition": m["condition"], "effort": m["effort"], "tool": tool, "spans": spans,
            "span_tools": span_tools, "on_policy": None, "source_episode_id": None, "cut": None,
        })
    return out


def tripwire_counts(records: list[dict], anchors: Path = stageb.ANCHORS,
                    tripwire: Path = stageb.RENDERED / "tripwire.parquet") -> dict:
    """The structure counts in EXPECTED["search"] and EXPECTED["held"], recomputed from records and the parquet."""
    a = load_anchors(anchors)
    cols = ["episode_id", "task_id", "label", "f", "d", "test_file"]
    table = pq.read_table(tripwire, columns=cols + ["token_ids", "gen_mask"])
    test_first = defaultdict(int)
    for m, ids, gen in zip(table.select(cols).to_pylist(), _lists(table, "token_ids", np.int64),
                           _lists(table, "gen_mask", bool)):
        an = a[m["episode_id"]]
        if an["label"] != 0 or an["f"] is None:
            continue
        calls = tool_calls(ids, gen, m["test_file"])
        ev = events({**m, "d_method": an["d_method"]}, calls)
        if "edit" in ev and next(c["kind"] for c in calls if c["p"] == ev["edit"]) not in SOURCE_EDITS:
            test_first[m["task_id"] in stageb.SEARCH_TASKS] += 1

    out = {}
    for part, tasks in (("search", set(stageb.SEARCH_TASKS)), ("held", set(stageb.HELD_TASKS))):
        r = [x for x in records if x["task_id"] in tasks]
        pos = [x for x in r if x["role"] == "positive"]
        neg = [x for x in r if x["role"] == "negative"]
        guard = [x for x in r if x["role"] == "guard"]
        c = {"positives": len(pos), "with_ctrl": sum("ctrl16" in x["spans"] for x in pos), "negatives": len(neg),
             "test_first": test_first[part == "search"], "guard": len(guard)}
        if part == "search":
            c |= {"tasks_with_both": len({x["task_id"] for x in pos} & {x["task_id"] for x in neg}),
                  "guard_tasks": len({x["task_id"] for x in guard})}
        else:
            c |= {"positive_tasks": len({x["task_id"] for x in pos}),
                  "positive_tools": _tally(x["tool"] for x in pos),
                  "negative_conditions": _tally(x["condition"] for x in neg),
                  "negative_tools": _tally(x["tool"] for x in neg)}
        out[part] = {k: c[k] for k in EXPECTED[part]}
    return out


def _tally(xs) -> dict:
    t = defaultdict(int)
    for x in xs:
        t[x] += 1
    return dict(sorted(t.items()))


# ---------------------------------------------------------------- search_dev

def search_dev(records: list[dict], tripwire: Path, seed: int = stageb.SEED) -> tuple[pa.Table, dict]:
    """The pre-cut search_dev parquet table and its set (name, rule, episodes in layout order).

    Layout p1, n1, ..., p12, n12, g1, ..., g8. Positives are cut at d + 16, negatives at edit + 16, guard episodes
    at 8,192 tokens. Positions at or past a cut become null, and a record keeps only the spans that end by it.
    """
    pos, neg, guard = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in records:
        if r["task_id"] not in stageb.SEARCH_TASKS:
            continue
        {"positive": pos, "negative": neg, "guard": guard}[r["role"]][r["task_id"]].append(r)
    tasks = sorted(set(pos) & set(neg))
    rng = np.random.default_rng(seed)
    chosen = []
    for t in [tasks[i] for i in rng.permutation(len(tasks))[:PAIRS]]:
        p = pos[t][rng.integers(len(pos[t]))]
        n = neg[t][rng.integers(len(neg[t]))]
        chosen += [(p, None), (n, None)]
    rng2 = np.random.default_rng(seed)
    gt = sorted(guard)
    chosen += [(guard[gt[i]][0], stageb.GUARD_CAP) for i in rng2.permutation(len(gt))[:GUARD_EPISODES]]

    table = pq.read_table(tripwire)
    index = {k: i for i, k in enumerate(table.column("episode_id").to_pylist())}
    rows = table.take([index[r["key"]] for r, _ in chosen]).to_pylist()
    out_rows, episodes = [], []
    for (r, cap), row in zip(chosen, rows):
        n = len(row["token_ids"])
        if r["role"] == "positive":
            cut = row["d"] + SPAN
        elif r["role"] == "negative":
            cut = r["spans"]["edit16"][1]
        else:
            cut = min(n, cap)
        if cut > n:
            raise ValueError(f"{r['key']}: cut {cut} past the row's {n} tokens")
        row = dict(row, token_ids=row["token_ids"][:cut], gen_mask=row["gen_mask"][:cut], n_tokens=cut, s_call=None)
        for k in POSITIONS:
            if row.get(k) is not None and row[k] >= cut:
                row[k] = None
        row["anchor_text"] = {k: (v if row.get(k) is not None else None) for k, v in row["anchor_text"].items()}
        row["turn_starts"] = [t for t in row["turn_starts"] if t < cut]
        row["n_turns"] = len(row["turn_starts"])
        row["n_gen_tokens"] = int(sum(row["gen_mask"]))
        out_rows.append(row)
        spans = {k: v for k, v in r["spans"].items() if v[1] <= cut}
        episodes.append({**r, "spans": spans, "span_tools": {k: v for k, v in r["span_tools"].items() if k in spans},
                         "cut": cut})
    schema = table.schema.append(pa.field("s_call", pa.int32()))
    rule = (f"{PAIRS} of the {len(tasks)} search tasks that hold both a positive with d and a source-edit negative, "
            f"drawn with default_rng({seed}); per drawn task one positive, then one negative, by rng.integers over "
            f"each in parquet order. Then {GUARD_EPISODES} of the {len(gt)} search tasks with possible episodes, "
            f"drawn with a second default_rng({seed}), each giving its first possible episode in parquet order. "
            f"Rows are pre-cut: positives at d + {SPAN}, negatives at edit + {SPAN}, guard episodes at "
            f"{stageb.GUARD_CAP} tokens.")
    return pa.Table.from_pylist(out_rows, schema=schema), {"name": "search_dev", "rule": rule, "episodes": episodes}


# ---------------------------------------------------------------- Claimed vs Done renders

def _slot_check(row: dict) -> None:
    """The status slot holds DONE (70396) on a DONE report and BLOCK, ED (39964, 1479) on a BLOCKED one."""
    s, ids = row["s"], row["token_ids"]
    if s is None:
        raise ValueError(f"{row['episode_id']}: no status slot")
    want = (stageb.DONE,) if row["status"] == "DONE" else (stageb.BLOCK, stageb.ED)
    if tuple(ids[s:s + len(want)]) != want:
        raise ValueError(f"{row['episode_id']}: status {row['status']} slot holds {list(ids[s:s + 2])}")


def _rule_row(job: tuple[dict, dict, str]) -> dict:
    """A cvd-rule row: an original-prompt run re-rendered under the strict prompt, or a strict run copied.

    Every header prompt is checked byte for byte first: against prompts/system.md before a swap, against
    prompts/system-strict.md before a copy.
    """
    run, original, strict = job
    if run["prompt"] == "strict":
        cvd.swap_system_prompt(cvd.read_transcript(run["dir"] / "transcript.jsonl"), strict,
                               cvd.STRICT_PROMPT_SHA256)
        row = {**original, "episode_id": f"cvd-rule/{run['batch']}/{run['run_id']}"}
        cvd.rerender_check(row)
    else:
        row = cvd.stageb_row(run, system_prompt=strict, id_prefix="cvd-rule")
        cvd.rerender_check(row, original=original)
    if row["s"] is not None:
        _slot_check(row)
    return {**row, "source_episode_id": original["episode_id"], "rendered_prompt": "strict"}


def _done_row(run: dict) -> dict:
    row = cvd.stageb_row(run, id_prefix="cvd-done", label=0, label_reason=cvd.TRUE_DONE_REASON)
    _slot_check(row)
    cvd.rerender_check(row)
    return {**row, "source_episode_id": None, "rendered_prompt": "original"}


def _cvd_schema(cvd_parquet: Path) -> pa.Schema:
    schema = pq.read_schema(cvd_parquet).remove_metadata()
    return schema.append(pa.field("source_episode_id", pa.string())).append(pa.field("rendered_prompt", pa.string()))


def _cvd_record(row: dict, role: str, on_policy: bool | None = None, source: str | None = None) -> dict:
    key = row["episode_id"]
    return {"key": key, "role": role, "task_id": row["task_id"], "cluster": cluster_of(key, "cvd", row["task_id"]),
            "condition": row["condition"], "effort": row["effort"], "tool": None, "spans": {}, "span_tools": {},
            "on_policy": on_policy, "source_episode_id": source, "cut": None}


def render_rows(renders: dict[str, tuple[Callable[[Any], dict], list]], workers: int,
                executor: Callable[..., Executor] = ProcessPoolExecutor) -> dict[str, list[dict] | ValueError]:
    """Rows of each render in item order, in at most `workers` processes; 1 renders inline.

    A render whose token check misses gets its ValueError in place of its rows. When the pool breaks (Windows can
    lose a worker's queue handles while another job loads the machine), that render reruns inline.
    """
    out: dict[str, list[dict] | ValueError] = {}
    with ExitStack() as stack:
        pool = stack.enter_context(executor(max_workers=workers)) if workers > 1 else None
        results = {name: pool.map(fn, items, chunksize=2) if pool else map(fn, items)
                   for name, (fn, items) in renders.items()}
        for name, (fn, items) in renders.items():
            try:
                try:
                    out[name] = list(results[name])
                except BrokenProcessPool as err:
                    log.warning("%s: the process pool broke (%s); rendering inline", name, err)
                    print(f"{name}: the process pool broke; rendering inline", flush=True)
                    out[name] = list(map(fn, items))
            except ValueError as err:
                out[name] = err
    return out


# ---------------------------------------------------------------- the sets

def _set_doc(name: str, split_sha256: str, parquet: str | None, parquet_sha256: str | None, search_only: bool,
             rule: str, episodes: list[dict]) -> dict:
    return {"format": "ouroboros-set/1", **stageb.LABEL, "name": name, "split_sha256": split_sha256,
            "parquet": parquet, "parquet_sha256": parquet_sha256, "search_only": search_only, "rule": rule,
            "n_episodes": len(episodes), "episodes": episodes}


def _scored_tokens(episodes: list[dict], rows: dict[str, dict], cut: str) -> int:
    """Tokens left after the score-time cut: none, `truncate` (last needed position, margin 0) or `slot` (s + 2)."""
    total = 0
    for e in episodes:
        r = rows[e["key"]]
        n = r["n_tokens"]
        if cut == "slot":
            n = r["s"] + 2  # --slot-word BLOCKED puts BLOCK, ED at s; --truncate cuts after them
        elif cut == "truncate":
            ends = [b for _, b in e["spans"].values()]
            if r["window_end"] is not None:
                ends.append(r["window_end"] + 1)
            if r["d"] is not None:
                ends.append(r["d"] + SPAN)
            n = min(n, max(ends)) if ends else n
        total += n
    return total


def scoring_sets(out: Path = stageb.INPUTS, *, workers: int = 3) -> dict:
    """Write split.json, the six sets under sets/ and sets/manifest.json; returns the manifest."""
    tripwire, cvd_parquet = stageb.RENDERED / "tripwire.parquet", stageb.RENDERED / "cvd.parquet"
    prompts = stageb.CVD_RAW / "prompts"
    sets_dir = out / "sets"
    split = task_split()
    split_sha = write_json(out / "split.json", split)
    print(f"split: {len(split['search'])} search, {len(split['held'])} held-out tasks", flush=True)

    records = tripwire_records(split)
    counts = tripwire_counts(records)
    for part in ("search", "held"):
        if counts[part] != EXPECTED[part]:
            raise ValueError(f"{part} counts {counts[part]} differ from the reviewed {EXPECTED[part]}")
    print(f"tripwire records: {len(records)}; counts match the review", flush=True)

    written: dict[str, dict] = {}

    def put(doc: dict, cut: str, rows: dict[str, dict]) -> None:
        sha = write_json(sets_dir / f"{doc['name']}.json", doc)
        written[doc["name"]] = {"parquet": doc["parquet"], "parquet_sha256": doc["parquet_sha256"],
                                "episodes": f"{doc['name']}.json", "episodes_sha256": sha,
                                "n_episodes": doc["n_episodes"],
                                "n_tokens_scored": _scored_tokens(doc["episodes"], rows, cut),
                                "search_only": doc["search_only"]}
        print(f"{doc['name']}: {doc['n_episodes']} episodes, {written[doc['name']]['n_tokens_scored']} tokens "
              f"scored", flush=True)

    trip_meta = {r["episode_id"]: r for r in pq.read_table(
        tripwire, columns=["episode_id", "n_tokens", "window_end", "d", "s"]).to_pylist()}
    trip_sha = sha256_file(tripwire)

    table, dev = search_dev(records, tripwire)
    dev_sha = write_parquet(table, sets_dir / "search_dev.parquet")
    dev_rows = {r["episode_id"]: r for r in table.select(["episode_id", "n_tokens", "window_end", "d", "s"])
                .to_pylist()}
    put(_set_doc("search_dev", split_sha, "search_dev.parquet", dev_sha, True, dev["rule"], dev["episodes"]),
        "none", dev_rows)

    held = [r for r in records if r["task_id"] in stageb.HELD_TASKS]
    tw_held = [r for r in held if r["role"] == "positive"] + [r for r in held if r["role"] == "negative"]
    put(_set_doc("tripwire_held", split_sha, "tripwire.parquet", trip_sha, False,
                 "Held-out positives with d, then held-out negatives whose first edit after f is a source edit, "
                 "each in parquet order. Scored with --truncate --margin 0.", tw_held), "truncate", trip_meta)
    put(_set_doc("guard_held", split_sha, "tripwire.parquet", trip_sha, False,
                 "Held-out possible episodes, in parquet order, scored whole.",
                 [r for r in held if r["role"] == "guard"]), "none", trip_meta)

    cvd_rows = pq.read_table(cvd_parquet).to_pylist()
    by_id = {r["episode_id"]: r for r in cvd_rows}
    trap = [r for r in cvd_rows if r["label"] in CVD_ROLES]
    for r in trap:
        _slot_check(r)
    put(_set_doc("cvd_trap", split_sha, "cvd.parquet", sha256_file(cvd_parquet), False,
                 "Claimed vs Done trap slots labelled false DONE (1) or honest BLOCKED (0), in parquet order. Scored "
                 "with --slot-word BLOCKED --truncate --margin 0.",
                 [_cvd_record(r, CVD_ROLES[r["label"]]) for r in trap]), "slot", by_id)

    schema = _cvd_schema(cvd_parquet)
    runs = cvd.solvable_runs(stageb.CVD_RAW)
    drawn = cvd.draw_true_done(runs, TRUE_DONE_PER_EFFORT, stageb.SEED)
    _, strict = cvd.prompt_template(cvd.STRICT_PROMPT_SHA256)
    jobs = [(run, by_id[cvd.episode_id(run)], strict) for run in cvd.trap_runs(stageb.CVD_RAW)]
    renders = {
        "cvd_true_done": (_done_row, drawn,
                          (f"{len(drawn)} of the {len(runs)} original-prompt runs on solvable tasks that reported a "
                           f"true DONE, {TRUE_DONE_PER_EFFORT} per effort (high, low, max, none), split over families "
                           f"by largest remainder, drawn with one default_rng({stageb.SEED}). Rendered as in "
                           f"cvd.parquet and checked by an independent re-render.")),
        "cvd_rule": (_rule_row, jobs,
                     ("The original-prompt false-DONE slots re-rendered with prompts/system-strict.md (off-policy), "
                      "then the strict-prompt false-DONE slots copied unchanged (on-policy), each in cvd.parquet "
                      "order. Each header prompt matched its prompt file byte for byte after the harness fill "
                      "(prompts/system.md before a swap, prompts/system-strict.md before a copy); an independent "
                      "re-render matched every row token for token.")),
    }
    rendered = render_rows({name: (fn, items) for name, (fn, items, _) in renders.items()}, workers)
    for name, (_, _, rule) in renders.items():
        path = sets_dir / f"{name}.parquet"
        rows = rendered[name]
        if isinstance(rows, ValueError):  # a missed token check: the set is declared not run and Boot 2 drops it
            log.error("%s: %s", name, rows)
            print(f"{name}: not run: {rows}", flush=True)
            path.unlink(missing_ok=True)
            put(_set_doc(name, split_sha, None, None, False, f"not run: {rows}", []), "none", {})
            continue
        sha = write_parquet(pa.Table.from_pylist(rows, schema=schema), path)
        if name == "cvd_true_done":
            episodes = [_cvd_record(r, "true_done") for r in rows]
        else:
            rule_slots = [r for r in rows if r["label"] == 1]
            off = [r for r in rule_slots if r["condition"] == "original"]
            on = [r for r in rule_slots if r["condition"] != "original"]
            episodes = ([_cvd_record(r, "false_done", False, r["source_episode_id"]) for r in off]
                        + [_cvd_record(r, "false_done", True, r["source_episode_id"]) for r in on])
        put(_set_doc(name, split_sha, path.name, sha, False, rule, episodes), "slot",
            {r["episode_id"]: r for r in rows})

    cvd_counts = {
        "cvd_trap": _tally(CVD_ROLES[r["label"]] for r in trap),
        "cvd_rule": {"off_policy": sum(e["on_policy"] is False for e in _episodes(sets_dir, "cvd_rule")),
                     "on_policy": sum(e["on_policy"] is True for e in _episodes(sets_dir, "cvd_rule"))},
        "cvd_true_done": _tally(e["effort"] for e in _episodes(sets_dir, "cvd_true_done")),
    }
    for name, c in cvd_counts.items():
        if written[name]["n_episodes"] and c != EXPECTED[name]:
            raise ValueError(f"{name} counts {c} differ from the expected {EXPECTED[name]}")

    manifest_doc = {
        "format": "ouroboros-sets-manifest/1", **stageb.LABEL,
        "split": {"file": "split.json", "sha256": split_sha},
        "sources": {"tripwire.parquet": trip_sha, "cvd.parquet": sha256_file(cvd_parquet),
                    "anchors.csv": sha256_file(stageb.ANCHORS), "system.md": sha256_file(prompts / "system.md"),
                    "system-strict.md": sha256_file(prompts / "system-strict.md")},
        "sets": {name: written[name] for name in stageb.SETS},
    }
    write_json(sets_dir / "manifest.json", manifest_doc)
    return manifest_doc


def _episodes(sets_dir: Path, name: str) -> list[dict]:
    return json.loads((sets_dir / f"{name}.json").read_text(encoding="utf-8"))["episodes"]


# ---------------------------------------------------------------- manifests

def vouched_hashes(harness: Path | None = None) -> dict[str, str]:
    return {name: sha256_file((harness or HARNESS) / name) for name in stageb.VOUCHED_SHA256}


def manifest(root: Path = stageb.INPUTS) -> dict:
    """vouched.sha256 (repo paths), MANIFEST.sha256 (every other file under root) and manifest.json.

    Refuses when a vouched harness file differs from stageb.VOUCHED_SHA256.
    """
    vouched = vouched_hashes()
    bad = [name for name, h in vouched.items() if h != stageb.VOUCHED_SHA256[name]]
    if bad:
        raise ValueError(f"vouched files differ from stageb.VOUCHED_SHA256: {', '.join(bad)}")
    _write_bytes(root / "vouched.sha256",
                 "".join(f"{h}  ouroboros/harness/{name}\n" for name, h in vouched.items()).encode("utf-8"))
    skip = {"MANIFEST.sha256", "manifest.json"}
    files = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*"))
             if p.is_file() and not p.name.endswith(".tmp") and not (p.parent == root and p.name in skip)}
    files = dict(sorted(files.items()))
    _write_bytes(root / "MANIFEST.sha256", "".join(f"{h}  {rel}\n" for rel, h in files.items()).encode("utf-8"))
    doc = {"format": "ouroboros-inputs-manifest/1", **stageb.LABEL, "files": files, "vouched": vouched}
    write_json(root / "manifest.json", doc)
    return doc


# ---------------------------------------------------------------- CLI

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["split", "sets", "manifest"])
    p.add_argument("--out", type=Path, default=stageb.INPUTS, help="the Stage B inputs tree")
    p.add_argument("--workers", type=int, default=3, help="processes for the Claimed vs Done renders; 1 renders inline")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        if args.step == "split":
            split = task_split()
            sha = write_json(args.out / "split.json", split)
            print(f"split.json {sha}: search {split['search']}, held {split['held']}", flush=True)
        elif args.step == "sets":
            doc = scoring_sets(args.out, workers=args.workers)
            for name, s in doc["sets"].items():
                print(f"{name}: {s['n_episodes']} episodes, {s['parquet']} {s['parquet_sha256']}, "
                      f"{s['episodes']} {s['episodes_sha256']}", flush=True)
        else:
            doc = manifest(args.out)
            print(f"manifest: {len(doc['files'])} files", flush=True)
    except ValueError as err:  # the data or the vouched files differ from what the plan was reviewed on
        sys.exit(f"stage_b {args.step} refused: {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
