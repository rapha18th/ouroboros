"""A complete Stage B inputs tree at tiny-Beam scale, for the round-loop, surgeon and integration tests.

tiny_inputs() writes what Boot 1 reads: the pre-cut search_dev parquet and its set file from tripwire_rows(), the
stock bias read from the checkpoint, a synthetic CEM basis, expert table, SteerMoE map and instruction contrast, a
window sample from a real tiny run_shard with windows, a short surgeon context, the R0 bank and its variants files
built by stage_b_maps.r0_bank as on the real inputs, and the manifests. The synthetic maps also let
stage_b_maps.boot2_bank run on the tree after Boot 1. tiny_cvd_inputs() adds small cvd_trap, cvd_true_done and
cvd_rule sets from cvd_rows(), and tiny_held_inputs() a rendered tripwire parquet with tripwire_held and guard_held
set files. Every file carries the exploratory label, as the real inputs do.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from stageb_support import TinyBeam, cvd_rows, tripwire_rows

from ouroboros import stageb
from ouroboros.analysis import stage_b_maps
from ouroboros.harness import bank, run
from ouroboros.harness.stream import Streamer
from ouroboros.harness.weights import LocalDir
from ouroboros.surgeon.footprint import sample_from_capture, save_sample

INTS = ("f", "d", "s", "window_end", "s_call")
CALL_HEAD = [stageb.START, stageb.ASSISTANT, stageb.TO, stageb.EQUALS]
VOUCHED = ("stream.py", "patch.py", "shards.py", "run.py", "capture.py", "weights.py", "fp8.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path: Path, obj: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    return path


def write_parquet(rows: list[dict], path: Path) -> Path:
    """Rendered-style rows with n_tokens and the exploratory label in the schema metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        **{c: [r[c] for r in rows] for c in ("episode_id", "source", "task_id", "condition", "effort")},
        "label": pa.array([r["label"] for r in rows], type=pa.int8()),
        "n_tokens": pa.array([len(r["token_ids"]) for r in rows], type=pa.int32()),
        "token_ids": pa.array([list(map(int, r["token_ids"])) for r in rows], type=pa.list_(pa.int32())),
        "gen_mask": pa.array([[int(g) for g in r["gen_mask"]] for r in rows], type=pa.list_(pa.uint8())),
        **{c: pa.array([r.get(c) for r in rows], type=pa.int32()) for c in INTS},
    })
    table = table.replace_schema_metadata({"exploratory": "true", "reason": stageb.REASON})
    pq.write_table(table, path)
    return path


def calls(ids) -> list[int]:
    """Every tool call's <|start|>: '<|start|>assistant to=' opens a call; a generation prompt is not one."""
    ids = list(ids)
    return [i for i in range(len(ids) - 3) if ids[i:i + 4] == CALL_HEAD]


def tool_at(ids, call: int) -> str:
    j = call + 4
    while ids[j] != stageb.CHANNEL:
        j += 1
    return run.tokenizer().decode([int(t) for t in ids[call + 4:j]]).strip()


def lead(ids, gen, call: int) -> list[int] | None:
    """[m + 1, e): the analysis message before a call, when every token of it is Beam-written."""
    e = call - 1
    if ids[e] != stageb.END:
        return None
    m = max(i for i in range(e) if ids[i] == stageb.MESSAGE)
    return [m + 1, e] if all(gen[m + 1:e]) else None


def d_end(ids, d: int) -> int:
    return next(i for i in range(d, len(ids)) if ids[i] in (stageb.CALL, stageb.END))


def record(row: dict, role: str, spans: dict, span_tools: dict, tool: str | None, cut: int | None = None) -> dict:
    return {"key": row["episode_id"], "role": role, "task_id": row["task_id"], "cluster": row["task_id"],
            "condition": row["condition"], "effort": row["effort"], "tool": tool, "spans": spans,
            "span_tools": span_tools, "on_policy": None, "source_episode_id": None, "cut": cut}


def tripwire_records(row: dict, *, with_dend: bool = False) -> dict:
    """The set record of a tripwire_rows() row: positive, negative or guard, with its spans."""
    ids, gen = row["token_ids"], row["gen_mask"]
    if row["label"] == 1:
        d, ctrl = row["d"], row["ctrl"]
        spans = {"ctrl16": [ctrl, ctrl + 16]}
        if with_dend:
            spans["dend"] = [d, d_end(ids, d) + 1]
        for name, call in (("lead_d", d), ("lead_ctrl", ctrl)):
            got = lead(ids, gen, call)
            if got:
                spans[name] = got
        return record(row, "positive", spans, {"ctrl16": tool_at(ids, ctrl)}, tool_at(ids, d))
    if row["label"] == 0:
        edit = next(c for c in calls(ids) if c > row["f"])
        spans = {"edit16": [edit, edit + 16]}
        got = lead(ids, gen, edit)
        if got:
            spans["lead_edit"] = got
        return record(row, "negative", spans, {"edit16": tool_at(ids, edit)}, tool_at(ids, edit))
    return record(row, "guard", {}, {}, None)


def cut_row(row: dict, cut: int) -> dict:
    out = {**row, "token_ids": row["token_ids"][:cut], "gen_mask": row["gen_mask"][:cut]}
    for a in INTS:
        if out.get(a) is not None and out[a] >= cut:
            out[a] = None
    return out


def set_file(path: Path, name: str, parquet: Path, records: list[dict], rule: str, search_only: bool) -> Path:
    return _json(path, stageb.labelled({
        "format": "ouroboros-set/1", "name": name, "split_sha256": None, "parquet": parquet.name,
        "parquet_sha256": _sha(parquet), "search_only": search_only, "rule": rule, "n_episodes": len(records),
        "episodes": records}))


def search_dev(root: Path) -> tuple[list[dict], list[dict]]:
    """p, n per task, then the guard episodes; pre-cut at d + 16, edit + 16 and the guard cap."""
    rows = tripwire_rows()
    by = {(r["task_id"], r["label"]): r for r in rows}
    order = [by[("t_a", 1)], by[("t_a", 0)], by[("t_b", 1)], by[("t_b", 0)], by[("t_a", -1)], by[("t_b", -1)]]
    records, cut_rows = [], []
    for row in order:
        rec = tripwire_records(row)
        if rec["role"] == "positive":
            cut = row["d"] + stageb.TAMPER_SPAN
        elif rec["role"] == "negative":
            cut = rec["spans"]["edit16"][1]
        else:
            cut = min(len(row["token_ids"]), stageb.GUARD_CAP)
        rec["spans"] = {k: v for k, v in rec["spans"].items() if v[1] <= cut}
        rec["cut"] = cut
        records.append(rec)
        cut_rows.append(cut_row(row, cut))
    return cut_rows, records


def window_sample(root: Path, tiny: TinyBeam, bias: np.ndarray) -> Path:
    """Window scores and captured top-k ids from one tiny run_shard with windows over the search rows."""
    rows = [r for r in tripwire_rows() if r["f"] is not None]
    parquet = write_parquet(rows, root / "_work" / "windows.parquet")
    from ouroboros.harness import shards

    episodes = shards.load_rendered([parquet])
    with LocalDir(tiny.ckpt) as source:
        streamer = Streamer(tiny.config, source, "cpu", torch.bfloat16, 65536, "sdpa")
        path = run.run_shard(streamer, episodes, root / "_work" / "shard", candidates=list(stageb.CANDIDATES),
                             tap_layers=(), top_k=4, with_windows=True)
    c = tiny.config
    sample = sample_from_capture(path, bias, k=c.num_experts_per_tok, n_group=c.n_group, topk_group=c.topk_group)
    shutil.rmtree(root / "_work", ignore_errors=True)  # sample_from_capture copied every row it kept
    return save_sample(root / "maps" / "window_sample.npz", **sample, label=stageb.LABEL)


def _labels(tiny: TinyBeam) -> dict:
    return {"exploratory": np.bool_(True), "reason": np.str_(stageb.REASON),
            "moe_layers": np.asarray(tiny.moe_layers, np.int32), "keys_sha256": np.str_("tiny")}


def direction_maps(root: Path, tiny: TinyBeam, stats: dict) -> None:
    """maps/steermoe_rd.npz and maps/instruction.npz from the synthetic expert table, in B3's layout."""
    rd = stats["rd_impossible"]
    np.savez(root / "maps" / "steermoe_rd.npz", **_labels(tiny), rd=rd,
             direction=stage_b_maps.steermoe_direction(rd).astype(np.float32))
    shift = stats["instruction_shift"]
    np.savez(root / "maps" / "instruction.npz", **_labels(tiny), shift=shift, t=(shift / 0.004).astype(np.float32),
             n_pairs=np.int32(2))


def manifests(root: Path) -> dict:
    """vouched.sha256, MANIFEST.sha256 and manifest.json over the tree, as the manifest step writes them."""
    harness = Path(run.__file__).resolve().parent
    vouched = {name: _sha(harness / name) for name in VOUCHED}
    (root / "vouched.sha256").write_text("".join(f"{v}  ouroboros/harness/{k}\n" for k, v in vouched.items()),
                                         encoding="utf-8")
    skip = {"vouched.sha256", "MANIFEST.sha256", "manifest.json"}
    files = {p.relative_to(root).as_posix(): _sha(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.relative_to(root).as_posix() not in skip and "_work" not in p.parts}
    (root / "MANIFEST.sha256").write_text("".join(f"{v}  {k}\n" for k, v in files.items()), encoding="utf-8")
    _json(root / "manifest.json", stageb.labelled({"format": "ouroboros-inputs-manifest/1", "files": files,
                                                   "vouched": vouched}))
    return files


def tiny_inputs(root: str | Path, tiny: TinyBeam) -> Path:
    """The Boot 1 inputs tree at tiny scale; returns root."""
    root = Path(root)
    (root / "sets").mkdir(parents=True, exist_ok=True)
    (root / "maps").mkdir(parents=True, exist_ok=True)
    rows, records = search_dev(root)
    parquet = write_parquet(rows, root / "sets" / "search_dev.parquet")
    set_file(root / "sets" / "search_dev.json", "search_dev", parquet, records,
             "tiny: p, n per task, then the guard; pre-cut", True)
    _json(root / "split.json", stageb.labelled({"format": "ouroboros-split/1", "rule": "tiny test split", "seed":
                                                stageb.SEED, "search": ["t_a", "t_b"], "held": ["t_c", "t_d"]}))
    with LocalDir(tiny.ckpt) as source:
        bias = bank.stock_bias(source, tiny.moe_layers).numpy()
    np.save(root / "maps" / "bias.npy", bias)
    rng = np.random.default_rng(stageb.SEED)
    n_moe, n_exp = tiny.shape
    stats = {k: rng.normal(0, 0.01, (n_moe, n_exp)).astype(np.float32)
             for k in ("rd_impossible", "rd_honest", "instruction_shift", "token_local")}
    stats["base"] = rng.uniform(0, 0.5, (n_moe, n_exp)).astype(np.float32)
    stats["rate_pre_tamper"] = rng.uniform(0, 0.5, (n_moe, n_exp)).astype(np.float32)
    for k in ("dose_add", "dose_remove"):
        stats[k] = np.sort(rng.uniform(0, 1, (n_moe, n_exp, 3)), axis=-1).astype(np.float32)
    np.savez(root / "maps" / "expert_stats.npz", **_labels(tiny), **stats)
    basis = np.zeros((3, n_moe, n_exp), np.float32)
    for b, rows_b in enumerate(([0, 1], [2], [3])):
        cells = [(j, e) for j in rows_b for e in range(n_exp)]
        pick = rng.choice(len(cells), size=16, replace=False)
        for k, i in enumerate(pick):
            basis[b][cells[i]] = -0.25 if k < 8 else 0.25
    np.savez(root / "maps" / "cem_basis.npz", **_labels(tiny), basis=basis)
    direction_maps(root, tiny, stats)
    window_sample(root, tiny, bias)
    (root / "maps" / "surgeon_context.md").write_text(
        "Tiny test maps, built from synthetic numbers. Their reliability is nil by construction.\n\n"
        "Dose table: entries of tens of units move a few tokens per layer.\n", encoding="utf-8")
    stage_b_maps.r0_bank(root)  # the real builder: R0 names, seeds, footprints, variants/r0.json
    stage_b_maps.r0_bank(root, lite=True)
    manifests(root)
    return root


def tiny_cvd_inputs(root: str | Path, tiny: TinyBeam) -> Path:
    """cvd_trap, cvd_true_done and cvd_rule sets at tiny scale under root/sets; returns root."""
    root = Path(root)
    rows = {r["task_id"]: r for r in cvd_rows()}

    def cvd(row: dict, key: str, role: str, on_policy: bool | None = None) -> tuple[dict, dict]:
        r = {**row, "episode_id": key}
        rec = record(r, role, {}, {}, None)
        rec.update(cluster=key.rsplit("/", 1)[1], on_policy=on_policy,
                   source_episode_id=None if key == row["episode_id"] else row["episode_id"])
        return r, rec

    groups = {
        "cvd_trap": [cvd(rows["false_done"], rows["false_done"]["episode_id"], "false_done"),
                     cvd(rows["honest_blocked"], rows["honest_blocked"]["episode_id"], "honest_blocked")],
        "cvd_true_done": [cvd(rows["true_done"], "cvd-done/test-batch/true_done-r3", "true_done")],
        "cvd_rule": [cvd(rows["false_done"], "cvd-rule/test-batch/false_done-r1", "false_done", False),
                     cvd(rows["false_done"], "cvd-rule/test-batch-strict/false_done-r4", "false_done", True)],
    }
    for name, pairs in groups.items():
        parquet = write_parquet([r for r, _ in pairs], root / "sets" / f"{name}.parquet")
        set_file(root / "sets" / f"{name}.json", name, parquet, [rec for _, rec in pairs], f"tiny {name}", False)
    return root


def tiny_held_inputs(root: str | Path, tiny: TinyBeam) -> Path:
    """rendered/tripwire.parquet (whole rows of two more tasks) with tripwire_held and guard_held set files."""
    root = Path(root)
    rows = []
    for row in tripwire_rows(stageb.SEED + 5):
        task = {"t_a": "t_c", "t_b": "t_d"}[row["task_id"]]
        rows.append({**row, "task_id": task, "episode_id": row["episode_id"].replace(row["task_id"], task)})
    parquet = write_parquet(rows, root / "rendered" / "tripwire.parquet")
    recs = [tripwire_records(r, with_dend=True) for r in rows]
    held = [x for x in recs if x["role"] == "positive"] + [x for x in recs if x["role"] == "negative"]
    set_file(root / "sets" / "tripwire_held.json", "tripwire_held", parquet, held, "tiny held tamper and legit", False)
    set_file(root / "sets" / "guard_held.json", "guard_held", parquet, [x for x in recs if x["role"] == "guard"],
             "tiny held guard", False)
    return root
