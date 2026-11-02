"""Stage B harness: one weight stream for the stock model and many patches, the patch bank, the sets, and the score
CLI. Everything runs on the tiny Beam, which keeps Beam's real vocabulary."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import shutil
import subprocess
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from stageb_support import CountingSource, one_thread, patch, tiny_beam, tripwire_rows, write_rendered  # noqa: F401

from ouroboros import stageb
from ouroboros.harness import bank, multi, run, score, sets, shards
from ouroboros.harness.capture import load
from ouroboros.harness.multi import LayoutGroup, Variant
from ouroboros.harness.patch import patched
from ouroboros.harness.stream import Streamer, build_layer

CANDIDATES = list(stageb.CANDIDATES)
ARRAYS = ("nll", "top_ids", "top_logprobs", "candidate_logprobs")
PATCHES = {"p3": {(3, 4): 2.0}, "p4": {(4, 7): -1.5, (4, 1): 1.0}, "p5": {(5, 9): 2.0}}
PROVENANCE = {"arm": "random", "round": None, "slot": None, "seed": 1, "parent": None, "built_by": "test",
              "inputs": {}}


def calls(row: dict) -> list[int]:
    """<|start|> of every Beam-written tool call (<|start|>assistant to=...)."""
    ids, gen = list(row["token_ids"]), list(row["gen_mask"])
    return [p for p in range(len(ids) - 2) if gen[p] and ids[p:p + 3] == [stageb.START, stageb.ASSISTANT, stageb.TO]]


def spans_of(row: dict) -> dict:
    """ctrl16 and edit16 as the sets define them, plus result8 over the fork tool result, which Beam did not write,
    so the spans add targets beyond run.py's."""
    spans = {} if row["f"] is None else {"result8": [row["f"] - 7, row["f"] + 1]}
    if row.get("ctrl") is not None:
        spans["ctrl16"] = [row["ctrl"], row["ctrl"] + 16]
    if row["label"] == 0:
        edit = next(p for p in calls(row) if p > row["f"])
        spans["edit16"] = [edit, edit + 16]
    return spans


def episodes(rows: list[dict], with_spans: bool = True) -> list[shards.Episode]:
    out = []
    for r in rows:
        e = shards.Episode(r["episode_id"], np.asarray(r["token_ids"], np.int64), np.asarray(r["gen_mask"], bool),
                           **{a: r[a] for a in shards.ANCHORS})
        out.append(sets.wrap(e, spans_of(r)) if with_spans else e)
    return out


def write_set(rows: list[dict], parquet, path, drop: int = 0):
    records = [{"key": r["episode_id"], "role": {1: "positive", 0: "negative", -1: "guard"}[r["label"]],
                "task_id": r["task_id"], "cluster": r["task_id"], "condition": r["condition"], "effort": r["effort"],
                "tool": None, "spans": spans_of(r), "span_tools": {}, "on_policy": None, "source_episode_id": None,
                "cut": None} for r in rows[:len(rows) - drop]]
    spec = {"format": sets.SET_FORMAT, **stageb.LABEL, "name": path.stem, "split_sha256": None,
            "parquet": parquet.name, "parquet_sha256": sets.sha256_file(parquet), "search_only": True,
            "rule": "test rows in parquet order", "n_episodes": len(records), "episodes": records}
    path.write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
    return path


def streamer_for(tiny, dtype=torch.bfloat16, chunk_tokens: int = 200) -> Streamer:
    return Streamer(tiny.config, CountingSource(tiny.ckpt), "cpu", dtype, chunk_tokens=chunk_tokens, attn="sdpa")


def stock_bias(tiny) -> torch.Tensor:
    return torch.stack([tiny.reference[f"model.layers.{i}.mlp.gate.e_score_correction_bias"] for i in tiny.moe_layers])


def plain(eps: list[sets.StageBEpisode]) -> list[shards.Episode]:
    """The same episodes as run.py loads them: no spans."""
    return [shards.Episode(e.key, e.token_ids, e.gen_mask, e.f, e.d, e.s, e.window_end, e.s_call) for e in eps]


def run_shard(streamer, eps, out, delta=None):
    return load(run.run_shard(streamer, eps, out, candidates=CANDIDATES, tap_layers=(), with_windows=False,
                              delta=delta))


@pytest.fixture(scope="module")
def rows():
    return tripwire_rows()


@pytest.fixture(scope="module")
def eps(rows):
    return episodes(rows[:3])  # task t_a: a positive with ctrl, a source-edit negative, a possible episode


@pytest.fixture(scope="module")
def bf16_pass(tiny_beam, eps):
    return patched_pass(tiny_beam, eps, torch.bfloat16)


def patched_pass(tiny_beam, eps, dtype) -> SimpleNamespace:
    """Stock, twin and patches forking at decoder layers 3, 4 and 5, in one pass that records its calls."""
    streamer = streamer_for(tiny_beam, dtype)
    group = LayoutGroup.build("g", eps, CANDIDATES)
    moe = tiny_beam.moe_layers
    variants = [Variant.stock("g"), Variant.make_twin("g", moe, 16),
                *(Variant.patch(name, "g", patch(tiny_beam, entries), moe) for name, entries in PATCHES.items())]
    applies, built, allocs = [], [], []
    apply, build = streamer.apply, streamer.build

    def counted_apply(layer, i, *args, **kwargs):
        applies.append(i)
        return apply(layer, i, *args, **kwargs)

    def kept_build(i, tensors):
        built.append((i, build(i, tensors)))
        return built[-1][1]

    streamer.apply, streamer.build = counted_apply, kept_build
    real = multi.host_tensor
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(multi, "host_tensor", lambda *a, **k: (allocs.append(built[-1][0]), real(*a, **k))[1])
        result = multi.run_pass(streamer, [group], variants, routes="ids")
    heads = multi.head_pass(streamer, [group], variants, result.hidden)
    return SimpleNamespace(streamer=streamer, group=group, variants=variants, result=result, heads=heads,
                           applies=applies, built=built, allocs=allocs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_multi_stock_equals_run_shard(tiny_beam, eps, tmp_path, dtype):
    streamer = streamer_for(tiny_beam, dtype)
    ref = run_shard(streamer, plain(eps), tmp_path / "shard")
    group = LayoutGroup.build("g", eps, CANDIDATES)
    assert len(group.plan.targets) > len(group.base_rows)  # the spans add targets beyond run.py's
    np.testing.assert_array_equal(group.plan.targets[group.base_rows], ref.array("targets"))
    variants = [Variant.stock("g")]
    result = multi.run_pass(streamer, [group], variants, routes="ids")
    stock = multi.head_pass(streamer, [group], variants, result.hidden)["g/stock"]
    for name in ARRAYS:
        np.testing.assert_array_equal(stock[name][group.base_rows], ref.array(name), err_msg=name)
    np.testing.assert_array_equal(result.route_ids["g/stock"], ref.array("route_ids"))
    assert not stock["kl_stock_v"].any() and not stock["kl_v_stock"].any() and stock["top1_agree"].all()
    assert not result.route_changed["g/stock"].any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_multi_patched_equals_independent_runs(request, tiny_beam, eps, tmp_path, dtype):
    p = request.getfixturevalue("bf16_pass") if dtype == torch.bfloat16 else patched_pass(tiny_beam, eps, dtype)
    assert p.streamer.source.loads == list(range(6))  # five variants, every layer read once
    assert [v.fork_layer for v in p.variants[1:]] == [3, 3, 4, 5]
    streamer = streamer_for(tiny_beam, dtype)
    for v in p.variants[2:]:
        ref = run_shard(streamer, plain(eps), tmp_path / v.name, v.delta)
        for name in ARRAYS:
            np.testing.assert_array_equal(p.heads[v.key][name][p.group.base_rows], ref.array(name),
                                          err_msg=f"{v.name} {name}")
        np.testing.assert_array_equal(p.result.route_ids[v.key], ref.array("route_ids"), err_msg=v.name)
        stock_ids = np.sort(p.result.route_ids["g/stock"], axis=-1)
        moved = (np.sort(p.result.route_ids[v.key], axis=-1) != stock_ids).any(axis=-1).sum(axis=1)
        np.testing.assert_array_equal(p.result.route_changed[v.key], moved)
        j = tiny_beam.moe_layers.index(v.fork_layer)
        assert moved[j] > 0 and not moved[:j].any()  # the patch moves routes from its fork on, never before


def test_fork_at_first_patched_layer(bf16_pass):
    p = bf16_pass
    forks = [v.fork_layer for v in p.variants if not v.is_stock]
    assert Counter(p.applies) == {i: 1 + sum(f <= i for f in forks) for i in range(6)}
    assert p.allocs == forks  # each buffer is allocated at its own fork layer, in variant order
    assert list(p.result.hidden) == ["g/twin", "g/p3", "g/p4", "g/p5", "g/stock"]


def test_variant_order_and_bias_isolation(bf16_pass, tiny_beam):
    p = bf16_pass
    for i, layer in p.built:
        if tiny_beam.config.mlp_layer_types[i] == "sparse":
            stock = tiny_beam.reference[f"model.layers.{i}.mlp.gate.e_score_correction_bias"]
            assert torch.equal(layer.mlp.gate.e_score_correction_bias, stock), i
    streamer = streamer_for(tiny_beam)
    reversed_order = [p.variants[0], *reversed(p.variants[1:])]
    result = multi.run_pass(streamer, [p.group], reversed_order, routes="ids")
    heads = multi.head_pass(streamer, [p.group], reversed_order, result.hidden)
    for v in p.variants:
        for name, a in heads[v.key].items():
            np.testing.assert_array_equal(a, p.heads[v.key][name], err_msg=f"{v.name} {name}")
        np.testing.assert_array_equal(result.route_ids[v.key], p.result.route_ids[v.key])
        np.testing.assert_array_equal(result.route_changed[v.key], p.result.route_changed[v.key])


def test_head_kl_matches_reference(bf16_pass):
    p = bf16_pass
    t = p.group.plan.targets

    def logp(key: str, rows: slice) -> torch.Tensor:
        h = p.result.hidden[key][torch.from_numpy(t[rows] - 1)]
        return torch.log_softmax(p.streamer.logits(h).double(), dim=-1)

    stock = p.heads["g/stock"]
    for v in p.variants[1:]:
        got = p.heads[v.key]
        for a in range(0, len(t), 64):
            rows = slice(a, a + 64)
            ls, lv = logp("g/stock", rows), logp(v.key, rows)
            np.testing.assert_allclose(got["kl_stock_v"][rows], (ls.exp() * (ls - lv)).sum(-1).numpy(), rtol=0,
                                       atol=1e-6)
            np.testing.assert_allclose(got["kl_v_stock"][rows], (lv.exp() * (lv - ls)).sum(-1).numpy(), rtol=0,
                                       atol=1e-6)
            # Top-1 is the first maximal logit, so exact BF16 ties resolve the same way in both.
            np.testing.assert_array_equal(got["top1_agree"][rows], (lv.argmax(-1) == ls.argmax(-1)).numpy())
    twin = p.heads["g/twin"]
    assert not twin["kl_stock_v"].any() and not twin["kl_v_stock"].any() and twin["top1_agree"].all()
    for name in ARRAYS:
        np.testing.assert_array_equal(twin[name], stock[name])
    assert not p.result.route_changed["g/twin"].any()
    assert p.heads["g/p3"]["kl_stock_v"].max() > 1e-4  # a real patch moves the next-token distribution


def test_reordered_stock_alignment(tiny_beam, rows):
    eps6 = episodes(rows)
    g = LayoutGroup.build("g", eps6, CANDIDATES)
    re = LayoutGroup.build("re", sets.permute(eps6), CANDIDATES, reference="g")
    assert [e.key for e in re.episodes] != [e.key for e in g.episodes]
    tok = sets.align_tokens(g.episodes, g.layout, re.episodes, re.layout)
    np.testing.assert_array_equal(g.layout.token_ids[tok], re.layout.token_ids)
    np.testing.assert_array_equal(g.layout.positions[tok], re.layout.positions)
    align = sets.align_targets(g.episodes, g.layout, g.plan, re.episodes, re.layout, re.plan)
    np.testing.assert_array_equal(g.plan.targets[align], tok[re.plan.targets])
    with pytest.raises(ValueError, match="keys"):
        sets.align_tokens(g.episodes[1:], g.layout, re.episodes, re.layout)

    streamer = streamer_for(tiny_beam, torch.float32)
    variants = [Variant.stock("g"), Variant.stock("re")]
    result = multi.run_pass(streamer, [g, re], variants)
    heads = multi.head_pass(streamer, [g, re], variants, result.hidden)
    np.testing.assert_allclose(heads["re/stock"]["nll"], heads["g/stock"]["nll"][align], rtol=0, atol=1e-5)
    assert heads["re/stock"]["kl_stock_v"].max() < 1e-6 and heads["re/stock"]["top1_agree"].all()
    assert result.route_changed["re/stock"].sum() <= 0.01 * re.layout.n_tokens * len(tiny_beam.moe_layers)


def test_patch_bank(tiny_beam, tmp_path, caplog):
    moe = tiny_beam.moe_layers
    with CountingSource(tiny_beam.ckpt) as source:
        bias = bank.stock_bias(source, moe)
        assert source.reads == ["tensors 4"]
        layer = build_layer(tiny_beam.config, 3, source.load_layer(3, "cpu", torch.float32), torch.device("cpu"),
                            torch.float32)
    assert torch.equal(bias, stock_bias(tiny_beam)) and bias.dtype == torch.float32

    units = [(3, 4, 25), (5, 9, -7)]
    a = bank.units_to_delta(units, moe, 16)
    assert bank.delta_to_units(a, moe) == units and a[1, 4] == np.float32(25e-4)
    for bad in ([(1, 0, 3)], [(3, 16, 3)], [(3, 0, 0)], [(3, 0, 1), (3, 0, 2)]):  # dense, expert, zero, repeat
        with pytest.raises(ValueError):
            bank.units_to_delta(bad, moe, 16)
    cells = np.random.default_rng(0).choice(64, 32, replace=False)
    direction = torch.zeros(64)
    direction[cells] = torch.from_numpy(np.where(np.random.default_rng(1).random(32) < 0.5, -1.0, 1.0)).float()
    direction = direction.view(4, 16)

    path = tmp_path / "bank.safetensors"
    meta = {"cem_r01a": {"provenance": {**PROVENANCE, "arm": "cem", "round": 1, "slot": "a"}, "units": units},
            "random_b010_k0": {"provenance": PROVENANCE, "footprint": {"share": 0.1, "per_layer": [0.1] * 4,
                                                                      "sample_sha256": "x"}}}
    head = bank.save_patch_bank(path, {"cem_r01a": a, "random_b010_k0": direction * 1e-3}, meta, bias, moe)
    tensors, got = bank.load_patch_bank(path, 4, 16, bias=bias)
    assert got == head and torch.equal(tensors["cem_r01a"], a) and list(head)[:3] == ["format", "exploratory",
                                                                                         "reason"]
    again = tmp_path / "again.safetensors"
    for _ in range(6):  # safetensors writes its metadata in hash order; the bank fixes the order
        bank.save_patch_bank(again, {"cem_r01a": a, "random_b010_k0": direction * 1e-3}, meta, bias, moe)
        assert again.read_bytes() == path.read_bytes()
    record = head["patches"]["cem_r01a"]
    assert record["units"] == [list(u) for u in units] and record["exploratory"] is True
    assert record["provenance"]["arm"] == "cem" and head["patches"]["random_b010_k0"]["footprint"]["share"] == 0.1
    assert record["effective_l2"] == pytest.approx(record["nominal_l2"], rel=1e-5) and record["nonzero"] == 2

    with pytest.raises(ValueError, match="floor"):
        bank.check_floor(patch(tiny_beam, {(3, 1): 5e-6}))
    with pytest.raises(ValueError, match="floor"):
        bank.save_patch_bank(tmp_path / "x.safetensors", {"jitter": patch(tiny_beam, {(3, 1): 5e-6})},
                             {"jitter": {"provenance": PROVENANCE}}, bias, moe)
    for name in ("stock", "twin", "Beam", "a-b"):
        with pytest.raises(ValueError, match="patch name"):
            bank.save_patch_bank(tmp_path / "x.safetensors", {name: a}, {name: {"provenance": PROVENANCE}}, bias, moe)
    with pytest.raises(ValueError, match="units"):
        bank.save_patch_bank(tmp_path / "x.safetensors", {"cem_r01b": a * 2}, {"cem_r01b": meta["cem_r01a"]}, bias,
                             moe)

    # The effective delta is what patched() leaves in the bias, bit for bit, also at Beam's scale.
    d = torch.randn(4, 16, generator=torch.Generator().manual_seed(2)) * 1e-3
    stock = layer.mlp.gate.e_score_correction_bias.clone()
    with patched(layer, d[1]):
        left = layer.mlp.gate.e_score_correction_bias - stock
    assert torch.equal(left, bank.effective_delta(bias, d)[1])
    beam_bias = 20 + torch.rand(4, 16, generator=torch.Generator().manual_seed(3)) * 8
    eff = bank.effective_delta(beam_bias, d)
    assert not torch.equal(eff, d) and (eff - d).abs().max() <= 2 ** -19  # within one FP32 step at 16 to 32

    for target in (1e-4, 0.01, 0.03):
        scaled = bank.scale_to_effective_l2(direction, bias, target)
        assert abs(bank.effective_l2(bias, scaled) - target) <= 1e-6 * target
        assert bank.patch_sha256(scaled) != bank.patch_sha256(direction)
    with caplog.at_level(logging.WARNING):
        coarse = bank.scale_to_effective_l2(direction, beam_bias, 0.03)
    assert abs(bank.effective_l2(beam_bias, coarse) - 0.03) <= 2e-4 * 0.03 * 5 and "coarser" in caplog.text
    with pytest.raises(ValueError, match="floor"):
        bank.scale_to_effective_l2(direction, bias, 1e-6)

    forged = json.loads(json.dumps(head))
    forged["patches"]["cem_r01a"]["sha256"] = "0" * 64
    save_file(dict(tensors), str(tmp_path / "forged.safetensors"),
              metadata={"format": "pt", "ouroboros": json.dumps(forged)})
    with pytest.raises(ValueError, match="sha256"):
        bank.load_patch_bank(tmp_path / "forged.safetensors", 4, 16)
    with pytest.raises(ValueError, match="stock bias"):
        bank.load_patch_bank(path, 4, 16, bias=bias + 1)
    with pytest.raises(ValueError, match=r"\[4, 16\]"):
        bank.load_patch_bank(path, 49, 256)
    unlabelled = {**head, "exploratory": False}
    save_file(dict(tensors), str(tmp_path / "unlabelled.safetensors"),
              metadata={"format": "pt", "ouroboros": json.dumps(unlabelled)})
    with pytest.raises(ValueError, match="exploratory"):
        bank.load_patch_bank(tmp_path / "unlabelled.safetensors", 4, 16)

    beam = bank.units_to_delta([(4, 2, 30)], moe, 16)
    merged = bank.merge_banks(tmp_path / "merged.safetensors",
                              [(tensors, head), ({"beam_r01a": beam}, {"beam_r01a": {
                                  "provenance": {**PROVENANCE, "arm": "beam", "round": 1, "slot": "a"},
                                  "units": [[4, 2, 30]], "hypothesis": "lift expert 2 at layer 4"}})], bias, moe)
    assert list(merged["patches"]) == ["cem_r01a", "random_b010_k0", "beam_r01a"]
    assert merged["patches"]["beam_r01a"]["hypothesis"] == "lift expert 2 at layer 4"
    assert merged["patches"]["cem_r01a"] == head["patches"]["cem_r01a"]
    with pytest.raises(ValueError, match="two banks"):
        bank.merge_banks(tmp_path / "dup.safetensors", [(tensors, head), (tensors, head)], bias, moe)


def test_sets_and_vouched(rows, tmp_path):
    eps6 = episodes(rows)
    keys = [e.key for e in eps6]
    assert [e.key for e in sets.select(eps6, keys[::-1])] == keys[::-1]
    with pytest.raises(KeyError):
        sets.select(eps6, ["tripwire/none"])
    with pytest.raises(KeyError):
        sets.select(eps6, [keys[0], keys[0]])
    order = np.random.default_rng(stageb.SEED).permutation(6)
    assert [e.key for e in sets.permute(eps6)] == [keys[i] for i in order]

    pos, neg, possible = eps6[:3]
    assert pos.span("ctrl16") == (rows[0]["ctrl"], rows[0]["ctrl"] + 16) and pos.span("dend") is None
    bare = episodes(rows[:1], with_spans=False)[0]
    np.testing.assert_array_equal(pos.base_targets(), bare.targets())
    targets = set(pos.targets().tolist())
    assert set(bare.targets().tolist()) <= targets and 0 not in targets and set(range(*pos.span("ctrl16"))) <= targets
    assert pos.last_needed() == bare.last_needed()  # ctrl16 ends before d + 16
    capped = possible.capped(50)
    assert (len(capped.token_ids), capped.cap, capped.full_length) == (50, 50, len(possible.token_ids))
    assert pos.capped(50) is pos and neg.capped(50) is neg and possible.capped(10_000) is possible
    with pytest.raises(ValueError, match="outside"):
        sets.wrap(bare, {"x": [0, len(bare.token_ids) + 1]})

    parquet = write_rendered(rows, tmp_path / "t.parquet")
    spec = write_set(rows, parquet, tmp_path / "t.json")
    got, info = sets.load_set(parquet, spec, "t")
    assert [e.key for e in got] == keys and [e.spans for e in got] == [e.spans for e in eps6]
    assert info["parquet_sha256"] == sets.sha256_file(parquet) and info["episodes_sha256"] == sets.sha256_file(spec)
    whole, info = sets.load_set(parquet, None, "t")
    assert [e.key for e in whole] == keys and not any(e.spans for e in whole) and info["records"] is None
    base = json.loads(spec.read_text(encoding="utf-8"))
    for change, match in (({"format": "other/1"}, "format"), ({"exploratory": False}, "exploratory"),
                          ({"parquet_sha256": "0" * 64}, "sha256"),
                          ({"episodes": [{**base["episodes"][0], "key": "tripwire/none"}], "n_episodes": 1},
                           "no episode")):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({**base, **change}), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            sets.load_set(parquet, bad, "t")
    assert sets.check_vouched() == stageb.VOUCHED_SHA256 == sets.vouched_hashes()


def test_truncation_keeps_spans(rows, tiny_beam, tmp_path):
    neg = episodes(rows[1:2])[0]
    bare = episodes(rows[1:2], with_spans=False)[0]
    a, b = neg.span("edit16")
    assert b > bare.last_needed()  # the first source edit lies past window_end, where run.py's truncation cuts
    cut = neg.truncated(0)
    assert len(cut.token_ids) == b and cut.spans == neg.spans and isinstance(cut, sets.StageBEpisode)
    assert set(range(a, b)) <= set(cut.targets().tolist())
    assert len(bare.truncated(0).token_ids) < b

    parquet = write_rendered(rows, tmp_path / "t.parquet")
    spec = write_set(rows, parquet, tmp_path / "t.json")
    args = score.parser().parse_args(["--weights", "w", "--out", "o", "--set", f"t={parquet}:{spec}", "--truncate"])
    (group,) = score.build_groups(args, tiny_beam.config, run.tokenizer())
    got = {e.key: e for e in group.episodes}[neg.key]
    assert len(got.token_ids) == b and got.span("edit16") == (a, b)
    t0 = group.layout.offsets[[e.key for e in group.episodes].index(neg.key)]
    assert set(range(t0 + a, t0 + b)) <= set(group.plan.targets.tolist())


def test_cap_and_search_dev_parity(tiny_beam, rows, tmp_path, monkeypatch):
    cut_rows = []
    for r in rows:  # cut as search_dev cuts: positives at d + 16, negatives at edit + 16, the rest to a cap
        if r["label"] == 1:
            cut = r["d"] + 16
        elif r["label"] == 0:
            cut = spans_of(r)["edit16"][1]
        else:
            cut = min(len(r["token_ids"]), 100)
        cut_rows.append({**r, "token_ids": r["token_ids"][:cut], "gen_mask": r["gen_mask"][:cut],
                         **{a: None if r[a] is None or r[a] >= cut else r[a] for a in shards.ANCHORS}})
    parquet = write_rendered(cut_rows, tmp_path / "search_dev.parquet")
    spec = write_set(cut_rows, parquet, tmp_path / "search_dev.json")
    source = CountingSource(tiny_beam.ckpt)
    monkeypatch.setattr(run, "open_source", lambda location, **kw: contextlib.nullcontext(source))
    monkeypatch.setattr(score, "open_source", lambda location, **kw: contextlib.nullcontext(source))
    common = ["--weights", "unused", "--config", str(tiny_beam.ckpt), "--device", "cpu", "--attn", "sdpa"]
    assert run.main([*common, "--rendered", str(parquet), "--out", str(tmp_path / "g1a"), "--no-windows", "--taps",
                     "--shard-gb", "100"]) == 0
    assert score.main([*common, "--set", f"search_dev={parquet}:{spec}", "--out", str(tmp_path / "pass")]) == 0
    shard = load(tmp_path / "g1a" / "shard_000")
    pj = json.loads((tmp_path / "pass" / "pass.json").read_text(encoding="utf-8"))
    assert shard.index["fingerprint"] == pj["groups"]["search_dev"]["fingerprint"]
    gdir = tmp_path / "pass" / "search_dev"
    targets = np.load(gdir / "targets.npy")
    rows_in = np.searchsorted(targets, shard.array("targets"))
    np.testing.assert_array_equal(targets[rows_in], shard.array("targets"))
    assert len(targets) > len(rows_in)  # result8 adds targets
    for name in ARRAYS:
        np.testing.assert_array_equal(np.load(gdir / "variant_stock" / f"{name}.npy")[rows_in], shard.array(name))

    full = write_rendered(rows, tmp_path / "full.parquet")
    args = score.parser().parse_args(["--weights", "w", "--out", "o", "--set", f"full={full}", "--cap", "full=60"])
    (group,) = score.build_groups(args, tiny_beam.config, run.tokenizer())
    for e, r in zip(group.episodes, rows):
        anchorless = all(r[a] is None for a in shards.ANCHORS)
        assert len(e.token_ids) == (60 if anchorless else len(r["token_ids"])), e.key
        assert e.cap == (60 if anchorless else None) and e.full_length == len(r["token_ids"])
    with pytest.raises(ValueError, match="set nope"):
        score.build_groups(score.parser().parse_args(["--weights", "w", "--out", "o", "--set", f"full={full}",
                                                      "--cap", "nope=60"]), tiny_beam.config, run.tokenizer())


@pytest.fixture
def cli(tiny_beam, rows, tmp_path, monkeypatch):
    source = CountingSource(tiny_beam.ckpt)
    monkeypatch.setattr(score, "open_source", lambda location, **kw: contextlib.nullcontext(source))
    parquet = write_rendered(rows[:3], tmp_path / "trip.parquet")
    spec = write_set(rows[:3], parquet, tmp_path / "trip.json")

    def write_bank(name: str, jitter: float) -> tuple[str, str]:
        path = tmp_path / f"{name}.safetensors"
        patches = {"random_b010_k0": patch(tiny_beam, {(3, 4): 2.0}), "jitter": patch(tiny_beam, {(5, 9): jitter})}
        bank.save_patch_bank(path, patches, {k: {"provenance": PROVENANCE} for k in patches}, stock_bias(tiny_beam),
                             tiny_beam.moe_layers)
        variants = tmp_path / f"{name}.json"
        variants.write_text(json.dumps({"format": score.VARIANTS_FORMAT, **stageb.LABEL,
                                        "bank_sha256": sets.sha256_file(path),
                                        "variants": [{"patch": "random_b010_k0", "groups": ["trip"]},
                                                     {"patch": "jitter", "groups": ["trip"]}]}), encoding="utf-8")
        return str(path), str(variants)

    bank_path, variants = write_bank("bank", 1e-4)
    base = ["--weights", "unused", "--config", str(tiny_beam.ckpt), "--device", "cpu", "--attn", "sdpa",
            "--chunk-tokens", "200", "--set", f"trip={parquet}:{spec}", "--twin"]
    return SimpleNamespace(base=base, bank=["--bank", bank_path, "--variants", variants], source=source,
                           tmp=tmp_path, write_bank=write_bank, rows=rows[:3], parquet=parquet, tiny=tiny_beam)


def test_plan_refuses_before_reading(cli, capsys):
    out = cli.tmp / "plan"
    with pytest.raises(SystemExit, match="plan refused.*host budget"):
        score.main([*cli.base, *cli.bank, "--host-gb", "1e-6", "--out", str(out)])
    fixed = multi.PREFETCH_GB + multi.GLOBALS_GB
    with pytest.raises(SystemExit, match="plan refused.*prefetch buffers and globals"):  # fits without the fixed GB
        score.main([*cli.base, *cli.bank, "--mem-available-gb", str(multi.RESERVE_GB + fixed - 0.1), "--out",
                    str(out)])
    assert cli.source.reads == [] and not out.exists()
    assert score.main([*cli.base, *cli.bank, "--mem-available-gb", str(multi.RESERVE_GB + fixed + 0.1), "--plan",
                       "--out", str(out)]) == 0
    assert "group trip: 3 episodes" in capsys.readouterr().out
    assert cli.source.reads == [] and not out.exists()


def test_score_cli_resume_and_refusals(cli, capsys, monkeypatch):
    out, mirror = cli.tmp / "pass", cli.tmp / "mirror"
    written = []
    write_json, save = score._write_json, score._save
    monkeypatch.setattr(score, "_write_json", lambda p, o: (written.append(p), write_json(p, o))[1])
    monkeypatch.setattr(score, "_save", lambda p, a: (written.append(p), save(p, a))[1])
    args = [*cli.base, *cli.bank, "--routes", "ids", "--out", str(out), "--sync", str(mirror)]
    assert score.main(args) == 0
    assert written[-1] == out / "pass.json" and written.count(out / "pass.json") == 1
    pj = json.loads((out / "pass.json").read_text(encoding="utf-8"))
    assert list(pj)[:3] == ["exploratory", "reason", "format"] and pj["exploratory"] is True  # label first (5.1)
    assert list(pj)[-1] == "settings" and pj["vouched_ok"] and pj["vouched_sha256"] == stageb.VOUCHED_SHA256
    assert list(pj["variants"]) == ["trip/stock", "trip/random_b010_k0", "trip/jitter", "trip/twin"]  # twin last
    assert pj["bias_sha256"] == bank.patch_sha256(stock_bias(cli.tiny)) == pj["settings"]["bias_sha256"]
    assert pj["variants"]["trip/twin"]["fork_layer"] == 3 and pj["variants"]["trip/jitter"]["fork_layer"] == 5
    assert pj["variants"]["trip/random_b010_k0"]["route_change_share"] > 0
    assert pj["groups"]["trip"]["n_episodes"] == 3 and pj["candidates"] == CANDIDATES
    gdir = out / "trip"
    eps_json = json.loads((gdir / "episodes.json").read_text(encoding="utf-8"))
    assert eps_json["exploratory"] is True and eps_json["format"] == score.EPISODES_FORMAT
    first = eps_json["episodes"][0]
    assert first["spans"] == spans_of(cli.rows[0]) and first["meta"]["role"] == "positive"
    assert "windows" not in first and first["target_rows"][0] == 0
    stock, twin = gdir / "variant_stock", gdir / "variant_twin"
    for name in (*ARRAYS, "route_ids"):
        np.testing.assert_array_equal(np.load(stock / f"{name}.npy"), np.load(twin / f"{name}.npy"))
    variant = json.loads((gdir / "variant_jitter" / "variant.json").read_text(encoding="utf-8"))
    assert list(variant)[:2] == ["exploratory", "reason"] and variant["effective_l2"] > 0
    assert len(variant["seconds_per_layer"]) == 6 and variant["route_tokens"] == pj["groups"]["trip"]["n_tokens"]
    assert np.load(gdir / "target_gen.npy").dtype == bool and np.load(gdir / "token_ids.npy").dtype == np.int32
    assert json.loads((mirror / "pass.json").read_text(encoding="utf-8")) == pj
    assert sorted(p.name for p in (mirror / "trip").iterdir()) == sorted(p.name for p in gdir.iterdir())

    reads = list(cli.source.reads)
    assert score.main(args) == 0
    shutil.rmtree(out)  # a wiped disk: the mirror alone vouches for the pass
    assert score.main(args) == 0
    assert capsys.readouterr().out.count("pass: done, skipping") == 2 and cli.source.reads == reads

    shutil.copytree(mirror, out)
    with pytest.raises(SystemExit, match="other settings"):
        score.main([*args, "--chunk-tokens", "100"])
    bank_path, variants = cli.write_bank("bank2", 2e-4)
    with pytest.raises(SystemExit, match="other settings"):
        score.main([*cli.base, "--bank", bank_path, "--variants", variants, "--routes", "ids", "--out", str(out)])
    shorter = write_set(cli.rows, cli.parquet, cli.tmp / "short.json", drop=1)
    with pytest.raises(SystemExit, match="other settings"):
        score.main([f"trip={cli.parquet}:{shorter}" if a.startswith("trip=") else a for a in args])
    assert cli.source.reads == reads

    _, stale = cli.write_bank("bank3", 3e-4)
    with pytest.raises(SystemExit, match="bank"):  # a variants file written for another bank
        score.main([*cli.base, "--bank", cli.bank[1], "--variants", stale, "--out", str(cli.tmp / "other")])
    with pytest.raises(SystemExit, match="bank"):
        score.main([*cli.base, "--variants", cli.bank[3], "--out", str(cli.tmp / "other")])

    redo = cli.tmp / "redo"
    (redo / "trip").mkdir(parents=True)
    (redo / "trip" / "junk.npy").write_text("x")  # a pass cut short, without pass.json
    empty = write_set(cli.rows, cli.parquet, cli.tmp / "empty.json", drop=len(cli.rows))  # a set read "not run"
    assert score.main([*cli.base, "--set", f"empty={cli.parquet}:{empty}", "--out", str(redo)]) == 0
    assert not (redo / "trip" / "junk.npy").exists() and (redo / "pass.json").exists()
    assert sorted(p.name for p in (redo / "trip").iterdir() if p.is_dir()) == ["variant_stock", "variant_twin"]
    pj = json.loads((redo / "pass.json").read_text(encoding="utf-8"))
    assert pj["groups"]["empty"]["n_episodes"] == 0 and np.load(redo / "empty" / "variant_twin" / "nll.npy").size == 0
    redo_args = [*cli.base, "--set", f"empty={cli.parquet}:{empty}", "--out", str(redo)]
    with pytest.raises(SystemExit, match="without --routes ids"):  # a skip would return no route_ids
        score.main([*redo_args, "--routes", "ids"])
    (redo / "pass.json").write_text('{"format": "ouroboros-pa', encoding="utf-8")  # a write killed halfway
    assert score.main(redo_args) == 0
    assert json.loads((redo / "pass.json").read_text(encoding="utf-8"))["settings"] == pj["settings"]
    assert not list(redo.glob("*.partial"))


def test_pass_mirror_runs_gcloud(monkeypatch, tmp_path):
    calls_made = []

    def fake_run(cmd, **kwargs):
        calls_made.append(cmd)
        missing = "ERROR: (gcloud.storage.cat) The following URLs matched no objects or files:"
        return subprocess.CompletedProcess(cmd, 1 if cmd[2] == "cat" else 0, "", missing)

    monkeypatch.setattr(score.shutil, "which", lambda name: "/usr/bin/gcloud")
    monkeypatch.setattr(score.subprocess, "run", fake_run)
    mirror = score.PassMirror("gs://bucket/pass/")
    assert mirror.pass_json() is None
    assert calls_made[0] == ["/usr/bin/gcloud", "storage", "cat", "gs://bucket/pass/pass.json"]
    out = tmp_path / "pass"
    (out / "g" / "variant_stock").mkdir(parents=True)
    (out / "g" / "variant_stock" / "nll.npy").write_text("x")
    (out / "pass.json").write_text("{}")
    mirror.put(out)
    assert calls_made[1] == ["/usr/bin/gcloud", "storage", "cp", "--recursive", str(out / "g"), "gs://bucket/pass/"]
    assert calls_made[2] == ["/usr/bin/gcloud", "storage", "cp", str(out / "pass.json"), "gs://bucket/pass/pass.json"]


def test_parse_specs_take_drive_letters():
    assert score.parse_set("cvd_trap=C:/x/cvd.parquet:D:/y/cvd_trap.json") == (
        "cvd_trap", score.Path("C:/x/cvd.parquet"), score.Path("D:/y/cvd_trap.json"))
    assert score.parse_set("search_dev=/srv/work/data/stage-b/sets/search_dev.parquet")[2] is None
    assert score.parse_cap("guard=8192") == ("guard", 8192)
    assert score.parse_reorder("guard_reordered=guard_held") == ("guard_reordered", "guard_held", stageb.SEED)
    assert score.parse_reorder("a=b:7") == ("a", "b", 7)
    for bad in ("Cvd=x.parquet", "x=y.csv"):
        with pytest.raises(argparse.ArgumentTypeError):
            score.parse_set(bad)
