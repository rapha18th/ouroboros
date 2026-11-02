"""Capture shapes and contents per the pre-registration: routing for every token, full scores and taps in windows."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_stream_support import make_episodes, one_thread, reference, tiny  # noqa: F401

from ouroboros.harness.capture import (
    CapturePlan,
    bf16_bits_to_float32,
    host_tensor,
    is_complete,
    load,
    to_numpy,
    topk_set_agreement,
)
from ouroboros.harness.run import run_shard
from ouroboros.harness.stream import Streamer

TAPS = (2, 5)
CANDIDATES = (7, 11)


@pytest.fixture(scope="module")
def shard(tiny, tmp_path_factory):
    config, model, source = tiny
    episodes = make_episodes()
    streamer = Streamer(config, source, "cpu", torch.float32, chunk_tokens=64, attn="sdpa")
    path = run_shard(streamer, episodes, tmp_path_factory.mktemp("cap") / "shard_000", candidates=CANDIDATES,
                     tap_layers=TAPS, top_k=5, meta={"note": "test"})
    refs = [reference(model, e.token_ids) for e in episodes]
    return config, episodes, streamer, load(path), refs


def test_files_shapes_and_dtypes(shard):
    config, episodes, streamer, out, _ = shard
    n_tok = sum(len(e.token_ids) for e in episodes)
    n_moe, k, n_exp = len(streamer.moe_layers), config.num_experts_per_tok, config.num_experts
    n_win = sum(b - a for e in episodes for a, b in e.windows())
    n_tgt = sum(len(e.targets()) for e in episodes)
    expect = {
        "route_ids": ((n_moe, n_tok, k), np.uint8),
        "route_scores": ((n_moe, n_tok, k), np.float16),
        "window_scores": ((n_moe, n_win, n_exp), np.float16),
        "tap_hidden": ((len(TAPS), n_win, config.hidden_size), np.float32),
        "windows": ((n_win,), np.int64),
        "targets": ((n_tgt,), np.int64),
        "nll": ((n_tgt,), np.float32),
        "top_ids": ((n_tgt, 5), np.int32),
        "top_logprobs": ((n_tgt, 5), np.float32),
        "candidate_logprobs": ((n_tgt, len(CANDIDATES)), np.float32),
        "target_gen": ((n_tgt,), np.bool_),
        "token_ids": ((n_tok,), np.int32),
        "gen_mask": ((n_tok,), np.bool_),
    }
    for name, (shape, dtype) in expect.items():
        a = out.array(name)
        assert a.shape == shape and a.dtype == dtype, name
    assert is_complete(out.path) and not out.path.with_name(out.path.name + ".partial").exists()
    assert out.index["note"] == "test" and out.index["tap_layers"] == list(TAPS)
    assert out.index["layers_run"] == list(range(config.num_hidden_layers))
    assert out.array("candidates").tolist() == list(CANDIDATES)


def test_index_maps_episodes_to_rows(shard):
    _, episodes, _, out, _ = shard
    windows, targets = out.array("windows"), out.array("targets")
    for ep, entry in zip(episodes, out.index["episodes"]):
        assert entry["key"] == ep.key and entry["length"] == len(ep.token_ids)
        assert (entry["f"], entry["d"], entry["s"], entry["window_end"]) == (ep.f, ep.d, ep.s, ep.window_end)
        w = windows[slice(*entry["window_rows"])] - entry["start"]
        assert w.tolist() == [p for a, b in ep.windows() for p in range(a, b)]
        t = targets[slice(*entry["target_rows"])] - entry["start"]
        assert t.tolist() == ep.targets().tolist()
        tokens = slice(entry["start"], entry["start"] + entry["length"])
        np.testing.assert_array_equal(out.array("token_ids")[tokens], ep.token_ids)
        np.testing.assert_array_equal(out.array("gen_mask")[tokens], ep.gen_mask)
        # Beam-written window rows: what Stage A features count.
        w_abs = windows[slice(*entry["window_rows"])]
        np.testing.assert_array_equal(out.array("gen_mask")[w_abs], ep.gen_mask[w])


def test_routing_for_every_token(shard):
    _, episodes, streamer, out, refs = shard
    ids, scores = out.array("route_ids"), out.array("route_scores")
    for entry, ref in zip(out.index["episodes"], refs):
        rows = slice(entry["start"], entry["start"] + entry["length"])
        for j, layer in enumerate(streamer.moe_layers):
            logits, want = (x.numpy() for x in ref["routers"][layer])
            np.testing.assert_array_equal(ids[j, rows], want)
            sig = torch.sigmoid(torch.from_numpy(logits)).numpy()
            np.testing.assert_allclose(scores[j, rows], np.take_along_axis(sig, want, 1), atol=1e-3)


def test_window_scores_and_taps(shard):
    _, episodes, streamer, out, refs = shard
    taps, wscores = out.tap_hidden(), out.array("window_scores")
    for entry, ref in zip(out.index["episodes"], refs):
        rows = slice(*entry["window_rows"])
        pos = out.array("windows")[rows] - entry["start"]
        for t, layer in enumerate(TAPS):
            np.testing.assert_allclose(taps[t, rows], ref["layers"][layer].numpy()[pos], atol=1e-5, rtol=1e-5)
        for j, layer in enumerate(streamer.moe_layers):
            sig = torch.sigmoid(ref["routers"][layer][0]).numpy()[pos]
            np.testing.assert_allclose(wscores[j, rows], sig, atol=1e-3)


def test_head_at_targets(shard):
    _, episodes, _, out, refs = shard
    for ep, entry, ref in zip(episodes, out.index["episodes"], refs):
        rows = slice(*entry["target_rows"])
        t = out.array("targets")[rows] - entry["start"]
        logp = torch.log_softmax(ref["logits"], dim=-1).numpy()[t - 1]
        np.testing.assert_allclose(out.array("nll")[rows], -logp[np.arange(len(t)), ep.token_ids[t]], atol=1e-5)
        np.testing.assert_allclose(out.array("candidate_logprobs")[rows], logp[:, list(CANDIDATES)], atol=1e-5)
        np.testing.assert_allclose(out.array("top_logprobs")[rows], -np.sort(-logp, axis=1)[:, :5], atol=1e-5)
        np.testing.assert_array_equal(out.array("target_gen")[rows], ep.gen_mask[t])


def test_bf16_bits_round_trip():
    x = torch.randn(4, 7).to(torch.bfloat16)
    bits = to_numpy(x)
    assert bits.dtype == np.uint16
    np.testing.assert_array_equal(bf16_bits_to_float32(bits), x.float().numpy())


def test_host_tensor_is_exact_size():
    t = host_tensor((3, 5), torch.bfloat16)
    assert t.shape == (3, 5) and t.dtype == torch.bfloat16 and t.is_contiguous()
    assert t.untyped_storage().nbytes() == 3 * 5 * 2
    t.fill_(1.5)
    np.testing.assert_array_equal(bf16_bits_to_float32(to_numpy(t)), np.full((3, 5), 1.5, np.float32))
    assert host_tensor((0, 4), torch.float32).shape == (0, 4)


def test_topk_set_agreement_ignores_order():
    a = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    b = np.array([[[3, 2, 1], [4, 5, 7]]], dtype=np.uint8)
    assert topk_set_agreement(a, b) == 0.5
    assert topk_set_agreement(a, a[:, :, ::-1]) == 1.0


def test_plan_requires_sorted_rows():
    with pytest.raises(ValueError):
        CapturePlan(np.array([3, 1]), np.array([], np.int64), np.array([], np.int64))


def test_tap_layers_must_exist(tiny, tmp_path):
    config, _, source = tiny
    streamer = Streamer(config, source, "cpu", torch.float32, chunk_tokens=64, attn="sdpa")
    with pytest.raises(ValueError, match="tap layers"):
        run_shard(streamer, make_episodes(), tmp_path / "s", candidates=[1], tap_layers=(4, 46))
