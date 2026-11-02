"""Router-bias patches: forcing an expert, exact revert, and layer-prefix reuse."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_stream_support import make_episodes, one_thread, tiny  # noqa: F401

from ouroboros.harness.capture import load
from ouroboros.harness.patch import PrefixCache, first_patched_layer, load_patch, patched, save_patch
from ouroboros.harness.run import run_shard
from ouroboros.harness.stream import Streamer, build_layer

COMPARED = ("route_ids", "route_scores", "window_scores", "tap_hidden", "nll", "top_ids", "top_logprobs",
            "candidate_logprobs")


def capture(tiny, out, delta=None, prefix=None, min_layer=0):
    config, _, source = tiny
    streamer = Streamer(config, source, "cpu", torch.float32, chunk_tokens=64, attn="sdpa")
    return load(run_shard(streamer, make_episodes(), out, candidates=[3, 4], tap_layers=(3, 6), top_k=4,
                          delta=delta, prefix_dir=prefix, prefix_min_layer=min_layer, meta={"weights": "tiny"}))


def zero_delta(config) -> torch.Tensor:
    return torch.zeros(sum(k == "sparse" for k in config.mlp_layer_types), config.num_experts)


def assert_identical(a, b, names=COMPARED):
    for name in names:
        np.testing.assert_array_equal(a.array(name), b.array(name), err_msg=name)


def test_plus_50_forces_expert_into_every_top8_and_revert_is_exact(tiny, tmp_path):
    config = tiny[0]
    stock = capture(tiny, tmp_path / "stock")
    j, expert = 1, 5
    stock_ids = stock.array("route_ids")
    assert not (stock_ids[j] == expert).any(axis=-1).all()  # not already chosen everywhere
    delta = zero_delta(config)
    delta[j, expert] = 50.0
    patched_run = capture(tiny, tmp_path / "patched", delta)
    ids = patched_run.array("route_ids")
    assert (ids[j] == expert).any(axis=-1).all()
    np.testing.assert_array_equal(ids[:j], stock_ids[:j])  # layers below the patch are untouched
    # Mixing weights come from the unbiased scores: the forced expert's captured score is its sigmoid score.
    assert (patched_run.array("route_scores")[j] < 1).all()
    reverted = capture(tiny, tmp_path / "reverted", zero_delta(config))
    assert_identical(stock, reverted)


def test_patched_context_restores_bias_bit_for_bit(tiny):
    config, _, source = tiny
    layer = build_layer(config, 3, source.load_layer(3, "cpu", torch.float32), torch.device("cpu"), torch.float32)
    bias = layer.mlp.gate.e_score_correction_bias
    stock = bias.clone()
    row = torch.randn(config.num_experts) * 1e-3 + 1 / 3
    with patched(layer, row):
        assert torch.equal(bias, stock + row)
    assert torch.equal(bias, stock)
    with pytest.raises(RuntimeError), patched(layer, row):
        raise RuntimeError
    assert torch.equal(bias, stock)


def test_prefix_reuse_matches_full_rerun(tiny, tmp_path):
    config, _, source = tiny
    first_moe = 2  # decoder layer 4
    a, b = zero_delta(config), zero_delta(config)
    a[first_moe, 7], a[4, 1] = 0.3, -0.2
    b[first_moe + 1, 9] = 0.5  # first patched layer 5: restarts from the prefix saved at 4
    full_a, full_b = capture(tiny, tmp_path / "full_a", a), capture(tiny, tmp_path / "full_b", b)

    prefix = tmp_path / "prefix"
    reuse_a = capture(tiny, tmp_path / "reuse_a", a, prefix)  # computes and saves the prefix
    assert (prefix / "input_04.npy").exists()
    assert_identical(full_a, reuse_a)
    loads = len(source.loads)
    reuse_b = capture(tiny, tmp_path / "reuse_b", b, prefix)
    assert source.loads[loads:] == [4, 5, 6]  # layers 0-3 came from the cache
    assert reuse_b.index["first_layer"] == 4 and reuse_b.index["layers_run"] == [4, 5, 6]
    assert_identical(full_b, reuse_b, ("nll", "top_ids", "top_logprobs", "candidate_logprobs"))
    # The capture of a resumed run covers the layers it ran: MoE layers from index 2, tap layer 6.
    for name in ("route_ids", "route_scores", "window_scores"):
        np.testing.assert_array_equal(full_b.array(name)[first_moe:], reuse_b.array(name)[first_moe:], err_msg=name)
    np.testing.assert_array_equal(full_b.array("tap_hidden")[1], reuse_b.array("tap_hidden")[1])


def test_prefix_cache_keeps_one_useful_layer(tiny, tmp_path):
    config, _, source = tiny
    prefix = tmp_path / "prefix"
    stock = capture(tiny, tmp_path / "stock", None, prefix)
    assert not list(prefix.glob("input_*.npy"))  # a stock run has no patched layer to save for
    deep, shallow = zero_delta(config), zero_delta(config)
    deep[3, 2] = 0.4  # decoder layer 5
    shallow[1, 2] = 0.4  # decoder layer 3
    capture(tiny, tmp_path / "skip", deep, prefix, min_layer=6)
    assert not list(prefix.glob("input_*.npy"))  # 5 is below min_layer
    capture(tiny, tmp_path / "deep", deep, prefix)
    assert [p.name for p in prefix.glob("input_*.npy")] == ["input_05.npy"]
    capture(tiny, tmp_path / "shallow", shallow, prefix)
    assert [p.name for p in prefix.glob("input_*.npy")] == ["input_03.npy"]  # the deeper prefix is evicted
    loads = len(source.loads)
    again = capture(tiny, tmp_path / "stock_again", None, prefix)  # a stock run reuses a saved stock prefix
    assert source.loads[loads:] == [3, 4, 5, 6] and again.index["first_layer"] == 3
    assert_identical(stock, again, ("nll", "top_ids", "top_logprobs", "candidate_logprobs"))


def test_prefix_cache_refuses_other_weights(tmp_path):
    identity = {"layout": "abc", "weights": "gs://b/rev1", "attn_implementation": "sdpa"}
    PrefixCache(tmp_path, identity, torch.bfloat16)
    PrefixCache(tmp_path, identity, torch.bfloat16)
    with pytest.raises(ValueError, match="weights"):
        PrefixCache(tmp_path, {**identity, "weights": "gs://b/rev2"}, torch.bfloat16)
    with pytest.raises(ValueError, match="dtype"):
        PrefixCache(tmp_path, identity, torch.float32)


def test_patch_file_round_trip_and_checks(tmp_path):
    delta = torch.zeros(5, 32)
    delta[2, 3] = 1.5
    save_patch(tmp_path / "p.safetensors", delta)
    assert torch.equal(load_patch(tmp_path / "p.safetensors", 5, 32), delta)
    with pytest.raises(ValueError, match="expected"):
        load_patch(tmp_path / "p.safetensors", 49, 256)
    delta[0, 0] = float("nan")
    save_patch(tmp_path / "bad.safetensors", delta)
    with pytest.raises(ValueError, match="non-finite"):
        load_patch(tmp_path / "bad.safetensors", 5, 32)


def test_first_patched_layer():
    moe = [3, 4, 5, 6]
    delta = torch.zeros(4, 8)
    assert first_patched_layer(None, moe) is None
    assert first_patched_layer(delta, moe) is None
    delta[2, 1] = -1e-3
    assert first_patched_layer(delta, moe) == 5
