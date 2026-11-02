"""The streaming forward reproduces transformers' BeamForCausalLM on the same weights, per episode."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_stream_support import (  # noqa: F401
    WINDOW,
    bf16_model,
    make_episodes,
    one_thread,
    reference,
    tiny,
    tiny_config,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.beam.modeling_beam import BeamAttention

from ouroboros.harness.capture import CapturePlan, load
from ouroboros.harness.run import finish, new_capture
from ouroboros.harness.stream import Layout, Streamer, build_layer, packed_sdpa_attention


def stream_everything(streamer, episodes, out_dir, candidates=(1, 2, 3)):
    """Capture every token: all layers tapped, every token a window row, every non-first token a target."""
    layout = Layout.pack([e.token_ids for e in episodes])
    starts = set(layout.offsets[:-1].tolist())
    targets = np.array([t for t in range(layout.n_tokens) if t not in starts], dtype=np.int64)
    plan = CapturePlan(np.arange(layout.n_tokens), targets, np.array(candidates),
                       tuple(range(streamer.config.num_hidden_layers)), top_k=4)
    capture = new_capture(streamer, layout, plan, out_dir)
    hidden = streamer.embed(layout)
    streamer.run_layers(hidden, layout, capture=capture)
    return layout, hidden, load(finish(streamer, capture, hidden, episodes, layout, {}))


def assert_same_experts(got: np.ndarray, ref_ids: np.ndarray, ref_logits: np.ndarray, bias: np.ndarray) -> None:
    """Identical top-k sets, except where the swapped experts tie exactly on the biased selection score."""
    g, w = np.sort(got, axis=-1), np.sort(ref_ids, axis=-1)
    for row in np.flatnonzero((g != w).any(axis=-1)):
        choice = 1 / (1 + np.exp(-ref_logits[row].astype(np.float64))) + bias
        swapped = sorted(set(g[row]) ^ set(w[row]))
        assert np.ptp(choice[swapped]) == 0, f"row {row}: experts {swapped} differ without an exact tie"


def test_layout_positions_and_chunks():
    layout = Layout.pack([np.arange(5), np.arange(3), np.arange(9), np.arange(2)])
    assert layout.positions.tolist() == [*range(5), *range(3), *range(9), *range(2)]
    assert layout.chunks(8) == [(0, 2), (2, 3), (3, 4)]
    assert layout.chunks(100) == [(0, 4)]
    assert layout.chunks(1) == [(0, 1), (1, 2), (2, 3), (3, 4)]


@pytest.mark.parametrize("window", [WINDOW, None])
def test_packed_attention_matches_transformers_masks(window):
    config = tiny_config()
    config._attn_implementation = "sdpa"
    module = BeamAttention(config, 0)
    lengths = [1, WINDOW - 1, WINDOW, WINDOW + 1, 40, 64]
    cu = torch.tensor(np.concatenate([[0], np.cumsum(lengths)]), dtype=torch.int32)
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 4, int(cu[-1]), 16, generator=g)
    k, v = (torch.randn(1, 2, int(cu[-1]), 16, generator=g) for _ in range(2))
    got, _ = packed_sdpa_attention(module, q, k, v, None, scaling=0.25, sliding_window=window, cu_seq_lens_q=cu)
    make_mask = create_causal_mask if window is None else create_sliding_window_causal_mask
    for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
        x = torch.zeros(1, b - a, config.hidden_size)
        mask = make_mask(config=config, inputs_embeds=x, attention_mask=None, past_key_values=None,
                         position_ids=torch.arange(b - a)[None])
        want, _ = sdpa_attention_forward(module, q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], mask, scaling=0.25)
        torch.testing.assert_close(got[:, a:b], want, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_build_layer_holds_transformers_weights(tiny, dtype):
    config, model, source = tiny
    want_model = model if dtype == torch.float32 else bf16_model(model)
    for i in (0, 2):  # one dense layer, one MoE layer with fused experts
        layer = build_layer(config, i, source.load_layer(i, "cpu", dtype), torch.device("cpu"), dtype)
        want = want_model.model.layers[i].state_dict()
        got = layer.state_dict()
        assert got.keys() == want.keys()
        for name in want:
            assert got[name].dtype == want[name].dtype, name
            assert torch.equal(got[name], want[name]), name
    assert layer.mlp.gate.weight.dtype == torch.float32
    assert layer.mlp.gate.e_score_correction_bias.dtype == torch.float32


def test_build_layer_rejects_missing_and_extra_tensors(tiny):
    config, _, source = tiny
    tensors = source.load_layer(2, "cpu", torch.float32)
    del tensors["mlp.experts.3.up_proj.weight"]
    with pytest.raises(KeyError):
        build_layer(config, 2, tensors, torch.device("cpu"), torch.float32)
    tensors = source.load_layer(0, "cpu", torch.float32)
    tensors["self_attn.extra"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unused"):
        build_layer(config, 0, tensors, torch.device("cpu"), torch.float32)


def test_stream_matches_transformers_fp32(tiny, tmp_path):
    config, model, source = tiny
    episodes = make_episodes()
    assert sum(len(e.token_ids) > WINDOW for e in episodes) >= 2
    streamer = Streamer(config, source, "cpu", torch.float32, chunk_tokens=64, attn="sdpa")
    assert len(Layout.pack([e.token_ids for e in episodes]).chunks(64)) == 2  # two episodes share a chunk
    layout, hidden, out = stream_everything(streamer, episodes, tmp_path / "shard")
    taps, ids, targets = out.tap_hidden(), out.array("route_ids"), out.array("targets")
    for e, ep in enumerate(episodes):
        lo, hi = int(layout.offsets[e]), int(layout.offsets[e + 1])
        ref = reference(model, ep.token_ids)
        for i in range(config.num_hidden_layers):
            np.testing.assert_allclose(taps[i, lo:hi], ref["layers"][i].numpy(), atol=1e-5, rtol=1e-5)
        for j, layer in enumerate(streamer.moe_layers):
            logits, want_ids = (x.numpy() for x in ref["routers"][layer])
            bias = model.model.layers[layer].mlp.gate.e_score_correction_bias.numpy()
            assert_same_experts(ids[j, lo:hi], want_ids, logits, bias)
        torch.testing.assert_close(streamer.logits(hidden[lo:hi]), ref["logits"], atol=1e-5, rtol=1e-5)
        rows = slice(*np.searchsorted(targets, [lo, hi]))
        logp = torch.log_softmax(ref["logits"], dim=-1).numpy()
        t = targets[rows]
        np.testing.assert_allclose(out.array("nll")[rows], -logp[t - lo - 1, ep.token_ids[t - lo]], atol=1e-5)
        np.testing.assert_allclose(out.array("candidate_logprobs")[rows], logp[t - lo - 1][:, [1, 2, 3]], atol=1e-5)
        np.testing.assert_array_equal(out.array("top_ids")[rows, 0], logp[t - lo - 1].argmax(-1))


def test_stream_matches_transformers_bf16(tiny, tmp_path):
    """BF16 rounding can flip near-tied experts, so this checks agreement rates, not exact sets."""
    config, model, source = tiny
    ref_model = bf16_model(model)
    episodes = make_episodes()
    streamer = Streamer(config, source, "cpu", torch.bfloat16, chunk_tokens=64, attn="sdpa")
    layout, _, out = stream_everything(streamer, episodes, tmp_path / "shard")
    assert out.array("tap_hidden").dtype == np.uint16
    taps, ids = out.tap_hidden(), out.array("route_ids")
    same, total = 0, 0
    for e, ep in enumerate(episodes):
        lo, hi = int(layout.offsets[e]), int(layout.offsets[e + 1])
        ref = reference(ref_model, ep.token_ids)
        for i in (0, 1):  # dense layers, before any routing can differ
            got, want = taps[i, lo:hi], ref["layers"][i].numpy()
            assert np.linalg.norm(got - want) <= 4e-3 * np.linalg.norm(want)
        for j, layer in enumerate(streamer.moe_layers):
            match = (np.sort(ids[j, lo:hi], -1) == np.sort(ref["routers"][layer][1].numpy(), -1)).all(-1)
            same, total = same + match.sum(), total + match.size
    assert same / total >= 0.97


def test_prefetching_source_streams_the_same(tiny, tmp_path):
    """A source with stream() (prefetch) feeds the same layers as load_layer."""
    config, _, source = tiny

    class Prefetching:
        def __init__(self):
            self.streamed = []

        def load_globals(self, device, dtype):
            return source.load_globals(device, dtype)

        def load_layer(self, i, device, dtype):
            raise AssertionError("stream() should be used")

        def stream(self, layers, device, dtype):
            for i in layers:
                self.streamed.append(i)
                yield i, source.load_layer(i, device, dtype)

    episodes = make_episodes()
    prefetching = Prefetching()
    _, a, _ = stream_everything(Streamer(config, source, "cpu", torch.float32, 64, "sdpa"), episodes, tmp_path / "a")
    _, b, _ = stream_everything(Streamer(config, prefetching, "cpu", torch.float32, 64, "sdpa"), episodes,
                                tmp_path / "b")
    assert prefetching.streamed == list(range(config.num_hidden_layers))
    assert torch.equal(a, b)

