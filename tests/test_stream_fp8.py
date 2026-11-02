"""End to end through P2a's loader: a tiny FP8 checkpoint in the real format, read by LocalDir with prefetch."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_stream_support import one_thread, reference  # noqa: F401

from ouroboros.harness.capture import CapturePlan, load
from ouroboros.harness.run import finish, new_capture
from ouroboros.harness.shards import Episode
from ouroboros.harness.stream import Layout, Streamer

tinyckpt = pytest.importorskip("ouroboros.harness.tinyckpt")
weights = pytest.importorskip("ouroboros.harness.weights")


def test_fp8_checkpoint_streams_like_transformers(tmp_path):
    config = tinyckpt.tiny_config()
    ckpt, values = tinyckpt.build_checkpoint(tmp_path / "ckpt", config)
    model = tinyckpt.reference_model(config, values)
    model.set_attn_implementation("sdpa")
    model.set_experts_implementation("eager")
    rng = np.random.default_rng(3)
    lengths = (3 * config.sliding_window + 5, 5, 2 * config.sliding_window)
    episodes = [Episode(str(k), rng.integers(0, 1000, n), np.ones(n, bool)) for k, n in enumerate(lengths)]
    layout = Layout.pack([e.token_ids for e in episodes])
    starts = set(layout.offsets[:-1].tolist())
    plan = CapturePlan(np.arange(layout.n_tokens), np.array([t for t in range(layout.n_tokens) if t not in starts]),
                       np.array([1]), tuple(range(config.num_hidden_layers)), top_k=1)
    with weights.LocalDir(ckpt) as source:
        streamer = Streamer(config, source, "cpu", torch.bfloat16, chunk_tokens=40, attn="sdpa")
        capture = new_capture(streamer, layout, plan, tmp_path / "shard")
        hidden = streamer.embed(layout)
        streamer.run_layers(hidden, layout, capture=capture)
        out = load(finish(streamer, capture, hidden, episodes, layout, {}))
    taps, ids = out.tap_hidden(), out.array("route_ids")
    same = total = 0
    for e, ep in enumerate(episodes):
        lo, hi = int(layout.offsets[e]), int(layout.offsets[e + 1])
        ref = reference(model, ep.token_ids)
        for i in (0, 1):  # dense layers: no routing to diverge yet
            want = ref["layers"][i].numpy()
            assert np.linalg.norm(taps[i, lo:hi] - want) <= 4e-3 * np.linalg.norm(want)
        for j, layer in enumerate(streamer.moe_layers):
            match = (np.sort(ids[j, lo:hi], -1) == np.sort(ref["routers"][layer][1].numpy(), -1)).all(-1)
            same, total = same + match.sum(), total + match.size
    assert same / total >= 0.95
