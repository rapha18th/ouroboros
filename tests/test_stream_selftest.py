"""The harness gate runs end to end on CPU with the tiny model: (a) and (c) pass, (b) is measured and reported."""

from __future__ import annotations

import contextlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from test_stream_support import WINDOW, one_thread, tiny  # noqa: F401
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import AttentionInterface

from ouroboros.harness import gpu_selftest
from ouroboros.harness.shards import Episode
from ouroboros.harness.stream import PACKED_SDPA, Layout, packed_sdpa_attention


def gate_groups(n_per_source=2):
    rng = np.random.default_rng(1)
    groups = []
    for source in ("tripwire", "cvd"):
        group = []
        for k in range(n_per_source + 1):
            n = 48 + 9 * k
            gen = np.zeros(n, bool)
            gen[n // 2:] = k < n_per_source  # the last episode has no sampled tokens and is never picked
            group.append(Episode(f"{source}{k}", rng.integers(0, 512, n), gen))
        groups.append(group)
    return groups


def test_gate_pool_keeps_billing_exact_rows(tmp_path):
    rows = {"episode_id": ["x", "y", "z"], "token_ids": [[1, 2], [3, 4], [5, 6]],
            "gen_mask": [[0, 1], [0, 1], [0, 1]]}
    pq.write_table(pa.table(rows), tmp_path / "plain.parquet")
    pq.write_table(pa.table({**rows, "billing_exact": [True, False, None]}), tmp_path / "billed.parquet")
    assert [e.key for e in gpu_selftest.gate_pool(tmp_path / "plain.parquet")] == ["x", "y", "z"]
    assert [e.key for e in gpu_selftest.gate_pool(tmp_path / "billed.parquet")] == ["x"]


def test_selftest_on_tiny_model(tiny, tmp_path):
    config, _, source = tiny
    gate = gpu_selftest.pick(gate_groups(), per_source=2)
    assert [e.key for e in gate] == ["tripwire0", "tripwire1", "cvd0", "cvd1"]
    results = gpu_selftest.selftest(config, source, gate, tmp_path, device="cpu", dtype=torch.float32,
                                    ref_len=2 * WINDOW + 5, chunk_tokens=64, attn="sdpa", rel_tol=1e-5)
    checks = {c["name"][:3] + str(k): c for k, c in enumerate(results["checks"])}
    a, nll, agree, c = checks["(a)0"], checks["(b)1"], checks["(b)2"], checks["(c)3"]
    assert a["pass"] and a["value"] < 1e-5
    assert c["pass"] and c["value"] == 1.0
    # A random model is far from Beam: its NLL sits near log(vocab), so (b) fails and says so.
    assert abs(nll["value"] - np.log(config.vocab_size)) < 0.5 and not nll["pass"]
    assert not results["pass"]
    assert len(results["layers"]) == config.num_hidden_layers
    assert results["sampled_tokens"] == sum(int(e.gen_mask[1:].sum()) for e in gate)
    assert json.loads((tmp_path / "selftest.json").read_text())["pass"] is False
    table = gpu_selftest.table(results)
    assert "PASS" in table and "FAIL" in table


def leaky_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, sliding_window=None,
                    **kwargs):
    """A packing bug: causal attention over the whole chunk, across episode boundaries."""
    n = query.shape[2]
    i, j = torch.arange(n)[:, None], torch.arange(n)[None, :]
    mask = j <= i
    if sliding_window is not None:
        mask &= j > i - sliding_window
    return sdpa_attention_forward(module, query, key, value, mask[None, None], scaling=scaling)


@pytest.mark.parametrize("bug", ["global positions", "attention across episodes"])
def test_check_a_catches_packing_bugs(tiny, tmp_path, monkeypatch, bug):
    config, _, source = tiny
    if bug == "global positions":
        monkeypatch.setattr(Layout, "positions", property(lambda self: np.arange(self.n_tokens, dtype=np.int64)))
    else:
        AttentionInterface.register(PACKED_SDPA, leaky_attention)
    try:
        results = gpu_selftest.selftest(config, source, gpu_selftest.pick(gate_groups(), per_source=2), tmp_path,
                                        device="cpu", dtype=torch.float32, ref_len=2 * WINDOW + 5, chunk_tokens=64,
                                        attn="sdpa", rel_tol=1e-5)
    finally:
        AttentionInterface.register(PACKED_SDPA, packed_sdpa_attention)
    a = results["checks"][0]
    assert a["name"].startswith("(a)") and not a["pass"] and a["value"] > 1e-4
    assert results["ref_offset"] == WINDOW + 3


def test_check_a_needs_lead_and_reference_in_one_chunk(tiny, tmp_path):
    config, _, source = tiny
    with pytest.raises(ValueError, match="chunk_tokens"):
        gpu_selftest.selftest(config, source, gpu_selftest.pick(gate_groups(), per_source=2), tmp_path, device="cpu",
                              ref_len=40, chunk_tokens=50, attn="sdpa")


def test_selftest_cli(tiny, tmp_path, monkeypatch, capsys):
    config, _, source = tiny
    config.save_pretrained(tmp_path / "config")
    monkeypatch.setattr(gpu_selftest, "open_source", lambda location: contextlib.nullcontext(source))
    paths = []
    for g, group in enumerate(gate_groups()):
        paths.append(tmp_path / f"src{g}.parquet")
        pq.write_table(pa.table({"episode_id": [e.key for e in group],
                                 "token_ids": [e.token_ids.tolist() for e in group],
                                 "gen_mask": [e.gen_mask.tolist() for e in group]}), paths[-1])
    code = gpu_selftest.main(["--weights", "unused", "--rendered", *map(str, paths), "--out", str(tmp_path / "gate"),
                              "--config", str(tmp_path / "config"), "--device", "cpu", "--attn", "sdpa",
                              "--per-source", "2", "--ref-len", "40"])
    assert code == 1  # (b) fails on a random model
    assert "harness gate: FAIL" in capsys.readouterr().out
