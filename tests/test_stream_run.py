"""The run CLI: shards on disk, skipping finished shards, mirroring them, and refusing changed settings."""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from test_stream_support import make_episodes, one_thread, tiny  # noqa: F401

from ouroboros.harness import run
from ouroboros.harness.capture import is_complete, load
from ouroboros.harness.patch import save_patch


@pytest.fixture
def cli(tiny, tmp_path, monkeypatch):
    config, _, source = tiny
    config.save_pretrained(tmp_path / "config")
    def open_source(location, **options):
        source.options = options
        return contextlib.nullcontext(source)

    monkeypatch.setattr(run, "open_source", open_source)
    episodes = make_episodes()
    pq.write_table(pa.table({
        "episode_id": [e.key for e in episodes],
        "token_ids": [e.token_ids.tolist() for e in episodes],
        "gen_mask": [e.gen_mask.tolist() for e in episodes],
        **{a: pa.array([getattr(e, a) for e in episodes], type=pa.int64()) for a in ("f", "d", "s", "window_end")},
    }), tmp_path / "rendered.parquet")
    base = ["--weights", "unused", "--rendered", str(tmp_path / "rendered.parquet"), "--config",
            str(tmp_path / "config"), "--device", "cpu", "--attn", "sdpa", "--taps", "2", "5",
            "--candidates", "3", "4", "--shard-gb", str(70 * 64 * 2 / 1e9)]  # 70 tokens per shard
    return base, source, tmp_path


def test_cli_writes_shards_and_skips_finished_ones(cli, capsys):
    base, source, tmp = cli
    out = tmp / "out"
    assert run.main([*base, "--out", str(out), "--shard", "1"]) == 0
    split = json.loads((out / "shards.json").read_text())
    assert split == {"000": ["a", "b"], "001": ["c"]}
    assert is_complete(out / "shard_001") and not (out / "shard_000").exists()
    shard = load(out / "shard_001")
    assert shard.index["hidden_dtype"] == "bfloat16" and shard.index["candidates"] == [3, 4]
    assert shard.array("route_ids").shape == (5, 120, 8)

    loads = len(source.loads)
    assert run.main([*base, "--out", str(out)]) == 0  # all: only shard 0 is new
    assert len(source.loads) == loads + 7
    assert run.main([*base, "--out", str(out)]) == 0
    assert len(source.loads) == loads + 7
    assert "shard 1: done, skipping" in capsys.readouterr().out

    with pytest.raises(SystemExit, match="different split"):
        run.main([*base[:-1], "1e-3", "--out", str(out)])
    with pytest.raises(SystemExit, match="captured with"):  # same split, other candidates
        run.main([*base, "--candidates", "3", "5", "--out", str(out)])
    with pytest.raises(SystemExit, match="captured with"):  # other weights
        run.main([*base, "--weights", "gs://bucket/other-revision", "--out", str(out)])
    with pytest.raises(SystemExit, match="captured with"):  # without windows
        run.main([*base, "--no-windows", "--out", str(out)])
    assert len(source.loads) == loads + 7
    assert shard.index["weights"] == run.weights_location("unused") and shard.index["attn_implementation"]


def test_cli_patch_with_prefix_cache(cli):
    base, source, tmp = cli
    delta = torch.zeros(5, 32)
    delta[2, 4] = 2.0
    save_patch(tmp / "p.safetensors", delta)
    args = [*base, "--patch", str(tmp / "p.safetensors"), "--prefix-dir", str(tmp / "prefix"), "--truncate",
            "--margin", "4", "--prefix-min-layer", "0"]
    assert run.main([*args, "--out", str(tmp / "patched")]) == 0
    shard = load(tmp / "patched" / "shard_000")
    assert shard.index["truncate"] and len(shard.index["patch"]["sha256"]) == 64
    c = load(tmp / "patched" / "shard_001").index["episodes"][0]
    assert (c["key"], c["length"], c["full_length"]) == ("c", 110 + 1 + 4, 120)  # cut after s plus the margin
    assert (tmp / "prefix" / "shard_000" / "input_04.npy").exists()
    plain = run.main([*base, "--truncate", "--margin", "4", "--out", str(tmp / "stock")])
    assert plain == 0
    stock = load(tmp / "stock" / "shard_000")
    np.testing.assert_array_equal(stock.array("route_ids")[:2], shard.array("route_ids")[:2])
    assert not np.array_equal(stock.array("route_ids")[2], shard.array("route_ids")[2])
    with pytest.raises(SystemExit, match="captured with"):  # a stock rerun never reuses patched shards
        run.main([*base, "--truncate", "--margin", "4", "--out", str(tmp / "patched")])


def test_sync_survives_a_wiped_disk(cli, capsys):
    base, source, tmp = cli
    mirror = tmp / "mirror"
    assert run.main([*base, "--out", str(tmp / "local"), "--sync", str(mirror)]) == 0
    for k in ("000", "001"):
        local, copied = tmp / "local" / f"shard_{k}", mirror / f"shard_{k}"
        assert sorted(p.name for p in copied.iterdir()) == sorted(p.name for p in local.iterdir())
        assert (copied / "index.json").read_text() == (local / "index.json").read_text()
    loads = len(source.loads)
    shutil.rmtree(tmp / "local")  # a spot preemption wipes the local NVMe
    (mirror / "shard_001" / "index.json").unlink()  # a copy cut short before its index
    assert run.main([*base, "--out", str(tmp / "local"), "--sync", str(mirror)]) == 0
    assert len(source.loads) == loads + 7  # only shard 1 runs again
    assert "shard 0: done, skipping" in capsys.readouterr().out
    assert (mirror / "shard_001" / "index.json").exists()
    with pytest.raises(SystemExit, match="captured with"):
        run.main([*base, "--candidates", "9", "--out", str(tmp / "fresh"), "--sync", str(mirror)])

    # A mirror holding another capture stops the run before --drop-local can delete the good local shard.
    other = tmp / "other_mirror"
    shutil.copytree(mirror, other)
    stale = json.loads((other / "shard_001" / "index.json").read_text())
    (other / "shard_001" / "index.json").write_text(json.dumps({**stale, "candidates": [3, 5]}))
    with pytest.raises(SystemExit, match="other_mirror"):
        run.main([*base, "--out", str(tmp / "local"), "--sync", str(other), "--drop-local"])
    assert is_complete(tmp / "local" / "shard_001")  # rerun after the wipe, so only this shard is local


def test_drop_local_keeps_only_the_mirror(cli):
    base, _, tmp = cli
    with pytest.raises(SystemExit):
        run.main([*base, "--out", str(tmp / "local"), "--drop-local"])
    assert run.main([*base, "--out", str(tmp / "local"), "--sync", str(tmp / "mirror"), "--drop-local"]) == 0
    assert not list((tmp / "local").glob("shard_*")) and is_complete(tmp / "mirror" / "shard_000")


def test_plan_prints_the_split_and_loads_nothing(cli, capsys):
    base, source, tmp = cli
    loads = len(source.loads)
    assert run.main([*base, "--out", str(tmp / "plan"), "--plan"]) == 0
    assert json.loads((tmp / "plan" / "shards.json").read_text()) == {"000": ["a", "b"], "001": ["c"]}
    out = capsys.readouterr().out
    assert "shard 0: 2 episodes" in out and "shard 1: 1 episodes" in out
    assert len(source.loads) == loads and not (tmp / "plan" / "shard_000").exists()


def test_transport_flags_reach_the_weight_source(cli):
    base, source, tmp = cli
    assert run.main([*base, "--out", str(tmp / "default"), "--shard", "1"]) == 0
    assert source.options == {}  # each source keeps its own defaults
    assert run.main([*base, "--out", str(tmp / "tuned"), "--shard", "1", "--weight-workers", "8",
                     "--chunk-mb", "16"]) == 0
    assert source.options == {"workers": 8, "chunk_bytes": 16 << 20}


def test_no_windows_and_no_taps(cli):
    base, _, tmp = cli
    assert run.main([*base, "--no-windows", "--taps", "--out", str(tmp / "lean")]) == 0
    shard = load(tmp / "lean" / "shard_000")
    assert shard.array("window_scores").shape == (5, 0, 32) and shard.array("tap_hidden").shape == (0, 0, 64)
    assert shard.index["capture_windows"] is False and shard.index["tap_layers"] == []
    assert all(e["windows"] == [] for e in shard.index["episodes"])
    assert shard.array("nll").size > 0 and shard.array("route_ids").shape == (5, 62, 8)


def test_mirror_runs_gcloud(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        missing = "ERROR: (gcloud.storage.cat) The following URLs matched no objects or files:"
        return subprocess.CompletedProcess(cmd, 1 if cmd[2] == "cat" else 0, "", missing)

    monkeypatch.setattr(run.shutil, "which", lambda name: "/usr/bin/gcloud")
    monkeypatch.setattr(run.subprocess, "run", fake_run)
    mirror = run.Mirror("gs://bucket/caps/")
    assert mirror.index("shard_000") is None
    assert calls[0] == ["/usr/bin/gcloud", "storage", "cat", "gs://bucket/caps/shard_000/index.json"]
    shard = tmp_path / "shard_000"
    shard.mkdir()
    for name in ("route_ids.npy", "index.json", "nll.npy"):
        (shard / name).write_text("x")
    mirror.put(shard)
    dest = "gs://bucket/caps/shard_000/"
    assert calls[1] == ["/usr/bin/gcloud", "storage", "cp", str(shard / "nll.npy"), str(shard / "route_ids.npy"), dest]
    assert calls[2] == ["/usr/bin/gcloud", "storage", "cp", str(shard / "index.json"), dest]
    monkeypatch.setattr(run.subprocess, "run",
                        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, "", "ERROR: HTTPError 403"))
    with pytest.raises(RuntimeError, match="403"):
        mirror.index("shard_000")


def test_slot_word_follows_the_space_at_the_slot():
    tok = run.tokenizer()
    assert run.slot_word_ids(tok, "BLOCKED", 70396) == [39964, 1479]  # after "DONE"
    assert run.slot_word_ids(tok, "BLOCKED", 54685) == [28677, 1479]  # after " DONE"
    assert run.slot_word_ids(tok, "DONE", 39964) == [70396]


def test_status_candidates_are_first_tokens_of_done_and_blocked():
    # "DONE", " DONE", "BLOCK"(ED), " BLOCK"(ED) in Beam's tokenizer at the pinned revision.
    assert run.status_candidates() == [70396, 54685, 39964, 28677]
