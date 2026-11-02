"""Episodes into shards under a RAM budget, windows, targets and truncation."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ouroboros.harness import shards
from ouroboros.harness.shards import PRE_SLOT, TAMPER_SPAN, Episode


def episode(n=2000, gen_from=100, **anchors) -> Episode:
    gen = np.zeros(n, bool)
    gen[gen_from:] = True
    return Episode("e", np.arange(n), gen, **anchors)


def test_windows_cover_primary_and_pre_slot_spans():
    assert episode(f=300, window_end=900).windows() == [(300, 901)]  # window_end is the 512th Beam token
    assert episode(s=1500).windows() == [(1500 - PRE_SLOT, 1500)]
    assert episode(s=100).windows() == [(0, 100)]
    assert episode(f=300, window_end=900, s=1000).windows() == [(300, 1000)]  # overlapping spans merge
    assert episode(f=10, window_end=5000).windows() == [(10, 2000)]  # clipped to the episode
    assert episode().windows() == []


def test_pre_slot_window_reaches_back_512_beam_tokens():
    gen = np.zeros(3000, bool)
    gen[::4] = True  # Beam wrote every fourth token
    ep = Episode("e", np.arange(3000), gen, s=2900)
    (a, b), = ep.windows()
    assert b == 2900 and gen[a:b].sum() == PRE_SLOT and a == 2900 - 4 * PRE_SLOT
    few = Episode("e", np.arange(3000), np.isin(np.arange(3000), [100, 2000]), s=2900)
    assert few.windows() == [(100, 2900)]  # fewer than 512 Beam tokens: back to the first one
    # With s_call, the window also holds the 512 Beam tokens before the finish call opens.
    call = Episode("e", np.arange(3000), gen, s=2900, s_call=2800)
    assert call.windows() == [(2800 - 4 * PRE_SLOT, 2900)]


def test_slot_word_scores_the_whole_word():
    gen = np.ones(30, bool)
    ids = np.arange(100, 130)
    done = Episode("e", ids, gen, s=20)
    blocked = done.with_slot_word([7, 8])  # Beam wrote 120 at s: the word replaces the tail
    assert blocked.token_ids.tolist() == [*range(100, 120), 7, 8]
    assert blocked.gen_mask.tolist() == [True] * 20 + [False, False] and blocked.full_length == 30
    assert blocked.targets().tolist() == list(range(1, 22)) and blocked.last_needed() == 22
    same = done.with_slot_word([120, 121])  # Beam wrote this word: the episode stays as it is
    assert same.token_ids is ids and same.slot_ids == (120, 121)
    assert episode(n=50).with_slot_word([7, 8]).slot_ids is None  # no slot
    assert len(blocked.truncated(0).token_ids) == 22


def test_targets_are_sampled_tokens_tamper_span_and_slot():
    ep = episode(n=50, gen_from=30, d=5, s=20)
    want = [*range(5, 5 + TAMPER_SPAN), 20, *range(30, 50)]
    assert ep.targets().tolist() == sorted(set(want))
    gen = np.ones(10, bool)
    assert Episode("x", np.arange(10), gen).targets().tolist() == list(range(1, 10))  # never position 0


def test_truncation_keeps_everything_needed():
    ep = episode(n=2000, f=100, window_end=700, d=900, s=None)
    cut = ep.truncated(margin=64)
    assert len(cut.token_ids) == 900 + TAMPER_SPAN + 64 and cut.full_length == 2000
    assert len(episode(n=2000, f=100, window_end=700).truncated(0).token_ids) == 701
    assert cut.windows() == ep.windows()
    assert set(ep.targets()[ep.targets() < len(cut.token_ids)]) == set(cut.targets())
    assert len(episode(n=2000).truncated(64).token_ids) == 2000  # no anchors: kept whole
    assert len(episode(n=1000, s=990).truncated(64).token_ids) == 1000


def test_split_keeps_episodes_whole_under_budget():
    lengths = [40, 30, 50, 200, 10, 10]
    groups = shards.split(lengths, max_tokens=100)
    assert groups == [[0, 1], [2], [3], [4, 5]]
    assert sorted(e for g in groups for e in g) == list(range(len(lengths)))
    assert all(sum(lengths[e] for e in g) <= 100 or len(g) == 1 for g in groups)


def test_budget_is_hidden_bytes():
    assert shards.budget_tokens(40.0, 5120) == 40_000_000_000 // 10240
    four = shards.split([16_000] * 780, shards.budget_tokens(40.0, 5120))  # about 12.5M tokens
    assert len(four) == 4


def test_load_rendered_reads_p1_parquet(tmp_path):
    table = pa.table({
        "episode_id": ["t1", "t2"],
        "token_ids": pa.array([[5, 6, 7, 8], [9, 10]], type=pa.list_(pa.int32())),
        "gen_mask": pa.array([[False, False, True, True], [False, True]], type=pa.list_(pa.bool_())),
        "f": pa.array([1, None], type=pa.int64()),
        "d": pa.array([2, None], type=pa.int64()),
        "s": pa.array([None, 1], type=pa.int64()),
        "window_end": pa.array([3, None], type=pa.int64()),
    })
    pq.write_table(table, tmp_path / "r.parquet", row_group_size=1)
    a, b = shards.load_rendered([tmp_path / "r.parquet"])
    assert (a.key, a.token_ids.tolist(), a.gen_mask.tolist(), a.f, a.d, a.s, a.window_end) == \
        ("t1", [5, 6, 7, 8], [False, False, True, True], 1, 2, None, 3)
    assert (b.key, b.token_ids.tolist(), b.s, b.f) == ("t2", [9, 10], 1, None)
    assert a.token_ids.dtype == np.int64 and a.gen_mask.dtype == bool

    pq.write_table(table.drop(["episode_id", "f", "d"]), tmp_path / "bare.parquet")
    keys = [e.key for e in shards.load_rendered([tmp_path / "bare.parquet"])]
    assert keys == ["bare:0", "bare:1"]
    with pytest.raises(ValueError, match="duplicate"):
        shards.load_rendered([tmp_path / "r.parquet", tmp_path / "r.parquet"])


def test_build_places_rows_in_shard_coordinates():
    eps = [episode(n=40, gen_from=30, f=5, window_end=10), episode(n=20, gen_from=15, s=12)]
    layout, plan = shards.build(eps, [1, 2], (4,), top_k=8)
    assert layout.n_tokens == 60
    assert plan.windows.tolist() == [*range(5, 11), *range(40, 52)]
    assert plan.targets.tolist() == [*range(30, 40), 52, *range(55, 60)]
    assert plan.candidates.tolist() == [1, 2] and plan.tap_layers == (4,)
    _, lean = shards.build(eps, [1, 2], (), top_k=8, with_windows=False)
    assert lean.windows.size == 0 and lean.targets.tolist() == plan.targets.tolist()
    index = shards.episode_index(eps, layout, lean)
    assert [e["windows"] for e in index] == [[], []] and [e["window_rows"] for e in index] == [[0, 0], [0, 0]]
