"""Stage A analysis core: windows, features, folds, nested CV, bootstrap, earliness and the CLI on synthetic data."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ouroboros.analysis import controls, data, features, monitors, populations, probe, report, stage_a, synthetic
from ouroboros.analysis.data import WINDOW_MAX, WINDOWS, window_positions
from ouroboros.analysis.populations import REGISTERED_COUNTS, build_study

PROJECT = Path(__file__).resolve().parents[2]
MOE = 6  # MoE layers in the synthetic capture; the real one has 49


# Windows and features


def brute_window(gen: np.ndarray, kind: str, a: dict) -> list[int]:
    if kind in ("primary", "f_any"):
        lo, hi = (a["f"], a["window_end"]) if kind == "primary" else (a["f_any"], a["window_end_any"])
        return [p for p in range(len(gen)) if gen[p] and lo < p <= hi]
    end = a["s"] if kind == "h3_s" else a["s_call"]
    pos = [p for p in range(len(gen)) if gen[p] and p < end][-WINDOW_MAX:]
    return [p for p in pos if p < a["first_finish"]] if kind == "h3_prefinish" else pos


def nth_after(gen: np.ndarray, p: int) -> int:
    after = [q for q in range(p + 1, len(gen)) if gen[q]]
    return after[min(len(after), WINDOW_MAX) - 1]


def test_window_positions_match_brute_force():
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(40):
        n = int(rng.integers(300, 2500))
        gen = rng.random(n) < rng.uniform(0.15, 0.95)
        gen[-1] = True
        f = int(rng.integers(0, n // 2))
        f_any = int(rng.integers(0, f + 1))
        s = int(rng.integers(n // 2, n))
        a = {"f": f, "window_end": nth_after(gen, f), "f_any": f_any, "window_end_any": nth_after(gen, f_any),
             "s": s, "s_call": int(rng.integers(1, s))}
        a["first_finish"] = int(rng.integers(a["s_call"] // 2, a["s_call"] + 1))
        for kind in WINDOWS:
            want = brute_window(gen, kind, a)
            if not want:
                with pytest.raises(ValueError):
                    window_positions(gen, kind, a)
                continue
            assert window_positions(gen, kind, a).tolist() == want
            checked += 1
    assert checked > 100
    gen = np.ones(2000, bool)
    assert window_positions(gen, "primary", {"f": None, "window_end": None}) is None
    with pytest.raises(ValueError):  # more than 512 Beam tokens
        window_positions(gen, "primary", {"f": 10, "window_end": 600})
    with pytest.raises(ValueError):  # empty
        window_positions(np.zeros(100, bool), "primary", {"f": 10, "window_end": 50})
    with pytest.raises(ValueError):  # past the episode
        window_positions(gen, "primary", {"f": 1990, "window_end": 2100})


def test_first_finish_finds_the_first_earlier_call_with_the_accepted_header():
    S, C, M = 1, 2, 3  # <|start|>, <|channel|>, <|message|>
    ids = np.array([S, 10, 12, C, 9,  # another role
                    S, 10, 11, M, 9,  # same header, closed by <|message|>
                    S, 10, 11, C, 9,
                    S, 10, 11, 7, C, 9,  # a longer header
                    S, 10, 11, C, 9])  # the accepted call
    s_call = 21
    assert data.first_finish(ids, s_call, S, (C, M)) == 5
    assert data.first_finish(ids[10:], s_call - 10, S, (C, M)) == 0
    assert data.first_finish(ids[15:], s_call - 15, S, (C, M)) == s_call - 15  # none earlier: s_call itself
    with pytest.raises(ValueError):
        data.first_finish(ids, s_call + 1, S, (C, M))


def test_routing_features_equal_a_brute_force_count(tmp_path):
    rng = np.random.default_rng(2)
    route = rng.integers(0, 256, size=(49, 300, 8), dtype=np.uint8)
    np.save(tmp_path / "route_ids.npy", route)
    mm = np.load(tmp_path / "route_ids.npy", mmap_mode="r")
    tokens = np.sort(rng.choice(np.arange(40, 260), size=37, replace=False))
    want = np.zeros(49 * 256)
    for t in tokens:
        for j in range(49):
            for k in range(8):
                want[j * 256 + int(route[j, t, k])] += 1
    got = data.routing_features(mm, tokens)
    np.testing.assert_array_equal(got, want / len(tokens))
    np.testing.assert_allclose(got.reshape(49, 256).sum(axis=1), 8)
    with pytest.raises(ValueError):
        data.routing_features(mm, tokens[::-1])


def test_window_row_lookup_asserts_every_token_has_a_row():
    windows = np.array([5, 6, 7, 10, 11, 20])
    assert data.window_rows(windows, np.array([6, 10, 20])).tolist() == [1, 3, 5]
    assert data.covered(windows, np.array([5, 6, 7]))
    for bad in ([6, 8], [21], [0, 5], [11, 25]):
        with pytest.raises(ValueError):
            data.window_rows(windows, np.array(bad))
        assert not data.covered(windows, np.array(bad))


def test_bf16_bits_widen_exactly_and_hidden_means_them():
    bits = np.array([0x3F80, 0xC020, 0x0000, 0x8000, 0x7F80, 0x3E20], np.uint16)
    assert data.bf16_to_float32(bits).tolist() == [1.0, -2.5, 0.0, -0.0, float("inf"), 0.15625]
    rng = np.random.default_rng(3)
    x = rng.normal(size=1000).astype(np.float32)
    top = (x.view(np.uint32) >> 16).astype(np.uint16)
    np.testing.assert_array_equal(data.bf16_to_float32(top), (x.view(np.uint32) & 0xFFFF0000).view(np.float32))
    tap = rng.integers(0x3000, 0x4000, size=(5, 10, 4), dtype=np.uint16)
    windows = np.arange(100, 110)
    got = data.hidden_features(tap, windows, np.array([101, 103, 104]))
    want = data.bf16_to_float32(tap[:, [1, 3, 4], :]).astype(np.float64).mean(axis=1).ravel()
    np.testing.assert_allclose(got, want, rtol=1e-12)
    with pytest.raises(ValueError):
        data.hidden_features(tap, windows, np.array([101, 111]))


def test_analysis_text_keeps_only_beam_reasoning():
    harmony = features.Harmony(features.tokenizer())
    tok = harmony.tok
    text = ("<|start|>assistant<|channel|>analysis<|message|>Hello there<|end|>"
            "<|start|>assistant to=run_command<|channel|>analysis <|constrain|>xml<|message|>CALL<|call|>"
            "<|start|>run_command to=assistant<|channel|>analysis<|message|>RESULT<|end|>"
            "<|start|>assistant<|channel|>analysis<|message|>Second thought, longer this time<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Bye<|return|>")
    ids = np.array(tok.encode(text, add_special_tokens=False).ids)
    every = np.arange(len(ids))
    assert features.analysis_text(harmony, ids, every) == ("Hello there" + features.GAP
                                                           + "Second thought, longer this time")
    assert features.window_text(harmony, ids, every) == text
    assert features.window_text(harmony, ids, np.r_[0:3, 5:7]) == (harmony.decode(ids[0:3]) + features.GAP
                                                                   + harmony.decode(ids[5:7]))
    # A window that opens mid-message still reads the rest of the message it opened in.
    m = np.flatnonzero(ids == harmony.message)[3]
    end = m + np.flatnonzero(ids[m:] == harmony.end)[0]
    assert features.analysis_text(harmony, ids, np.arange(m + 3, len(ids))) == harmony.decode(ids[m + 3:end])


# Populations and folds


def test_folds_follow_the_registered_formula():
    tasks = [f"t{i:02d}" for i in range(44)]
    folds = populations.outer_folds(list(reversed(tasks)) + tasks[:3])
    perm = np.random.default_rng(20261028).permutation(44)
    assert folds == {tasks[perm[i]]: i % 5 for i in range(44)}
    assert folds == populations.outer_folds(tasks)
    assert np.bincount(list(folds.values())).tolist() == [9, 9, 9, 9, 8]
    for o in range(5):
        train = sorted(t for t in tasks if folds[t] != o)
        inner = populations.inner_folds(train[::-1], o)
        perm = np.random.default_rng(20261028 + o).permutation(len(train))
        assert inner == {train[perm[i]]: i % 4 for i in range(len(train))}


def test_registered_counts_hold_on_the_real_anchors():
    anchors = PROJECT / "prereg" / "anchors.csv"
    rendered = [PROJECT / "data" / "rendered" / f"{n}.parquet" for n in ("tripwire", "cvd")]
    if not anchors.exists() or not all(p.exists() for p in rendered):
        pytest.skip("real anchors and parquets not present")
    study = build_study(anchors, rendered)  # counts only; no score touches the labels here
    assert study.registered
    assert {k: study.counts[k] for k in REGISTERED_COUNTS} == REGISTERED_COUNTS
    assert len(study.folds) == 44


# Nested CV and bootstrap


def test_nested_cv_never_tunes_or_trains_on_held_out_tasks(monkeypatch):
    rng = np.random.default_rng(4)
    tasks = np.array([f"t{i % 12:02d}" for i in range(96)], dtype=object)
    y = rng.integers(0, 2, 96)
    X = rng.normal(size=(96, 6))
    X[:, 0] += y
    folds = populations.outer_folds(tasks)
    fold = np.array([folds[t] for t in tasks])
    calls = []
    real = probe._fit_score

    def spy(X_, y_, train, test, C, cols, keep):
        calls.append((train.copy(), test.copy(), C))
        return real(X_, y_, train, test, C, cols, keep)

    monkeypatch.setattr(probe, "_fit_score", spy)
    cv = probe.nested_cv(X, y, tasks, folds)
    per_outer = 4 * len(probe.C_GRID)
    assert len(calls) == 5 * per_outer + 5
    for c, (train, test, C) in enumerate(calls[:5 * per_outer]):
        o = c // per_outer
        assert (fold[train] != o).all() and (fold[test] != o).all()  # the held-out fold stays out of tuning
        assert not set(tasks[train]) & set(tasks[test])  # inner folds are grouped by task
    for o, (train, test, C) in enumerate(calls[5 * per_outer:]):
        assert np.array_equal(test, np.flatnonzero(fold == o)) and np.array_equal(train, np.flatnonzero(fold != o))
        assert C == cv.C[o]
    assert np.isfinite(cv.oof).all() and cv.auroc(y) > 0.6
    assert probe.choose_c(np.array([0.5, 0.7, 0.7, np.nan])) == 1  # ties go to the smaller C
    assert probe.choose_c(np.full(3, np.nan)) == 0


def test_oof_scores_are_standardized_within_each_held_out_fold():
    rng = np.random.default_rng(9)
    tasks = np.array([f"t{i % 15:02d}" for i in range(150)], dtype=object)
    y = rng.integers(0, 2, 150)
    X = rng.normal(size=(150, 30))
    X[:, 0] += y
    folds = populations.outer_folds(tasks)
    cv = probe.nested_cv(X, y, tasks, folds, keep_models=True)
    for o in range(5):
        k = cv.fold == o
        assert cv.oof[k].mean() == pytest.approx(0, abs=1e-12) and cv.oof[k].std() == pytest.approx(1)
        np.testing.assert_allclose(cv.models[o].score(X[k]), cv.oof[k], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose((cv.oof_raw[k] - cv.fold_mean[o]) / cv.fold_sd[o], cv.oof[k])
        assert cv.fold_auroc[o] == pytest.approx(roc_auc_score(y[k], cv.oof_raw[k]))
    assert cv.auroc(y) == pytest.approx(roc_auc_score(y, cv.oof))
    assert cv.auroc_raw(y) == pytest.approx(roc_auc_score(y, cv.oof_raw))
    # A per-fold affine map of the raw scores changes the raw pooled AUROC and leaves the statistic alone.
    mapped = cv.oof_raw * np.array([1, 50, 0.02, 3, 7])[cv.fold] + np.array([0, -9, 4, 1, 2])[cv.fold]
    again = np.empty_like(mapped)
    for o in range(5):
        mean, sd = probe.fold_standardize(mapped[cv.fold == o])
        again[cv.fold == o] = (mapped[cv.fold == o] - mean) / sd
    np.testing.assert_allclose(again, cv.oof, atol=1e-9)
    assert probe.fold_standardize(np.zeros(0)) == (0.0, 1.0) and probe.fold_standardize(np.ones(3)) == (1.0, 1.0)


def test_full_fit_runs_every_fit_on_one_blas_thread(monkeypatch):
    from threadpoolctl import threadpool_info

    rng = np.random.default_rng(10)
    tasks = np.array([f"t{i % 10}" for i in range(80)], dtype=object)
    y = rng.integers(0, 2, 80)
    X = rng.normal(size=(80, 20)) + 0.5 * y[:, None]
    seen = []
    real = probe._fit_score

    def spy(*args):
        seen.append(max((i["num_threads"] for i in threadpool_info()), default=1))
        return real(*args)

    monkeypatch.setattr(probe, "_fit_score", spy)
    model, _ = probe.full_fit(X, y, tasks, populations.outer_folds(tasks))
    assert len(seen) == 5 * len(probe.C_GRID) + 1 and max(seen) == 1
    assert model.n_iter > 0


def test_parallel_nested_cv_matches_serial():
    rng = np.random.default_rng(5)
    tasks = np.array([f"t{i % 10}" for i in range(80)], dtype=object)
    y = rng.integers(0, 2, 80)
    X = rng.normal(size=(80, 40)) + 0.3 * y[:, None]
    folds = populations.outer_folds(tasks)
    a = probe.nested_cv(X, y, tasks, folds, jobs=1)
    b = probe.nested_cv(X, y, tasks, folds, jobs=2)
    np.testing.assert_array_equal(a.oof, b.oof)
    np.testing.assert_array_equal(a.oof_raw, b.oof_raw)
    assert a.C == b.C


def test_bootstrap_is_cluster_level_and_paired():
    rng = np.random.default_rng(6)
    n = 60
    clusters = np.array([f"c{i % 9}" for i in range(n)], dtype=object)
    y = rng.integers(0, 2, n)
    s1 = rng.normal(size=n) + y
    s2 = np.round(rng.normal(size=n), 1)  # ties
    W = probe.cluster_weights(clusters, 200, seed=7)
    for c in set(clusters):  # every episode of a cluster carries its cluster's count
        idx = np.flatnonzero(clusters == c)
        assert (W[:, idx] == W[:, idx[:1]]).all()
    first = [int(np.flatnonzero(clusters == c)[0]) for c in sorted(set(clusters))]
    assert (W[:, first].sum(axis=1) == 9).all()  # each replicate draws 9 clusters
    wa = probe.weighted_auroc(y, s2, W)
    for b in range(30):  # equal to sklearn on the replicated episodes
        rep = np.repeat(np.arange(n), W[b])
        if len(set(y[rep])) == 2:
            assert wa[b] == pytest.approx(roc_auc_score(y[rep], s2[rep]), abs=1e-12)
        else:
            assert np.isnan(wa[b])
    g = probe.gap_summary(y, s1, s2, clusters, B=500, seed=11)
    W = probe.cluster_weights(clusters, 500, seed=11)
    d = probe.weighted_auroc(y, s1, W) - probe.weighted_auroc(y, s2, W)
    assert g["ci"] == pytest.approx(np.percentile(d[np.isfinite(d)], [2.5, 97.5]).tolist())
    assert g["gap"] == pytest.approx(roc_auc_score(y, s1) - roc_auc_score(y, s2))
    a1 = probe.auc_summary(y, s1, clusters, B=500, seed=11)
    assert a1["auroc"] == pytest.approx(roc_auc_score(y, s1)) and a1["n_clusters"] == 9
    only = (clusters == "c0").astype(int)  # positives in one cluster: replicates without it hold one class
    out = probe.auc_summary(only, s1, clusters, B=300, seed=1)
    W = probe.cluster_weights(clusters, 300, seed=1)
    assert out["skipped"] == int((W[:, 0] == 0).sum()) > 0


# End to end on synthetic data


def _args(world: synthetic.Paths, out: Path) -> list[str]:
    return ["--rendered", str(world.rendered[0]), str(world.rendered[1]), "--anchors", str(world.anchors),
            "--out", str(out)]


def _fake_qwen_run(records, out_path, model_id, revision, threads, max_batch_tokens=8192, log=print):
    done = set(report.monitor_scores(Path(out_path).parents[1]).get("qwen", {}))
    with open(out_path, "a", encoding="utf-8") as fh:
        for r in records:
            if tuple(r["key"].split("|", 1)) not in done:
                fh.write(json.dumps({"key": r["key"], "score": len(r["text"]) % 7 - 3.0,
                                     "n_prompt_tokens": len(r["text"])}) + "\n")
    return {"model_id": model_id, "revision": revision, "yes_id": 1, "no_id": 2, "n_scored": len(records),
            "seconds": 0.0}


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> synthetic.Paths:
    return synthetic.write(tmp_path_factory.mktemp("world"), seed=0, n_moe=MOE)


@pytest.fixture(scope="module")
def run(world, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("run")
    args = _args(world, out)
    assert stage_a.main(["features", "--capture", str(world.capture), *args]) == 0
    assert stage_a.main(["probes", *args, "--jobs", "2"]) == 0
    assert stage_a.main(["controls", *args, "--jobs", "2", "--limit", "2"]) == 0
    assert stage_a.main(["keyword", *args]) == 0
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(monitors, "qwen_run", _fake_qwen_run)
        assert stage_a.main(["qwen", *args]) == 0
    assert stage_a.main(["report", *args]) == 0
    return out


def _study(world: synthetic.Paths) -> populations.Study:
    return build_study(world.anchors, world.rendered)


def test_synthetic_world_reads_back(world, run):
    study = _study(world)
    assert not study.registered and study.counts["h1_pos"] > 5 and study.counts["h1_neg"] > 5
    for p in study.populations.values():  # folds group whole tasks
        for k, f, t in zip(p.keys, p.fold, p.task):
            assert f == (study.folds[t] if study.episodes[k]["source"] == "tripwire" else -1)
    feats = features.load_features(run / "features")
    assert feats.kinds == set(features.KINDS)
    early = study.populations["h1_f_any"]
    assert 0 < feats.hidden_ok("f_any", early.keys).sum() < len(early)  # f_any windows that start before f miss rows
    texts = features.load_texts(run / "features")
    row = texts[(study.populations["h1"].keys[0], "primary")]
    assert "pytest -q tests" in row["text"] and features.GAP in row["text"]
    assert "pytest" not in row["analysis_text"] and "passed in" not in row["analysis_text"]
    assert "<|" not in row["analysis_text"]


def test_planted_signal_is_found_and_shuffled_labels_are_not(run):
    routing = json.loads((run / "probes" / "routing__h1.json").read_text(encoding="utf-8"))
    # Every held-out fold separates perfectly. Every C ties at inner AUROC 1, so C = 1e-5 wins and the raw scores
    # sit near each fold model's own intercept. Standardizing each held-out fold puts the folds on one scale.
    assert min(routing["fold_auroc"]) > 0.95
    assert routing["auroc"] > 0.95 and routing["auroc"] >= routing["auroc_raw_pooled"]
    shuffled = json.loads((run / "controls" / "shuffled.json").read_text(encoding="utf-8"))["global"]
    assert shuffled["n"] == 2 and max(shuffled["null"]) < routing["auroc"]
    assert shuffled["p"] == pytest.approx(1 / 3)
    sets = json.loads((run / "controls" / "expert_sets.json").read_text(encoding="utf-8"))["sets"]
    assert all(v["selected_auroc"] is not None for v in sets.values())


def test_shuffled_routing_signal_scores_near_chance():
    """Routing features with a planted expert: found under the true labels, near 0.5 under shuffled ones."""
    rng = np.random.default_rng(8)
    n, per = 160, 64
    tasks = np.array([f"t{i % 16:02d}" for i in range(n)], dtype=object)
    y = (np.arange(n) // 16) % 2  # both classes in every task
    folds = populations.outer_folds(tasks)
    route = rng.integers(0, 256, size=(2, n * per, 8), dtype=np.uint8)
    X = np.stack([data.routing_features(route, np.arange(i * per, (i + 1) * per)) for i in range(n)])
    X[y == 1, 77::256] += 0.3
    assert probe.nested_cv(X, y, tasks, folds).auroc(y) > 0.95
    nulls = []
    for _ in range(4):
        shuffled = rng.permutation(y)
        nulls.append(probe.nested_cv(X, shuffled, tasks, folds).auroc(shuffled))
    assert abs(np.mean(nulls) - 0.5) < 0.1


def test_earliness_scores_other_episodes_with_the_fold_model_of_their_task(world, run):
    study = _study(world)
    early = study.populations["earliness"]
    h1 = set(study.populations["h1"].keys)
    scores = report.read_scores(run / "scores" / "routing__earliness.csv")
    oof = report.read_scores(run / "scores" / "routing__h1.csv")
    assert set(scores) == set(early.keys)
    extra = [k for k in early.keys if k not in h1]
    assert extra and all(scores[k] == oof[k] for k in early.keys if k in h1)
    feats = features.load_features(run / "features")
    meta = json.loads((run / "probes" / "routing__h1.json").read_text(encoding="utf-8"))
    pop = study.populations["h1"]
    X = feats.matrix("routing", "primary", pop.keys)
    fold = np.array([study.folds[t] for t in pop.task])

    def standardized(o: int, x: np.ndarray) -> float:  # fold o's model, on its held-out fold's scale
        model = probe.make_model(meta["fold_C"][o]).fit(X[fold != o], pop.label[fold != o])
        return (float(model.decision_function(x)[0]) - meta["fold_score_mean"][o]) / meta["fold_score_sd"][o]

    for k in extra:
        o = study.folds[study.episodes[k]["task_id"]]
        x = feats.matrix("routing", "primary", [k])
        assert scores[k] == pytest.approx(standardized(o, x), rel=1e-6, abs=1e-9)
        for other in set(range(5)) - {o}:
            assert scores[k] != pytest.approx(standardized(other, x), rel=1e-6, abs=1e-9)
        break


def test_cli_writes_every_output(world, run):
    study = _study(world)
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert not results["blinded"]
    for h in ("H1", "H2", "H3"):
        assert results["decisions"][h]["decision"] in ("pass", "fail")
    assert results["decisions"]["H3"]["decision"] == "pass"  # one fully fitted model: no pooling across folds
    assert results["decisions"]["H1"]["decision"] == "pass"
    assert min(results["within_fold_auroc"]["routing__h1"]["fold_auroc"]) > 0.95
    assert results["decisions"]["H1"]["exploratory"]["raw_pooled_auroc"] is not None
    md = (run / "report.md").read_text(encoding="utf-8")
    assert "per held-out fold" in md and "## Deviations" in md and "exploratory: every h3_prefinish cell" in md
    assert set(results["auroc"]) == {"h1", "h1_f_any", "h3_scall", "h3_s", "h3_prefinish"}
    assert set(results["auroc"]["h1"]) == {"routing", "bag", "hidden", "keyword", "qwen"}
    assert set(results["auroc"]["h3_prefinish"]) == {"routing", "bag", "hidden", "keyword", "qwen"}
    assert "hidden" not in results["auroc"]["h1_f_any"]
    assert results["populations"]["populations"]["h3_prefinish"]["exploratory"]
    cov = json.loads((run / "probes" / f"{report.F_ANY_HIDDEN}.json").read_text(encoding="utf-8"))
    assert 0 < cov["n_covered"] == cov["n_covered_equal_primary"] < cov["of"]
    assert cov["n_covered"] + cov["n_uncovered_f_any_before_f"] + cov["n_uncovered_without_f"] == cov["of"]
    assert f"{cov['of'] - cov['n_covered']} of the {cov['of']} h1_f_any" in results["deviations"][0]
    assert results["partial"] == [] and results["missing"] == []
    assert results["populations"]["fold_of_task"] == study.folds
    assert len(results["controls"]["layers"]["layers"]) == 2
    assert set(results["controls"]["expert_sets"]["sets"]) == {"16", "64", "256", "1024"}
    assert results["earliness"]["routing"]["n_negatives"] == study.counts["earliness_neg"]
    assert results["capture"] and results["interpretations"] and results["exploratory"]
    with open(run / "oof.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0]) == report.OOF_COLUMNS
    for name, p in study.populations.items():
        sel = [r for r in rows if r["population"] == name]
        assert [r["episode_id"] for r in sel] == p.keys
        assert all(r["routing"] != "" and r["keyword"] != "" and r["qwen"] != "" for r in sel)
    w = np.load(run / "probe_weights.npz")
    assert w["coef"].shape == (MOE, 256) and w["coef_raw"].shape == (MOE, 256)
    np.testing.assert_allclose(w["coef_raw"], w["coef"] / w["scale"])
    assert (run / "report.md").read_text(encoding="utf-8").startswith("# Ouroboros Stage A")
    qmeta = json.loads((run / "monitors" / "qwen_meta.json").read_text(encoding="utf-8"))
    assert qmeta["aliases"]  # f_any windows equal to their primary window are scored once
    assert qmeta["finished"] and qmeta["questions"] == monitors.QUESTIONS and qmeta["enable_thinking"] is False
    assert results["monitors"]["qwen"]["questions"] == monitors.QUESTIONS


def test_h3_prefinish_leaves_out_an_earlier_finish_call(world, run):
    study = _study(world)
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run / "features").glob("shard_*.json"))]
    early = [k for m in metas for k in m["h3_prefinish"]["keys"]]
    assert len(early) == 2 and sum(m["h3_prefinish"]["differs_from_h3_scall"] for m in metas) == 2
    assert sorted(study.episodes[k]["label"] for k in early) == [0, 1]
    feats = features.load_features(run / "features")
    keys = study.populations["h3_prefinish"].keys
    assert feats.hidden_ok("h3_prefinish", keys).all()  # inside h3_scall, whose rows the capture holds
    texts = features.load_texts(run / "features")
    harmony = features.Harmony(features.tokenizer())
    for k in keys:
        scall, pre = texts[(k, "h3_scall")]["text"], texts[(k, "h3_prefinish")]["text"]
        if k in early:
            assert "to=finish" in scall and "to=finish" not in pre and scall.startswith(pre)
        else:
            assert pre == scall and "to=finish" not in scall
    checked = 0
    for d in data.shard_dirs(world.capture):
        shard = data.Shard.open(d)
        ids_all, gen_all = shard.array("token_ids"), shard.array("gen_mask").astype(bool)
        for ep in shard.index["episodes"]:
            if ep["key"] not in early:
                continue
            ids = np.asarray(ids_all[ep["start"]:ep["start"] + ep["length"]])
            gen = gen_all[ep["start"]:ep["start"] + ep["length"]]
            first = data.first_finish(ids, ep["s_call"], harmony.start, (harmony.channel, harmony.message))
            a = {**study.episodes[ep["key"]], "first_finish": first}
            scall, pre = data.window_positions(gen, "h3_scall", a), data.window_positions(gen, "h3_prefinish", a)
            assert first < ep["s_call"] and first in scall and pre.max() < first
            assert np.array_equal(pre, scall[scall < first])
            checked += 1
    assert checked == 2


def test_report_marks_a_partly_scored_monitor(world, run, tmp_path):
    part = tmp_path / "run"
    shutil.copytree(run, part)
    qw = part / "monitors" / "qwen_scores.jsonl"
    lines = qw.read_text(encoding="utf-8").splitlines()
    qw.write_text("\n".join(lines[:30]) + "\n", encoding="utf-8")
    assert stage_a.main(["report", *_args(world, part)]) == 0
    results = json.loads((part / "results.json").read_text(encoding="utf-8"))
    n_h1 = results["populations"]["populations"]["h1"]["n"]
    assert any(x.startswith("qwen on h1: ") and x.endswith(f" of {n_h1}") for x in results["partial"])
    md = (part / "report.md").read_text(encoding="utf-8")
    assert "## Partly run" in md and "qwen on h1: " in md
    full = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert results["decisions"]["H1"] == full["decisions"]["H1"]  # a partial monitor leaves the decisions alone


def _decision_row(md: str, h: str) -> str:
    return next(line for line in md.splitlines() if line.startswith(f"| {h} |"))


def _auroc_cells(md: str) -> dict[str, dict[str, str]]:
    """AUROC by population: population -> score -> cell text."""
    out = {}
    for line in md.splitlines():
        if line.startswith("| h") and " (n = " in line.split("|")[1]:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            out[cells[0].split(" (n = ")[0]] = dict(zip(report.SCORES, cells[1:]))
    return out


def test_bag_of_tokens_cells_outside_h1_are_exploratory(run):
    assert not report._exploratory("h1", "bag")
    assert all(report._exploratory(p, "bag") for p in ("h1_f_any", "h3_scall", "h3_s", "h3_prefinish"))
    assert not any(report._exploratory(p, c) for p in ("h1", "h1_f_any", "h3_scall") for c in report.SCORES
                   if c != "bag")
    assert not report._exploratory("h3_s", "routing")
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert any(x.startswith("bag-of-tokens AUROC on h1_f_any, h3_scall and h3_s") for x in results["exploratory"])
    md = (run / "report.md").read_text(encoding="utf-8")
    cells = _auroc_cells(md)
    assert set(cells) == {"h1", "h1_f_any", "h3_scall", "h3_s", "h3_prefinish"}
    for pop, cols in cells.items():
        for c, text in cols.items():
            assert text.endswith(" *") == report._exploratory(pop, c), (pop, c, text)
    assert not cells["h1"]["bag"].endswith("*") and cells["h1_f_any"]["bag"].endswith(" *")
    assert cells["h3_scall"]["bag"].endswith(" *") and not cells["h3_scall"]["routing"].endswith("*")
    assert "every bag-of-tokens cell outside h1" in md


def test_report_is_interim_while_a_registered_item_is_missing(world, run, tmp_path):
    full = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert full["title"] == "Ouroboros Stage A" and full["missing"] == []
    md = (run / "report.md").read_text(encoding="utf-8")
    assert md.startswith("# Ouroboros Stage A\n") and "## Not yet run" not in md
    part = tmp_path / "run"
    shutil.copytree(run, part)
    (part / "scores" / "bag__h3_scall.csv").unlink()  # an exploratory cell leaves the title alone
    assert stage_a.main(["report", *_args(world, part)]) == 0
    results = json.loads((part / "results.json").read_text(encoding="utf-8"))
    assert results["missing"] == ["bag on h3_scall (exploratory)"] and results["title"] == "Ouroboros Stage A"
    shutil.rmtree(part / "controls")  # every registered control
    assert stage_a.main(["report", *_args(world, part)]) == 0
    results = json.loads((part / "results.json").read_text(encoding="utf-8"))
    assert results["title"] == "Ouroboros Stage A (interim)" and results["controls"] == {}
    assert results["missing"] == ["bag on h3_scall (exploratory)", "shuffled labels",
                                  "random expert sets of equal size", "one layer at a time"]
    md = (part / "report.md").read_text(encoding="utf-8")
    assert md.startswith("# Ouroboros Stage A (interim)\n")
    assert ("## Not yet run\n\nbag on h3_scall (exploratory), shuffled labels, random expert sets of equal size, "
            "one layer at a time.") in md
    assert "## Controls\n\n- Effort strata: " in md  # the strata keep their heading without the other controls
    for h in ("H1", "H2", "H3"):
        assert results["decisions"][h] == full["decisions"][h]


def test_stage_b_follows_h1_and_chance_is_named(world, run, tmp_path):
    full = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert full["decisions"]["H1"]["decision"] == "pass"
    assert full["decisions"]["stage_b"] == {"entry_rule": "H1 holds and the harness gate passes", "met": True,
                                            "reason": report.STAGE_B_REASON["pass"], "decision": "met"}
    md = (run / "report.md").read_text(encoding="utf-8")
    assert ("- Stage B entry rule: H1 must hold. H1 holds, and the harness gate passed before capture, so the entry "
            "rule is met.") in md
    assert "consistent with chance" not in _decision_row(md, "H1")
    n_h1 = full["populations"]["populations"]["h1"]["n"]
    assert f"The full probe trains on the {n_h1} H1 episodes, read as \"all of Tripwire\"." in _decision_row(md, "H3")
    assert report.stage_b("missing") == {"entry_rule": report.STAGE_B_RULE, "met": False,
                                         "reason": report.STAGE_B_REASON["missing"], "decision": "pending"}
    for h1 in ("pass", "fail", "missing"):  # a blinded H1 says nothing about the entry rule
        assert report.stage_b(h1, blinded=True) == {"entry_rule": report.STAGE_B_RULE, "met": False,
                                                    "reason": report.STAGE_B_REASON["blinded"], "decision": "pending"}
    # Constant routing and bag scores on h1: AUROC 0.5 in every replicate, so H1 and H2 fail at chance.
    part = tmp_path / "run"
    shutil.copytree(run, part)
    for kind in ("routing", "bag"):
        path = part / "scores" / f"{kind}__h1.csv"
        keys = list(report.read_scores(path))
        report.write_scores(path, keys, np.zeros(len(keys)))
    assert stage_a.main(["report", *_args(world, part)]) == 0
    results = json.loads((part / "results.json").read_text(encoding="utf-8"))
    d = results["decisions"]
    assert d["H1"]["auroc"] == pytest.approx(0.5) and d["H1"]["ci"] == pytest.approx([0.5, 0.5])
    assert d["H1"]["decision"] == "fail" and d["H2"]["decision"] == "fail"
    assert d["stage_b"] == {"entry_rule": "H1 holds and the harness gate passes", "met": False,
                            "reason": report.STAGE_B_REASON["fail"], "decision": "unmet"}
    md = (part / "report.md").read_text(encoding="utf-8")
    assert ("- Stage B entry rule: H1 must hold. H1 fails, so the pre-registered Stage B (H4, H5, H6 and the guard) "
            "stays untested.") in md
    assert "consistent with chance" in _decision_row(md, "H1")
    assert "The bag-of-tokens baseline sits at chance on h1." in _decision_row(md, "H2")
    assert d["H3"] == full["decisions"]["H3"]


def test_chance_notes_follow_the_interval(run):
    r = json.loads((run / "results.json").read_text(encoding="utf-8"))
    d, bag = r["decisions"], r["auroc"]["h1"]["bag"]
    for h in ("H1", "H3"):  # S6: a failed decision whose CI holds 0.5
        for decision, ci, named in (("fail", [0.45, 0.62], True), ("fail", [0.5, 0.5], True),
                                    ("fail", [0.52, 0.63], False), ("fail", [0.2, 0.49], False),
                                    ("pass", [0.45, 0.9], False)):
            d[h].update(decision=decision, ci=ci)
            assert ("consistent with chance" in _decision_row(report.render_markdown(r), h)) == named, (h, ci)
    for ci, named in (([0.42, 0.56], True), ([0.5, 0.7], True), ([0.55, 0.7], False), ([0.3, 0.49], False)):
        bag["ci"] = ci  # S7: the bag-of-tokens AUROC on h1
        row = _decision_row(report.render_markdown(r), "H2")
        assert ("The bag-of-tokens baseline sits at chance on h1." in row) == named, ci


def test_report_names_populations_and_the_results_beside_each_primary(run):
    r = json.loads((run / "results.json").read_text(encoding="utf-8"))
    md = (run / "report.md").read_text(encoding="utf-8")
    pops = r["populations"]["populations"]
    section = md.split("## Populations\n\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert [x.split(":", 1)[0] for x in section] == [f"- {p}" for p in ("h1", "h1_f_any", "earliness", "h3_scall",
                                                                          "h3_s", "h3_prefinish")]
    assert f"{pops['h1']['n_pos']} positives" in section[0] and f"the {pops['h1']['n_neg']} negatives" in section[0]
    fa, v1 = r["decisions"]["H1"]["f_any_sensitivity"], r["decisions"]["H3"]["h3_s"]
    assert (f"- Declared sensitivity analysis: routing on h1_f_any (n = {fa['n']}), AUROC {fa['auroc']:.3f}"
            in md) and "which decides H1." in md
    assert f"- Version 1 window: routing on h3_s (n = {v1['n']}), AUROC {v1['auroc']:.3f}" in md
    assert "which decides H3." in md
    e, h1_pos = pops["earliness"], pops["h1"]["n_pos"]
    assert (f"The population is {e['n_pos']} positives ({h1_pos} in H1 plus {e['n_pos'] - h1_pos} with gap under "
            f"500) against the {e['n_neg']} H1 negatives.") in md
    skipped = [b for b in r["earliness"]["routing"]["bins"] if b["skipped"]]
    assert ("Bootstrap replicates skipped for holding one class: " in md) == bool(skipped)
    assert "pre-registration version 2.1 records in changelog item 2" in r["deviations"][0]


def test_features_shard_by_shard_as_the_vm_runs_them(world, run, tmp_path):
    """The VM runs features --shard K with every kind and deletes the shard's route_ids and tap_hidden right after.
    The merged features equal one call over every shard, and later steps read no capture array."""
    capture = tmp_path / "capture"
    shutil.copytree(world.capture, capture)
    out = tmp_path / "run"
    args = ["--capture", str(capture), *_args(world, out)]
    for d in data.shard_dirs(capture):
        k = int(d.name.split("_")[1])
        assert stage_a.main(["features", *args, "--shard", str(k), "--kinds", "routing,hidden,bag,text"]) == 0
        for name in ("route_ids", "tap_hidden"):
            (d / f"{name}.npy").unlink()
    a, b = features.load_features(run / "features"), features.load_features(out / "features")
    assert a.kinds == b.kinds == set(features.KINDS)
    for w in WINDOWS:
        x, y = a.windows[w], b.windows[w]
        assert x.keys == y.keys
        for field in ("ntok", "routing", "hidden", "hidden_ok"):
            np.testing.assert_array_equal(getattr(x, field), getattr(y, field))
        assert all(np.array_equal(p, q) for p, q in zip(x.bag, y.bag))
    assert features.load_texts(run / "features") == features.load_texts(out / "features")
    assert stage_a.main(["probes", *args, "--kinds", "bag"]) == 0
    assert (out / "scores" / "bag__h1.csv").exists()
    with pytest.raises(FileNotFoundError, match="route_ids"):
        stage_a.main(["features", *args[:-1], str(tmp_path / "other"), "--shard", "0"])
    assert stage_a.main(["features", *args[:-1], str(tmp_path / "other"), "--shard", "0", "--kinds", "bag,text"]) == 0


def test_steps_resume_and_keep_their_blind_seed(world, run, capsys):
    before = (run / "scores" / "routing__h1.csv").read_bytes()
    assert stage_a.main(["probes", *_args(world, run), "--kinds", "routing"]) == 0
    assert "skipping" in capsys.readouterr().out
    assert (run / "scores" / "routing__h1.csv").read_bytes() == before
    with pytest.raises(SystemExit):
        stage_a.main(["report", *_args(world, run), "--blind-seed", "3"])


def test_blind_seed_permutes_labels_within_each_population(world, tmp_path):
    assert stage_a.main(["report", *_args(world, tmp_path), "--blind-seed", "3"]) == 0
    results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert results["blinded"] and results["blind_seed"] == 3
    assert results["decisions"]["stage_b"]["decision"] == "pending" and not results["decisions"]["stage_b"]["met"]
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Blinded run" in md
    assert f"- Stage B entry rule: H1 must hold. {report.STAGE_B_REASON['blinded']}" in md
    real = _study(world)
    blind = build_study(world.anchors, world.rendered, blind_seed=3)
    for name, p in blind.populations.items():
        q = real.populations[name]
        assert p.keys == q.keys and sorted(p.label) == sorted(q.label)
        assert sorted(np.nan_to_num(p.gap, nan=-1e9)) == sorted(np.nan_to_num(q.gap, nan=-1e9))
        assert ((p.label == 1) == np.isfinite(p.gap)).all() or name.startswith("h3")
    assert any(not np.array_equal(blind.populations[n].label, real.populations[n].label) for n in blind.populations)


def test_registered_counts_fail_loudly(world, monkeypatch):
    monkeypatch.setattr(populations, "REGISTERED_ANCHORS_SHA256", data.sha256(world.anchors))
    with pytest.raises(ValueError, match="registered counts"):
        build_study(world.anchors, world.rendered)


def test_controls_helpers():
    y = np.array([1, 0, 1, 0, 1, 0])
    tasks = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    for perm in controls.within_task_permutations(y, tasks, 5).values():
        for t in "abc":
            assert sorted(perm[tasks == t]) == sorted(y[tasks == t])
    assert np.array_equal(controls.global_permutations(y, 2)["perm001"], np.random.default_rng(20261029).permutation(y))
    sets = controls.selected_sets(np.array([[0.1, -3.0, 2.0, 0.5], [1.0, 1.0, 0.0, -2.0]]), 2)
    assert [s.tolist() for s in sets] == [[1, 2], [0, 3]]
    assert np.array_equal(controls.random_set(12544, 16, 3), controls.random_set(12544, 16, 3))
    assert len(set(controls.random_set(12544, 1024, 0))) == 1024
    s = controls.null_summary([0.4, 0.5, 0.9], 0.8)
    assert s["n_ge_observed"] == 1 and s["p"] == pytest.approx(2 / 4)


def test_decisions_pass_on_the_threshold_despite_float_rounding():
    assert 0.04999999999999993 < 0.05 and report._decision(0.04999999999999993, 0.05) == "pass"
    assert report._decision(0.75, 0.75) == "pass" and report._decision(0.7499, 0.75) == "fail"
    assert report._decision(0.9, 0.75, extra=False) == "fail" and report._decision(None, 0.75) == "missing"
