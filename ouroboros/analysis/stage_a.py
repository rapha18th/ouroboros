"""Stage A analysis, one resumable step at a time.

    python -m ouroboros.analysis.stage_a STEP --capture DIR --rendered TRIPWIRE CVD --anchors CSV --out DIR \\
        [--shard K] [--jobs N] [--kinds routing,hidden,bag,text] [--blind-seed N] [--limit N] \\
        [--qwen-model ID] [--qwen-revision SHA] [--threads N]

Steps, in order:
- features: per capture shard (--shard K, or every finished shard under --capture), writes
  features/shard_K.npz, shard_K.json and texts_K.jsonl. A missing array is an error unless --kinds leaves its kind
  out.
- probes: nested CV of the routing, bag and hidden probes on h1 and of routing and bag on h1_f_any, full fits
  scored on Claimed vs Done (h3_scall, h3_s, h3_prefinish), earliness scores, probe_weights.npz. The capture holds
  hidden rows in an f_any window only when it equals the primary window, so the hidden nested CV runs on h1 only;
  the step writes those coverage counts to probes/f_any_hidden_coverage.json.
- controls: shuffled labels, within-task shuffles, random and selected expert sets, one layer at a time.
- keyword, qwen: the two text monitors on every window the populations use.
- report: oof.csv, results.json and report.md from everything written so far.

Each step skips work it finds finished. --blind-seed permutes the labels within each population before any step
reads them; a run directory keeps the seed it started with. --limit caps permutations, random sets, layers and Qwen
windows for quick runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ouroboros.analysis import controls, data, features, monitors, report
from ouroboros.analysis.populations import Population, Study, build_study
from ouroboros.analysis.probe import Linear, full_fit, nested_cv

STEPS = ("features", "probes", "controls", "keyword", "qwen", "report")
PROBE_KINDS = ("routing", "bag", "hidden")
WINDOW_ORDER = {w: i for i, w in enumerate(data.WINDOWS)}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report._clean(obj), indent=1), encoding="utf-8")
    tmp.replace(path)


def load_study(args: argparse.Namespace) -> Study:
    """Populations for this run; the run directory keeps the inputs and blind seed it started with."""
    study = build_study(args.anchors, args.rendered, args.blind_seed)
    out = Path(args.out)
    ident = {"blind_seed": args.blind_seed, "anchors_sha256": study.anchors_sha256,
             "rendered_sha256": [data.sha256(p) for p in args.rendered]}
    path = out / "run.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != ident:
            sys.exit(f"{path} records {old}; this call has {ident}. Use the original flags or a new --out.")
    else:
        _write_json(path, ident)
    for note in study.notes:
        log(note)
    log("populations: " + ", ".join(f"{n} {p.counts()['n_pos']}+/{p.counts()['n_neg']}-"
                                     for n, p in study.populations.items())
        + (f" (blinded, seed {args.blind_seed})" if args.blind_seed is not None else ""))
    return study


def kinds_of(args: argparse.Namespace, allowed: Sequence[str]) -> list[str]:
    kinds = features.check_kinds(args.kinds.split(",")) if args.kinds else list(allowed)
    return [k for k in kinds if k in allowed]


# features


def step_features(args: argparse.Namespace) -> None:
    if not args.capture:
        sys.exit("features needs --capture")
    anchors = data.load_anchors(args.anchors)
    rendered = data.load_tokens(args.rendered)
    kinds = kinds_of(args, features.KINDS)
    tok = features.tokenizer(args.tokenizer) if "text" in kinds else None
    for d in data.shard_dirs(args.capture, args.shard):
        features.compute_shard(d, anchors, Path(args.out) / "features", kinds, rendered, tok, log)


# probes


def _matrix(feats: features.Features, kind: str, window: str, keys: list[str], vocab=None) -> np.ndarray:
    return feats.matrix(kind, window, keys, vocab)


LINEAR_FIELDS = ("coef", "mean", "scale", "intercept", "C", "n_iter")


def _save_linear(path: Path, models: list[Linear], **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, coef=np.stack([m.coef for m in models]), mean=np.stack([m.mean for m in models]),
             scale=np.stack([m.scale for m in models]), intercept=np.array([m.intercept for m in models]),
             C=np.array([m.C for m in models]), n_iter=np.array([m.n_iter for m in models]), **extra)


def _load_linear(path: Path) -> tuple[list[Linear], dict]:
    z = np.load(path)
    n_iter = z["n_iter"] if "n_iter" in z.files else np.zeros(len(z["C"]), int)
    models = [Linear(z["mean"][i], z["scale"][i], z["coef"][i], float(z["intercept"][i]), float(z["C"][i]),
                     int(n_iter[i])) for i in range(len(z["C"]))]
    return models, {k: z[k] for k in z.files if k not in LINEAR_FIELDS}


def run_probe(out: Path, feats: features.Features, kind: str, pop: Population, study: Study, jobs: int,
              vocab: np.ndarray | None = None) -> tuple[list[Linear], np.ndarray | None]:
    """Nested CV of one probe; writes probes/<kind>__<pop>.json/.npz and scores/<kind>__<pop>.csv.

    The scores and fold models are fold-standardized (probe.nested_cv); the npz keeps the raw OOF scores beside
    them, and the json the raw pooled AUROC (exploratory)."""
    name = f"{kind}__{pop.name}"
    meta_path = out / "probes" / f"{name}.json"
    npz_path = out / "probes" / f"{name}.npz"
    if meta_path.exists() and npz_path.exists():
        log(f"probe {name}: done, skipping")
        models, extra = _load_linear(npz_path)
        return models, extra.get("vocab")
    t0 = time.perf_counter()
    if kind == "bag" and vocab is None:
        vocab = feats.vocabulary(pop.window, pop.keys)
    X = _matrix(feats, kind, pop.window, pop.keys, vocab)
    cv = nested_cv(X, pop.label, pop.task, study.folds, jobs=jobs, keep_models=True)
    report.write_scores(out / "scores" / f"{name}.csv", pop.keys, cv.oof)
    _save_linear(npz_path, cv.models, oof=cv.oof, oof_raw=cv.oof_raw, fold_mean=cv.fold_mean, fold_sd=cv.fold_sd,
                 **({"vocab": vocab} if vocab is not None else {}))
    meta = {"name": name, "kind": kind, "population": pop.name, "window": pop.window, **pop.counts(),
            "n_features": X.shape[1], "fold_C": cv.C, "fold_auroc": cv.fold_auroc,
            "fold_n": [int((cv.fold == o).sum()) for o in range(len(cv.C))], "auroc": cv.auroc(pop.label),
            "auroc_raw_pooled": cv.auroc_raw(pop.label), "fold_score_mean": cv.fold_mean,
            "fold_score_sd": cv.fold_sd, "fold_n_iter": [m.n_iter for m in cv.models],
            "inner_mean_auroc": cv.inner_mean, "exploratory": pop.exploratory,
            "blinded": study.blind_seed is not None, "seconds": round(time.perf_counter() - t0, 1)}
    _write_json(meta_path, meta)
    log(f"probe {name}: n {len(pop)}, {X.shape[1]} features, C {cv.C}, {meta['seconds']} s")
    return cv.models, vocab


def run_full(out: Path, feats: features.Features, kind: str, pop: Population, study: Study, jobs: int,
             vocab: np.ndarray | None) -> Linear:
    """The fully fitted probe on all of `pop`; scores every Claimed vs Done window with it."""
    name = f"{kind}__{pop.name}_full"
    meta_path = out / "probes" / f"{name}.json"
    npz_path = out / "probes" / f"{name}.npz"
    if meta_path.exists() and npz_path.exists():
        model = _load_linear(npz_path)[0][0]
    else:
        t0 = time.perf_counter()
        X = _matrix(feats, kind, pop.window, pop.keys, vocab)
        model, cv_mean = full_fit(X, pop.label, pop.task, study.folds, jobs=jobs)
        _save_linear(npz_path, [model], cv_mean=cv_mean, **({"vocab": vocab} if vocab is not None else {}))
        _write_json(meta_path, {"name": name, "kind": kind, "population": pop.name, "C": model.C,
                                "n_iter": model.n_iter, "cv_mean_auroc": cv_mean, "n": len(pop),
                                "blinded": study.blind_seed is not None,
                                "seconds": round(time.perf_counter() - t0, 1)})
        log(f"full fit {name}: C {model.C}")
    for target in ("h3_scall", "h3_s", "h3_prefinish"):
        p = study.populations[target]
        path = out / "scores" / f"{kind}__{target}.csv"
        if not path.exists():
            report.write_scores(path, p.keys, model.score(_matrix(feats, kind, p.window, p.keys, vocab)))
    return model


def earliness_scores(out: Path, feats: features.Features, kind: str, study: Study, models: list[Linear],
                     vocab: np.ndarray | None) -> None:
    """H1 episodes keep their OOF score; the rest take the outer model whose held-out fold holds their task. The
    fold models carry their fold's H1 standardization, so every score here sits on the same scale."""
    path = out / "scores" / f"{kind}__earliness.csv"
    if path.exists():
        return
    oof = report.read_scores(out / "scores" / f"{kind}__h1.csv")
    early = study.populations["earliness"]
    extra = [k for k in early.keys if k not in oof]
    scores = dict(oof)
    if extra:
        X = _matrix(feats, kind, early.window, extra, vocab)
        folds = np.array([study.folds[study.episodes[k]["task_id"]] for k in extra])
        for o in np.unique(folds):
            for k, s in zip(np.array(extra)[folds == o], models[o].score(X[folds == o])):
                scores[k] = s
    report.write_scores(path, early.keys, np.array([scores[k] for k in early.keys]))


def save_expert_map(out: Path, model: Linear, n_moe: int, blinded: bool) -> None:
    shape = (n_moe, data.N_EXPERTS)
    coef = model.coef.reshape(shape)
    mean, scale = model.mean.reshape(shape), model.scale.reshape(shape)
    np.savez(out / "probe_weights.npz", coef=coef, coef_raw=coef / scale, intercept=model.intercept,
             intercept_raw=model.intercept - float((coef * mean / scale).sum()), mean=mean, scale=scale, C=model.C,
             decoder_layers=np.arange(3, 3 + n_moe), blinded=blinded)


def step_probes(args: argparse.Namespace) -> None:
    study = load_study(args)
    out = Path(args.out)
    feats = features.load_features(out / "features")
    kinds = kinds_of(args, PROBE_KINDS)
    missing = [k for k in kinds if not feats.has(k)]
    if missing:
        sys.exit(f"features lack {missing}; rerun features with them or leave them out with --kinds")
    pops = study.populations
    if "hidden" in kinds:
        write_f_any_coverage(out, feats, study)
    for kind in kinds:
        models, vocab = run_probe(out, feats, kind, pops["h1"], study, args.jobs)
        model = run_full(out, feats, kind, pops["h1"], study, args.jobs, vocab)
        earliness_scores(out, feats, kind, study, models, vocab)
        if kind == "routing":
            save_expert_map(out, model, len(model.coef) // data.N_EXPERTS, study.blind_seed is not None)
        if kind != "hidden":
            run_probe(out, feats, kind, pops["h1_f_any"], study, args.jobs)
    report.write_oof(out, study)


def write_f_any_coverage(out: Path, feats: features.Features, study: Study) -> None:
    """How many h1_f_any windows hold every hidden row, and why the rest do not (anchors only, no labels)."""
    base = study.populations["h1_f_any"]
    ok = feats.hidden_ok(base.window, base.keys)
    eps = [study.episodes[k] for k in base.keys]
    no_f = np.array([e["f"] is None for e in eps])
    before = np.array([e["f"] is not None and e["f_any"] < e["f"] for e in eps])
    cov = {"n_covered": int(ok.sum()), "of": len(base), "n_covered_equal_primary": int((ok & ~no_f & ~before).sum()),
           "n_uncovered_f_any_before_f": int((~ok & before).sum()), "n_uncovered_without_f": int((~ok & no_f).sum()),
           "note": "tap_hidden holds rows only in [f, window_end] and the H3 spans; an h1_f_any window holds every "
                   "row only when f_any == f, and then it equals the primary window"}
    _write_json(out / "probes" / f"{report.F_ANY_HIDDEN}.json", cov)
    log(f"f_any hidden rows: {cov['n_covered']} of {cov['of']} h1_f_any windows whole, all equal to the primary "
        f"window: {cov['n_covered_equal_primary'] == cov['n_covered']}")


# controls


def step_controls(args: argparse.Namespace) -> None:
    study = load_study(args)
    out = Path(args.out)
    probe_npz = out / "probes" / "routing__h1.npz"
    probe_json = out / "probes" / "routing__h1.json"
    if not (probe_npz.exists() and probe_json.exists()):
        sys.exit("controls need the routing probe on h1; run the probes step first")
    feats = features.load_features(out / "features")
    pop = study.populations["h1"]
    X = feats.matrix("routing", pop.window, pop.keys)
    y, tasks = pop.label, pop.task
    observed = json.loads(probe_json.read_text(encoding="utf-8"))["auroc"]
    fold_models, _ = _load_linear(probe_npz)
    limit = args.limit
    n_perm = controls.N_PERMUTATIONS if limit is None else min(limit, controls.N_PERMUTATIONS)
    n_sets = controls.N_RANDOM_SETS if limit is None else min(limit, controls.N_RANDOM_SETS)
    n_moe = X.shape[1] // data.N_EXPERTS
    n_layers = n_moe if limit is None else min(limit, n_moe)
    cdir = out / "controls"
    cdir.mkdir(parents=True, exist_ok=True)
    blinded = study.blind_seed is not None

    t0 = time.perf_counter()
    glob = controls.run_replicates(cdir / "shuffled_global.jsonl", {i: (p, None) for i, p in
                                   controls.global_permutations(y, n_perm).items()}, X, tasks, study.folds,
                                   args.jobs, log)
    within = controls.run_replicates(cdir / "shuffled_within_task.jsonl", {i: (p, None) for i, p in
                                     controls.within_task_permutations(y, tasks, n_perm).items()}, X, tasks,
                                     study.folds, args.jobs, log)
    _write_json(cdir / "shuffled.json", {
        "global": {**controls.null_summary([r["auroc"] for r in glob.values()], observed), "declared": True},
        "within_task": {**controls.null_summary([r["auroc"] for r in within.values()], observed),
                        "exploratory": True},
        "blinded": blinded, "seconds": round(time.perf_counter() - t0, 1)})
    log(f"shuffled labels: {n_perm} global and {n_perm} within-task permutations")

    t0 = time.perf_counter()
    sets = controls.expert_sets(cdir / "expert_sets.jsonl", X, y, tasks, study.folds,
                                np.stack([m.coef for m in fold_models]), n_sets, args.jobs, log=log)
    _write_json(cdir / "expert_sets.json", {"sets": sets, "n_random": n_sets, "blinded": blinded,
                                            "seconds": round(time.perf_counter() - t0, 1)})
    log(f"expert sets: K {list(sets)}, {n_sets} random sets each")

    t0 = time.perf_counter()
    lay = controls.layers(cdir / "layers.jsonl", X, y, tasks, pop.cluster, study.folds, n_layers, args.jobs, log=log)
    _write_json(cdir / "layers.json", {**lay, "n_layers": n_layers, "blinded": blinded,
                                       "seconds": round(time.perf_counter() - t0, 1)})
    log(f"single layers: {n_layers}")


# monitors


def needed_windows(study: Study) -> list[tuple[str, str]]:
    """Every (episode, window) a population reads, primary first."""
    need = {(k, p.window) for p in study.populations.values() for k in p.keys}
    return sorted(need, key=lambda kw: (WINDOW_ORDER[kw[1]], kw[0]))


def step_keyword(args: argparse.Namespace) -> None:
    study = load_study(args)
    out = Path(args.out)
    texts = features.load_texts(out / "features")
    path = out / "monitors" / "keyword.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, w in needed_windows(study):
        if (key, w) not in texts:
            sys.exit(f"no text for {key} window {w}; run features with the text kind")
        rows.append((key, w, int(bool(monitors.keyword_fires(texts[(key, w)]["analysis_text"])))))
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["episode_id", "window", "score"])
        writer.writerows(rows)
    tmp.replace(path)
    _write_json(out / "monitors" / "keyword_meta.json", {"n_windows": len(rows), "fired": sum(r[2] for r in rows),
                                                         "text": "analysis_text"})
    log(f"keyword: {len(rows)} windows scored")
    report.write_oof(out, study)


def step_qwen(args: argparse.Namespace) -> None:
    study = load_study(args)
    out = Path(args.out)
    texts = features.load_texts(out / "features")
    records, aliases, seen = [], {}, {}
    for key, w in needed_windows(study):
        if (key, w) not in texts:
            sys.exit(f"no text for {key} window {w}; run features with the text kind")
        text = texts[(key, w)]["text"]
        k = f"{key}|{w}"
        if (key, text) in seen:
            aliases[k] = seen[(key, text)]
            continue
        seen[(key, text)] = k
        records.append({"key": k, "source": study.episodes[key]["source"], "text": text})
    if args.limit is not None:
        records = records[:args.limit]
    path = out / "monitors" / "qwen_scores.jsonl"
    meta_path = out / "monitors" / "qwen_meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    threads = args.threads or 0  # 0 keeps torch's default, the physical cores
    log(f"qwen: {len(records)} distinct windows, {len(aliases)} duplicates scored once")
    common = {"n_records": len(records), "aliases": aliases, "limit": args.limit, "questions": monitors.QUESTIONS,
              "enable_thinking": False}
    _write_json(meta_path, {**common, "finished": False})  # aliases reach the report while scoring runs
    meta = monitors.qwen_run(records, path, args.qwen_model, args.qwen_revision, threads, log=log)
    _write_json(meta_path, {**meta, **common, "n_in_file": len(monitors.read_scores(path)), "finished": True})
    report.write_oof(out, study)


# report


def step_report(args: argparse.Namespace) -> None:
    study = load_study(args)
    out = Path(args.out)
    rows = report.write_oof(out, study)
    config = {"argv": sys.argv, "anchors": str(args.anchors), "rendered": [str(p) for p in args.rendered],
              "capture": args.capture, "jobs": args.jobs, "limit": args.limit, "blind_seed": args.blind_seed,
              "seed": 20261028, "C_grid": "10 ** arange(-5, 1.01, 0.5)",
              "run": json.loads((out / "run.json").read_text(encoding="utf-8"))}
    results = report.build_results(out, study, rows, config)
    _write_json(out / "results.json", results)
    (out / "report.md").write_text(report.render_markdown(results), encoding="utf-8")
    d = results["decisions"]
    log("report: " + ", ".join(f"{h} {v['decision']}" for h, v in d.items())
        + (" (blinded)" if results["blinded"] else ""))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=STEPS)
    p.add_argument("--capture", help="directory holding shard_000, shard_001, ...")
    p.add_argument("--rendered", nargs=2, required=True, metavar=("TRIPWIRE", "CVD"), help="rendered parquets")
    p.add_argument("--anchors", required=True, help="anchors.csv")
    p.add_argument("--out", required=True, help="run directory")
    p.add_argument("--shard", type=int, help="features: this shard only")
    p.add_argument("--jobs", type=int, default=1, help="parallel fits (one BLAS thread each)")
    p.add_argument("--kinds", help="comma list of routing,hidden,bag,text (default: all)")
    p.add_argument("--blind-seed", type=int, help="permute labels within each population with this seed")
    p.add_argument("--limit", type=int, help="cap permutations, random sets, layers and Qwen windows")
    p.add_argument("--qwen-model", default=monitors.QWEN_MODEL, help="Hub id or local directory")
    p.add_argument("--qwen-revision", default=monitors.QWEN_REVISION, help="Hub commit sha, recorded")
    p.add_argument("--threads", type=int, help="torch threads for qwen (default: torch's, the physical cores)")
    p.add_argument("--tokenizer", default=str(features.VENDOR / "tokenizer.json"), help="Beam tokenizer.json")
    args = p.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    {"features": step_features, "probes": step_probes, "controls": step_controls, "keyword": step_keyword,
     "qwen": step_qwen, "report": step_report}[args.step](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
