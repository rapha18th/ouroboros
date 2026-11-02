"""Controls on the H1 routing probe, effort strata and the earliness curve.

- shuffled labels (declared): permutation i draws default_rng(20261028 + i).permutation of the H1 labels; each runs
  the full nested CV. p = (1 + #null >= observed) / (n + 1).
- within-task shuffled labels (exploratory): default_rng(20261028 + 500000 + i) permutes the labels inside each
  task, tasks in sorted order.
- random expert sets (interpretation): for K in 16, 64, 256, 1024, set r is default_rng(20261028 + 1000 K + r)
  .choice of K features without replacement, shared by the 5 folds. The selected set of outer fold o is the top K
  by |coef| of that fold's routing model (standardized units); both run the nested CV restricted to their set.
  Fold o's selected set comes from fold o's whole training part, so the selected arm's inner C search scores inner
  folds that helped choose the set: its inner AUROCs run high and its C is picked on them. The outer held-out fold
  stays clean, so the selected AUROC is a held-out estimate; in synthetic checks the leak moved it by under 0.015 on
  average, with no consistent sign.
- one layer at a time: MoE layer j (decoder layer 3 + j) alone, 256 features, with a B = 2,000 bootstrap CI.
- effort strata: the H1 OOF AUROC inside each effort level.
- earliness: each gap bin's positives against every negative, and the cumulative curve over gap >= g (exploratory).
  Gaps count rendered token positions between the window end and d.

Every nested CV here pools fold-standardized OOF scores (probe.nested_cv), as the observed H1 statistic does.

Replicates append to a jsonl file as they finish, so a restart resumes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, parallel_config

from ouroboros.analysis.data import N_EXPERTS
from ouroboros.analysis.probe import auc_summary, nested_cv, shared
from ouroboros.analysis.populations import SEED

SET_SIZES = (16, 64, 256, 1024)
N_PERMUTATIONS = 100
N_RANDOM_SETS = 50
LAYER_B = 2_000
WITHIN_TASK_OFFSET = 500_000
GAP_BINS = ((-np.inf, 0), (0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, np.inf))
GAP_CUMULATIVE = (0, 250, 500, 1000, 2000, 4000, 8000)
PACKING_NOTE = ("Routing depends on how episodes are packed: two stock captures that pack the same episode "
                "differently agree bit for bit at decoder layer 3 and on about 60% of top-8 sets at decoder layers "
                "43 to 51 (MoE index 40 to 48). Stage A window features agree at Pearson 0.999 across packings. "
                "Read the deep layers' single-layer AUROCs with that in mind.")


def _replicate(X: np.ndarray, y: np.ndarray, tasks: np.ndarray, folds: dict[str, int], cols) -> dict:
    cv = nested_cv(X, y, tasks, folds, jobs=1, cols=cols)
    return {"auroc": cv.auroc(y), "C": cv.C, "oof": cv.oof.tolist()}


def _read_jsonl(path: Path) -> dict:
    done = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    done[row["id"]] = row
    return done


def run_replicates(path: Path, specs: dict[str, tuple[np.ndarray, object]], X: np.ndarray, tasks: np.ndarray,
                   folds: dict[str, int], jobs: int, log: Callable[[str], None] = print,
                   keep_oof: bool = False) -> dict[str, dict]:
    """Nested CV for every spec id -> (labels, cols), resuming from and appending to `path`."""
    done = _read_jsonl(path)
    todo = [i for i in specs if i not in done]
    if todo:
        log(f"{path.name}: {len(todo)} of {len(specs)} replicates to run")
        with (shared(X, jobs) as Xs, open(path, "a", encoding="utf-8") as fh,
              parallel_config(backend="loky", inner_max_num_threads=1)):  # one BLAS thread per worker
            calls = (delayed(_replicate)(Xs, specs[i][0], tasks, folds, specs[i][1]) for i in todo)
            results = (Parallel(n_jobs=jobs, return_as="generator")(calls) if jobs > 1 else
                       (_replicate(Xs, specs[i][0], tasks, folds, specs[i][1]) for i in todo))
            for i, r in zip(todo, results):
                row = {"id": i, "auroc": r["auroc"], "C": r["C"], **({"oof": r["oof"]} if keep_oof else {})}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                done[i] = row
    return {i: done[i] for i in specs}


def null_summary(null: Sequence[float], observed: float) -> dict:
    null = np.asarray(null, dtype=np.float64)
    n = len(null)
    return {"n": n, "observed": observed, "mean": float(null.mean()) if n else None,
            "sd": float(null.std(ddof=1)) if n > 1 else None,
            "p95": float(np.percentile(null, 95)) if n else None,
            "n_ge_observed": int((null >= observed).sum()),
            "p": float((1 + (null >= observed).sum()) / (n + 1)), "null": null.tolist()}


def global_permutations(y: np.ndarray, n: int) -> dict[str, np.ndarray]:
    return {f"perm{i:03d}": np.random.default_rng(SEED + i).permutation(y) for i in range(n)}


def within_task_permutations(y: np.ndarray, tasks: np.ndarray, n: int) -> dict[str, np.ndarray]:
    out = {}
    for i in range(n):
        rng = np.random.default_rng(SEED + WITHIN_TASK_OFFSET + i)
        perm = np.asarray(y).copy()
        for t in sorted(set(tasks)):
            idx = np.flatnonzero(tasks == t)
            perm[idx] = perm[idx][rng.permutation(len(idx))]
        out[f"perm{i:03d}"] = perm
    return out


def random_set(n_features: int, K: int, r: int) -> np.ndarray:
    return np.sort(np.random.default_rng(SEED + 1000 * K + r).choice(n_features, size=K, replace=False))


def selected_sets(fold_coef: np.ndarray, K: int) -> list[np.ndarray]:
    """Per outer fold, the K features with the largest |coef| (stable order on ties), sorted."""
    return [np.sort(np.argsort(-np.abs(c), kind="stable")[:K]) for c in fold_coef]


def expert_sets(path: Path, X: np.ndarray, y: np.ndarray, tasks: np.ndarray, folds: dict[str, int],
                fold_coef: np.ndarray, n_sets: int, jobs: int, sizes: Sequence[int] = SET_SIZES,
                log: Callable[[str], None] = print) -> dict:
    specs = {}
    for K in sizes:
        specs[f"K{K}_selected"] = (y, selected_sets(fold_coef, K))
        for r in range(n_sets):
            specs[f"K{K}_r{r:02d}"] = (y, random_set(X.shape[1], K, r))
    done = run_replicates(path, specs, X, tasks, folds, jobs, log)
    out = {}
    for K in sizes:
        sel = done[f"K{K}_selected"]["auroc"]
        rand = np.array([done[f"K{K}_r{r:02d}"]["auroc"] for r in range(n_sets)])
        out[str(K)] = {"selected_auroc": sel, "selected_C": done[f"K{K}_selected"]["C"],
                       "random": null_summary(rand, sel)}
    return out


def layer_specs(y: np.ndarray, n_layers: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {f"layer{j:02d}": (y, np.arange(j * N_EXPERTS, (j + 1) * N_EXPERTS)) for j in range(n_layers)}


def layers(path: Path, X: np.ndarray, y: np.ndarray, tasks: np.ndarray, clusters: np.ndarray,
           folds: dict[str, int], n_layers: int, jobs: int, first_moe_layer: int = 3,
           log: Callable[[str], None] = print) -> dict:
    done = run_replicates(path, layer_specs(y, n_layers), X, tasks, folds, jobs, log, keep_oof=True)
    rows = []
    for j in range(n_layers):
        r = done[f"layer{j:02d}"]
        s = auc_summary(y, np.array(r["oof"]), clusters, B=LAYER_B)
        rows.append({"moe_index": j, "decoder_layer": first_moe_layer + j, "C": r["C"], **s})
    return {"layers": rows, "note": PACKING_NOTE}


def strata(y: np.ndarray, s: np.ndarray, clusters: np.ndarray, groups: np.ndarray) -> dict:
    """AUROC with its cluster-bootstrap CI inside each group level."""
    return {str(g): auc_summary(y[groups == g], s[groups == g], clusters[groups == g])
            for g in sorted(set(groups.astype(str)))}


def earliness(y: np.ndarray, s: np.ndarray, gap: np.ndarray, clusters: np.ndarray) -> dict:
    """Per gap bin and cumulatively (gap >= g): the positives there against every negative."""
    y, s, gap, clusters = np.asarray(y), np.asarray(s, dtype=np.float64), np.asarray(gap), np.asarray(clusters)
    neg = y == 0

    def against(pos: np.ndarray) -> dict:
        keep = pos | neg
        return auc_summary(y[keep], s[keep], clusters[keep])

    has = (y == 1) & np.isfinite(gap)
    bins = []
    for lo, hi in GAP_BINS:
        pos = has & (gap >= lo) & (gap < hi)
        bins.append({"lo": None if np.isinf(lo) else lo, "hi": None if np.isinf(hi) else hi, **against(pos)})
    cumulative = [{"gap_at_least": g, **against(has & (gap >= g))} for g in GAP_CUMULATIVE]
    return {"bins": bins, "cumulative": cumulative, "cumulative_exploratory": True, "n_negatives": int(neg.sum()),
            "n_positives_without_gap": int(((y == 1) & ~np.isfinite(gap)).sum())}
