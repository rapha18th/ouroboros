"""One classifier for every probe, nested task-grouped cross-validation, full fits and cluster bootstraps.

Classifier: StandardScaler then L2 logistic regression (lbfgs, max_iter 10000, tol 1e-8), float64. l1_ratio=0 is
sklearn 1.9's spelling of penalty="l2", which it deprecates. sklearn passes tol to L-BFGS-B as the gradient test on
the mean loss; its default, 1e-4, stops fits at C >= 1 well short of the optimum without a warning.
- Outer: the 5 task folds. OOF score = decision_function on the held-out fold, standardized by that fold's own mean
  and sd (no labels). Each fold model picks its own C, so raw scores differ in scale and intercept across folds;
  the pooled AUROC ranks the standardized scores. The fold models carry the same map, and oof_raw keeps the raw
  scores.
- Inner, per outer fold o: the training tasks in 4 folds (populations.inner_folds). C is the grid value with the
  highest mean inner-fold AUROC, ties to the smaller C, refit on the whole training part.
- Full fit: C chosen the same way with the 5 outer folds as the CV splits, then fit on every episode. Its scores
  stay raw: one model, no pooling.
- Bootstrap: resample clusters with replacement, B times from default_rng(seed); each replicate's AUROC is that of
  the concatenated episodes, computed from integer cluster counts as weights. Single-class replicates are skipped
  and counted. Two scores on the same episodes share every replicate, so a gap is paired.

Every fit runs with one BLAS thread; joblib spreads independent fits over processes, so results do not depend on
the worker count.
"""

from __future__ import annotations

import atexit
import gc
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ouroboros.analysis.populations import N_FOLDS, N_INNER, SEED, inner_folds

C_GRID = 10.0 ** np.arange(-5, 1.01, 0.5)
B_DEFAULT = 10_000
TOL = 1e-8  # lbfgs gradient tolerance on the mean loss


def make_model(C: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegression(C=C, l1_ratio=0.0, solver="lbfgs", max_iter=10000, tol=TOL))])


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """roc_auc_score, or NaN when y holds one class."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


@dataclass
class Linear:
    """A fitted pipeline as plain arrays: score = ((x - mean) / scale) @ coef + intercept."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray  # standardized units
    intercept: float
    C: float
    n_iter: int = 0  # lbfgs iterations of the fit

    @classmethod
    def of(cls, model: Pipeline, C: float) -> Linear:
        sc, lr = model.named_steps["scale"], model.named_steps["lr"]
        return cls(sc.mean_.copy(), sc.scale_.copy(), lr.coef_[0].copy(), float(lr.intercept_[0]), float(C),
                   int(lr.n_iter_[0]))

    def standardized(self, mean: float, sd: float) -> Linear:
        """The same model with (score - mean) / sd folded into its coef and intercept."""
        return replace(self, coef=self.coef / sd, intercept=(self.intercept - mean) / sd)

    def score(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, dtype=np.float64) - self.mean) / self.scale) @ self.coef + self.intercept


def _take(X: np.ndarray, rows: np.ndarray, cols: np.ndarray | None) -> np.ndarray:
    return X[rows] if cols is None else X[np.ix_(rows, cols)]


def _fit_score(X: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, C: float,
               cols: np.ndarray | None, keep: bool) -> tuple[np.ndarray, Linear | None]:
    """Fit on `train`, return decision scores on `test` (and the model as arrays when `keep`)."""
    model = make_model(C).fit(_take(X, train, cols), y[train])
    scores = model.decision_function(_take(X, test, cols)) if len(test) else np.zeros(0)
    return scores, (Linear.of(model, C) if keep else None)


def parallel(fn: Callable, calls: Sequence[tuple], jobs: int = 1) -> list:
    """fn(*args) for every args tuple, in order, with one BLAS thread per fit; in-process when jobs is 1.

    The thread limit is set once per call here (threadpoolctl takes tens of milliseconds per limit), and loky
    workers start with it.
    """
    if jobs == 1:
        with threadpool_limits(1):
            return [fn(*args) for args in calls]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=jobs)(delayed(fn)(*args) for args in calls)


@contextmanager
def shared(X: np.ndarray, jobs: int) -> Iterator[np.ndarray]:
    """X as a read-only memmap that workers open by file name, so joblib neither hashes nor copies it per call."""
    if jobs == 1 or isinstance(X, np.memmap):
        yield X
        return
    folder = tempfile.mkdtemp(prefix="ouroboros-")
    path = os.path.join(folder, "X.npy")
    np.save(path, np.ascontiguousarray(X))
    m = np.load(path, mmap_mode="r")
    try:
        yield m
    finally:
        del m
        gc.collect()
        shutil.rmtree(folder, ignore_errors=True)
        if os.path.exists(folder):  # a worker still holds the file (Windows); retry at exit
            _LEFTOVER.append(folder)


_LEFTOVER: list[str] = []


@atexit.register
def _remove_leftovers() -> None:
    for folder in _LEFTOVER:
        shutil.rmtree(folder, ignore_errors=True)


def choose_c(mean_auc: np.ndarray) -> int:
    """Index of the best mean AUROC, ties to the smaller C; 0 when every mean is NaN."""
    if np.isnan(mean_auc).all():
        return 0
    return int(np.nanargmax(mean_auc))


def _nanmean(a: np.ndarray, axis: int) -> np.ndarray:
    n = (~np.isnan(a)).sum(axis=axis)
    return np.where(n > 0, np.nansum(a, axis=axis) / np.maximum(n, 1), np.nan)


@dataclass
class CVResult:
    oof: np.ndarray  # OOF decision score per episode, standardized within its held-out fold
    fold: np.ndarray  # outer fold per episode
    C: list[float]  # chosen per outer fold
    fold_auroc: list[float]
    inner_mean: np.ndarray  # [5, 13] mean inner AUROC per C
    models: list[Linear] | None  # fold models, scoring on the standardized scale
    oof_raw: np.ndarray  # OOF decision score per episode as its fold model gave it
    fold_mean: list[float]  # held-out raw score mean and sd per outer fold, the standardization
    fold_sd: list[float]

    def auroc(self, y: np.ndarray) -> float:
        """The registered statistic: AUROC of the pooled fold-standardized OOF scores."""
        return auroc(y, self.oof)

    def auroc_raw(self, y: np.ndarray) -> float:
        """AUROC of the pooled raw OOF scores (exploratory)."""
        return auroc(y, self.oof_raw)


def fold_standardize(s: np.ndarray) -> tuple[float, float]:
    """Mean and sd (ddof 0) of one held-out fold's scores; sd 1 when the scores are constant or absent."""
    if len(s) == 0:
        return 0.0, 1.0
    mean, sd = float(s.mean()), float(s.std())
    return mean, (sd if sd > 0 else 1.0)


def _per_fold(cols, n_folds: int) -> list[np.ndarray | None]:
    if cols is None or isinstance(cols, np.ndarray):
        return [cols] * n_folds
    return list(cols)


def nested_cv(X: np.ndarray, y: np.ndarray, tasks: Sequence[str], folds: dict[str, int], jobs: int = 1,
              cols: np.ndarray | Sequence[np.ndarray] | None = None, keep_models: bool = False) -> CVResult:
    """Nested CV over the task folds. `cols` restricts the features, shared or one array per outer fold."""
    y = np.asarray(y, dtype=np.int64)
    tasks = np.asarray(tasks, dtype=object)
    fold = np.array([folds[t] for t in tasks])
    cols_o = _per_fold(cols, N_FOLDS)
    splits = []  # (outer, inner, train, validation)
    for o in range(N_FOLDS):
        train = np.flatnonzero(fold != o)
        inner = inner_folds(tasks[train], o)
        part = np.array([inner[t] for t in tasks[train]])
        for i in range(N_INNER):
            a, b = train[part != i], train[part == i]
            if len(a) and len(b):
                splits.append((o, i, a, b))
    inner_auc = np.full((N_FOLDS, N_INNER, len(C_GRID)), np.nan)
    oof = np.full(len(y), np.nan)
    oof_raw = np.full(len(y), np.nan)
    models, fold_mean, fold_sd = [], [], []
    with shared(X, jobs) as Xs:
        calls = [(Xs, y, a, b, C, cols_o[o], False) for o, i, a, b in splits for C in C_GRID]
        where = [(o, i, ci, b) for o, i, a, b in splits for ci in range(len(C_GRID))]
        for (o, i, ci, b), (s, _) in zip(where, parallel(_fit_score, calls, jobs)):
            inner_auc[o, i, ci] = auroc(y[b], s)
        inner_mean = _nanmean(inner_auc, axis=1)
        chosen = [float(C_GRID[choose_c(inner_mean[o])]) for o in range(N_FOLDS)]
        refits = [(Xs, y, np.flatnonzero(fold != o), np.flatnonzero(fold == o), chosen[o], cols_o[o], keep_models)
                  for o in range(N_FOLDS)]
        for o, (s, m) in enumerate(parallel(_fit_score, refits, jobs)):
            mean, sd = fold_standardize(s)  # label-free, per held-out fold
            oof_raw[fold == o] = s
            oof[fold == o] = (s - mean) / sd
            models.append(None if m is None else m.standardized(mean, sd))
            fold_mean.append(mean)
            fold_sd.append(sd)
    fold_auc = [auroc(y[fold == o], oof[fold == o]) for o in range(N_FOLDS)]
    return CVResult(oof, fold, chosen, fold_auc, inner_mean, models if keep_models else None, oof_raw, fold_mean,
                    fold_sd)


def full_fit(X: np.ndarray, y: np.ndarray, tasks: Sequence[str], folds: dict[str, int], jobs: int = 1,
             cols: np.ndarray | None = None) -> tuple[Linear, np.ndarray]:
    """C from the 5 outer folds as CV splits, then a fit on every episode; returns the model and the CV means."""
    y = np.asarray(y, dtype=np.int64)
    fold = np.array([folds[t] for t in tasks])
    splits = [(o, np.flatnonzero(fold != o), np.flatnonzero(fold == o)) for o in range(N_FOLDS)]
    splits = [(o, a, b) for o, a, b in splits if len(a) and len(b)]
    auc = np.full((N_FOLDS, len(C_GRID)), np.nan)
    with shared(X, jobs) as Xs:
        calls = [(Xs, y, a, b, C, cols, False) for o, a, b in splits for C in C_GRID]
        where = [(o, ci, b) for o, a, b in splits for ci in range(len(C_GRID))]
        for (o, ci, b), (s, _) in zip(where, parallel(_fit_score, calls, jobs)):
            auc[o, ci] = auroc(y[b], s)
    mean = _nanmean(auc, axis=0)
    C = float(C_GRID[choose_c(mean)])
    everyone = np.arange(len(y))
    with threadpool_limits(1):  # one BLAS thread, as every other fit, so the result does not depend on the cores
        _, model = _fit_score(X, y, everyone, everyone[:0], C, cols, True)
    return model, mean


# Bootstrap


def cluster_weights(clusters: Sequence, B: int, seed: int = SEED) -> np.ndarray:
    """[B, n] integer weights: how often each episode's cluster was drawn in each replicate."""
    names, idx = np.unique(np.asarray(clusters, dtype=object).astype(str), return_inverse=True)
    G = len(names)
    draws = np.random.default_rng(seed).integers(0, G, size=(B, G))
    counts = np.bincount((draws + G * np.arange(B)[:, None]).ravel(), minlength=B * G).reshape(B, G)
    return counts[:, idx]


def weighted_auroc(y: np.ndarray, s: np.ndarray, W: np.ndarray) -> np.ndarray:
    """AUROC per row of integer weights W [B, n], equal to roc_auc_score on the replicated episodes; NaN when a
    replicate holds one class."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    W = np.atleast_2d(W)
    order = np.argsort(s, kind="stable")
    ss = s[order]
    starts = np.flatnonzero(np.r_[True, np.diff(ss) != 0])  # tie groups
    out = np.empty(W.shape[0])
    for lo in range(0, W.shape[0], 1000):
        w = W[lo:lo + 1000][:, order].astype(np.float64)
        pos = np.add.reduceat(w * y[order], starts, axis=1)
        neg = np.add.reduceat(w * ~y[order], starts, axis=1)
        below = np.cumsum(neg, axis=1) - neg
        n_pos, n_neg = pos.sum(axis=1), neg.sum(axis=1)
        u = (pos * (below + 0.5 * neg)).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[lo:lo + 1000] = np.where((n_pos > 0) & (n_neg > 0), u / (n_pos * n_neg), np.nan)
    return out


def percentile_ci(values: np.ndarray) -> list[float]:
    v = values[np.isfinite(values)]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])] if len(v) else [float("nan")] * 2


def auc_summary(y: np.ndarray, s: np.ndarray, clusters: Sequence, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """Point AUROC with its percentile 95% cluster-bootstrap CI, over the episodes with a finite score."""
    y, s, clusters = np.asarray(y), np.asarray(s, dtype=np.float64), np.asarray(clusters, dtype=object)
    ok = np.isfinite(s)
    y, s, clusters = y[ok], s[ok], clusters[ok]
    out = {"auroc": auroc(y, s), "n": int(len(y)), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
           "n_clusters": int(len(np.unique(clusters.astype(str)))), "B": B}
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {**out, "ci": [float("nan")] * 2, "skipped": B}
    reps = weighted_auroc(y, s, cluster_weights(clusters, B, seed))
    return {**out, "ci": percentile_ci(reps), "skipped": int(np.isnan(reps).sum())}


def gap_summary(y: np.ndarray, s1: np.ndarray, s2: np.ndarray, clusters: Sequence, B: int = B_DEFAULT,
                seed: int = SEED) -> dict:
    """AUROC(s1) - AUROC(s2) with a paired cluster-bootstrap CI, over episodes where both scores are finite."""
    y, clusters = np.asarray(y), np.asarray(clusters, dtype=object)
    s1, s2 = np.asarray(s1, dtype=np.float64), np.asarray(s2, dtype=np.float64)
    ok = np.isfinite(s1) & np.isfinite(s2)
    y, s1, s2, clusters = y[ok], s1[ok], s2[ok], clusters[ok]
    out = {"gap": auroc(y, s1) - auroc(y, s2), "n": int(len(y)), "B": B,
           "n_clusters": int(len(np.unique(clusters.astype(str))))}
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {**out, "ci": [float("nan")] * 2, "skipped": B}
    W = cluster_weights(clusters, B, seed)
    reps = weighted_auroc(y, s1, W) - weighted_auroc(y, s2, W)
    return {**out, "ci": percentile_ci(reps), "skipped": int(np.isnan(reps).sum())}
