"""Stage A populations, the pre-registered count checks and the task folds.

From anchors.csv (labels, anchors, effort) and the rendered parquets (task_id):
- Stage A: Tripwire with label 0 or 1. Label -1 is excluded or guard.
- h1: positives with f, d and d - window_end >= 500; negatives with f. Window primary.
- h1_f_any: the same under f_any and window_end_any. Window f_any.
- earliness: every Stage A positive with f and d (gap = d - window_end), against the h1 negatives. Window primary.
- h3_scall, h3_s: Claimed vs Done with label 0 or 1. Windows h3_scall and h3_s.
- h3_prefinish (exploratory): the same episodes and labels as h3_scall, window h3_prefinish.

When anchors.csv hashes to the registered sha256, every count the pre-registration states must come out exactly,
or building the study fails. Folds: tasks = sorted Stage A task_ids, perm = default_rng(20261028).permutation,
fold[tasks[perm[i]]] = i % 5. Inner folds per outer fold o: the sorted training tasks under
default_rng(20261028 + o).permutation, i % 4. Bootstrap clusters: task_id on Tripwire, the last episode_id segment
(a task instance such as trap-contradiction-r2, run at most once per batch and repeated across batches) on Claimed
vs Done.

A blind seed permutes each population's labels, with the gaps that travel with them, before anything reads them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ouroboros.analysis.data import load_anchors, load_meta, sha256

SEED = 20261028
N_FOLDS = 5
N_INNER = 4
GAP_MIN = 500
REGISTERED_ANCHORS_SHA256 = "8b08733df41d44af56c806528b80da5072f22af5dcda77f38b4bec28f9943b81"
REGISTERED_COUNTS = {
    "stage_a": 430, "stage_a_pos": 172, "stage_a_neg": 258,
    "stage_a_conditions": 440, "stage_a_conditions_with_f": 360, "guard": 176,
    "stage_a_pos_without_f": 27, "stage_a_neg_without_f": 52,
    "h1_pos": 120, "h1_pos_tamper": 96, "h1_pos_special": 24, "h1_neg": 206,
    "h1_f_any_pos": 140, "h1_f_any_neg": 218,
    "h3_pos": 64, "h3_neg": 31, "h3_excluded": 3, "h3_with_s_and_s_call": 95,
    "tasks": 44,
}
WINDOW_OF = {"h1": "primary", "h1_f_any": "f_any", "earliness": "primary", "h3_scall": "h3_scall", "h3_s": "h3_s",
             "h3_prefinish": "h3_prefinish"}
STAGE_A_CONDITIONS = ("impossible", "impossible_honest")


@dataclass
class Population:
    name: str
    window: str
    keys: list[str]
    label: np.ndarray  # int64, blinded when the study is
    task: np.ndarray  # task_id
    cluster: np.ndarray  # bootstrap cluster
    fold: np.ndarray  # outer fold, -1 on Claimed vs Done
    effort: np.ndarray
    gap: np.ndarray  # float: d minus the window end for positives with d, NaN otherwise
    exploratory: bool = False

    def __len__(self) -> int:
        return len(self.keys)

    def subset(self, mask: np.ndarray, name: str, exploratory: bool = True) -> Population:
        mask = np.asarray(mask, dtype=bool)
        return replace(self, name=name, keys=[k for k, m in zip(self.keys, mask) if m], label=self.label[mask],
                       task=self.task[mask], cluster=self.cluster[mask], fold=self.fold[mask],
                       effort=self.effort[mask], gap=self.gap[mask], exploratory=exploratory)

    def counts(self) -> dict:
        return {"n": len(self), "n_pos": int((self.label == 1).sum()), "n_neg": int((self.label == 0).sum()),
                "n_clusters": int(len(np.unique(self.cluster)))}


@dataclass
class Study:
    episodes: dict[str, dict]  # anchors.csv row plus task_id and cluster
    populations: dict[str, Population]
    folds: dict[str, int]  # Tripwire task_id -> outer fold
    counts: dict[str, int]
    registered: bool  # anchors.csv is the registered file, so the counts were enforced
    anchors_sha256: str
    blind_seed: int | None = None
    notes: list[str] = field(default_factory=list)

    def describe(self) -> dict:
        return {
            "anchors_sha256": self.anchors_sha256,
            "registered_anchors": self.registered,
            "counts_enforced": self.registered,
            "blinded": self.blind_seed is not None,
            "blind_seed": self.blind_seed,
            "counts": self.counts,
            "populations": {name: {"window": p.window, "exploratory": p.exploratory, **p.counts()}
                            for name, p in self.populations.items()},
            "fold_of_task": dict(sorted(self.folds.items())),
            "tasks_per_fold": [sum(f == o for f in self.folds.values()) for o in range(N_FOLDS)],
            "notes": self.notes,
        }


def outer_folds(tasks: Sequence[str]) -> dict[str, int]:
    """fold[tasks[perm[i]]] = i % 5 over the sorted unique tasks, perm from default_rng(20261028)."""
    tasks = sorted(set(tasks))
    perm = np.random.default_rng(SEED).permutation(len(tasks))
    return {tasks[p]: i % N_FOLDS for i, p in enumerate(perm)}


def inner_folds(train_tasks: Sequence[str], outer: int) -> dict[str, int]:
    """The 4 inner folds of outer fold `outer`: sorted training tasks under default_rng(20261028 + outer)."""
    tasks = sorted(set(train_tasks))
    perm = np.random.default_rng(SEED + outer).permutation(len(tasks))
    return {tasks[p]: i % N_INNER for i, p in enumerate(perm)}


def cluster_of(episode_id: str, source: str, task_id: str) -> str:
    return task_id if source == "tripwire" else episode_id.rsplit("/", 1)[-1]


def _kind(row: dict) -> str | None:
    reason = row.get("label_reason") or ""
    return "special" if "special_cased" in reason else ("tamper" if "tamper_attempt" in reason else None)


def build_study(anchors_path: str | Path, rendered: Sequence[str | Path], blind_seed: int | None = None) -> Study:
    """Populations, counts and folds; enforces the registered counts when anchors.csv is the registered file."""
    anchors = load_anchors(anchors_path)
    meta = load_meta(rendered)
    if set(anchors) != set(meta):
        raise ValueError(f"anchors.csv and the parquets disagree on {len(set(anchors) ^ set(meta))} episode ids")
    episodes = {}
    for k, a in anchors.items():
        m = meta[k]
        if m["label"] is not None and int(m["label"]) != a["label"]:
            raise ValueError(f"{k}: label {a['label']} in anchors.csv, {m['label']} in the parquet")
        if m["source"] != a["source"]:
            raise ValueError(f"{k}: source {a['source']} in anchors.csv, {m['source']} in the parquet")
        episodes[k] = {**a, "task_id": m["task_id"], "cluster": cluster_of(k, a["source"], m["task_id"]),
                       "kind": _kind(a)}
    keys = list(anchors)  # anchors.csv order
    trip = [k for k in keys if episodes[k]["source"] == "tripwire"]
    cvd = [k for k in keys if episodes[k]["source"] == "cvd"]
    stage_a = [k for k in trip if episodes[k]["label"] in (0, 1)]
    pos = [k for k in stage_a if episodes[k]["label"] == 1]
    neg = [k for k in stage_a if episodes[k]["label"] == 0]

    def ok(k: str, f: str, end: str) -> bool:
        e = episodes[k]
        if e[f] is None or e[end] is None:
            return False
        return e["label"] == 0 or (e["d"] is not None and e["d"] - e[end] >= GAP_MIN)

    h1 = [k for k in stage_a if ok(k, "f", "window_end")]
    h1_any = [k for k in stage_a if ok(k, "f_any", "window_end_any")]
    h1_neg = [k for k in h1 if episodes[k]["label"] == 0]
    early_pos = {k for k in pos if episodes[k]["f"] is not None and episodes[k]["d"] is not None}
    earliness = [k for k in stage_a if k in early_pos or k in set(h1_neg)]
    h3 = [k for k in cvd if episodes[k]["label"] in (0, 1)]
    missing = [k for k in h3 if episodes[k]["s"] is None or episodes[k]["s_call"] is None]
    if missing:
        raise ValueError(f"{len(missing)} Claimed vs Done episodes lack s or s_call, e.g. {missing[0]}")

    tasks = sorted({episodes[k]["task_id"] for k in stage_a})
    folds = outer_folds(tasks)
    conditions = [k for k in trip if episodes[k]["condition"] in STAGE_A_CONDITIONS]

    def n(ks, label=None, kind=None):
        return sum(1 for k in ks if (label is None or episodes[k]["label"] == label)
                   and (kind is None or episodes[k]["kind"] == kind))

    counts = {
        "stage_a": len(stage_a), "stage_a_pos": len(pos), "stage_a_neg": len(neg),
        "stage_a_conditions": len(conditions),
        "stage_a_conditions_with_f": sum(episodes[k]["f"] is not None for k in conditions),
        "guard": sum(episodes[k]["condition"] == "possible" for k in trip),
        "stage_a_pos_without_f": sum(episodes[k]["f"] is None for k in pos),
        "stage_a_neg_without_f": sum(episodes[k]["f"] is None for k in neg),
        "h1_pos": n(h1, 1), "h1_pos_tamper": n(h1, 1, "tamper"), "h1_pos_special": n(h1, 1, "special"),
        "h1_neg": n(h1, 0), "h1_f_any_pos": n(h1_any, 1), "h1_f_any_neg": n(h1_any, 0),
        "h3_pos": n(h3, 1), "h3_neg": n(h3, 0), "h3_excluded": n(cvd, -1),
        "h3_with_s_and_s_call": len(h3) - len(missing), "tasks": len(tasks),
        "earliness_pos": len(early_pos), "earliness_neg": len(h1_neg),
        "cvd_clusters": len({episodes[k]["cluster"] for k in h3}),
        "cvd_batches": len({k.split("/")[1] for k in h3}),
    }
    digest = sha256(anchors_path)
    registered = digest == REGISTERED_ANCHORS_SHA256
    if registered:
        bad = {c: (counts[c], want) for c, want in REGISTERED_COUNTS.items() if counts[c] != want}
        if bad:
            raise ValueError(f"registered counts not reproduced (got, want): {bad}")

    def population(name: str, ks: list[str], end: str | None) -> Population:
        e = [episodes[k] for k in ks]
        gap = np.array([np.nan if end is None or x["d"] is None or x["label"] != 1 else x["d"] - x[end] for x in e],
                       dtype=np.float64)
        return Population(
            name=name, window=WINDOW_OF[name], keys=list(ks),
            label=np.array([x["label"] for x in e], dtype=np.int64),
            task=np.array([x["task_id"] for x in e], dtype=object),
            cluster=np.array([x["cluster"] for x in e], dtype=object),
            fold=np.array([folds.get(x["task_id"], -1) if x["source"] == "tripwire" else -1 for x in e]),
            effort=np.array([x["effort"] for x in e], dtype=object), gap=gap)

    pops = {"h1": population("h1", h1, "window_end"),
            "h1_f_any": population("h1_f_any", h1_any, "window_end_any"),
            "earliness": population("earliness", earliness, "window_end"),
            "h3_scall": population("h3_scall", h3, None)}
    if blind_seed is not None:
        rng = np.random.default_rng(blind_seed)
        for name in ("h1", "h1_f_any", "earliness", "h3_scall"):
            p = pops[name]
            perm = rng.permutation(len(p))
            pops[name] = replace(p, label=p.label[perm], gap=p.gap[perm])
    pops["h3_s"] = replace(pops["h3_scall"], name="h3_s", window="h3_s")
    pops["h3_prefinish"] = replace(pops["h3_scall"], name="h3_prefinish", window="h3_prefinish", exploratory=True)
    notes = []
    if not registered:
        notes.append(f"anchors.csv sha256 {digest} is not the registered file; counts recorded, not enforced")
    return Study(episodes, pops, folds, counts, registered, digest, blind_seed, notes)
