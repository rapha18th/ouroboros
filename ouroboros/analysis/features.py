"""Window features, one capture shard at a time, and the merged view the probes read.

Every feature reads only the Beam-written tokens of its window (data.window_positions):
- routing: per MoE layer j and expert e, the count of e among the 8 selected ids over the window's tokens, divided
  by the token count; [49 * 256], index j * 256 + e.
- bag: the window's token ids; the probe turns them into frequencies over its population's vocabulary.
- hidden: per tap layer, the mean float32 hidden state over the window's rows; [5 * H]. A window with rows missing
  from the capture (f_any in some episodes) is marked missing; on any other window a missing row is an error.
- text: the window decoded with the Beam tokenizer, special tokens kept, contiguous runs joined by "\\n[...]\\n".
  analysis_text keeps only the content of Beam's own analysis messages (<|start|>assistant<|channel|>analysis
  <|message|> up to <|end|> or <|call|>), which leaves out tool calls (their channel header carries <|constrain|>)
  and tool results (their role is the tool).

h3_prefinish needs first_finish, the <|start|> of the first finish call before the accepted one; the shard step finds
it in the token ids (data.first_finish) and counts the windows where it differs from h3_scall.

features/shard_K.npz holds the arrays, features/shard_K.json the episode keys per window and the shard's
fingerprint (written last, so its presence marks a finished shard), features/texts_K.jsonl one line per window.
Each file goes through a temporary name and os.replace.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from ouroboros.analysis.data import (INDEX_ANCHORS, WINDOWS, Shard, covered, first_finish, hidden_features,
                                     routing_features, window_positions)

KINDS = ("routing", "hidden", "bag", "text")
VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "beam"
GAP = "\n[...]\n"
NEEDS = {"routing": ("route_ids",), "hidden": ("tap_hidden", "windows"), "bag": (), "text": ()}
ALWAYS = ("token_ids", "gen_mask")


def tokenizer(path: str | Path = VENDOR / "tokenizer.json") -> Tokenizer:
    return Tokenizer.from_file(str(path))


class Harmony:
    """Control-token ids of the Beam tokenizer and the analysis-content mask of an episode."""

    def __init__(self, tok: Tokenizer):
        self.tok = tok
        ids = {name: tok.token_to_id(name) for name in
               ("<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|call|>", "<|return|>")}
        if any(v is None for v in ids.values()):
            raise ValueError("the tokenizer lacks Harmony control tokens")
        self.start, self.end, self.message, self.channel = (ids[n] for n in
                                                            ("<|start|>", "<|end|>", "<|message|>", "<|channel|>"))
        self.stops = np.array([ids["<|end|>"], ids["<|call|>"], ids["<|return|>"], ids["<|start|>"]])

    def decode(self, ids: Iterable[int]) -> str:
        return self.tok.decode([int(i) for i in ids], skip_special_tokens=False)

    def analysis_mask(self, token_ids: np.ndarray) -> np.ndarray:
        """True on the content tokens of every message with role assistant and channel header exactly analysis."""
        ids = np.asarray(token_ids)
        mask = np.zeros(len(ids), bool)
        starts = np.flatnonzero(ids == self.start)
        channels = np.flatnonzero(ids == self.channel)
        messages = np.flatnonzero(ids == self.message)
        stops = np.flatnonzero(np.isin(ids, self.stops))

        def after(positions: np.ndarray, x: int) -> int:
            j = np.searchsorted(positions, x, side="right")
            return int(positions[j]) if j < len(positions) else len(ids)

        for i, p in enumerate(starts):
            nxt = starts[i + 1] if i + 1 < len(starts) else len(ids)
            c = after(channels, p)
            m = after(messages, c)
            if not p < c < m < nxt:
                continue
            if self.decode(ids[p + 1:c]).strip() != "assistant" or self.decode(ids[c + 1:m]) != "analysis":
                continue
            mask[m + 1:after(stops, m)] = True
        return mask


def runs(positions: np.ndarray) -> list[np.ndarray]:
    """Maximal runs of consecutive positions."""
    positions = np.asarray(positions)
    if len(positions) == 0:
        return []
    cuts = np.flatnonzero(np.diff(positions) != 1) + 1
    return np.split(positions, cuts)


def window_text(harmony: Harmony, token_ids: np.ndarray, positions: np.ndarray) -> str:
    """The window's tokens decoded with special tokens kept, contiguous runs joined by a [...] line."""
    return GAP.join(harmony.decode(token_ids[r]) for r in runs(positions))


def analysis_text(harmony: Harmony, token_ids: np.ndarray, positions: np.ndarray,
                  mask: np.ndarray | None = None) -> str:
    """Only the analysis-message content inside the window, runs joined by a [...] line."""
    mask = harmony.analysis_mask(token_ids) if mask is None else mask
    return window_text(harmony, token_ids, np.asarray(positions)[mask[positions]])


def check_kinds(kinds: Sequence[str]) -> list[str]:
    bad = [k for k in kinds if k not in KINDS]
    if bad:
        raise ValueError(f"unknown kinds {bad}; choose from {KINDS}")
    return list(kinds)


def compute_shard(shard_dir: str | Path, anchors: dict[str, dict], out_dir: str | Path, kinds: Sequence[str] = KINDS,
                  rendered: dict[str, tuple[np.ndarray, np.ndarray]] | None = None, tok: Tokenizer | None = None,
                  log: Callable[[str], None] = print) -> Path:
    """Features of every window of every episode in one shard; skipped when a finished file covers `kinds`."""
    kinds = check_kinds(kinds)
    shard = Shard.open(shard_dir)
    k = shard.number
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"shard_{k}.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if set(kinds) <= set(old["kinds"]) and set(WINDOWS) <= set(old["windows"]):
            log(f"features shard {k}: done, skipping")
            return meta_path
    for name in ALWAYS + tuple(a for kind in kinds for a in NEEDS[kind]):
        if not shard.has(name):
            raise FileNotFoundError(f"{shard.path} lacks {name}.npy; exclude its kind with --kinds")
    token_ids = np.asarray(shard.array("token_ids"))
    gen_mask = np.asarray(shard.array("gen_mask")).astype(bool)
    route_ids = shard.array("route_ids") if "routing" in kinds else None
    tap_hidden = shard.array("tap_hidden") if "hidden" in kinds else None
    windows = np.asarray(shard.array("windows")) if "hidden" in kinds else None
    bf16 = shard.index.get("hidden_dtype", "bfloat16") == "bfloat16"
    harmony = Harmony(tok or tokenizer())  # control-token ids for first_finish; texts only with the text kind
    header_stops = (harmony.channel, harmony.message)
    prefinish = []  # episodes whose h3_prefinish window differs from h3_scall

    acc: dict[str, dict] = {w: {"keys": [], "ntok": [], "routing": [], "hidden": [], "hidden_ok": [], "bag": []}
                            for w in WINDOWS}
    texts = []
    for ep in shard.index["episodes"]:
        key = ep["key"]
        if key not in anchors:
            raise KeyError(f"{key} of shard {k} is not in anchors.csv")
        a = anchors[key]
        for name in INDEX_ANCHORS:
            if ep.get(name) != a[name]:
                raise ValueError(f"{key}: {name} is {ep.get(name)} in index.json, {a[name]} in anchors.csv")
        start, length = int(ep["start"]), int(ep["length"])
        if a["n_tokens"] is not None and length != a["n_tokens"]:
            raise ValueError(f"{key}: {length} tokens in the shard, {a['n_tokens']} in anchors.csv")
        ids = token_ids[start:start + length]
        gen = gen_mask[start:start + length]
        if rendered is not None:
            r_ids, r_gen = rendered[key]
            if not (np.array_equal(ids, r_ids) and np.array_equal(gen, r_gen)):
                raise ValueError(f"{key}: shard tokens differ from the rendered parquet")
        mask = harmony.analysis_mask(ids) if "text" in kinds else None
        if a["s_call"] is not None:
            a = {**a, "first_finish": first_finish(ids, a["s_call"], harmony.start, header_stops)}
            if a["first_finish"] != a["s_call"]:
                prefinish.append(key)
        for w in WINDOWS:
            pos = window_positions(gen, w, a)
            if pos is None:
                continue
            t = start + pos
            slot = acc[w]
            slot["keys"].append(key)
            slot["ntok"].append(len(pos))
            slot["bag"].append(ids[pos].astype(np.int32))
            if route_ids is not None:
                slot["routing"].append(routing_features(route_ids, t))
            if tap_hidden is not None:
                # Only f_any may miss rows; every other window was captured whole.
                if w == "f_any" and not covered(windows, t):
                    slot["hidden"].append(np.full(tap_hidden.shape[0] * tap_hidden.shape[2], np.nan))
                    slot["hidden_ok"].append(False)
                else:
                    slot["hidden"].append(hidden_features(tap_hidden, windows, t, bf16))
                    slot["hidden_ok"].append(True)
            if "text" in kinds:
                texts.append({"episode_id": key, "window": w, "text": window_text(harmony, ids, pos),
                              "analysis_text": analysis_text(harmony, ids, pos, mask)})

    arrays = {}
    summary = {}
    for w, slot in acc.items():
        n = len(slot["keys"])
        summary[w] = n
        arrays[f"ntok__{w}"] = np.array(slot["ntok"], dtype=np.int64)
        arrays[f"bag_ids__{w}"] = np.concatenate(slot["bag"]) if n else np.zeros(0, np.int32)
        arrays[f"bag_off__{w}"] = np.concatenate([[0], np.cumsum(slot["ntok"])]).astype(np.int64)
        if route_ids is not None:
            arrays[f"routing__{w}"] = (np.stack(slot["routing"]) if n else
                                       np.zeros((0, route_ids.shape[0] * 256)))
        if tap_hidden is not None:
            arrays[f"hidden__{w}"] = (np.stack(slot["hidden"]) if n else
                                      np.zeros((0, tap_hidden.shape[0] * tap_hidden.shape[2])))
            arrays[f"hidden_ok__{w}"] = np.array(slot["hidden_ok"], dtype=bool)
    tmp = out_dir / f"shard_{k}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, out_dir / f"shard_{k}.npz")
    if "text" in kinds:
        tmp = out_dir / f"texts_{k}.jsonl.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for row in texts:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, out_dir / f"texts_{k}.jsonl")
    coverage = {w: [int(sum(acc[w]["hidden_ok"])), len(acc[w]["keys"])] for w in WINDOWS} if tap_hidden is not None else None
    meta = {"shard": k, "path": str(shard.path), "kinds": kinds, "keys": {w: acc[w]["keys"] for w in WINDOWS},
            "windows": summary, "hidden_coverage": coverage, "capture": shard.fingerprint(),
            "rendered_checked": rendered is not None,
            "h3_prefinish": {"differs_from_h3_scall": len(prefinish), "of": summary["h3_prefinish"],
                             "keys": prefinish}}
    tmp = out_dir / f"shard_{k}.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    os.replace(tmp, meta_path)
    log(f"features shard {k}: {len(shard.index['episodes'])} episodes, windows {summary}; h3_prefinish differs "
        f"from h3_scall in {len(prefinish)}")
    return meta_path


@dataclass
class WindowFeatures:
    keys: list[str] = field(default_factory=list)
    ntok: list[np.ndarray] = field(default_factory=list)
    routing: list[np.ndarray] = field(default_factory=list)
    hidden: list[np.ndarray] = field(default_factory=list)
    hidden_ok: list[np.ndarray] = field(default_factory=list)
    bag: list[np.ndarray] = field(default_factory=list)  # one id array per window
    pos: dict[str, int] = field(default_factory=dict)

    def row(self, keys: Sequence[str]) -> np.ndarray:
        missing = [k for k in keys if k not in self.pos]
        if missing:
            raise KeyError(f"no window features for {len(missing)} episodes, e.g. {missing[0]}")
        return np.array([self.pos[k] for k in keys], dtype=np.int64)


@dataclass
class Features:
    """Every finished shard's features, merged per window."""

    windows: dict[str, WindowFeatures]
    kinds: set[str]
    shards: list[dict]

    def has(self, kind: str) -> bool:
        return kind in self.kinds

    def matrix(self, kind: str, window: str, keys: Sequence[str], vocab: np.ndarray | None = None) -> np.ndarray:
        """Feature rows for `keys` in order, float64."""
        if kind not in self.kinds:
            raise ValueError(f"features hold no {kind}; rerun the features step with it")
        wf = self.windows[window]
        rows = wf.row(keys)
        if kind == "routing":
            return np.asarray(wf.routing[rows], dtype=np.float64)
        if kind == "hidden":
            if not wf.hidden_ok[rows].all():
                raise ValueError(f"{int((~wf.hidden_ok[rows]).sum())} {window} windows lack hidden rows")
            return np.asarray(wf.hidden[rows], dtype=np.float64)
        if kind == "bag":
            if vocab is None:
                raise ValueError("a bag matrix needs a vocabulary")
            return bag_matrix([wf.bag[r] for r in rows], wf.ntok[rows], vocab)
        raise ValueError(f"no matrix for {kind}")

    def vocabulary(self, window: str, keys: Sequence[str]) -> np.ndarray:
        wf = self.windows[window]
        return np.unique(np.concatenate([wf.bag[r] for r in wf.row(keys)])).astype(np.int64)

    def hidden_ok(self, window: str, keys: Sequence[str]) -> np.ndarray:
        wf = self.windows[window]
        return wf.hidden_ok[wf.row(keys)] if self.has("hidden") else np.zeros(len(keys), bool)


def bag_matrix(ids: Sequence[np.ndarray], ntok: np.ndarray, vocab: np.ndarray) -> np.ndarray:
    """Token frequency (count over window length) over `vocab`; ids outside it count toward the length only."""
    X = np.zeros((len(ids), len(vocab)))
    for i, w in enumerate(ids):
        j = np.searchsorted(vocab, w)
        inside = j < len(vocab)
        inside[inside] = vocab[j[inside]] == w[inside]
        X[i] = np.bincount(j[inside], minlength=len(vocab)) / ntok[i]
    return X


def load_features(feature_dir: str | Path) -> Features:
    """Merge every finished features/shard_K; kinds are those every shard holds."""
    feature_dir = Path(feature_dir)
    metas = sorted(feature_dir.glob("shard_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    if not metas:
        raise FileNotFoundError(f"no features under {feature_dir}; run the features step")
    windows = {w: WindowFeatures() for w in WINDOWS}
    kinds = set(KINDS)
    shards = []
    for path in metas:
        meta = json.loads(path.read_text(encoding="utf-8"))
        kinds &= set(meta["kinds"])
        shards.append(meta)
        z = np.load(path.with_suffix(".npz"))
        for w in WINDOWS:
            keys = meta["keys"][w]
            wf = windows[w]
            wf.keys += keys
            off = z[f"bag_off__{w}"]
            ids = z[f"bag_ids__{w}"]
            wf.bag += [ids[a:b] for a, b in zip(off[:-1], off[1:])]
            wf.ntok.append(z[f"ntok__{w}"])
            if f"routing__{w}" in z:
                wf.routing.append(z[f"routing__{w}"])
            if f"hidden__{w}" in z:
                wf.hidden.append(z[f"hidden__{w}"])
                wf.hidden_ok.append(z[f"hidden_ok__{w}"])
    for w, wf in windows.items():
        wf.pos = {k: i for i, k in enumerate(wf.keys)}
        if len(wf.pos) != len(wf.keys):
            raise ValueError(f"window {w}: an episode appears in two shards")
        wf.ntok = np.concatenate(wf.ntok)
        wf.routing = np.concatenate(wf.routing) if "routing" in kinds else None
        wf.hidden = np.concatenate(wf.hidden) if "hidden" in kinds else None
        wf.hidden_ok = np.concatenate(wf.hidden_ok) if "hidden" in kinds else None
    return Features(windows, kinds, shards)


def load_texts(feature_dir: str | Path) -> dict[tuple[str, str], dict]:
    """(episode_id, window) -> {"text", "analysis_text"} from every texts_K.jsonl."""
    out = {}
    paths = sorted(Path(feature_dir).glob("texts_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no texts under {feature_dir}; run the features step with the text kind")
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                out[(row["episode_id"], row["window"])] = row
    return out
