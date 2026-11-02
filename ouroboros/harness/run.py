"""Capture shards of rendered episodes, resumable per shard.

    python -m ouroboros.harness.run --weights <dir or gs://...> --rendered a.parquet [b.parquet ...] \\
        --shard k [k ...] --out DIR [--sync gs://bucket/prefix] [--patch file.safetensors] [--prefix-dir DIR]

Shard k lands in DIR/shard_k (k zero-padded to 3 digits). A finished shard holds index.json and is skipped on
the next call; an unfinished one is redone from the start. DIR/shards.json pins the episode-to-shard split, and
each finished shard records the settings that shape its contents (weights, attention kernels, dtype, patch,
truncation, status word, candidates, taps, windows), so a rerun with other settings stops instead of mixing
captures.

A spot preemption wipes the local NVMe. --sync copies each finished shard to a gs:// prefix (or a directory),
index.json last, and a shard whose index.json reached it counts as done on the next call. Stage B passes can
add --drop-local to free the disk once a shard is copied, and --no-windows with an empty --taps to skip the
window scores and tap hidden states they never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import BeamConfig

from ouroboros.harness import shards
from ouroboros.harness.capture import INDEX, Capture, CapturePlan, is_complete, load
from ouroboros.harness.patch import PREFIX_MIN_LAYER, PrefixCache, load_patch, run_patched
from ouroboros.harness.shards import Episode
from ouroboros.harness.stream import Layout, Streamer, attention_backend
from ouroboros.harness.weights import open_source

VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "beam"
TAP_LAYERS = (4, 16, 28, 40, 46)
STATUS_WORDS = ("DONE", " DONE", "BLOCKED", " BLOCKED")
DTYPE = torch.bfloat16


def tokenizer(path: str | Path = VENDOR / "tokenizer.json") -> Tokenizer:
    return Tokenizer.from_file(str(path))


def status_candidates(tok: Tokenizer | None = None) -> list[int]:
    """First token of each status word, with and without a leading space.

    BLOCKED is two tokens (BLOCK, ED), so these score its first token only; --slot-word BLOCKED adds the rest.
    """
    tok = tok or tokenizer()
    ids = [tok.encode(w, add_special_tokens=False).ids[0] for w in STATUS_WORDS]
    return list(dict.fromkeys(ids))


def slot_word_ids(tok: Tokenizer, word: str, slot_token: int) -> list[int]:
    """`word` as it would stand at a status slot, with a leading space when the token Beam wrote there has one."""
    space = " " if tok.decode([slot_token]).startswith(" ") else ""
    return tok.encode(space + word, add_special_tokens=False).ids


def new_capture(streamer: Streamer, layout: Layout, plan: CapturePlan, out_dir: Path) -> Capture:
    c = streamer.config
    if any(not 0 <= t < c.num_hidden_layers for t in plan.tap_layers):
        raise ValueError(f"tap layers {plan.tap_layers} outside the model's {c.num_hidden_layers} layers")
    return Capture(out_dir, layout.n_tokens, plan, n_moe=len(streamer.moe_layers), n_experts=c.num_experts,
                   experts_per_token=c.num_experts_per_tok, hidden_size=c.hidden_size, hidden_dtype=streamer.dtype)


def finish(streamer: Streamer, capture: Capture, hidden: torch.Tensor, episodes: Sequence[Episode],
           layout: Layout, meta: dict) -> Path:
    """Score the target tokens with lm_head, then write the shard.

    The shard also keeps every token id and gen_mask, so window rows can be narrowed to the tokens Beam wrote.
    """
    plan = capture.plan
    head = streamer.head(hidden, layout.token_ids, plan.targets, plan.candidates, plan.top_k)
    gen = np.concatenate([e.gen_mask for e in episodes]).astype(bool)
    index = {
        **meta,
        "n_tokens": layout.n_tokens,
        "fingerprint": layout.fingerprint(),
        "moe_layers": streamer.moe_layers,
        "attn_implementation": streamer.config._attn_implementation,
        "episodes": shards.episode_index(episodes, layout, plan),
    }
    per_token = {"token_ids": layout.token_ids.astype(np.int32), "gen_mask": gen, "target_gen": gen[plan.targets]}
    return capture.finalize({**head, **per_token}, index)


def run_shard(streamer: Streamer, episodes: Sequence[Episode], out_dir: str | Path, *, candidates: Sequence[int],
              tap_layers: Sequence[int] = TAP_LAYERS, top_k: int = 16, with_windows: bool = True,
              delta: torch.Tensor | None = None, prefix_dir: str | Path | None = None,
              prefix_min_layer: int = PREFIX_MIN_LAYER, meta: dict | None = None) -> Path:
    """Forward one shard through every layer with capture; with prefix_dir, reuse a saved stock prefix."""
    meta = meta or {}
    layout, plan = shards.build(episodes, candidates, tap_layers, top_k, with_windows)
    capture = new_capture(streamer, layout, plan, Path(out_dir))
    if prefix_dir is None:
        hidden = streamer.embed(layout)
        streamer.run_layers(hidden, layout, delta=delta, capture=capture)
        first = 0
    else:
        # A prefix is reusable only for the same tokens, weights, attention kernels and dtype.
        identity = {"layout": layout.fingerprint(), "weights": meta.get("weights"),
                    "attn_implementation": streamer.config._attn_implementation}
        cache = PrefixCache(prefix_dir, identity, streamer.dtype, prefix_min_layer)
        hidden, first = run_patched(streamer, layout, delta, cache, capture)
    return finish(streamer, capture, hidden, episodes, layout, {**meta, "first_layer": first})


class Mirror:
    """Finished shards copied to a gs:// prefix through the gcloud CLI, or to a local directory."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.gcloud = None
        if self.url.startswith("gs://"):
            self.gcloud = shutil.which("gcloud")
            if self.gcloud is None:
                raise FileNotFoundError("--sync to gs:// needs the gcloud CLI")

    def index(self, name: str) -> dict | None:
        """The index of a mirrored shard, or None when the shard has not arrived whole."""
        if self.gcloud is None:
            path = Path(self.url) / name / INDEX
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        url = f"{self.url}/{name}/{INDEX}"
        proc = subprocess.run([self.gcloud, "storage", "cat", url], capture_output=True, text=True)
        if proc.returncode == 0:
            return json.loads(proc.stdout)
        if "matched no objects" in proc.stderr:
            return None
        raise RuntimeError(f"gcloud storage cat {url}: {proc.stderr.strip()}")

    def put(self, shard: Path) -> None:
        """Copy a finished shard with index.json last, so a copy cut short never looks finished."""
        files = sorted(p for p in shard.iterdir() if p.name != INDEX)
        for batch in (files, [shard / INDEX]):
            if self.gcloud is None:
                dest = Path(self.url) / shard.name
                dest.mkdir(parents=True, exist_ok=True)
                for p in batch:
                    shutil.copyfile(p, dest / p.name)
            else:
                subprocess.run([self.gcloud, "storage", "cp", *map(str, batch), f"{self.url}/{shard.name}/"],
                               check=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weights_location(location: str) -> str:
    """The checkpoint as recorded in each shard: the gs:// URL (it names the revision) or the absolute directory."""
    return location.rstrip("/") if location.startswith("gs://") else str(Path(location).resolve())


def settings(index: dict) -> dict:
    """The parts of a shard's index that shape its contents; a finished shard is reused only when they match."""
    return {
        "weights": index.get("weights"),
        "attn_implementation": index["attn_implementation"],
        "hidden_dtype": index["hidden_dtype"],
        "truncate": index["truncate"],
        "margin": index["margin"] if index["truncate"] else None,
        "slot_word": index.get("slot_word"),
        "candidates": list(index["candidates"]),
        "patch_sha256": (index.get("patch") or {}).get("sha256"),
        "tap_layers": list(index["tap_layers"]),
        "capture_windows": index.get("capture_windows"),
        "top_k_logprobs": index["top_k_logprobs"],
    }


def add_source_options(p: argparse.ArgumentParser) -> None:
    """Weight-transport tuning, as `python -m ouroboros.harness.weights` benches it."""
    p.add_argument("--weight-workers", type=int, help="parallel weight reads (default: the source's own)")
    p.add_argument("--chunk-mb", type=int, help="megabytes per weight read (default: the source's own)")


def source_options(args: argparse.Namespace) -> dict:
    """open_source keyword arguments for the transport flags that were given."""
    options = {"workers": args.weight_workers, "chunk_bytes": args.chunk_mb and args.chunk_mb << 20}
    return {k: v for k, v in options.items() if v}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="checkpoint directory or gs://bucket/prefix")
    p.add_argument("--rendered", nargs="+", required=True, help="P1 parquet files")
    p.add_argument("--shard", nargs="+", default=["all"], help="shard indices, or 'all'")
    p.add_argument("--out", required=True, help="local directory for the shards being written")
    p.add_argument("--sync", help="copy each finished shard to this gs://bucket/prefix or directory")
    p.add_argument("--drop-local", action="store_true", help="delete a shard's local copy once --sync holds it")
    p.add_argument("--patch", help="safetensors file holding a [moe_layers, experts] FP32 'delta'")
    p.add_argument("--prefix-dir", help="cache of stock hidden states at the input of the first patched layer")
    p.add_argument("--prefix-min-layer", type=int, default=PREFIX_MIN_LAYER,
                   help="save a prefix only when the first patched layer is at least this deep")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--config", default=str(VENDOR), help="directory holding Beam's config.json")
    p.add_argument("--shard-gb", type=float, default=40.0, help="host memory for one shard's hidden states")
    p.add_argument("--truncate", action="store_true", help="cut each episode after its last needed position")
    p.add_argument("--margin", type=int, default=64, help="tokens kept after the last needed position")
    p.add_argument("--slot-word", help="score this status word in full at every status slot, e.g. BLOCKED")
    p.add_argument("--chunk-tokens", type=int, default=65536)
    p.add_argument("--attn", choices=["auto", "flash", "sdpa"], default="auto")
    p.add_argument("--taps", type=int, nargs="*", default=list(TAP_LAYERS), help="tap layers; none skips them")
    p.add_argument("--no-windows", action="store_true", help="skip the window scores and tap hidden states")
    p.add_argument("--candidates", type=int, nargs="+", help="token ids scored at every target")
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--plan", action="store_true", help="write and print the shard split, then stop")
    add_source_options(p)
    args = p.parse_args(argv)
    if args.drop_local and not args.sync:
        p.error("--drop-local needs --sync")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = BeamConfig.from_pretrained(args.config)
    tok = tokenizer()
    episodes = shards.load_rendered(args.rendered)
    if args.slot_word:
        episodes = [e if e.s is None else e.with_slot_word(slot_word_ids(tok, args.slot_word, int(e.token_ids[e.s])))
                    for e in episodes]
    if args.truncate:
        episodes = [e.truncated(args.margin) for e in episodes]
    groups = shards.split([len(e.token_ids) for e in episodes], shards.budget_tokens(args.shard_gb, config.hidden_size))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split = {f"{k:03d}": [episodes[e].key for e in g] for k, g in enumerate(groups)}
    pinned = out / "shards.json"
    if pinned.exists() and json.loads(pinned.read_text(encoding="utf-8")) != split:
        sys.exit(f"{pinned} holds a different split; use the original flags or a new --out")
    pinned.write_text(json.dumps(split, indent=1), encoding="utf-8")
    if args.plan:
        for k, g in enumerate(groups):
            n = sum(len(episodes[e].token_ids) for e in g)
            print(f"shard {k}: {len(g)} episodes, {n} tokens, "
                  f"{n * config.hidden_size * shards.HIDDEN_BYTES / 1e9:.1f} GB of hidden states", flush=True)
        return 0

    candidates = args.candidates or status_candidates(tok)
    meta = {"weights": weights_location(args.weights), "rendered": [str(r) for r in args.rendered],
            "truncate": args.truncate, "margin": args.margin, "slot_word": args.slot_word, "candidates": candidates,
            "capture_windows": not args.no_windows, "patch": None}
    delta = None
    if args.patch:
        n_moe = sum(kind == "sparse" for kind in config.mlp_layer_types)
        delta = load_patch(args.patch, n_moe, config.num_experts)
        meta["patch"] = {"file": str(args.patch), "sha256": _sha256(Path(args.patch))}
    wanted_settings = settings({**meta, "attn_implementation": attention_backend(args.attn, torch.device(args.device)),
                                "hidden_dtype": str(DTYPE).removeprefix("torch."), "tap_layers": args.taps,
                                "top_k_logprobs": args.top_k})
    mirror = Mirror(args.sync) if args.sync else None

    def publish(path: Path, mirrored: bool) -> None:
        if mirror is not None and not mirrored:
            mirror.put(path)
        if args.drop_local:
            shutil.rmtree(path)

    wanted = range(len(groups)) if args.shard == ["all"] else [int(k) for k in args.shard]
    todo = []
    for k in wanted:
        if k >= len(groups):
            sys.exit(f"shard {k} does not exist; the split has {len(groups)} shards")
        name = f"shard_{k:03d}"
        local = out / name
        mirrored = mirror.index(name) if mirror is not None else None
        # Both copies must match: --drop-local must never delete a local shard the mirror does not hold.
        found = {str(local): load(local).index} if is_complete(local) else {}
        if mirrored is not None:
            found[f"{args.sync}/{name}"] = mirrored
        if not found:
            todo.append(k)
            continue
        for where, index in found.items():
            if [e["key"] for e in index["episodes"]] != split[f"{k:03d}"]:
                sys.exit(f"{where} holds other episodes than shard {k} of this split")
            if settings(index) != wanted_settings:
                sys.exit(f"{where} was captured with {settings(index)}, not {wanted_settings}; "
                         "use a new --out and --sync")
        if is_complete(local):
            publish(local, mirrored is not None)
        print(f"shard {k}: done, skipping", flush=True)
    if not todo:
        return 0

    with open_source(args.weights, **source_options(args)) as source:
        streamer = Streamer(config, source, args.device, DTYPE, args.chunk_tokens, args.attn)
        for k in todo:
            t0 = time.perf_counter()
            group = [episodes[e] for e in groups[k]]
            prefix = None if args.prefix_dir is None else Path(args.prefix_dir) / f"shard_{k:03d}"
            path = run_shard(streamer, group, out / f"shard_{k:03d}", candidates=candidates, tap_layers=args.taps,
                             top_k=args.top_k, with_windows=not args.no_windows, delta=delta, prefix_dir=prefix,
                             prefix_min_layer=args.prefix_min_layer, meta={**meta, "shard": k})
            publish(path, False)
            n = sum(len(e.token_ids) for e in group)
            print(f"shard {k}: {len(group)} episodes, {n} tokens, {time.perf_counter() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
