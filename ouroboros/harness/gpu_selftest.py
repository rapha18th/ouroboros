"""The pre-registered harness gate, run first on the A100. Prints a pass/fail table and writes selftest.json.

(a) Every decoder layer's streamed output matches transformers' own path for that layer (one unpacked sequence,
    SDPA with masks from masking_utils) on a 2,048-token sequence, within BF16 tolerance. Both paths share the
    layer's weights and its input, so this checks packing, chunking, attention backend and routing per layer.
    The reference is packed second in its chunk, behind a lead episode, so attention that crosses an episode
    boundary or positions that fail to restart per episode change its rows.
(b) On 8 episodes, 4 per source, the teacher-forced mean NLL of Beam's sampled tokens is at most 1.0 nat per
    token, and argmax agreement with those tokens is at least 0.70.
(c) Two runs of the same shard agree on the top-8 expert set at 99.9% of token-layers or more.

    python -m ouroboros.harness.gpu_selftest --weights gs://... --rendered tripwire.parquet cvd.parquet --out DIR

Pass one parquet per source; the gate draws 4 episodes with sampled tokens from each, seed 20261028, among the
rows P1 marks billing_exact (their sampled tokens are Beam's own, not the route's re-serialization).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import BeamConfig
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.beam.modeling_beam import BeamDecoderLayer, compute_attention_scaling

from ouroboros.harness import shards
from ouroboros.harness.capture import load, topk_set_agreement
from ouroboros.harness.run import VENDOR, add_source_options, finish, new_capture, run_shard, source_options
from ouroboros.harness.shards import Episode
from ouroboros.harness.stream import Streamer
from ouroboros.harness.weights import open_source

REL_TOL = 1e-2  # relative L2 error of a layer's output, about 1.3 BF16 ulps
NLL_MAX = 1.0
ARGMAX_MIN = 0.70
TOPK_MIN = 0.999
SEED = 20261028


def pick(groups: Sequence[Sequence[Episode]], per_source: int, seed: int = SEED) -> list[Episode]:
    """`per_source` episodes holding sampled tokens from each source, drawn with a fixed seed."""
    rng = np.random.default_rng(seed)
    out = []
    for group in groups:
        pool = [e for e in group if e.gen_mask[1:].any()]
        if len(pool) < per_source:
            raise ValueError(f"a source has {len(pool)} episodes with sampled tokens, need {per_source}")
        out += [pool[k] for k in sorted(rng.choice(len(pool), per_source, replace=False))]
    return out


def gate_pool(path: str | Path) -> list[Episode]:
    """A source's episodes, narrowed to P1's billing_exact rows when the parquet has that column."""
    episodes = shards.load_rendered([path])
    if "billing_exact" not in pq.read_schema(path).names:
        return episodes
    table = pq.read_table(path, columns=["episode_id", "billing_exact"])
    exact = {k for k, ok in zip(table["episode_id"].to_pylist(), table["billing_exact"].to_pylist()) if ok}
    return [e for e in episodes if e.key in exact]


@torch.no_grad()
def reference_layer(streamer: Streamer, layer: BeamDecoderLayer, i: int, h: torch.Tensor) -> torch.Tensor:
    """transformers' path for decoder layer i on one unpacked sequence h [n, H], as BeamModel.forward runs it."""
    config = streamer.config
    backend = config._attn_implementation
    config._attn_implementation = "sdpa"
    try:
        x = h.unsqueeze(0)
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        layer_type = config.layer_types[i]
        make_mask = create_causal_mask if layer_type == "full_attention" else create_sliding_window_causal_mask
        mask = make_mask(config=config, inputs_embeds=x, attention_mask=None, past_key_values=None, position_ids=pos)
        if config.rope_parameters[layer_type] is None:
            pe, scaling = None, compute_attention_scaling(pos)
        else:
            pe, scaling = streamer.rotary(x, pos, layer_type), None
        return layer(x, position_embeddings=pe, attention_mask=mask, position_ids=pos, attention_scaling=scaling)[0]
    finally:
        config._attn_implementation = backend


def selftest(config: BeamConfig, source, gate: Sequence[Episode], out_dir: str | Path, *, device: str = "cuda",
             dtype: torch.dtype = torch.bfloat16, ref_len: int = 2048, chunk_tokens: int = 65536,
             attn: str = "auto", rel_tol: float = REL_TOL) -> dict:
    """Run checks (a), (b) and (c); return the results (also written to out_dir/selftest.json)."""
    out_dir = Path(out_dir)
    base = next((e for e in gate if len(e.token_ids) >= ref_len), None)
    if base is None:
        raise ValueError(f"no gate episode has {ref_len} tokens")
    # The lead is the tail of another gate episode, longer than half the reference so it spans a sliding window.
    lead_len = ref_len // 2 + 1
    other = next((e for e in gate if e is not base and len(e.token_ids) >= lead_len), base)
    lead = Episode("lead", other.token_ids[-lead_len:], np.zeros(lead_len, bool))
    reference = Episode("reference", base.token_ids[:ref_len], np.zeros(ref_len, bool))
    episodes = [lead, reference, *gate]
    streamer = Streamer(config, source, device, dtype, chunk_tokens, attn)
    layout, plan = shards.build(episodes, candidates=[], tap_layers=(), top_k=1, with_windows=False)
    if layout.chunks(chunk_tokens)[0][1] < 2:
        raise ValueError(f"chunk_tokens {chunk_tokens} must hold the lead and the reference ({lead_len + ref_len})")
    rows = slice(lead_len, lead_len + ref_len)
    meta = {"selftest": True}

    # Run 1: stream every layer and check it against the reference path on the reference rows.
    capture = new_capture(streamer, layout, plan, out_dir / "run1")
    hidden = streamer.embed(layout)
    layers = []
    clock = time.perf_counter()
    for i, tensors in streamer.weights(0, config.num_hidden_layers):
        t0 = time.perf_counter()
        layer = streamer.build(i, tensors)
        h_in = hidden[rows].to(streamer.device, copy=True)  # apply() updates hidden in place
        t1 = time.perf_counter()
        streamer.apply(layer, i, hidden, layout, capture=capture)
        if streamer.device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        want = reference_layer(streamer, layer, i, h_in).float()
        got = hidden[rows].to(streamer.device).float()
        # The residual dominates a layer's output; the update (output minus input) shows a layer's own error.
        layers.append({"layer": i, "rel_err": float((got - want).norm() / want.norm()),
                       "update_rel_err": float((got - want).norm() / (want - h_in.float()).norm()),
                       "max_abs_err": float((got - want).abs().max()),
                       "load_s": round(t0 - clock, 3), "build_s": round(t1 - t0, 3), "run_s": round(t2 - t1, 3)})
        del layer, tensors, h_in, want, got
        streamer.free()
        clock = time.perf_counter()
    run1 = load(finish(streamer, capture, hidden, episodes, layout, meta))
    del hidden

    # Run 2: the same shard again, end to end.
    run2 = load(run_shard(streamer, episodes, out_dir / "run2", candidates=[], tap_layers=(), top_k=1,
                          with_windows=False, meta=meta))

    gen = run1.array("target_gen")
    targets = run1.array("targets")
    nll = run1.array("nll")
    agree = run1.array("top_ids")[:, 0] == layout.token_ids[targets]
    per_episode = []
    for e in run1.index["episodes"][2:]:
        rows = slice(*e["target_rows"])
        g = gen[rows]
        per_episode.append({"key": e["key"], "sampled_tokens": int(g.sum()),
                            "mean_nll": float(nll[rows][g].mean()), "argmax_agreement": float(agree[rows][g].mean())})
    worst = max(layers, key=lambda r: r["rel_err"])
    checks = [
        ("(a) layer output vs transformers, max rel L2", worst["rel_err"], "<=", rel_tol),
        ("(b) mean NLL of sampled tokens (nat)", float(nll[gen].mean()), "<=", NLL_MAX),
        ("(b) argmax agreement with sampled tokens", float(agree[gen].mean()), ">=", ARGMAX_MIN),
        ("(c) top-8 set agreement, two runs", topk_set_agreement(run1.array("route_ids"), run2.array("route_ids")),
         ">=", TOPK_MIN),
    ]
    results = {
        "checks": [{"name": n, "value": v, "threshold": f"{op} {t}", "pass": bool(v <= t if op == "<=" else v >= t)}
                   for n, v, op, t in checks],
        "worst_layer": worst["layer"],
        "max_update_rel_err": max(r["update_rel_err"] for r in layers),
        "device": str(streamer.device),
        "dtype": str(dtype).removeprefix("torch."),
        "attn_implementation": streamer.config._attn_implementation,
        "ref_len": ref_len,
        "ref_offset": lead_len,
        "sampled_tokens": int(gen.sum()),
        "episodes": per_episode,
        "layers": layers,
    }
    results["pass"] = all(c["pass"] for c in results["checks"])
    (out_dir / "selftest.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return results


def table(results: dict) -> str:
    rows = [("check", "value", "threshold", "result")]
    rows += [(c["name"], f"{c['value']:.6g}", c["threshold"], "PASS" if c["pass"] else "FAIL")
             for c in results["checks"]]
    widths = [max(len(r[k]) for r in rows) for k in range(4)]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(r, widths)) for r in rows]
    lines.append(f"(info) layer update vs transformers, max rel L2: {results['max_update_rel_err']:.6g}")
    lines.append(f"harness gate: {'PASS' if results['pass'] else 'FAIL'} "
                 f"({results['attn_implementation']}, {results['device']}, {results['dtype']})")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="checkpoint directory or gs://bucket/prefix")
    p.add_argument("--rendered", nargs="+", required=True, help="one P1 parquet per source")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--config", default=str(VENDOR))
    p.add_argument("--per-source", type=int, default=4)
    p.add_argument("--ref-len", type=int, default=2048)
    p.add_argument("--chunk-tokens", type=int, default=65536)
    p.add_argument("--attn", choices=["auto", "flash", "sdpa"], default="auto")
    add_source_options(p)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = BeamConfig.from_pretrained(args.config)
    gate = pick([gate_pool(r) for r in args.rendered], args.per_source)
    with open_source(args.weights, **source_options(args)) as source:
        results = selftest(config, source, gate, args.out, device=args.device, ref_len=args.ref_len,
                           chunk_tokens=args.chunk_tokens, attn=args.attn)
    print(table(results), flush=True)
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
