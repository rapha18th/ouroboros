"""Score the stock model and many router-bias patches over Stage B sets, in one weight stream.

    python -m ouroboros.harness.score --weights <dir or gs://...> --set NAME=PARQUET[:EPISODES] [--set ...] \\
        --out DIR [--bank FILE --variants FILE] [--twin] [--reorder NEW=SRC[:SEED]] [--cap NAME=TOKENS] \\
        [--slot-word WORD] [--truncate] [--margin 0] [--routes none|ids] [--sync gs://bucket/prefix] [--plan]

Each --set is one layout group: a rendered parquet and, optionally, a set file listing its episodes in layout order
with their target spans. Each group's episodes take --slot-word, then --truncate, then --cap. --reorder adds a group
holding another group's episodes in a seeded order, whose stock is compared with the source group's stock: the
packing floor. Stock runs in every group first, --variants lists the bank patches each group scores, and --twin adds
a zero patch forked at decoder layer 3 after them.

DIR receives one directory per group (episodes.json, the token arrays, the targets, and variant_<patch>/ with each
variant's arrays), then pass.json last. A rerun skips when the local or mirrored pass.json was written with equal
settings, and refuses when they differ. A directory without pass.json is cleared and redone. --sync copies the
finished pass to a gs:// prefix or a directory, pass.json last.

Stage B is exploratory: its registered entry rule is unmet, and every file this writes says so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import BeamConfig

from ouroboros import stageb
from ouroboros.harness import bank, multi, run, sets, shards
from ouroboros.harness.multi import LayoutGroup, PassResult, Variant
from ouroboros.harness.stream import Streamer, attention_backend, moe_layers
from ouroboros.harness.weights import open_source  # module-level name, so tests can monkeypatch it

PASS = "pass.json"
PASS_FORMAT = "ouroboros-pass/1"
EPISODES_FORMAT = "ouroboros-pass-episodes/1"
VARIANTS_FORMAT = "ouroboros-variants/1"
DTYPE = run.DTYPE
SET_SPEC = re.compile(r"^(?P<name>[a-z0-9_]+)=(?P<parquet>.+?\.parquet)(?::(?P<episodes>.+\.json))?$")
CAP_SPEC = re.compile(r"^(?P<name>[a-z0-9_]+)=(?P<tokens>\d+)$")
REORDER_SPEC = re.compile(r"^(?P<new>[a-z0-9_]+)=(?P<src>[a-z0-9_]+)(?::(?P<seed>\d+))?$")
PASS_KEYS = ("exploratory", "reason", "format", "argv", "vouched_sha256", "vouched_ok", "code_sha256", "weights",
             "attn_implementation", "hidden_dtype", "device", "chunk_tokens", "top_k", "head_batch", "candidates",
             "slot_word", "truncate", "margin", "caps", "bias_sha256", "bank", "variants_file", "groups", "variants",
             "plan", "started_utc", "finished_utc", "seconds", "settings")
SETTING_KEYS = ("weights", "attn_implementation", "hidden_dtype", "chunk_tokens", "top_k", "head_batch", "candidates",
                "slot_word", "truncate", "margin", "caps", "bias_sha256", "vouched_sha256")
CODE_ROOT = Path(__file__).resolve().parents[1]  # the ouroboros package
log = logging.getLogger(__name__)


def parse_set(spec: str) -> tuple[str, Path, Path | None]:
    """NAME=PARQUET[:EPISODES]; paths may start with a drive letter."""
    m = SET_SPEC.match(spec)
    if m is None:
        raise argparse.ArgumentTypeError(f"--set takes NAME=PARQUET[:EPISODES.json], got {spec!r}")
    return m["name"], Path(m["parquet"]), Path(m["episodes"]) if m["episodes"] else None


def parse_cap(spec: str) -> tuple[str, int]:
    """NAME=TOKENS."""
    m = CAP_SPEC.match(spec)
    if m is None or int(m["tokens"]) < 1:
        raise argparse.ArgumentTypeError(f"--cap takes NAME=TOKENS, got {spec!r}")
    return m["name"], int(m["tokens"])


def parse_reorder(spec: str) -> tuple[str, str, int]:
    """NEW=SRC[:SEED]; the seed defaults to stageb.SEED."""
    m = REORDER_SPEC.match(spec)
    if m is None:
        raise argparse.ArgumentTypeError(f"--reorder takes NEW=SRC[:SEED], got {spec!r}")
    return m["new"], m["src"], int(m["seed"]) if m["seed"] else stageb.SEED


def build_groups(args: argparse.Namespace, config: BeamConfig, tok: Tokenizer) -> list[LayoutGroup]:
    """One group per --set in order, then one per --reorder. ValueError names the set at fault."""
    candidates = args.candidates or run.status_candidates(tok)
    caps = dict(parse_cap(c) if isinstance(c, str) else c for c in args.cap or [])
    groups: dict[str, LayoutGroup] = {}
    for spec in args.set:
        name, parquet, episodes_file = parse_set(spec) if isinstance(spec, str) else spec
        if name in groups:
            raise ValueError(f"set {name} is given twice")
        try:
            episodes, info = sets.load_set(parquet, episodes_file, name)
        except (ValueError, OSError) as e:
            raise ValueError(f"set {name}: {e}") from None
        if args.slot_word:
            episodes = [e if e.s is None else e.with_slot_word(run.slot_word_ids(tok, args.slot_word,
                                                                                 int(e.token_ids[e.s])))
                        for e in episodes]
        if args.truncate:
            episodes = [e.truncated(args.margin) for e in episodes]
        if name in caps:
            episodes = [e.capped(caps[name]) for e in episodes]
        meta = {"set": name, "parquet": info["parquet"], "parquet_sha256": info["parquet_sha256"],
                "episodes_file": info["episodes_file"], "episodes_sha256": info["episodes_sha256"],
                "reorder_seed": None, "records": {r["key"]: r for r in info["records"] or []}}
        groups[name] = LayoutGroup.build(name, episodes, candidates, args.top_k, meta=meta)
    for spec in args.reorder or []:
        new, src, seed = parse_reorder(spec) if isinstance(spec, str) else spec
        if new in groups or src not in groups:
            raise ValueError(f"set {src} must be given once and {new} must be a new name for --reorder")
        g = groups[src]
        groups[new] = LayoutGroup.build(new, sets.permute(g.episodes, seed), candidates, args.top_k, reference=src,
                                        meta={**g.meta, "reorder_seed": seed})
    unknown = set(caps) - set(groups)
    if unknown:
        raise ValueError(f"set {min(unknown)} has a --cap but no --set")
    return list(groups.values())


def build_variants(groups: Sequence[LayoutGroup], bank: Mapping[str, torch.Tensor], spec: dict | None,
                   twin: bool, moe_layers: Sequence[int], n_experts: int) -> list[Variant]:
    """Per group: stock, then the twin with --twin, then the listed patches in file order."""
    listed: dict[str, list[str]] = {g.name: [] for g in groups}
    seen: set[tuple[str, str]] = set()
    for item in (spec or {}).get("variants", []):
        name = item["patch"]
        if name in (multi.STOCK, multi.TWIN):
            raise ValueError(f"bank: {name} runs implicitly and is never listed")
        if name not in bank:
            raise ValueError(f"bank has no patch {name}")
        for group in item["groups"]:
            if group not in listed:
                raise ValueError(f"set {group} is absent from this pass; variant {name} lists it")
            if (name, group) in seen:
                raise ValueError(f"bank: variant {name} is listed twice for set {group}")
            seen.add((name, group))
            listed[group].append(name)
    out = []
    for g in groups:
        out.append(Variant.stock(g.name))
        out.extend(Variant.patch(name, g.name, bank[name], moe_layers) for name in listed[g.name])
        if twin:
            # last in its group: a patch that left its bias changed at a layer would make the twin differ from
            # stock, so the twin's bit equality also checks that every patch restored the bias
            out.append(Variant.make_twin(g.name, moe_layers, n_experts))
    return out


def settings(pass_json: dict) -> dict:
    """The parts of a pass that shape its outputs; a finished pass is reused only when they match.

    bias_sha256 here is the bank's stock-bias hash (None for a pass without a bank), since the check runs before any
    weight is read; the weights location stands for the checkpoint.
    """
    out = {k: pass_json.get(k) for k in SETTING_KEYS}
    if not out["truncate"]:
        out["margin"] = None
    out["groups"] = {name: {k: g.get(k) for k in ("parquet_sha256", "episodes_sha256", "reference", "reorder_seed")}
                     for name, g in pass_json["groups"].items()}
    out["variants"] = {key: {k: v.get(k) for k in ("sha256", "twin", "fork_layer")}
                       for key, v in pass_json["variants"].items()}
    return out


class PassMirror:
    """A finished pass copied to a gs:// prefix through the gcloud CLI, or to a local directory."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.gcloud = None
        if self.url.startswith("gs://"):
            self.gcloud = shutil.which("gcloud")
            if self.gcloud is None:
                raise FileNotFoundError("--sync to gs:// needs the gcloud CLI")

    def pass_json(self) -> dict | None:
        """The mirrored pass.json, or None when the pass has not arrived whole."""
        if self.gcloud is None:
            return _read_pass(Path(self.url) / PASS)
        url = f"{self.url}/{PASS}"
        proc = subprocess.run([self.gcloud, "storage", "cat", url], capture_output=True, text=True)
        if proc.returncode == 0:
            return json.loads(proc.stdout)
        if "matched no objects" in proc.stderr:
            return None
        raise RuntimeError(f"gcloud storage cat {url}: {proc.stderr.strip()}")

    def put(self, out: Path) -> None:
        """Copy every file but pass.json, then pass.json, so a copy cut short never looks finished."""
        out = Path(out)
        items = sorted(p for p in out.iterdir() if p.name != PASS)
        if self.gcloud is None:
            dest = Path(self.url)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / PASS).unlink(missing_ok=True)
            for p in items:
                if p.is_dir():
                    shutil.copytree(p, dest / p.name, dirs_exist_ok=True)
                else:
                    shutil.copyfile(p, dest / p.name)
            shutil.copyfile(out / PASS, dest / f"{PASS}.partial")
            os.replace(dest / f"{PASS}.partial", dest / PASS)
            return
        if items:
            subprocess.run([self.gcloud, "storage", "cp", "--recursive", *map(str, items), f"{self.url}/"],
                           check=True)
        subprocess.run([self.gcloud, "storage", "cp", str(out / PASS), f"{self.url}/{PASS}"], check=True)


def _utc(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def _write_json(path: Path, obj: dict) -> None:
    """Written to a .partial file and renamed, so a kill mid-write never leaves a torn file that looks finished."""
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_pass(path: Path) -> dict | None:
    """A pass.json, or None when it is absent or unreadable (a torn write): the pass is then redone."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        log.warning("%s is unreadable; the pass is redone", path)
        return None


def _routes(pass_json: dict) -> str:
    """The --routes value a finished pass ran with, read from its argv."""
    argv = list(pass_json.get("argv") or [])
    for i, a in enumerate(argv):
        if a == "--routes" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--routes="):
            return a.split("=", 1)[1]
    return "none"


def _save(path: Path, a: np.ndarray) -> None:
    np.save(path, a)


def code_sha256(root: Path = CODE_ROOT) -> str:
    """sha256 over sorted "relative path sha256" lines of every .py file in the package."""
    lines = sorted(f"{p.relative_to(root.parent).as_posix()} {sets.sha256_file(p)}" for p in root.rglob("*.py"))
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def _episode_records(g: LayoutGroup) -> list[dict]:
    records = g.meta.get("records") or {}
    out = []
    index = shards.episode_index(g.episodes, g.layout, g.plan)
    for ep, entry in zip(g.episodes, index):
        entry = {k: v for k, v in entry.items() if k not in ("windows", "window_rows")}
        n = len(ep.token_ids)
        spans = {name: [a, min(b, n)] for name, a, b in getattr(ep, "spans", ()) if a < n}
        record = records.get(ep.key)
        meta = None if record is None else {k: v for k, v in record.items() if k not in ("key", "spans", "cut")}
        out.append({**entry, "spans": spans, "meta": meta})
    return out


def _variant_stats(v: Variant, bias, n_moe: int) -> dict:
    if v.is_stock:
        return {"nonzero": 0, "nominal_l2": 0.0, "effective_l2": 0.0, "effective_l2_per_layer": [0.0] * n_moe}
    stats = {"nonzero": int((v.delta != 0).sum()), "nominal_l2": float(torch.linalg.vector_norm(v.delta.double()))}
    if bias is None:
        return {**stats, "effective_l2": None, "effective_l2_per_layer": None}
    return {**stats, "effective_l2": bank.effective_l2(bias, v.delta),
            "effective_l2_per_layer": bank.effective_l2_per_layer(bias, v.delta)}


def write_pass(out: Path, groups: Sequence[LayoutGroup], variants: Sequence[Variant], result: PassResult,
               heads: Mapping[str, dict], meta: dict) -> Path:
    """Write every group directory, then pass.json last.

    `meta` holds the pass.json fields known before the run. Its "bias" (the stock bias, for effective L2) and
    "started" (epoch seconds) entries are used and not written.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    bias = meta.pop("bias", None)
    started = meta.pop("started", None)
    by_name = {g.name: g for g in groups}
    n_moe = len(next(iter(result.route_changed.values()))) if result.route_changed else 0
    group_meta, variant_meta = {}, {}
    for g in groups:
        gdir = out / g.name
        gdir.mkdir(parents=True, exist_ok=True)
        gen = (np.concatenate([e.gen_mask for e in g.episodes]).astype(bool) if g.episodes else np.zeros(0, bool))
        _write_json(gdir / "episodes.json", {**stageb.LABEL, "format": EPISODES_FORMAT, "group": g.name,
                                             "set": g.meta.get("set"), "reference": g.reference,
                                             "episodes": _episode_records(g)})
        _save(gdir / "token_ids.npy", g.layout.token_ids.astype(np.int32))
        _save(gdir / "gen_mask.npy", gen)
        _save(gdir / "targets.npy", g.plan.targets.astype(np.int64))
        _save(gdir / "target_gen.npy", gen[g.plan.targets])
        _save(gdir / "candidates.npy", g.plan.candidates.astype(np.int64))
        if g.reference is not None:
            r = by_name[g.reference]
            _save(gdir / "align_targets.npy", sets.align_targets(r.episodes, r.layout, r.plan, g.episodes, g.layout,
                                                                 g.plan))
            _save(gdir / "align_tokens.npy", sets.align_tokens(r.episodes, r.layout, g.episodes, g.layout))
        for v in (v for v in variants if v.group == g.name):
            vdir = gdir / f"variant_{v.name}"
            vdir.mkdir(parents=True, exist_ok=True)
            for name, a in heads[v.key].items():
                _save(vdir / f"{name}.npy", a)
            if v.key in result.route_ids:
                _save(vdir / "route_ids.npy", result.route_ids[v.key])
            changed = [int(x) for x in result.route_changed[v.key]]
            tokens = g.layout.n_tokens
            share = sum(changed) / (tokens * len(changed)) if tokens and changed else 0.0
            per_layer = [float(x) for x in result.seconds[v.key]]
            stats = _variant_stats(v, bias, n_moe)
            record = stageb.labelled({
                "group": g.name, "patch": v.name, "key": v.key, "sha256": v.sha256, "twin": v.twin,
                "fork_layer": v.fork_layer, **stats, "route_changed": changed, "route_tokens": tokens,
                "route_change_share": share, "seconds": float(sum(per_layer)), "seconds_per_layer": per_layer,
            })
            _write_json(vdir / "variant.json", record)
            variant_meta[v.key] = {k: record[k] for k in ("patch", "group", "sha256", "twin", "fork_layer",
                                                          "effective_l2", "effective_l2_per_layer", "route_changed",
                                                          "route_change_share", "seconds")}
        group_meta[g.name] = _group_meta(g)
    finished = time.time()
    meta.update(groups=group_meta, variants=variant_meta, finished_utc=_utc(finished),
                seconds=finished - (finished if started is None else started))
    _write_json(out / PASS, {**{k: meta.pop(k) for k in PASS_KEYS if k in meta}, **meta})
    return out / PASS


def _group_meta(g: LayoutGroup) -> dict:
    m = g.meta
    return {"set": m.get("set"), "parquet": m.get("parquet"), "parquet_sha256": m.get("parquet_sha256"),
            "episodes_file": m.get("episodes_file"), "episodes_sha256": m.get("episodes_sha256"),
            "reference": g.reference, "reorder_seed": m.get("reorder_seed"), "n_episodes": len(g.episodes),
            "n_tokens": g.layout.n_tokens, "n_targets": len(g.plan.targets), "fingerprint": g.layout.fingerprint()}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="checkpoint directory or gs://bucket/prefix")
    p.add_argument("--set", action="append", required=True, type=parse_set,
                   help="NAME=PARQUET[:EPISODES.json], once per layout group")
    p.add_argument("--out", required=True, help="local directory for the pass")
    p.add_argument("--config", default=str(run.VENDOR), help="directory holding Beam's config.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--attn", choices=["auto", "flash", "sdpa"], default="auto")
    p.add_argument("--chunk-tokens", type=int, default=65536)
    p.add_argument("--bank", help="patch bank (safetensors)")
    p.add_argument("--variants", help="variants file naming the bank patches each group scores")
    p.add_argument("--twin", action="store_true", help="add a zero patch forked at decoder layer 3 to every group")
    p.add_argument("--reorder", action="append", type=parse_reorder, help="NEW=SRC[:SEED]: SRC's episodes reordered")
    p.add_argument("--cap", action="append", type=parse_cap, help="NAME=TOKENS: cut anchorless episodes of a group")
    p.add_argument("--slot-word", help="score this status word in full at every status slot, e.g. BLOCKED")
    p.add_argument("--truncate", action="store_true", help="cut each episode after its last needed position")
    p.add_argument("--margin", type=int, default=0, help="tokens kept after the last needed position")
    p.add_argument("--candidates", type=int, nargs="+", help="token ids scored at every target")
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--head-batch", type=int, default=1024)
    p.add_argument("--host-gb", type=float, default=multi.HOST_GB, help="host memory for hidden states")
    p.add_argument("--mem-available-gb", type=float, help="MemAvailable to plan against (tests); default /proc")
    p.add_argument("--routes", choices=["none", "ids"], default="none", help="ids also writes every variant's routes")
    p.add_argument("--sync", help="copy the finished pass to this gs://bucket/prefix or directory")
    p.add_argument("--plan", action="store_true", help="print the memory plan, then stop")
    run.add_source_options(p)
    return p


def _load_variants(path: str | None, bank_path: str | None, bank_sha: str | None) -> dict | None:
    if path is None:
        return None
    if bank_path is None:
        sys.exit("bank: --variants needs --bank")
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("format") != VARIANTS_FORMAT or spec.get("exploratory") is not True:
        sys.exit(f"bank: {path} is no labelled {VARIANTS_FORMAT} file")
    if spec.get("bank_sha256") != bank_sha:
        sys.exit(f"bank {bank_path} has sha256 {bank_sha}, but {path} was written for {spec.get('bank_sha256')}")
    return spec


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.time()
    config = BeamConfig.from_pretrained(args.config)
    tok = run.tokenizer()
    moe, n_experts = moe_layers(config), config.num_experts
    try:
        groups = build_groups(args, config, tok)
    except ValueError as e:
        sys.exit(str(e))
    patches, bank_head, bank_sha = {}, None, None
    if args.bank:
        try:
            patches, bank_head = bank.load_patch_bank(args.bank, len(moe), n_experts)
        except (ValueError, OSError) as e:
            sys.exit(f"bank {args.bank}: {e}")
        bank_sha = sets.sha256_file(args.bank)
    spec = _load_variants(args.variants, args.bank, bank_sha)
    try:
        variants = build_variants(groups, patches, spec, args.twin, moe, n_experts)
    except ValueError as e:
        sys.exit(str(e) if str(e).startswith(("bank", "set ")) else f"bank {args.bank}: {e}")
    dtype_bytes = torch.empty((), dtype=DTYPE).element_size()
    try:
        plan = multi.plan_pass(groups, variants, hidden_size=config.hidden_size, host_gb=args.host_gb,
                               mem_available_gb=args.mem_available_gb, dtype_bytes=dtype_bytes)
    except multi.PlanRefused as e:
        sys.exit(f"plan refused: {e}")
    for g in groups:
        n = plan["per_group"][g.name]
        print(f"group {g.name}: {len(g.episodes)} episodes, {n['n_tokens']} tokens, {len(g.plan.targets)} targets, "
              f"{n['n_variants']} patched variants, {n['hidden_gb']:.2f} GB of hidden states", flush=True)
    mem = plan["mem_available_gb"]
    print(f"plan: {plan['hidden_gb']:.2f} GB of hidden states within {args.host_gb:g} GB, plus "
          f"{plan['fixed_gb']:.1f} GB fixed; MemAvailable {'unknown' if mem is None else f'{mem:.1f} GB'}", flush=True)
    if args.plan:
        return 0

    vouched = sets.vouched_hashes()
    meta = {
        "format": PASS_FORMAT, **stageb.LABEL, "argv": argv, "vouched_sha256": vouched,
        "vouched_ok": vouched == stageb.VOUCHED_SHA256, "code_sha256": code_sha256(),
        "weights": run.weights_location(args.weights),
        "attn_implementation": attention_backend(args.attn, torch.device(args.device)),
        "hidden_dtype": str(DTYPE).removeprefix("torch."), "device": args.device, "chunk_tokens": args.chunk_tokens,
        "top_k": args.top_k, "head_batch": args.head_batch, "candidates": [int(c) for c in groups[0].plan.candidates],
        "slot_word": args.slot_word, "truncate": args.truncate, "margin": args.margin,
        "caps": dict(sorted(args.cap or [])), "bias_sha256": None if bank_head is None else bank_head["bias_sha256"],
        "bank": None if args.bank is None else {"path": str(args.bank), "sha256": bank_sha},
        "variants_file": None if args.variants is None else {"path": str(args.variants),
                                                             "sha256": sets.sha256_file(args.variants)},
        "groups": {g.name: _group_meta(g) for g in groups},
        "variants": {v.key: {"patch": v.name, "group": v.group, "sha256": v.sha256, "twin": v.twin,
                             "fork_layer": v.fork_layer} for v in variants},
        "plan": {k: plan[k] for k in ("hidden_gb", "host_gb", "mem_available_gb", "fixed_gb")},
        "started_utc": _utc(started),
    }
    wanted = settings(meta)
    if not meta["vouched_ok"]:
        log.warning("vouched harness files differ from the gate's: %s", vouched)

    out = Path(args.out)
    mirror = PassMirror(args.sync) if args.sync else None
    local = _read_pass(out / PASS)
    mirrored = mirror.pass_json() if mirror is not None else None
    found = {where: pj for where, pj in ((str(out), local), (args.sync, mirrored)) if pj is not None}
    for where, pj in found.items():
        if pj.get("settings") != wanted:
            sys.exit(f"{where} was scored with other settings than this run; use a new --out and --sync")
        if args.routes == "ids" and _routes(pj) != "ids":  # routes stay out of settings (contract 5.5)
            sys.exit(f"{where} was scored without --routes ids, so it holds no route_ids; use a new --out and --sync")
    if found:
        if local is not None and mirror is not None and mirrored is None:
            mirror.put(out)
        print("pass: done, skipping", flush=True)
        return 0
    if out.exists():
        shutil.rmtree(out)  # an unfinished pass is redone from the start

    with open_source(args.weights, **run.source_options(args)) as source:
        streamer = Streamer(config, source, args.device, DTYPE, args.chunk_tokens, args.attn)
        bias = bank.stock_bias(source, moe)
        bias_sha = bank.patch_sha256(bias)
        if bank_head is not None and bank_head["bias_sha256"] != bias_sha:
            sys.exit(f"bank {args.bank} was built on stock bias {bank_head['bias_sha256']}, but the checkpoint's "
                     f"is {bias_sha}")
        result = multi.run_pass(streamer, groups, variants, routes=args.routes)
        heads = multi.head_pass(streamer, groups, variants, result.hidden, batch=args.head_batch)
    meta.update(bias_sha256=bias_sha, settings=wanted, bias=bias, started=started)
    del meta["groups"], meta["variants"]
    write_pass(out, groups, variants, result, heads, meta)
    if mirror is not None:
        mirror.put(out)
    print(f"pass: {len(groups)} groups, {len(variants)} variants, {time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
