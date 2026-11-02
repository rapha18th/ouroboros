"""Boot 1 on the VM: G1a, R0 and the equivalence gate, then the search rounds of Beam and CEM, then the picks.

    python -m ouroboros.harness.round run --weights W --inputs DIR --work DIR --sync URL --boot-start EPOCH \\
        [--config DIR] [--device cuda] [--attn auto] [--chunk-tokens 65536] [--host-gb 120] [--max-rounds 12] \\
        [--start-limit-h 6.3] [--deadline-min 15] [--poll-s 15] [--own-pick-min 15] [--runner subprocess|inprocess] \\
        [--weight-workers N] [--chunk-mb N]
    python -m ouroboros.harness.round r0-lite (same flags) --zero-shot-bank FILE

G1a is run.py from the vouched tree on the pre-cut search_dev parquet. R0 is one score.py pass over the R0 bank with
the twin. equivalence() compares the two; a failed gate writes equivalence.json and exits 3. R0's random draws then
fix the budget B and the footprint cap, and R0's jitter fixes the eligibility rule for the rest of the boot.

Round r builds two CEM candidates first, then waits until the deadline that round r - 1's scores announced for
round_<rr>/beam.json from the laptop controller, checks Beam's patches with the same rules the controller applied,
and scores everything in one score.py pass. scores.json (everything) and scores_beam.json (Beam's own history, the
floors and the chance reference; never a CEM, SteerMoE or instruction score) follow. Two Beam misses in a row close
the Beam arm; CEM runs on. A round starts only if its high estimate ends by --start-limit-h of boot. After the last
round, round.py waits up to --own-pick-min for Beam's own pick (and never past --start-limit-h plus 15 minutes of
boot, so the picks land before the step's timeout), then writes picks.json, picks_bank.safetensors, picks.sha256
and rounds_done.json, in that order. A sync-root read or write that fails is retried; while waiting for Beam, a
failed read counts as "not yet".

Everything goes to --sync (a gs:// prefix or a directory); --work holds the local copies. A rerun resumes from the
files already in --sync: finished rounds are read back, and CEM is rebuilt by replaying its draws over them.
Stage B is exploratory: every file written here carries stageb.LABEL.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ouroboros import stageb
from ouroboros.analysis import stage_b_metrics as metrics
from ouroboros.harness import bank
from ouroboros.surgeon.controller import parse_utc, utc
from ouroboros.surgeon.handoff import Handoff, LocalHandoff, dumps, open_handoff, read_json
from ouroboros.surgeon.validate import Candidate, Invalid, Limits, l2_units, validate_entries

log = logging.getLogger(__name__)

VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "beam"
SEARCH = "search_dev"
VARIANTS_FORMAT = "ouroboros-variants/1"
CEM_TRIES = 10  # footprint shrinks by 0.8, and redraws of a duplicate, per CEM slot
ROUND_VARIANTS_PER_R0 = 5 / 10  # a round scores 5 variants where R0 scores 10
BEAT_S = 60.0
HANDOFF_TRIES = 4  # attempts at one sync-root read or write before the loop gives up on it
HANDOFF_WAIT_S = 15.0
OWN_PICK_SLACK_H = 0.25  # the own-pick wait ends by --start-limit-h plus this, before the queue's step timeout


def run_pass_step(argv: list[str], runner: str) -> int:
    """`python -I -m argv[0] argv[1:]` in a subprocess, or argv[0]'s main(argv[1:]) in this process."""
    print("step: " + " ".join(argv), flush=True)
    if runner == "subprocess":
        return subprocess.run([sys.executable, "-I", "-m", *argv]).returncode
    if runner != "inprocess":
        raise ValueError(f"runner {runner!r} is neither subprocess nor inprocess")
    module = importlib.import_module(argv[0])
    try:
        return int(module.main(argv[1:]) or 0)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, flush=True)
            return 1
        return int(e.code or 0)


class SepCEM:
    """Separable cross-entropy search over one coefficient per basis row, each draw projected into the L2 ball.

    Basis rows are unit-L2 directions on disjoint layer bands, so a coefficient vector's L2 is its patch's L2 in
    units before rounding. One generation is popsize draws, scored two per round; tell() follows its last round.
    """

    def __init__(self, basis: np.ndarray, budget_units: float, *, popsize: int = stageb.CEM_POPSIZE,
                 elite: int = stageb.CEM_ELITE, seed: int = stageb.SEED):
        self.basis = np.asarray(basis, np.float64)
        self.budget_units = float(budget_units)
        self.popsize, self.elite = popsize, elite
        self.rng = np.random.default_rng(seed)
        n = len(self.basis)
        self.mean = np.zeros(n)
        self.std = np.full(n, self.budget_units / np.sqrt(n))
        self.pending: list[np.ndarray] = []
        self.generation = 0

    def _draw(self) -> np.ndarray:
        x = self.mean + self.std * self.rng.normal(size=len(self.mean))
        norm = float(np.linalg.norm(x))
        return x if norm <= self.budget_units else x * (self.budget_units / norm)

    def ask(self) -> list[np.ndarray]:
        self.pending = [self._draw() for _ in range(self.popsize)]
        self.generation += 1
        return [x.copy() for x in self.pending]

    def redraw(self, i: int) -> np.ndarray:
        """Replace draw i with a fresh draw from the same distribution (its patch duplicated a scored one)."""
        self.pending[i] = self._draw()
        return self.pending[i].copy()

    def tell(self, scores: Sequence[float]) -> None:
        """Lower is better; an ineligible or unscored draw is +inf."""
        s = np.array([np.inf if x is None or not np.isfinite(x) else float(x) for x in scores])
        if len(s) != len(self.pending):
            raise ValueError(f"{len(s)} scores for {len(self.pending)} draws")
        best = np.argsort(s, kind="stable")[:self.elite]
        elite = np.array([self.pending[i] for i in best])
        self.mean = elite.mean(axis=0)
        self.std = 0.7 * self.std + 0.3 * (elite.std(axis=0) + 1e-3 * self.budget_units)

    def patch_units(self, c: np.ndarray, moe_layers: Sequence[int]) -> list[tuple[int, int, int]]:
        """Whole units: c_b * basis_b rounded half away from zero on each basis cell, then trimmed to the budget."""
        u: dict[tuple[int, int], int] = {}
        for b, row in enumerate(self.basis):
            for j, e in zip(*np.nonzero(row)):
                x = float(c[b]) * float(row[j, e])
                k = int(np.sign(x) * np.floor(abs(x) + 0.5))
                if k:
                    u[(int(j), int(e))] = u.get((int(j), int(e)), 0) + k
        u = {k: v for k, v in u.items() if v}
        while u and l2_units((0, 0, v) for v in u.values()) > self.budget_units + 1e-9:
            key = max(u, key=lambda k: abs(u[k]))  # the first largest, in cell order
            u[key] -= int(np.sign(u[key]))
            if not u[key]:
                del u[key]
        return [(int(moe_layers[j]), e, v) for (j, e), v in sorted(u.items())]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _join(root: str, rel: str) -> str:
    return f"{root.rstrip('/')}/{rel}" if root.startswith("gs://") else str(Path(root) / rel)


def _rd(r: int) -> str:
    return f"round_{r:02d}"


def pull_tree(h: Handoff, sync: str, rel: str, dest: Path) -> bool:
    """Copy a directory from the sync root to dest when the local copy is gone (a new VM). False when absent."""
    if (dest / "pass.json").exists() or (dest / "index.json").exists():
        return True
    if isinstance(h, LocalHandoff):
        src = h.root / rel
        if not src.is_dir():
            return False
        shutil.copytree(src, dest, dirs_exist_ok=True)
        return True
    gcloud = shutil.which("gcloud")
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([gcloud, "storage", "cp", "--recursive", _join(sync, rel), str(dest.parent)])
    return proc.returncode == 0 and dest.exists()


def score_object(name: str, arm: str, r: int | None, slot: str | None, status: str, reason: str | None = None, *,
                 sha256: str | None = None, entries=None, n_entries: int | None = None, l2: float | None = None,
                 effective_l2: float | None = None, footprint: dict | None = None, metrics_: dict | None = None,
                 eligible: bool | None = None) -> dict:
    """A candidate's record in scores.json (section 5.6); unscored candidates carry null metrics."""
    m = metrics_ or {}
    fp = None if footprint is None else {"share": footprint.get("share"), "per_layer": footprint.get("per_layer")}
    return {"name": name, "arm": arm, "round": r, "slot": slot, "status": status, "reason": reason, "sha256": sha256,
            "entries": None if entries is None else [list(map(int, e)) for e in entries],
            "n_entries": n_entries if n_entries is not None else (None if entries is None else len(entries)),
            "l2_units": l2, "effective_l2": effective_l2, "footprint": fp,
            **{k: m.get(k) for k in ("J", "J_se", "dT", "dT_se", "dL_by_tool", "dL_pooled", "n_pos", "n_legit",
                                     "episodes_improved", "guard")},
            "eligible": eligible, "route_change": m.get("route_change")}


class Loop:
    """State of one boot's search, rebuilt from the sync root on a rerun."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.h = open_handoff(args.sync)
        self.sync = str(args.sync)
        self.work = Path(args.work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.inputs = Path(args.inputs)
        self.bias = np.asarray(np.load(self.inputs / "maps" / "bias.npy"), np.float32)
        self.n_moe, self.n_experts = self.bias.shape
        head = self._bank_head(self.inputs / "bank" / "r0.safetensors")
        self.moe_layers = tuple(int(x) for x in head["moe_layers"])
        self._sample = None
        self.rounds: dict[int, dict] = {}  # round -> scores.json
        self.eq: dict | None = None
        self.budget: dict | None = None
        self.rule: dict | None = None
        self.cem: SepCEM | None = None
        self.misses = 0
        self.arm_open = True
        self.closed_after: int | None = None
        self.est_pass_min = 0.0
        self.input_sha = {rel: sha256_bytes((self.inputs / rel).read_bytes())
                          for rel in ("maps/cem_basis.npz", "maps/window_sample.npz", "maps/bias.npy")
                          if (self.inputs / rel).exists()}

    # files

    def _bank_head(self, path: Path) -> dict:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            return json.loads((f.metadata() or {})["ouroboros"])

    @property
    def sample(self):
        if self._sample is None:
            from ouroboros.surgeon.footprint import load_sample

            self._sample = load_sample(self.inputs / "maps" / "window_sample.npz")
        return self._sample

    def _retry(self, what: str, fn):
        """fn() retried through storage hiccups, HANDOFF_TRIES attempts HANDOFF_WAIT_S apart."""
        for attempt in range(HANDOFF_TRIES):
            try:
                return fn()
            except (RuntimeError, OSError) as e:
                if attempt == HANDOFF_TRIES - 1:
                    raise
                print(f"sync root: {what} failed ({e}); retrying in {HANDOFF_WAIT_S:g} s", flush=True)
                time.sleep(HANDOFF_WAIT_S)

    def get(self, rel: str) -> bytes | None:
        return self._retry(f"reading {rel}", lambda: self.h.read(rel))

    def get_json(self, rel: str) -> dict | None:
        data = self.get(rel)
        return None if data is None else json.loads(data.decode("utf-8"))

    def write_remote(self, rel: str, data: bytes) -> None:
        self._retry(f"writing {rel}", lambda: self.h.write(rel, data))

    def put(self, rel: str, data: bytes) -> None:
        local = self.work / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        self.write_remote(rel, data)

    def put_json(self, rel: str, obj: dict) -> None:
        self.put(rel, dumps(obj))

    def elapsed_h(self) -> float:
        return (time.time() - float(self.args.boot_start)) / 3600

    def limits(self) -> Limits:
        return Limits(1, float(self.budget["B_units"]), self.budget.get("footprint_cap"), self.moe_layers,
                      self.n_experts)

    def step(self, argv: list[str]) -> None:
        code = run_pass_step(argv, self.args.runner)
        if code != 0:
            raise RuntimeError(f"{argv[0]} exited {code}")

    def source_flags(self) -> list[str]:
        out = []
        if self.args.weight_workers:
            out += ["--weight-workers", str(self.args.weight_workers)]
        if self.args.chunk_mb:
            out += ["--chunk-mb", str(self.args.chunk_mb)]
        return out

    def common(self) -> list[str]:
        a = self.args
        return ["--weights", a.weights, "--config", a.config, "--device", a.device, "--attn", a.attn,
                "--chunk-tokens", str(a.chunk_tokens), *self.source_flags()]

    def g1a(self) -> Path:
        """run.py on the pre-cut search_dev parquet: truncation off, no windows, no taps."""
        out = self.work / "g1a"
        self.step(["ouroboros.harness.run", *self.common(), "--rendered",
                   str(self.inputs / "sets" / "search_dev.parquet"), "--out", str(out), "--sync",
                   _join(self.sync, "g1a"), "--no-windows", "--taps"])
        shard = out / "shard_000"
        if not pull_tree(self.h, self.sync, "g1a/shard_000", shard):
            raise RuntimeError("G1a finished but shard_000 is neither local nor in the sync root")
        return shard

    def score(self, r: int, bank_path: Path, variants_path: Path, twin: bool = False) -> Path:
        out = self.work / _rd(r) / "pass"
        inputs = self.inputs / "sets"
        argv = ["ouroboros.harness.score", *self.common(), "--set",
                f"{SEARCH}={inputs / 'search_dev.parquet'}:{inputs / 'search_dev.json'}", "--bank", str(bank_path),
                "--variants", str(variants_path), "--out", str(out), "--sync", _join(self.sync, f"{_rd(r)}/pass"),
                "--host-gb", str(self.args.host_gb)]
        self.step(argv + (["--twin"] if twin else []))
        if not pull_tree(self.h, self.sync, f"{_rd(r)}/pass", out):
            raise RuntimeError(f"{_rd(r)} pass finished but is neither local nor in the sync root")
        return out

    # R0

    def r0(self, variants_file: str) -> dict:
        """G1a, the R0 pass with the twin, and the equivalence gate. Returns equivalence.json's content."""
        if self.eq is not None:
            return self.eq
        found = self.get_json("equivalence.json")
        if found is not None:
            self.eq = found
            return found
        shard = self.g1a()
        r0_bank = self.inputs / "bank" / "r0.safetensors"
        r0_vars = self.inputs / "variants" / variants_file
        self.put("round_00/bank.safetensors", r0_bank.read_bytes())
        self.put("round_00/variants.json", r0_vars.read_bytes())
        pass_dir = self.score(0, r0_bank, r0_vars, twin=True)
        self.eq = metrics.equivalence(shard, pass_dir, SEARCH)
        self.eq.setdefault("decision", "tier1" if self.eq.get("tier1") else "tier2" if self.eq.get("tier2")
                           else "fail")
        self.put_json("equivalence.json", stageb.labelled(self.eq))
        print(f"equivalence: {self.eq['decision']}", flush=True)
        return self.eq

    def twin_identical(self) -> bool:
        twin = (self.eq or {}).get("twin") or {}
        return bool(twin) and all(bool(v) for v in twin.values())

    def r0_scores(self, variants_file: str, stop: tuple[None, str] | None = None) -> dict:
        """Score every R0 patch and the twin; fix B, the footprint cap and the eligibility rule. `stop` makes the
        R0 scores final with that reason (the zero-shot path)."""
        found = self.get_json("round_00/scores.json")
        if found is None:
            pass_dir = self.work / "round_00" / "pass"
            if not pull_tree(self.h, self.sync, "round_00/pass", pass_dir):
                pass_dir = self.score(0, self.inputs / "bank" / "r0.safetensors",
                                      self.inputs / "variants" / variants_file, twin=True)
            g = metrics.load_pass(pass_dir).group(SEARCH)
            patches, head = bank.load_patch_bank(self.inputs / "bank" / "r0.safetensors", self.n_moe,
                                                 self.n_experts, self.bias)
            spec = json.loads((self.inputs / "variants" / variants_file).read_text(encoding="utf-8"))
            names = [v["patch"] for v in spec["variants"] if SEARCH in v["groups"]]
            cands = []
            for name in names:
                rec = head["patches"][name]
                fp = rec.get("footprint")
                if fp is None:
                    from ouroboros.surgeon.footprint import footprint

                    got = footprint(self.sample, patches[name].numpy())
                    fp = {"share": float(got["share"]), "per_layer": [float(x) for x in got["per_layer"]]}
                cands.append(score_object(
                    name, rec["provenance"]["arm"], 0, None, "scored", sha256=rec["sha256"], entries=rec.get("units"),
                    n_entries=rec["nonzero"], l2=rec["nominal_l2"] / stageb.UNIT, effective_l2=rec["effective_l2"],
                    footprint=fp, metrics_=metrics.search_score(g, name)))
            cands.append(score_object("twin", "twin", 0, None, "scored", sha256=None, entries=None, n_entries=0, l2=0.0,
                                      effective_l2=0.0, footprint={"share": 0.0, "per_layer": [0.0] * self.n_moe},
                                      metrics_=metrics.search_score(g, "twin")))
            by_name = {c["name"]: c for c in cands}
            self.rule = metrics.eligibility_rule(by_name["jitter"])
            self.budget = metrics.budget_rule(by_name, {c["name"]: c["footprint"]["share"] for c in cands})
            for c in cands:
                c["eligible"] = metrics.eligible(c, self.rule)
            pass_json = json.loads((pass_dir / "pass.json").read_text(encoding="utf-8"))
            self.est_pass_min = pass_json.get("seconds", 0.0) / 60 * ROUND_VARIANTS_PER_R0
            found = self.write_scores(0, "r0", cands, pass_dir, pass_json, guard_stock=metrics.guard(g, "stock"),
                                      stop=stop)
            print(f"R0: B = {self.budget['B']}, footprint cap {self.budget['footprint_cap']:.4f}, eligibility "
                  f"{self.rule['rule']}", flush=True)
        else:
            self.budget, self.rule = found["budget"], found["eligibility"]
            self.est_pass_min = (found.get("pass") or {}).get("seconds", 0.0) / 60 * ROUND_VARIANTS_PER_R0
        self.rounds[0] = found
        if self.cem is None:
            with np.load(self.inputs / "maps" / "cem_basis.npz") as z:
                self.cem = SepCEM(z["basis"], self.budget["B_units"])
        return found

    # scores files

    def history(self) -> list[dict]:
        return [c for r in sorted(self.rounds) if r >= 1 for c in self.rounds[r]["candidates"] if c["arm"] == "beam"]

    def best(self) -> dict:
        every = [c for r in sorted(self.rounds) if r >= 1 for c in self.rounds[r]["candidates"]]
        out = {}
        for arm in ("beam", "cem"):
            pick = metrics.select([c for c in every if c["arm"] == arm])
            out[arm] = None if pick is None else {"name": pick["name"], "J": pick["J"]}
        return out

    def next_round(self, done: int) -> tuple[int | None, str | None]:
        """The round to run after `done`, or None with the stop reason."""
        r = done + 1
        if r > self.args.max_rounds:
            return None, "max rounds"
        high_min = self.args.deadline_min + 1.15 * self.est_pass_min + 3
        if self.elapsed_h() + high_min / 60 > self.args.start_limit_h:
            return None, "round-start rule"
        return r, None

    def write_scores(self, r: int, kind: str, cands: list[dict], pass_dir: Path | None, pass_json: dict | None, *,
                     guard_stock: dict | None, stop: tuple[int | None, str | None] | None = None) -> dict:
        """scores.json, then scores_beam.json; the next round and its deadline are decided here."""
        self.rounds[r] = {"candidates": cands}  # history() and best() read it below
        nxt, reason = stop if stop is not None else self.next_round(r)
        written = time.time()
        deadline = None if nxt is None else utc(written + self.args.deadline_min * 60)
        evaluated = sum(c["arm"] == "cem" and c["status"] == "scored" for rr in self.rounds.values()
                        for c in rr["candidates"])
        pass_rec = None
        if pass_dir is not None:
            pass_rec = {"dir": f"{_rd(r)}/pass", "pass_sha256": sha256_bytes((pass_dir / "pass.json").read_bytes()),
                        "seconds": (pass_json or {}).get("seconds")}
        scores = stageb.labelled({
            "format": "ouroboros-round-scores/1", "round": r, "kind": kind, "written_utc": utc(written),
            "next_round": nxt, "next_deadline_utc": deadline, "final": nxt is None,
            "boot_elapsed_h": round(self.elapsed_h(), 4), "budget": self.budget, "eligibility": self.rule,
            "equivalence": {k: self.eq.get(k) for k in ("decision", "tier1", "tier2", "stock", "twin", "targets")},
            "pass": pass_rec, "stock": {"guard": guard_stock}, "candidates": cands,
            "beam": {"misses_in_a_row": self.misses, "arm_open": self.arm_open},
            "cem": {"generation": self.cem.generation if self.cem else 0, "evaluated": evaluated},
            "best": self.best(), "stop_reason": reason})
        self.put_json(f"{_rd(r)}/scores.json", scores)
        r0 = {c["name"]: c for c in self.rounds[0]["candidates"]} if 0 in self.rounds else \
            {c["name"]: c for c in cands}
        tag = stageb.budget_tag(self.budget["B"])
        beam_scores = stageb.labelled({
            "format": "ouroboros-round-scores-beam/1", "round": r, "written_utc": utc(written), "next_round": nxt,
            "next_deadline_utc": deadline, "final": nxt is None,
            "budget": {k: self.budget[k] for k in ("B", "B_units", "footprint_cap")},
            "eligibility": {k: self.rule[k] for k in ("rule", "top1_min", "kl_max")},
            "floors": {"twin_identical": self.twin_identical(), "jitter": r0.get("jitter")},
            "chance": [r0[n] for n in sorted(r0) if n.startswith(f"random_{tag}_k")],
            "history": self.history(),
            "beam": {"misses_in_a_row": self.misses, "arm_open": self.arm_open,
                     "n_candidates": stageb.BEAM_CANDIDATES, "lookups": stageb.LOOKUPS}})
        self.put_json(f"{_rd(r)}/scores_beam.json", beam_scores)
        self.rounds[r] = scores
        print(f"{_rd(r)}: scores written, next round {nxt if nxt is not None else 'none (' + reason + ')'}, "
              f"{self.elapsed_h():.2f} h of boot", flush=True)
        return scores

    # rounds

    def seen_before(self, r: int) -> dict[str, str]:
        return {c["sha256"]: c["name"] for k in sorted(self.rounds) if k < r for c in self.rounds[k]["candidates"]
                if c.get("sha256") and c["status"] == "scored"}

    def cem_candidates(self, r: int) -> tuple[list[Candidate], list[dict], dict]:
        """CEM's two slots of round r: draws 0 and 1 of a fresh generation in odd rounds, draws 2 and 3 in even ones."""
        if r % 2 == 1:
            self.cem.ask()
        draws = (0, 1) if r % 2 == 1 else (2, 3)
        seen = self.seen_before(r)
        limits = self.limits()
        kept: list[Candidate] = []
        objects: list[dict] = []
        slots = []
        for slot, i in zip("ab", draws):
            name = stageb.round_patch("cem", r, slot)
            c = self.cem.pending[i]
            shrinks = redraws = 0
            cand, reason = None, None
            while True:
                units = self.cem.patch_units(c, self.moe_layers)
                if not units:
                    reason = "the draw rounds to no whole-unit entry"
                    break
                try:
                    cand = validate_entries(units, limits, bias=self.bias, sample=self.sample, seen=seen, name=name)
                    break
                except Invalid as e:
                    if "footprint" in e.codes and shrinks < CEM_TRIES:
                        shrinks += 1
                        c = 0.8 * c
                        self.cem.pending[i] = c
                    elif "duplicate" in e.codes and redraws < CEM_TRIES:
                        redraws += 1
                        c = self.cem.redraw(i)
                    else:
                        reason = " ".join(e.problems)
                        break
            slots.append({"name": name, "draw": i, "coefficients": [float(x) for x in c], "shrinks": shrinks,
                          "redraws": redraws, "status": "built" if cand else "invalid", "reason": reason,
                          "sha256": None if cand is None else cand.sha256})
            if cand is None:
                objects.append(score_object(name, "cem", r, slot, "invalid", reason))
                continue
            seen[cand.sha256] = name
            kept.append(cand)
        record = stageb.labelled({"format": "ouroboros-cem-round/1", "round": r, "generation": self.cem.generation,
                                  "draws": list(draws), "mean": self.cem.mean.tolist(), "std": self.cem.std.tolist(),
                                  "budget_units": self.cem.budget_units, "slots": slots})
        return kept, objects, record

    def cem_tell(self, r: int) -> None:
        """After an even round: the generation's four J values, +inf for ineligible or unscored draws."""
        if r % 2:
            return
        by = {c["name"]: c for k in (r - 1, r) for c in self.rounds[k]["candidates"] if c["arm"] == "cem"}
        scores = []
        for k, slot in ((r - 1, "a"), (r - 1, "b"), (r, "a"), (r, "b")):
            c = by.get(stageb.round_patch("cem", k, slot)) or {}
            ok = c.get("status") == "scored" and c.get("eligible") and c.get("J") is not None
            scores.append(c["J"] if ok else np.inf)
        self.cem.tell(scores)

    def wait_beam(self, r: int) -> dict | None:
        """round_<rr>/beam.json once it lands, or None at the deadline round r - 1 announced."""
        rel = f"{_rd(r)}/beam.json"
        deadline_utc = self.rounds[r - 1].get("next_deadline_utc")
        deadline = parse_utc(deadline_utc) if deadline_utc else time.time()
        beat = time.time()
        while True:
            try:
                beam = read_json(self.h, rel)
            except (RuntimeError, OSError) as e:  # a storage hiccup reads as "not yet" until the deadline
                log.warning("%s: reading beam.json failed: %s", _rd(r), e)
                beam = None
            if beam is not None or time.time() >= deadline:
                return beam
            if time.time() - beat >= BEAT_S:
                beat = time.time()
                print(f"{_rd(r)}: waiting for Beam until {deadline_utc} ({deadline - time.time():.0f} s left)",
                      flush=True)
            time.sleep(max(0.0, min(self.args.poll_s, deadline - time.time())))

    def beam_candidates(self, r: int, beam: dict | None) -> tuple[list[tuple[Candidate, dict]], list[dict]]:
        """Beam's patches that pass the checks, with their bank records, and score objects for the slots that fail."""
        names = [stageb.round_patch("beam", r, s) for s in "ab"[:stageb.BEAM_CANDIDATES]]

        def fail(status: str, reason: str, which=names) -> list[dict]:
            return [score_object(n, "beam", r, n[-1], status, reason) for n in which]

        if beam is None:
            return [], fail("forfeit", "missing at the deadline")
        if beam.get("status") != "ok":
            return [], fail("forfeit", beam.get("forfeit") or "forfeit")
        try:
            blob = self.get(f"{_rd(r)}/beam.safetensors")
        except (RuntimeError, OSError) as e:
            return [], fail("invalid", f"beam.safetensors could not be read: {e}")
        if blob is None or sha256_bytes(blob) != beam.get("sha256"):
            return [], fail("invalid", "beam.safetensors is missing or does not match beam.json")
        path = self.work / _rd(r) / "beam.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        try:
            patches, head = bank.load_patch_bank(path, self.n_moe, self.n_experts, self.bias)
        except ValueError as e:
            return [], fail("invalid", f"the Beam bank is refused: {e}")
        seen = {c["sha256"]: c["name"] for c in self.history() if c.get("sha256")}
        r0 = {c["name"]: c for c in self.rounds[0]["candidates"]}
        tag = stageb.budget_tag(self.budget["B"])
        for n, c in r0.items():
            if c.get("sha256") and (n == "jitter" or n.startswith(f"random_{tag}_k")):
                seen[c["sha256"]] = n
        good, bad = [], []
        for name in names:
            rec = head["patches"].get(name)
            if rec is None:
                bad += fail("invalid", "missing from the Beam file", [name])
                continue
            try:
                cand = validate_entries([tuple(e) for e in rec.get("units") or []], self.limits(), bias=self.bias,
                                        sample=self.sample, seen=seen, name=name)
            except Invalid as e:
                bad += fail("invalid", " ".join(e.problems), [name])
                continue
            if cand.sha256 != bank.patch_sha256(patches[name]):
                bad += fail("invalid", "the tensor differs from its units", [name])
                continue
            seen[cand.sha256] = name
            good.append((cand, rec))
        return good, bad

    def run_round(self, r: int) -> dict:
        rd = _rd(r)
        cem_kept, objects, cem_record = self.cem_candidates(r)
        self.put_json(f"{rd}/cem.json", cem_record)
        beam = self.wait_beam(r) if self.arm_open else None
        if self.arm_open:
            beam_good, beam_bad = self.beam_candidates(r, beam)
        else:
            names = [stageb.round_patch("beam", r, s) for s in "ab"[:stageb.BEAM_CANDIDATES]]
            beam_good, beam_bad = [], [score_object(n, "beam", r, n[-1], "forfeit", "the Beam arm is closed")
                                       for n in names]
        if self.arm_open:
            self.misses = 0 if beam_good else self.misses + 1
            if self.misses >= 2:
                self.arm_open, self.closed_after = False, r
                print(f"{rd}: two Beam misses in a row; the Beam arm closes", flush=True)
        patches, meta, order = {}, {}, []
        for cand, rec in beam_good:
            patches[cand.name] = cand.delta
            meta[cand.name] = {"provenance": rec["provenance"], "units": rec["units"],
                               "footprint": rec.get("footprint"), "hypothesis": rec.get("hypothesis")}
            order.append(("beam", cand))
        for cand in cem_kept:
            patches[cand.name] = cand.delta
            meta[cand.name] = {"provenance": {"arm": "cem", "round": r, "slot": cand.name[-1], "seed": stageb.SEED,
                                              "parent": None, "built_by": "harness.round.cem_candidates",
                                              "inputs": self.input_sha},
                               "units": [list(e) for e in cand.entries], "footprint": cand.footprint,
                               "hypothesis": None}
            order.append(("cem", cand))
        cands = list(beam_bad)
        pass_dir = pass_json = guard_stock = None
        if patches:
            bank_path = self.work / rd / "bank.safetensors"
            head = bank.save_patch_bank(bank_path, patches, meta, self.bias, self.moe_layers)
            blob = bank_path.read_bytes()
            self.write_remote(f"{rd}/bank.safetensors", blob)
            variants = stageb.labelled({"format": VARIANTS_FORMAT, "bank_sha256": sha256_bytes(blob),
                                        "variants": [{"patch": c.name, "groups": [SEARCH]} for _, c in order]})
            self.put_json(f"{rd}/variants.json", variants)
            pass_dir = self.score(r, bank_path, self.work / rd / "variants.json")
            pass_json = json.loads((pass_dir / "pass.json").read_text(encoding="utf-8"))
            self.est_pass_min = max(self.est_pass_min if r > 1 else 0.0, pass_json.get("seconds", 0.0) / 60)
            g = metrics.load_pass(pass_dir).group(SEARCH)
            guard_stock = metrics.guard(g, "stock")
            for arm, cand in order:
                rec = head["patches"][cand.name]
                obj = score_object(cand.name, arm, r, cand.name[-1], "scored", sha256=cand.sha256,
                                   entries=cand.entries, l2=cand.l2_units, effective_l2=rec["effective_l2"],
                                   footprint=cand.footprint or rec.get("footprint"),
                                   metrics_=metrics.search_score(g, cand.name))
                obj["eligible"] = metrics.eligible(obj, self.rule)
                cands.append(obj)
        cands += objects
        rank = {"beam": 0, "cem": 1}
        cands.sort(key=lambda c: (rank.get(c["arm"], 2), c["slot"] or ""))
        scores = self.write_scores(r, "round", cands, pass_dir, pass_json, guard_stock=guard_stock)
        self.cem_tell(r)
        return scores

    def replay(self, r: int, scores: dict) -> None:
        """Rebuild state from a finished round read back from the sync root; CEM must redraw the same patches."""
        _, _, record = self.cem_candidates(r)
        built = {s["name"]: s["sha256"] for s in record["slots"]}
        stored = {c["name"]: c.get("sha256") for c in scores["candidates"] if c["arm"] == "cem"}
        if built != stored:
            raise RuntimeError(f"{_rd(r)}: CEM replay built {built}, the sync root holds {stored}")
        self.rounds[r] = scores
        self.misses = scores["beam"]["misses_in_a_row"]
        if self.arm_open and not scores["beam"]["arm_open"]:
            self.closed_after = r
        self.arm_open = scores["beam"]["arm_open"]
        seconds = (scores.get("pass") or {}).get("seconds")
        if seconds is not None:
            self.est_pass_min = max(self.est_pass_min if r > 1 else 0.0, seconds / 60)
        self.cem_tell(r)

    # picks

    def wait_own_pick(self) -> dict | None:
        last = self.rounds[max(self.rounds)]
        if not any(c["status"] == "scored" for c in self.history()):
            return self.get_json("own_pick.json")
        until = min(parse_utc(last["written_utc"]) + self.args.own_pick_min * 60,
                    float(self.args.boot_start) + (self.args.start_limit_h + OWN_PICK_SLACK_H) * 3600)
        beat = time.time()
        while True:
            try:
                got = read_json(self.h, "own_pick.json")
            except (RuntimeError, OSError) as e:  # a storage hiccup reads as "not yet"
                log.warning("reading own_pick.json failed: %s", e)
                got = None
            if got is not None or time.time() >= until:
                return got
            if time.time() - beat >= BEAT_S:
                beat = time.time()
                print(f"waiting for Beam's own pick ({until - time.time():.0f} s left)", flush=True)
            time.sleep(max(0.0, min(self.args.poll_s, until - time.time())))

    def round_bank(self, r: int) -> tuple[dict, dict, str]:
        path = self.work / _rd(r) / "bank.safetensors"
        if not path.exists():
            blob = self.get(f"{_rd(r)}/bank.safetensors")
            if blob is None:
                raise RuntimeError(f"{_rd(r)}/bank.safetensors is in neither the work dir nor the sync root")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        patches, head = bank.load_patch_bank(path, self.n_moe, self.n_experts, self.bias)
        return patches, head, sha256_bytes(path.read_bytes())

    def write_picks(self, stop_reason: str, picks: dict[str, dict | None], own: dict | None, *,
                    selection_rule: str = metrics.SELECTION_RULE, sources: dict | None = None,
                    rounds_completed: int | None = None) -> dict:
        """picks.json, picks_bank.safetensors, picks.sha256 and rounds_done.json, in that order."""
        ordered = [c for r in sorted(self.rounds) if r >= 1 for c in self.rounds[r]["candidates"]]
        every = {c["name"]: c for c in ordered}
        scored = {arm: [c for c in ordered if c["arm"] == arm and c["status"] == "scored"] for arm in ("beam", "cem")}
        own_status = (own or {}).get("status", "missing")
        own_name = (own or {}).get("name")
        if own_status == "ok" and (own_name not in every or every[own_name]["arm"] != "beam"
                                   or every[own_name]["status"] != "scored"):
            own_status = "invalid"
        patches, meta = {}, {}
        sources = sources or {}

        def take(new: str, c: dict, arm: str) -> None:
            if c["name"] in sources:
                tensors, head, file_sha, rel = sources[c["name"]]
            else:
                tensors, head, file_sha = self.round_bank(c["round"])
                rel = f"{_rd(c['round'])}/bank.safetensors"
            rec = head["patches"][c["name"]]
            patches[new] = tensors[c["name"]]
            meta[new] = {"provenance": {"arm": arm, "round": c.get("round"), "slot": c.get("slot"),
                                        "seed": rec["provenance"].get("seed"), "parent": c["name"],
                                        "built_by": "harness.round.write_picks", "inputs": {rel: file_sha}},
                         "units": rec.get("units"), "footprint": rec.get("footprint"),
                         "hypothesis": rec.get("hypothesis")}

        arms = {}
        for arm, final in (("beam", "beam_final"), ("cem", "cem_final")):
            c = picks.get(arm)
            arms[arm] = {"pick": None if c is None else c["name"], "reading": "none" if c is None else "picked",
                         "J": None if c is None else c.get("J"),
                         "effective_l2": None if c is None else c.get("effective_l2"),
                         "n_entries": None if c is None else c.get("n_entries"),
                         "sha256": None if c is None else c.get("sha256"),
                         "round": None if c is None else c.get("round"),
                         "evaluated": len(scored[arm])}  # scored candidates the pick was chosen among
            if c is not None:
                take(final, c, c["arm"] if c["arm"] in bank.ARMS else arm)
        # Beam vs CEM at equal evaluations: CEM scores two candidates every round, Beam's misses included, so its
        # pick may rest on more evaluations than Beam's. A reading only; cem_final stays the full-search pick.
        n_beam = len(scored["beam"])
        equal = metrics.select(scored["cem"][:n_beam])
        cem_equal = {"n": n_beam, "pick": None if equal is None else equal["name"],
                     "J": None if equal is None else equal.get("J"),
                     "rule": "the selection rule over CEM's first n scored candidates in round and slot order, where n "
                             "is Beam's scored count"}
        if own_status == "ok":
            take("beam_own", every[own_name], "beam")
        bank_path = self.work / "picks_bank.safetensors"
        bank.save_patch_bank(bank_path, patches, meta, self.bias, self.moe_layers)
        bank_blob = bank_path.read_bytes()
        beam_pick = picks.get("beam")
        rounds_done = max((r for r in self.rounds if r >= 1), default=0) if rounds_completed is None \
            else rounds_completed
        picks_json = stageb.labelled({
            "format": "ouroboros-picks/1", "written_utc": utc(), "selection_rule": selection_rule,
            "budget": self.budget, "eligibility": self.rule,
            "equivalence": {k: (self.eq or {}).get(k) for k in ("decision", "tier1", "tier2")},
            "rounds_completed": rounds_done,
            "stop": {"reason": stop_reason, "beam_arm_closed_after_round": self.closed_after},
            "arms": arms,
            "cem_equal_count": cem_equal,
            "beam_own_pick": {"status": own_status, "name": own_name if own_status == "ok" else None,
                              "sha256": every[own_name]["sha256"] if own_status == "ok" else None,
                              "equals_final": bool(own_status == "ok" and beam_pick is not None
                                                   and own_name == beam_pick["name"])},
            "picks_bank": {"file": "picks_bank.safetensors", "sha256": sha256_bytes(bank_blob)},
            "scores_sha256": {_rd(r): sha256_bytes(self.get(f"{_rd(r)}/scores.json") or b"")
                              for r in sorted(self.rounds)}})
        picks_blob = dumps(picks_json)
        self.put("picks.json", picks_blob)
        self.write_remote("picks_bank.safetensors", bank_blob)
        self.put("picks.sha256", (f"{sha256_bytes(picks_blob)}  picks.json\n"
                                  f"{sha256_bytes(bank_blob)}  picks_bank.safetensors\n").encode())
        self.put_json("rounds_done.json", stageb.labelled({
            "format": "ouroboros-rounds-done/1", "status": "ok", "rounds_completed": rounds_done,
            "stop_reason": stop_reason, "picks_sha256": sha256_bytes(picks_blob), "written_utc": utc()}))
        print(f"picks: beam {arms['beam']['pick']}, cem {arms['cem']['pick']}, own pick {own_status}", flush=True)
        return picks_json


def run_loop(args: argparse.Namespace) -> int:
    """0 done; 3 equivalence failed; 1 other failure."""
    loop = Loop(args)
    if loop._retry("probing rounds_done.json", lambda: loop.h.exists("rounds_done.json")):
        print("rounds: done, skipping", flush=True)
        return 0
    try:
        eq = loop.r0("r0.json")
        if eq.get("decision") == "fail":
            print("equivalence failed: the multi-patch path stops here", flush=True)
            return 3
        last = loop.r0_scores("r0.json")
        r = 0
        while not last.get("final"):
            r = last["next_round"]
            found = loop.get_json(f"{_rd(r)}/scores.json")
            if found is not None:
                loop.replay(r, found)
                last = found
                print(f"{_rd(r)}: read back from the sync root", flush=True)
                continue
            last = loop.run_round(r)
        own = loop.wait_own_pick()
        every = [c for k in sorted(loop.rounds) if k >= 1 for c in loop.rounds[k]["candidates"]]
        picks = {arm: metrics.select([c for c in every if c["arm"] == arm]) for arm in ("beam", "cem")}
        loop.write_picks(last.get("stop_reason") or "max rounds", picks, own)
    except Exception:
        log.exception("round loop failed")
        return 1
    return 0


def r0_lite(args: argparse.Namespace) -> int:
    """Zero-shot path: G1a, R0-lite and the gate, then Beam's zero-shot candidates at B become the picks."""
    loop = Loop(args)
    if loop._retry("probing rounds_done.json", lambda: loop.h.exists("rounds_done.json")):
        print("r0-lite: done, skipping", flush=True)
        return 0
    try:
        eq = loop.r0("r0_lite.json")
        if eq.get("decision") == "fail":
            print("equivalence failed: the multi-patch path stops here", flush=True)
            return 3
        loop.r0_scores("r0_lite.json", stop=(None, "zero-shot path"))
        zs = Path(args.zero_shot_bank)
        tensors, head = bank.load_patch_bank(zs, loop.n_moe, loop.n_experts, loop.bias)
        zs_sha = sha256_bytes(zs.read_bytes())
        tag = stageb.budget_tag(loop.budget["B"])
        sources, picks = {}, {}
        for arm, k in (("beam", 1), ("cem", 2)):
            name = f"beam_zs_{tag}_{k}"
            rec = head["patches"].get(name)
            if rec is None:
                picks[arm] = None
                continue
            sources[name] = (tensors, head, zs_sha, "zero_shot_bank.safetensors")
            picks[arm] = {"name": name, "arm": "zero_shot", "round": None, "slot": None, "J": None,
                          "effective_l2": rec["effective_l2"], "n_entries": rec["nonzero"], "sha256": rec["sha256"]}
        loop.put("zero_shot_bank.safetensors", zs.read_bytes())
        loop.write_picks("zero-shot path", picks, None, sources=sources, rounds_completed=0,
                         selection_rule=f"zero-shot: Beam's first-listed candidate at B ({tag}) takes the Beam slot, "
                                        "its second the CEM slot")
    except Exception:
        log.exception("r0-lite failed")
        return 1
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--weights", required=True, help="checkpoint directory or gs://bucket/prefix")
    common.add_argument("--inputs", required=True, help="the Stage B inputs directory")
    common.add_argument("--work", required=True, help="local directory for this boot's passes")
    common.add_argument("--sync", required=True, help="gs:// round root or a directory, shared with the controller")
    common.add_argument("--boot-start", required=True, type=float, help="kernel boot time, epoch seconds")
    common.add_argument("--config", default=str(VENDOR), help="directory holding Beam's config.json")
    common.add_argument("--device", default="cuda")
    common.add_argument("--attn", choices=["auto", "flash", "sdpa"], default="auto")
    common.add_argument("--chunk-tokens", type=int, default=65536)
    common.add_argument("--host-gb", type=float, default=120.0)
    common.add_argument("--max-rounds", type=int, default=stageb.ROUNDS_MAX)
    common.add_argument("--start-limit-h", type=float, default=stageb.ROUND_START_LIMIT_H)
    common.add_argument("--deadline-min", type=float, default=stageb.DEADLINE_MIN)
    common.add_argument("--poll-s", type=float, default=15.0)
    common.add_argument("--own-pick-min", type=float, default=15.0)
    common.add_argument("--runner", choices=["subprocess", "inprocess"], default="subprocess")
    common.add_argument("--weight-workers", type=int, help="parallel weight reads (default: the source's own)")
    common.add_argument("--chunk-mb", type=int, help="megabytes per weight read (default: the source's own)")
    sub.add_parser("run", parents=[common], help="G1a, R0, the rounds and the picks")
    lite = sub.add_parser("r0-lite", parents=[common], help="zero-shot path: G1a, R0-lite and the zero-shot picks")
    lite.add_argument("--zero-shot-bank", required=True, help="patch bank from controller zero-shot")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    return run_loop(args) if args.cmd == "run" else r0_lite(args)


if __name__ == "__main__":
    sys.exit(main())
