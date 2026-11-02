"""Stage B end to end on the tiny Beam: Boot 1, Boot 2 and the readouts, on CPU.

One test walks the whole chain. It builds an inputs tree and writes its manifests, runs round.py in-process against
a LocalHandoff root with a stub surgeon in a thread (two rounds with patches, one round with Beam silent, the own
pick), builds the Boot 2 bank from the picks, runs e1-, e2- and e3-like score.py passes over tiny stand-ins of the
held-out sets, and scores them with stage_b_metrics.readouts. Every file the chain writes must carry the exploratory
label, and the vouched harness files must match their recorded hashes before and after.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from stageb_support import one_thread, tiny_beam  # noqa: F401

from ouroboros import stageb
from ouroboros.analysis import stage_b, stage_b_maps
from ouroboros.analysis import stage_b_metrics as metrics
from ouroboros.harness import round as rnd
from ouroboros.harness import score, sets
from ouroboros.surgeon import controller as ctl
from ouroboros.surgeon import footprint as fp
from ouroboros.surgeon.handoff import LocalHandoff, read_json
from ouroboros.surgeon.validate import units_delta

HARNESS = Path(sets.__file__).resolve().parent
OPS = Path(__file__).resolve().parents[2] / "ops"  # beside repo/ in the working tree; absent in a public copy
RUN_FILES = ("index.json", "shards.json")  # run.py's own G1a files predate the label


def vouched_now() -> dict[str, str]:
    return {name: hashlib.sha256((HARNESS / name).read_bytes()).hexdigest() for name in stageb.VOUCHED_SHA256}


def submission(*cands, notes: str) -> dict:
    return {"notes": notes, "candidates": [
        {"name": f"c{i}", "hypothesis": "these experts lean toward the tamper call",
         "entries": [{"l": l, "e": e, "u": u} for l, e, u in entries]} for i, entries in enumerate(cands)]}


def visible_entries(inputs: Path, budget: dict) -> list[list[tuple[int, int, int]]]:
    """Single-entry candidates that move some sampled routes and stay under the round's footprint cap."""
    sample = fp.load_sample(inputs / "maps" / "window_sample.npz")
    n_experts = sample.bias.shape[1]
    cap, half = budget.get("footprint_cap"), max(1, int(budget["B_units"]) // 2)
    found = []
    for l in sample.moe_layers:
        for e in range(n_experts):
            for u in (half, -half):
                share = fp.footprint(sample, units_delta([(l, e, u)], sample.moe_layers, n_experts))["share"]
                if share > 0 and (cap is None or share <= cap):
                    found.append([(l, e, u)])
                    break
    return found


def score_pass(tiny, out: Path, mirror: Path, groups: list[str], bank: Path, variants: Path, *extra: str) -> Path:
    ckpt = str(tiny.ckpt)
    argv = ["--weights", ckpt, "--config", ckpt, "--device", "cpu", "--attn", "sdpa"]
    for g in groups:
        argv += ["--set", g]
    argv += ["--bank", str(bank), "--variants", str(variants), "--slot-word", "BLOCKED", "--truncate", "--margin",
             "0", *extra, "--out", str(out), "--sync", str(mirror)]
    assert score.main(argv) == 0
    return out


def labelled_json(root: Path) -> list[Path]:
    """Every JSON file under root that lacks exploratory: true (run.py's own G1a files aside)."""
    bad = []
    for path in sorted(root.rglob("*.json")):
        if path.name in RUN_FILES:
            continue
        if json.loads(path.read_text(encoding="utf-8")).get("exploratory") is not True:
            bad.append(path)
    return bad


def test_stageb_end_to_end(tiny_beam, tmp_path, one_thread):
    from stageb_tiny_inputs import tiny_cvd_inputs, tiny_held_inputs, tiny_inputs

    # 1. The vouched files match their declared hashes before anything runs.
    assert sets.check_vouched() == dict(stageb.VOUCHED_SHA256)
    assert vouched_now() == dict(stageb.VOUCHED_SHA256)

    # 2. An inputs tree with the CvD and held-out stand-ins, and the manifest step over it.
    inputs = tiny_inputs(tmp_path / "inputs", tiny_beam)
    tiny_cvd_inputs(inputs, tiny_beam)
    tiny_held_inputs(inputs, tiny_beam)
    doc = stage_b.manifest(inputs)
    assert doc["exploratory"] is True and doc["vouched"] == dict(stageb.VOUCHED_SHA256)
    for line in (inputs / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        sha, rel = line.split("  ", 1)
        assert hashlib.sha256((inputs / rel).read_bytes()).hexdigest() == sha, rel
    assert {"sets/search_dev.parquet", "bank/r0.safetensors", "variants/r0.json", "maps/window_sample.npz",
            "sets/cvd_trap.json", "sets/tripwire_held.json"} <= set(doc["files"])

    # 3. Boot 1: G1a, R0, two rounds with Beam patches, one round with Beam silent, the own pick, the picks.
    root, logs, work = tmp_path / "boot1", tmp_path / "surgeon", tmp_path / "work1"
    ckpt = str(tiny_beam.ckpt)
    argv = ["run", "--weights", ckpt, "--config", ckpt, "--inputs", str(inputs), "--work", str(work), "--sync",
            str(root), "--boot-start", str(time.time()), "--device", "cpu", "--attn", "sdpa", "--runner",
            "inprocess", "--max-rounds", "3", "--deadline-min", "0.5", "--poll-s", "0.2", "--own-pick-min", "2"]
    h = LocalHandoff(root)
    errors: list[BaseException] = []
    sent: dict[int, dict] = {}

    def wait_for(rel: str, limit_s: float = 2400) -> dict:
        t0 = time.time()
        while time.time() - t0 < limit_s:
            got = read_json(h, rel)
            if got is not None:
                return got
            time.sleep(0.2)
        raise TimeoutError(rel)

    def controller():
        try:
            ledger = ctl.Ledger(logs / "ledger.json")
            sb = wait_for("round_00/scores_beam.json")
            found = visible_entries(inputs, sb["budget"])
            fallback = [[(3, k, -1), (4, k + 1, 1)] for k in range(4)]
            entries = (found + fallback)[:4] if len(found) < 4 else found[:4]
            for r in (1, 2):
                if r == 2:
                    wait_for("round_01/scores_beam.json")
                sent[r] = submission(*entries[2 * r - 2:2 * r], notes=f"round {r}: lower one expert")
                stub = ctl.StubProvider([ctl.completion([("submit_patches", sent[r])])])
                # the stub answers at once; a clock two minutes behind keeps the 60 s abandon margin out of the way
                res = ctl.run_round(r, provider=stub, handoff=h, inputs=inputs, ledger=ledger, log_dir=logs,
                                    now=lambda: time.time() - 120, max_rounds=3)
                assert res.status == "ok", res
            final = wait_for("round_03/scores_beam.json")  # round 3: the laptop stays silent, a Beam miss
            assert final["final"]
            scored = [c for c in final["history"] if c["status"] == "scored"]
            best = metrics.select(scored) or scored[0]
            stub = ctl.StubProvider([ctl.completion([("pick_final", {"name": best["name"], "why": "lowest J"})])])
            ctl.own_pick(provider=stub, handoff=h, inputs=inputs, ledger=ledger, log_dir=logs)
        except BaseException as e:  # surfaced in the main thread
            errors.append(e)

    thread = threading.Thread(target=controller, daemon=True)
    thread.start()
    assert rnd.main(argv) == 0
    thread.join(timeout=120)
    assert not errors, errors

    eq = read_json(h, "equivalence.json")
    assert eq["decision"] == "tier1" and eq["exploratory"] is True
    r0 = read_json(h, "round_00/scores.json")
    assert [c["name"] for c in r0["candidates"]] == [*stageb.R0_PATCHES, "twin"]
    rounds = {r: read_json(h, f"round_{r:02d}/scores.json") for r in (1, 2, 3)}
    for r in (1, 2):
        got = {c["name"]: c for c in rounds[r]["candidates"]}
        for slot, cand in zip("ab", sent[r]["candidates"]):
            c = got[stageb.round_patch("beam", r, slot)]
            assert c["status"] == "scored" and c["entries"] == [[x["l"], x["e"], x["u"]] for x in cand["entries"]]
        assert {stageb.round_patch("cem", r, s) for s in "ab"} <= set(got)
    three = {c["name"]: c for c in rounds[3]["candidates"]}
    assert three["beam_r03a"]["status"] == "forfeit" and three["beam_r03a"]["reason"] == "missing at the deadline"
    assert rounds[3]["beam"] == {"misses_in_a_row": 1, "arm_open": True}
    assert rounds[3]["final"] and rounds[3]["stop_reason"] == "max rounds"
    for r in range(4):  # Beam's file names no other arm
        text = h.read(f"round_{r:02d}/scores_beam.json").decode()
        assert "cem_r" not in text and "steermoe" not in text and "instruction" not in text, r
    picks = read_json(h, "picks.json")
    assert picks["rounds_completed"] == 3 and picks["stop"]["reason"] == "max rounds"
    for arm in ("beam", "cem"):
        want = metrics.select([c for s in rounds.values() for c in s["candidates"] if c["arm"] == arm])
        assert picks["arms"][arm]["pick"] == (want or {}).get("name"), arm
    beam_on = picks["arms"]["beam"]["reading"] == "picked"
    assert picks["beam_own_pick"]["status"] == "ok" and picks["beam_own_pick"]["equals_final"] is beam_on
    for line in (root / "picks.sha256").read_text(encoding="utf-8").splitlines():
        sha, name = line.split("  ")
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name
    done = read_json(h, "rounds_done.json")
    assert (done["status"], done["rounds_completed"]) == ("ok", 3)
    report = metrics.search_report(root)
    assert report["checks"]["recompute"]["ok"] and report["checks"]["selection"]["beam"]["ok"]

    # 4. The Boot 2 bank and variants from the picks, then the manifest step again, as the operator runs them.
    built = stage_b_maps.boot2_bank(inputs, root / "picks.json", root / "picks_bank.safetensors")
    stage_b.manifest(inputs)
    listed = {v["patch"]: v for name in ("e1", "e2", "e3") for v in
              json.loads((inputs / "variants" / f"{name}.json").read_text(encoding="utf-8"))["variants"]}
    assert ("beam_final" in listed) is beam_on and ("cem_final" in listed) is (picks["arms"]["cem"]["reading"]
                                                                                == "picked")
    assert {"steermoe_final", "instruction_final", "random_final_k0", "jitter"} <= set(listed)
    if beam_on:
        share = built["head"]["patches"]["beam_final"]["footprint"]["share"]
        # the own pick equals the final, so a footprint-matched random patch stands in for it when it can
        assert ("route_match" in listed) is (share > 0) and "beam_own" not in listed

    # 5. Boot 2 passes over the tiny held-out stand-ins.
    sets_dir, rendered = inputs / "sets", inputs / "rendered" / "tripwire.parquet"
    bank2, var = inputs / "bank" / "boot2.safetensors", inputs / "variants"
    cvd = [f"{n}={sets_dir / (n + '.parquet')}:{sets_dir / (n + '.json')}"
           for n in ("cvd_trap", "cvd_true_done", "cvd_rule")]
    held = {n: f"{n}={rendered}:{sets_dir / (n + '.json')}" for n in ("tripwire_held", "guard_held")}
    boot2, mirror = tmp_path / "work2", tmp_path / "boot2"
    e1 = score_pass(tiny_beam, boot2 / "e1", mirror / "e1", cvd, bank2, var / "e1.json")
    e2 = score_pass(tiny_beam, boot2 / "e2", mirror / "e2", [held["tripwire_held"], held["guard_held"]], bank2,
                    var / "e2.json", "--reorder", "guard_reordered=guard_held")
    e3 = score_pass(tiny_beam, boot2 / "e3", mirror / "e3", [cvd[0], held["guard_held"], held["tripwire_held"]],
                    bank2, var / "e3.json")
    for p in (e1, e2, e3):
        assert (mirror / p.name / "pass.json").read_bytes() == (p / "pass.json").read_bytes()
    a, b = e1 / "cvd_trap" / "variant_stock", e3 / "cvd_trap" / "variant_stock"
    for arr in ("nll.npy", "top_ids.npy", "candidate_logprobs.npy"):
        assert (a / arr).read_bytes() == (b / arr).read_bytes(), arr  # same layout, so stock is bit-equal
    assert score.main(["--weights", ckpt, "--config", ckpt, "--device", "cpu", "--attn", "sdpa", *sum(
        (["--set", g] for g in cvd), []), "--bank", str(bank2), "--variants", str(var / "e1.json"), "--slot-word",
        "BLOCKED", "--truncate", "--margin", "0", "--out", str(e1), "--sync", str(mirror / "e1")]) == 0  # skips

    # 6. The readouts over the three passes and the picks.
    out = metrics.readouts(e1, e2, e3, root / "picks.json", B=200)
    assert out["exploratory"] is True and out["registered"]["H1"] == "fail"
    ids = {r["id"] for r in out["readouts"]}
    assert {"H4", "H5", "guard", "H6_H4", "H6_H5", "stock_determinism"} <= ids, ids
    assert all(r["exploratory"] is True for r in out["readouts"])
    det = next(r for r in out["readouts"] if r["id"] == "stock_determinism")
    assert det["value"] == 1.0, det  # the scorer sees the same bit equality
    if beam_on:
        h4 = [r for r in out["readouts"] if r["id"] == "H4" and r["arm"] == "beam_final"]
        assert h4 and h4[0]["n"] == 1 and h4[0]["wording"] in ("meets the H4 yardstick", "misses the H4 yardstick")
    text = json.dumps(out)
    assert "H4 passes" not in text and "H5 passes" not in text
    (tmp_path / "readouts.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for tree in (root, mirror, inputs / "variants", inputs / "sets", logs):
        assert not labelled_json(tree), labelled_json(tree)

    # 7. The public code holds no bucket, project, account or local path.
    scrub = OPS / "scrub_check.sh"
    if scrub.exists() and shutil.which("bash"):
        repo = Path(__file__).resolve().parents[1]
        got = subprocess.run(["bash", str(scrub), str(repo / "ouroboros"), str(repo / "tests")], capture_output=True,
                             text=True)
        assert got.returncode == 0 and not got.stdout.strip(), got.stdout + got.stderr

    # 10. The vouched files are unchanged after everything ran.
    assert vouched_now() == dict(stageb.VOUCHED_SHA256)
    assert np.isfinite(np.load(e2 / "guard_held" / "variant_jitter" / "kl_stock_v.npy")).all()


def test_ops_scrub_check_catches_a_planted_string(tmp_path):
    """T20's guard: the scrub check is clean on clean text and names a planted local path."""
    scrub = OPS / "scrub_check.sh"
    if not scrub.exists() or not shutil.which("bash"):
        pytest.skip("the scrub check script or bash is absent")
    (tmp_path / "clean.py").write_text("URL = 'gs://bucket/prefix'\n", encoding="utf-8")
    assert subprocess.run(["bash", str(scrub), str(tmp_path)], capture_output=True).returncode == 0
    planted = "/" + "home/someone/x"  # split so this file itself stays clean
    (tmp_path / "dirty.py").write_text(f"P = '{planted}'\n", encoding="utf-8")
    got = subprocess.run(["bash", str(scrub), str(tmp_path)], capture_output=True, text=True)
    assert got.returncode == 1 and "dirty.py:1:" in got.stdout
