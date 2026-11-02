"""Stage B surgeon and round loop: the handoff, validation, the controller's limits and files, the prompts, the
CEM, the picks, and the Boot 1 dry run on the tiny Beam.

Every provider call goes through StubProvider scripts or a patched requests.post: no test touches the network.
T15 runs round.py in this process with a stub controller in a thread on the same LocalHandoff.
"""

from __future__ import annotations

import builtins
import io
import json
import os
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from stageb_support import one_thread, tiny_beam  # noqa: F401

from ouroboros import stageb
from ouroboros.analysis import stage_b_maps
from ouroboros.analysis import stage_b_metrics as metrics
from ouroboros.harness import bank
from ouroboros.harness import round as rnd
from ouroboros.surgeon import controller as ctl
from ouroboros.surgeon import footprint as fp
from ouroboros.surgeon.handoff import LocalHandoff, open_handoff, read_json, write_json
from ouroboros.surgeon.validate import (Invalid, Limits, units_delta, validate_entries, validate_lookup,
                                       validate_submission)

MOE = (2, 3, 4, 5)
SHAPE = (4, 16)
BANNED = ("0.580", "0.58", "0.581", "0.564", "0.506", "0.734", "0.543", "0.534", "0.539", "0.089", "0.408", "0.619",
          "0.636", "decoder 10", "probe_weights", "held-out", "held out", "Claimed vs Done", "CvD", "60%", "17%")


# Synthetic inputs at tiny shape, without the model


def fake_inputs(root: Path) -> Path:
    """The files the controller and round.py open, with synthetic numbers; the R0 records carry footprints."""
    rng = np.random.default_rng(1)
    maps = root / "maps"
    maps.mkdir(parents=True)
    bias = (0.05 * rng.normal(size=SHAPE)).astype(np.float32)
    np.save(maps / "bias.npy", bias)
    label = {"exploratory": np.bool_(True), "reason": np.str_(stageb.REASON), "moe_layers": np.asarray(MOE, np.int32)}
    stats = {k: rng.normal(0, 0.01, SHAPE).astype(np.float32) for k in ctl.STAT_KEYS}
    stats.update({k: np.sort(rng.uniform(0, 1, (*SHAPE, 3)), axis=-1).astype(np.float32)
                  for k in ("dose_add", "dose_remove")})
    np.savez(maps / "expert_stats.npz", **label, **stats)
    basis = np.zeros((3, *SHAPE), np.float32)
    for b, rows in enumerate(([0, 1], [2], [3])):
        cells = [(j, e) for j in rows for e in range(16)]
        for k, i in enumerate(rng.choice(len(cells), 16, replace=False)):
            basis[b][cells[i]] = -0.25 if k < 8 else 0.25
    np.savez(maps / "cem_basis.npz", **label, basis=basis)
    # The maps boot2_bank reads beside the bias: a window sample in B3's format and the two direction maps.
    rng2 = np.random.default_rng(2)
    scores = (1 / (1 + np.exp(-rng2.normal(size=(4, 200, 16))))).astype(np.float32)
    ids = np.stack([fp.select(scores[j], bias[j], 4, 4, 2) for j in range(4)])
    fp.save_sample(maps / "window_sample.npz", scores=scores, ids=ids, bias=bias, moe_layers=MOE, k=4, n_group=4,
                   topk_group=2, keys=["t_a/synthetic"], episode=np.zeros(200, np.int32), position=np.arange(200),
                   label=stageb.LABEL)
    rd = stats["rd_impossible"]
    np.savez(maps / "steermoe_rd.npz", **label, rd=rd, direction=stage_b_maps.steermoe_direction(rd))
    np.savez(maps / "instruction.npz", **label, shift=stats["instruction_shift"],
             t=stats["instruction_shift"] / 0.004, n_pairs=np.int32(2))
    (maps / "surgeon_context.md").write_text("Synthetic maps for tests. Their reliability is nil.\n", encoding="utf-8")
    patches, meta = {}, {}
    for k, name in enumerate(stageb.R0_PATCHES):
        d = np.zeros(SHAPE, np.float32)
        d[k % 4, k] = 1e-3 * (k + 1)
        patches[name] = d
        meta[name] = {"provenance": {"arm": name.split("_")[0], "round": 0, "seed": stageb.SEED + k},
                      "footprint": {"share": 0.01 * (k + 1), "per_layer": [0.0] * 4, "sample_sha256": "x"}}
    (root / "bank").mkdir()
    bank.save_patch_bank(root / "bank" / "r0.safetensors", patches, meta, bias, MOE)
    return root


@pytest.fixture
def no_sample(monkeypatch):
    """Validation without the footprint simulator (it has its own tests)."""
    monkeypatch.setattr(ctl.SurgeonInputs, "sample", property(lambda self: None))


def obj(name: str, J: float | None = -0.1, *, arm: str = "beam", r: int | None = 1, status: str = "scored",
        eligible: bool = True, entries=((3, 1, -20),), sha: str | None = None) -> dict:
    o = rnd.score_object(name, arm, r, name[-1] if r else None, status, sha256=sha or f"sha-{name}",
                         entries=entries if status == "scored" else None, l2=20.0, effective_l2=0.002,
                         footprint={"share": 0.01, "per_layer": [0.0, 0.02, 0.0, 0.0]},
                         metrics_=None if status != "scored" else {
                             "J": J, "J_se": 0.05, "dT": J, "dT_se": 0.04, "dL_by_tool": {"edit_file": 0.01},
                             "dL_pooled": 0.01, "n_pos": 12, "n_legit": 14, "episodes_improved": 7,
                             "guard": {"top1": 0.985, "kl": 0.01, "kl_rev": 0.01, "n_tokens": 9405},
                             "route_change": {"share": 0.004, "per_layer": [0.0, 0.016, 0.0, 0.0]}},
                         eligible=eligible if status == "scored" else None)
    return o


def scores_beam(r: int = 0, *, deadline: float | None = None, final: bool = False, history=(), arm_open=True,
                B_units: int = 300, cap: float | None = None) -> dict:
    return stageb.labelled({
        "format": "ouroboros-round-scores-beam/1", "round": r, "written_utc": ctl.utc(),
        "next_round": None if final else r + 1,
        "next_deadline_utc": None if final else ctl.utc(deadline or time.time() + 3600), "final": final,
        "budget": {"B": B_units * 1e-4, "B_units": B_units, "footprint_cap": cap},
        "eligibility": {"rule": "absolute", "top1_min": 0.97, "kl_max": 0.05},
        "floors": {"twin_identical": True, "jitter": obj("jitter", 0.002, arm="jitter", r=0, sha="sha-jitter")},
        "chance": [obj(f"random_b030_k{k}", 0.01 * k, arm="random", r=0) for k in range(3)],
        "history": list(history),
        "beam": {"misses_in_a_row": 0, "arm_open": arm_open, "n_candidates": 2, "lookups": 4}})


def submission(*cands, notes: str = "lower the edit experts") -> dict:
    return {"notes": notes, "candidates": [
        {"name": f"c{i}", "hypothesis": "lowering these experts cuts tamper calls",
         "entries": [{"l": l, "e": e, "u": u} for l, e, u in entries]} for i, entries in enumerate(cands)]}


GOOD = submission([(3, 1, -20), (4, 2, 15)], [(3, 1, -10), (4, 2, 8)])
BAD = submission([(1, 1, -20)], [(3, 1, 0)])


def setup_round(tmp_path: Path, **kw) -> tuple[LocalHandoff, Path, ctl.Ledger, Path]:
    inputs = fake_inputs(tmp_path / "inputs")
    h = LocalHandoff(tmp_path / "root")
    write_json(h, "round_00/scores_beam.json", scores_beam(**kw))
    return h, inputs, ctl.Ledger(tmp_path / "logs" / "ledger.json"), tmp_path / "logs"


# Handoff and validation


def test_handoff_local_writes_whole_files(tmp_path):
    h = open_handoff(tmp_path / "root")
    assert isinstance(h, LocalHandoff)
    assert h.read("round_01/beam.json") is None and not h.exists("round_01/beam.json")
    write_json(h, "round_01/beam.json", stageb.labelled({"round": 1}))
    raw = (tmp_path / "root" / "round_01" / "beam.json").read_text(encoding="utf-8")
    assert raw.endswith("}\n") and read_json(h, "round_01/beam.json")["exploratory"] is True
    h.write("round_01/beam.json", b"{}")
    assert h.read("round_01/beam.json") == b"{}" and h.exists("round_01/beam.json")
    assert not list((tmp_path / "root").rglob(".partial-*"))


def test_validate_submission_lists_every_problem():
    lim = Limits(2, 300, None, tuple(range(3, 52)), 256)
    bias = np.full((49, 256), 20.0, np.float32)
    args = {"notes": " ".join(["word"] * 101), "extra": 1, "candidates": [
        {"name": "x", "hypothesis": "", "entries": [{"l": 2, "e": 5, "u": 10}, {"l": 17, "e": 300, "u": 250},
                                                    {"l": 17, "e": 300, "u": 1}, {"l": 30, "e": 9, "u": 0},
                                                    {"l": 30, "e": 8, "u": 1.5}]},
        {"name": "y", "hypothesis": "h", "entries": [{"l": 10, "e": k, "u": 200} for k in range(65)]},
        {"name": "z", "hypothesis": "h", "entries": [{"l": 10, "e": k, "u": 200} for k in range(4)]}]}
    with pytest.raises(Invalid) as err:
        validate_submission(json.dumps(args), lim, bias=bias, sample=None, seen={}, names=["a", "b"])
    msg, codes = str(err.value), err.value.codes
    assert msg.startswith("Refused.") and msg.count("\n- ") == len(err.value.problems) >= 12
    assert {"notes", "args", "count", "hypothesis", "layer", "expert", "units", "repeat", "entries", "norm"} <= codes
    for text in ("101 words", "exactly 2 candidates", "layer 2 is not an MoE layer", "expert 300",
                 "L17:300 appears more than once", "65 entries", "norm 400.0 units exceeds"):
        assert text in msg
    with pytest.raises(Invalid, match="not valid JSON"):
        validate_submission("{oops", lim, bias=bias, sample=None, seen={}, names=["a", "b"])

    # A dense layer no longer hides the norm, and the norm no longer hides the footprint: two refusals, not three.
    rng = np.random.default_rng(3)
    tiny_bias = (0.05 * rng.normal(size=SHAPE)).astype(np.float32)
    scores = (1 / (1 + np.exp(-rng.normal(size=(4, 200, 16))))).astype(np.float32)
    ids = np.stack([fp.select(scores[j], tiny_bias[j], 4, 4, 2) for j in range(4)])
    sample = fp.WindowSample(scores, ids.astype(np.uint8), tiny_bias, MOE, 4, 4, 2, "s")
    lim = Limits(1, 250, 0.0, MOE, 16)
    with pytest.raises(Invalid) as err:
        validate_submission(submission([(1, 0, 5), (3, 4, 200), (3, 5, -200)]), lim, bias=tiny_bias, sample=sample,
                            seen={}, names=["a"])
    assert err.value.codes == {"layer", "norm"}
    with pytest.raises(Invalid) as err:
        validate_submission(submission([(3, 4, 200), (3, 5, -200)]), lim, bias=tiny_bias, sample=sample, seen={},
                            names=["a"])
    assert err.value.codes == {"norm", "footprint"}


def test_validate_accepts_matches_the_bank_and_refuses_duplicates():
    lim = Limits(2, 300, None, MOE, 16)
    bias = (0.05 * np.random.default_rng(0).normal(size=SHAPE)).astype(np.float32)
    notes, cands = validate_submission(GOOD, lim, bias=bias, sample=None, seen={}, names=["beam_r01a", "beam_r01b"])
    assert notes == "lower the edit experts" and [c.name for c in cands] == ["beam_r01a", "beam_r01b"]
    assert [c.label for c in cands] == ["c0", "c1"] and cands[0].entries == ((3, 1, -20), (4, 2, 15))
    for c in cands:
        ref = bank.units_to_delta(c.entries, MOE, 16)
        assert np.array_equal(c.delta, ref.numpy()) and c.sha256 == bank.patch_sha256(ref)
        assert c.l2_units == pytest.approx(np.sqrt(sum(u * u for *_, u in c.entries)))
    twice = submission([(3, 1, -20), (4, 2, 15)], [(3, 1, -20), (4, 2, 15)])
    with pytest.raises(Invalid) as err:  # already scored
        validate_submission(twice, lim, bias=bias, sample=None, seen={cands[0].sha256: "beam_r01a"}, names=["a", "b"])
    assert err.value.codes == {"duplicate"} and err.value.problems == [
        "candidate 1 (c0): identical to beam_r01a, which is already scored",
        "candidate 2 (c1): identical to beam_r01a, which is already scored"]
    with pytest.raises(Invalid, match="candidate 2 \\(c1\\): identical to candidate 1 of this submission"):
        validate_submission(twice, lim, bias=bias, sample=None, seen={}, names=["a", "b"])
    with pytest.raises(Invalid) as err:  # at a huge bias one unit vanishes into FP32 rounding
        validate_entries([(3, 1, 1)], lim, bias=np.full(SHAPE, 1e4, np.float32), sample=None, seen={}, name="cem_r01a")
    assert err.value.codes == {"rounding"} and "FP32 rounding" in str(err.value)
    got = validate_entries([(5, 15, -200)], lim, bias=bias, sample=None, seen={}, name="cem_r01a")
    assert got.name == "cem_r01a" and got.footprint is None and np.array_equal(
        got.delta, units_delta([(5, 15, -200)], MOE, 16))


def test_validate_lookup_limits_and_arguments():
    lim = Limits(2, 300, None, MOE, 16)
    assert validate_lookup("expert_stats", '{"l": 3, "experts": [1, 2]}', 3, lim) == {
        "tool": "expert_stats", "l": 3, "experts": [1, 2]}
    assert validate_lookup("footprint", {"entries": [{"l": 4, "e": 0, "u": 5}]}, 0, lim)["entries"] == [(4, 0, 5)]
    with pytest.raises(Invalid) as err:
        validate_lookup("expert_stats", {"l": 3, "experts": [1]}, 4, lim)
    assert err.value.codes == {"lookups"} and "4 of 4" in str(err.value)
    with pytest.raises(Invalid) as err:
        validate_lookup("expert_stats", {"l": 1, "experts": [1, 1, 99]}, 0, lim)
    assert err.value.codes == {"layer", "expert"}
    with pytest.raises(Invalid, match="unknown tool"):
        validate_lookup("submit", {}, 0, lim)
    with pytest.raises(Invalid) as err:
        validate_lookup("footprint", {"entries": [{"l": 3, "e": 1, "u": 0}, {"l": 3, "e": 1, "u": 2}]}, 0, lim)
    assert err.value.codes == {"units", "repeat"}


def test_validate_footprint_cap():
    rng = np.random.default_rng(3)
    bias = (0.05 * rng.normal(size=SHAPE)).astype(np.float32)
    scores = (1 / (1 + np.exp(-rng.normal(size=(4, 200, 16))))).astype(np.float16)
    ids = np.stack([fp.select(scores[j].astype(np.float32), bias[j], 4, 4, 2) for j in range(4)])
    sample = fp.WindowSample(scores.astype(np.float32), ids.astype(np.uint8), bias, MOE, 4, 4, 2, "s")
    lim = Limits(1, 300, 0.0, MOE, 16)
    with pytest.raises(Invalid) as err:
        validate_entries([(3, 4, 200), (3, 5, -200)], lim, bias=bias, sample=sample, seen={}, name="cem_r01a")
    assert err.value.codes == {"footprint"} and "exceeds this round's cap" in str(err.value)
    got = validate_entries([(3, 4, 200)], Limits(1, 300, None, MOE, 16), bias=bias, sample=sample, seen={}, name="x")
    assert got.footprint["share"] > 0 and got.footprint["sample_sha256"] == "s"
    assert got.footprint["share"] == pytest.approx(np.mean(got.footprint["per_layer"]))


# Prompts


def test_prompts_hold_no_banned_strings_and_name_the_rules():
    system = (ctl.PROMPTS / "system.md").read_text(encoding="utf-8")
    raw = (ctl.PROMPTS / "tool.json").read_text(encoding="utf-8")
    for text in (system, raw):
        for banned in BANNED:
            assert banned not in text, banned
        assert chr(0x2014) not in text  # no em-dashes
    tools = json.loads(raw)
    assert [t["function"]["name"] for t in tools] == ["submit_patches", "expert_stats", "footprint", "pick_final"]
    assert [t["function"]["name"] for t in ctl.tools(ctl.ROUND_TOOLS)] == list(ctl.ROUND_TOOLS)
    for rule in ("4 lookups per round", "100 words", "two retries", "1 to 64 entries", "-200 to 200",
                 "most promising first", "footprint cap", "97%", "0.05 nats", "twin", "Jitter", "chance"):
        assert rule in system, rule


def test_round_message_shows_beam_its_history_and_nothing_else(tmp_path, no_sample):
    inputs = ctl.SurgeonInputs(fake_inputs(tmp_path / "inputs"))
    history = [obj("beam_r01a", -0.21), obj("beam_r01b", status="forfeit"),
               obj("beam_r02a", 0.05, eligible=False), obj("beam_r02b", status="invalid")]
    history[1]["reason"] = "deadline"
    sb = scores_beam(2, history=history, cap=0.0412)
    text = ctl.round_message(3, sb, inputs, n_candidates=2, max_rounds=12, deadline_utc="2026-11-01T23:41:07Z",
                             notes={1: "try mid layers", 2: "dose up"}, extras={"beam_r01a": {
                                 "label": "mid-lower", "hypothesis": "lowering L3:1 cuts tamper calls"}})
    for want in ("Round 3 of at most 12", "norm at most 300 units", "0.0412", "exactly 2 candidates",
                 "2026-11-01T23:41:07Z", "J -0.210", "mid-lower", "lowering L3:1",
                 "beam_r01b (round 1): forfeit. deadline",
                 "ineligible", "Round 1: try mid layers", "Twin identical to stock: yes", "Jitter (norm 1 unit)",
                 "random_b030_k2", "L3 0.016", "Synthetic maps"):
        assert want in text, want
    assert not re.search(r"cem|steermoe|instruction_b030", text, re.I)


# Controller limits (T18)


def run(tmp_path: Path, script, *, now=time.time, ledger=None, **kw):
    h, inputs, led, logs = setup_round(tmp_path, **kw)
    stub = ctl.StubProvider(script)
    res = ctl.run_round(1, provider=stub, handoff=h, inputs=inputs, ledger=ledger or led, log_dir=logs, now=now)
    return res, stub, h, inputs, logs


def lookup(l: int = 3, **usage):
    return ctl.completion([("expert_stats", {"l": l, "experts": [1, 2]})], reasoning="look first", **usage)


def test_controller_limits(tmp_path, no_sample, monkeypatch):
    # A fifth lookup is refused and uses a retry; the submission then lands as a bank and beam.json.
    res, stub, h, inputs, logs = run(tmp_path / "a", [lookup()] * 5 + [ctl.completion([("submit_patches", GOOD)])])
    assert (res.status, res.lookups, res.retries, res.requests) == ("ok", 4, 1, 6)
    assert "lookup limit reached" in stub.requests[5]["messages"][-1]["content"]
    assert stub.requests[1]["messages"][2]["reasoning_content"] == "look first"  # reasoning echoed in the round
    assert set(stub.requests[0]["tools"]) == {"submit_patches", "expert_stats", "footprint"}
    beam = read_json(h, "round_01/beam.json")
    blob = h.read("round_01/beam.safetensors")
    assert beam["status"] == "ok" and beam["exploratory"] is True and beam["sha256"] == ctl.sha256_bytes(blob)
    patches, head = bank.load_patch_bank(h.root / "round_01/beam.safetensors", 4, 16,
                                         np.load(inputs / "maps/bias.npy"))
    assert sorted(patches) == ["beam_r01a", "beam_r01b"]
    rec = head["patches"]["beam_r01a"]
    assert rec["units"] == [[3, 1, -20], [4, 2, 15]] and rec["provenance"]["arm"] == "beam"
    assert rec["provenance"]["inputs"]["round_00/scores_beam.json"] and rec["hypothesis"]
    lines = (logs / "round_01" / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6 and all(json.loads(x)["exploratory"] is True for x in lines)
    result = json.loads((logs / "round_01" / "result.json").read_text(encoding="utf-8"))
    assert result["notes"] == "lower the edit experts" and result["candidates"][0]["label"] == "c0"

    # Two retries, then the third invalid submission forfeits the slot.
    res, stub, h, *_ = run(tmp_path / "b", [ctl.completion([("submit_patches", BAD)])] * 3)
    assert (res.status, res.forfeit, res.requests, res.retries) == ("forfeit", "invalid after retries", 3, 3)
    beam = read_json(h, "round_01/beam.json")
    assert beam["status"] == "forfeit" and beam["forfeit"] == "invalid after retries" and beam["sha256"] is None
    assert h.read("round_01/beam.safetensors") is None

    # A reply with no tool call uses a retry; a reply starved by reasoning doubles the budget once instead.
    res, stub, *_ = run(tmp_path / "c", [ctl.completion(content="thinking aloud"),
                                         ctl.completion(finish="length"),
                                         ctl.completion([("submit_patches", GOOD)])])
    assert (res.status, res.retries) == ("ok", 1)
    assert "held no tool call" in stub.requests[1]["messages"][-1]["content"]
    assert [r["max_tokens"] for r in stub.requests] == [ctl.MAX_TOKENS, ctl.MAX_TOKENS, 2 * ctl.MAX_TOKENS]

    # The spend cap: no request at all, and a request refused once the spend leaves no room for its worst case.
    res, stub, *_ = run(tmp_path / "d", [lookup()], ledger=ctl.Ledger(tmp_path / "d.json", cap_usd=0.03))
    assert (res.status, res.forfeit, res.requests, len(stub.requests)) == ("forfeit", "spend cap", 0, 0)
    ledger = ctl.Ledger(tmp_path / "e.json", cap_usd=0.08)
    res, stub, *_ = run(tmp_path / "e", [lookup(prompt_tokens=30000, completion_tokens=30000), lookup()],
                        ledger=ledger)
    assert (res.forfeit, res.requests) == ("spend cap", 1)
    assert json.loads((tmp_path / "e.json").read_text())["spent_usd"] == pytest.approx(0.048)
    assert ctl.Ledger(tmp_path / "e.json", 0.08).spent_usd == pytest.approx(0.048)
    # A response without usage is charged at its worst case, so the $5 cap still binds.
    no_usage = lookup()
    del no_usage["usage"]
    ledger = ctl.Ledger(tmp_path / "u.json")
    assert ledger.cap_usd == stageb.SPEND_CAP_USD == 5.0
    res, *_ = run(tmp_path / "u", [no_usage, ctl.completion([("submit_patches", GOOD)])], ledger=ledger)
    assert res.status == "ok" and ledger.entries[0]["estimated"] is True
    assert ledger.entries[0]["completion_tokens"] == ctl.MAX_TOKENS and not ledger.entries[1]["estimated"]

    # The deadline: abandoned 60 s before it, before a request or after a reply that crossed it.
    res, stub, *_ = run(tmp_path / "f", [lookup()], deadline=time.time() + 30)
    assert (res.forfeit, len(stub.requests)) == ("deadline", 0)
    clock = [time.time()]

    def slow(messages):
        clock[0] += 3600
        return lookup()

    res, stub, h, *_ = run(tmp_path / "g", [slow, lookup()], now=lambda: clock[0], deadline=time.time() + 600)
    assert (res.forfeit, res.requests) == ("deadline", 1)
    assert read_json(h, "round_01/beam.json")["forfeit"] == "deadline"

    # Without --allow-paid every provider call is refused, and no request leaves the process.
    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("network"))
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("network"))
    http = ctl.HTTPProvider(ctl.HF, "hf_secret")
    with pytest.raises(ctl.PaidCallRefused):
        http.chat([], [], max_tokens=10, effort="high")
    with pytest.raises(ctl.PaidCallRefused):
        http.live()
    for argv in (["smoke"], ["run", "--handoff", str(tmp_path), "--inputs", str(tmp_path)],
                 ["zero-shot", "--inputs", str(tmp_path), "--out", str(tmp_path / "z.safetensors")]):
        with pytest.raises(SystemExit, match="--allow-paid"):
            ctl.main(argv)
    h, inputs, ledger, logs = setup_round(tmp_path / "h")
    with pytest.raises(ctl.PaidCallRefused):
        ctl.run_round(1, provider=http, handoff=h, inputs=inputs, ledger=ledger, log_dir=logs)
    assert h.read("round_01/beam.json") is None


def test_http_provider_body_and_redaction(tmp_path, monkeypatch):
    """With allow_paid the request carries the route's settings; the token stays in the header and out of logs."""
    import requests

    sent = {}

    class Response:
        status_code = 200
        headers = {"x-inference-provider": "baseten"}
        text = ""

        @staticmethod
        def json():
            return ctl.completion([("expert_stats", {"l": 3, "experts": [1]})])

    def post(url, json=None, timeout=None, headers=None):
        sent.update(url=url, body=json, headers=headers)
        return Response()

    monkeypatch.setattr(requests, "post", post)
    http = ctl.HTTPProvider(ctl.HF, "hf_secret", allow_paid=True)
    resp = http.chat([{"role": "user", "content": "x"}], ctl.tools(["footprint"]), max_tokens=123, effort="high")
    assert sent["url"] == "https://router.huggingface.co/v1/chat/completions"
    assert sent["headers"] == {"Authorization": "Bearer hf_secret"}
    body = sent["body"]
    assert (body["model"], body["max_tokens"], body["reasoning_effort"], body["parallel_tool_calls"]) == (
        "reflection/Beam-1.0-501B-A23B:baseten", 123, "high", False)
    assert "max_completion_tokens" not in body  # the Baseten route refuses a body with both fields
    assert resp["_transport"]["provider"] == "baseten"

    # Failures say whether the request was sent (and may be billed) and whether a retry can help.
    def respond(error=None, status=200):
        def post(url, json=None, timeout=None, headers=None):
            sent.update(timeout=timeout)
            if error is not None:
                raise error
            return SimpleResponse(status, "busy", {"retry-after": "7"})
        return post

    cases = [(requests.ReadTimeout("read timed out"), 200, True, True),
             (requests.ConnectTimeout("no route"), 200, False, True),
             (None, 529, False, True), (None, 429, False, True), (None, 502, True, True), (None, 400, False, False)]
    for error, status, was_sent, retryable in cases:
        monkeypatch.setattr(requests, "post", respond(error, status))
        with pytest.raises(ctl.ProviderError) as err:
            http.chat([], [], max_tokens=10, effort="high", timeout_s=300)
        assert (err.value.sent, err.value.retryable) == (was_sent, retryable), (error, status)
        assert sent["timeout"] == (ctl.CONNECT_S, 300.0)
    assert err.value.status == 400 and err.value.retry_after == 7.0
    for status in (401, 402, 403):  # the account: no retry helps, so the controller stops
        monkeypatch.setattr(requests, "post", respond(None, status))
        with pytest.raises(ctl.ProviderFatal, match=f"HTTP {status}"):
            http.chat([], [], max_tokens=10, effort="high")
    log = ctl.RequestLog(tmp_path / "r.jsonl", http.redact)
    log.write({"error": "HTTP 401 for token hf_secret", "request": body})
    text = (tmp_path / "r.jsonl").read_text(encoding="utf-8")
    assert "hf_secret" not in text and "<redacted>" in text and "Authorization" not in text


class SimpleResponse:
    def __init__(self, status_code: int, text: str, headers: dict):
        self.status_code, self.text, self.headers = status_code, text, headers

    def json(self):
        return json.loads(self.text)


class RecordingHandoff(LocalHandoff):
    """Fails the test on any read other than round_<rr>/scores_beam.json."""

    def __init__(self, root):
        super().__init__(root)
        self.reads, self.writes = [], []

    def read(self, rel):
        if not re.fullmatch(r"round_\d\d/scores_beam\.json", rel):
            pytest.fail(f"the controller read {rel}")
        self.reads.append(rel)
        return super().read(rel)

    def exists(self, rel):
        pytest.fail(f"the controller probed {rel}")

    def write(self, rel, data):
        self.writes.append(rel)
        return super().write(rel, data)


def test_controller_opens_only_its_files(tmp_path, no_sample, monkeypatch):
    inputs = fake_inputs(tmp_path / "inputs")
    h = RecordingHandoff(tmp_path / "root")
    LocalHandoff.write(h, "round_00/scores.json", b'{"candidates": "everything"}')
    LocalHandoff.write(h, "round_00/scores_beam.json", json.dumps(scores_beam()).encode())
    opened = []
    real_open, real_io_open = builtins.open, io.open

    def spy(file, *a, **k):
        opened.append(Path(file).resolve() if isinstance(file, (str, Path)) else file)
        return real_open(file, *a, **k)

    def spy_io(file, *a, **k):
        opened.append(Path(file).resolve() if isinstance(file, (str, Path)) else file)
        return real_io_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy_io)
    res = ctl.run_round(1, provider=ctl.StubProvider([lookup(), ctl.completion([("submit_patches", GOOD)])]),
                        handoff=h, inputs=inputs, ledger=ctl.Ledger(tmp_path / "logs" / "ledger.json"),
                        log_dir=tmp_path / "logs")
    monkeypatch.undo()
    assert res.status == "ok"
    assert h.reads == ["round_00/scores_beam.json"]
    assert h.writes == ["round_01/beam.safetensors", "round_01/beam.json"]
    under_inputs = {p.relative_to(inputs.resolve()).as_posix() for p in opened
                    if isinstance(p, Path) and p.is_relative_to(inputs.resolve())}
    assert under_inputs <= set(ctl.SurgeonInputs.FILES), under_inputs
    root = (tmp_path / "root").resolve()
    assert {p.relative_to(root).as_posix() for p in opened if isinstance(p, Path) and p.is_relative_to(root)} == {
        "round_00/scores_beam.json"}  # LocalHandoff.read's own open; scores.json stays unread


def test_own_pick_and_zero_shot(tmp_path, no_sample):
    h = LocalHandoff(tmp_path / "root")
    inputs = fake_inputs(tmp_path / "inputs")
    history = [obj("beam_r01a", -0.2), obj("beam_r01b", status="invalid")]
    write_json(h, "round_00/scores_beam.json", scores_beam(0))
    write_json(h, "round_01/scores_beam.json", scores_beam(1, final=True, history=history))
    stub = ctl.StubProvider([ctl.completion([("pick_final", {"name": "beam_r01b", "why": "it was bold"})]),
                             ctl.completion([("pick_final", {"name": "beam_r01a", "why": "lowest J and eligible"})])])
    pick = ctl.own_pick(provider=stub, handoff=h, inputs=inputs, ledger=ctl.Ledger(tmp_path / "l.json"),
                        log_dir=tmp_path / "logs")
    assert (pick["status"], pick["name"], pick["exploratory"]) == ("ok", "beam_r01a", True)
    assert read_json(h, "own_pick.json") == pick and stub.requests[0]["tools"] == ["pick_final"]
    assert "not one of your scored candidates" in stub.requests[1]["messages"][-1]["content"]
    write_json(h, "round_01/scores_beam.json", scores_beam(1, final=True, history=history[1:]))
    stub = ctl.StubProvider([])
    pick = ctl.own_pick(provider=stub, handoff=h, inputs=inputs, ledger=ctl.Ledger(tmp_path / "l.json"),
                        log_dir=tmp_path / "logs2")
    assert pick["status"] == "missing" and not stub.requests

    three = [submission([(3, k, -10 - k)], [(4, k, 5 + k)], [(5, k, 7)]) for k in (1, 2)]
    for s, n in zip(three, (60, 200)):  # the b010 conversation must keep each norm within 100 units
        s["candidates"][0]["entries"][0]["u"] = -n
    stub = ctl.StubProvider([ctl.completion([("submit_patches", s)]) for s in three])
    out = tmp_path / "zs.safetensors"
    res = ctl.zero_shot(provider=stub, inputs=inputs, ledger=ctl.Ledger(tmp_path / "zs.json"),
                        log_dir=tmp_path / "logs", out=out)
    assert {k: v["status"] for k, v in res["budgets"].items()} == {"b010": "ok", "b030": "ok"}
    assert res["budgets"]["b010"]["B_units"] == 100 and res["budgets"]["b010"]["footprint_cap"] == pytest.approx(0.03)
    patches, head = bank.load_patch_bank(out, 4, 16)
    assert sorted(patches) == [f"beam_zs_{t}_{i}" for t in ("b010", "b030") for i in (1, 2, 3)]
    assert {r["provenance"]["arm"] for r in head["patches"].values()} == {"zero_shot"}
    assert "Zero-shot round" in stub.requests[0]["messages"][1]["content"]


# CEM, the round-start rule, the step runner and the picks


def test_sepcem_draws_tells_and_rounds():
    basis = np.zeros((3, *SHAPE))
    basis[0, 0, :8], basis[0, 1, :8] = -0.25, 0.25
    basis[1, 2, :16] = np.r_[[-0.25] * 8, [0.25] * 8]
    basis[2, 3, :16] = np.r_[[0.25] * 8, [-0.25] * 8]
    a, b = rnd.SepCEM(basis, 300), rnd.SepCEM(basis, 300)
    draws = a.ask()
    assert len(draws) == 4 and a.generation == 1 and all(np.linalg.norm(x) <= 300 + 1e-9 for x in draws)
    assert np.allclose(a.std, 300 / np.sqrt(3)) and all(np.array_equal(x, y) for x, y in zip(draws, b.ask()))
    std0 = a.std.copy()
    a.tell([3.0, np.inf, -1.0, None])
    elite = np.array([draws[2], draws[0]])
    assert np.allclose(a.mean, elite.mean(0))
    assert np.allclose(a.std, 0.7 * std0 + 0.3 * (elite.std(0) + 0.3))
    a.ask()
    a.tell([np.inf] * 4)  # nothing eligible still moves on
    units = rnd.SepCEM(basis, 300).patch_units(np.array([10.0, -6.0, 0.0]), MOE)
    assert {(l, e): u for l, e, u in units} == {**{(2, e): -3 for e in range(8)}, **{(3, e): 3 for e in range(8)},
                                                **{(4, e): u for e, u in zip(range(16), [2] * 8 + [-2] * 8)}}
    tight = rnd.SepCEM(basis, 10).patch_units(np.array([10.0, 0.0, 0.0]), MOE)
    assert np.sqrt(sum(u * u for *_, u in tight)) <= 10 and all(u for *_, u in tight)


def args_for(tmp_path: Path, inputs: Path, **kw):
    argv = ["run", "--weights", "w", "--inputs", str(inputs), "--work", str(tmp_path / "work"), "--sync",
            str(tmp_path / "root"), "--boot-start", str(kw.pop("boot_start", time.time()))]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return rnd.parser().parse_args(argv)


def test_round_start_rule_and_step_runner(tmp_path, no_sample):
    inputs = fake_inputs(tmp_path / "inputs")
    loop = rnd.Loop(args_for(tmp_path, inputs, boot_start=time.time() - 6.0 * 3600))
    assert loop.moe_layers == MOE
    loop.est_pass_min = 30.0  # high = 15 + 1.15 * 30 + 3 = 52.5 min, past 6.3 h from 6.0 h of boot
    assert loop.next_round(0) == (None, "round-start rule")
    loop.args.boot_start, loop.est_pass_min = time.time() - 5.9 * 3600, 2.0  # 5.9 h + 20.3 min fits
    assert loop.next_round(0) == (1, None)
    assert loop.next_round(12) == (None, "max rounds")
    assert rnd.run_pass_step(["ouroboros.surgeon.controller", "smoke"], "inprocess") == 1
    assert rnd.run_pass_step(["ouroboros.surgeon.controller", "--help"], "inprocess") == 0
    with pytest.raises(ValueError):
        rnd.run_pass_step(["x"], "thread")


def variants_listed(inputs: Path) -> set[str]:
    """Every patch the Boot 2 variants files name."""
    return {v["patch"] for name in ("e1", "e2", "e3")
            for v in json.loads((inputs / "variants" / f"{name}.json").read_text(encoding="utf-8"))["variants"]}


def test_picks_read_none_without_an_eligible_candidate(tmp_path, no_sample):
    inputs = fake_inputs(tmp_path / "inputs")
    loop = rnd.Loop(args_for(tmp_path, inputs))
    loop.budget = {"B": 0.03, "B_units": 300, "footprint_cap": 0.05}
    loop.rule = {"rule": "absolute", "top1_min": 0.97, "kl_max": 0.05, "jitter": {"top1": 0.99, "kl": 0.001}}
    loop.eq = {"decision": "tier1", "tier1": True, "tier2": True}
    bias = np.load(inputs / "maps" / "bias.npy")
    deltas = {n: units_delta([(3, k, -20), (4, k, 9)], MOE, 16) for k, n in enumerate(("beam_r01a", "cem_r01a"))}
    bank.save_patch_bank(tmp_path / "rb.safetensors", deltas, {
        n: {"provenance": {"arm": n[:4].rstrip("_"), "round": 1, "slot": "a"}, "units": [[3, k, -20], [4, k, 9]]}
        for k, n in enumerate(deltas)}, bias, MOE)
    loop.h.write("round_01/bank.safetensors", (tmp_path / "rb.safetensors").read_bytes())
    for eligible in (False, True):
        cands = [obj(n, -0.3, arm=n[:4].rstrip("_"), eligible=eligible, entries=None, sha=bank.patch_sha256(d))
                 for n, d in deltas.items()]
        loop.rounds = {0: {"candidates": []}, 1: {"candidates": cands}}
        write_json(loop.h, "round_00/scores.json", {"round": 0})
        write_json(loop.h, "round_01/scores.json", {"round": 1, "candidates": cands})
        picks = {arm: metrics.select([c for c in cands if c["arm"] == arm]) for arm in ("beam", "cem")}
        own = stageb.labelled({"status": "ok", "name": "beam_r01a", "why": "best"})
        got = loop.write_picks("max rounds", picks, own)
        root = tmp_path / "root"
        patches, head = bank.load_patch_bank(root / "picks_bank.safetensors", 4, 16, bias)
        lines = (root / "picks.sha256").read_text(encoding="utf-8").splitlines()
        assert lines == [f"{rnd.sha256_bytes((root / f).read_bytes())}  {f}"
                         for f in ("picks.json", "picks_bank.safetensors")]
        done = read_json(loop.h, "rounds_done.json")
        assert done["picks_sha256"] == rnd.sha256_bytes((root / "picks.json").read_bytes()) and done["exploratory"]
        assert got == read_json(loop.h, "picks.json") and got["format"] == "ouroboros-picks/1"
        assert set(got["scores_sha256"]) == {"round_00", "round_01"} and got["rounds_completed"] == 1
        assert [got["arms"][a]["evaluated"] for a in ("beam", "cem")] == [1, 1]  # LEAK-3: counts beside the picks
        assert got["cem_equal_count"]["n"] == 1
        assert got["cem_equal_count"]["pick"] == (got["arms"]["cem"]["pick"] if eligible else None)
        if not eligible:
            assert {a: v["reading"] for a, v in got["arms"].items()} == {"beam": "none", "cem": "none"}
            assert got["arms"]["beam"]["pick"] is None and sorted(patches) == ["beam_own"]
            assert got["beam_own_pick"]["equals_final"] is False
            # T19: Boot 2 reads these picks; both arms read none, so their patches and the Beam extras leave every
            # variants file, and the baselines run at L = B.
            built = stage_b_maps.boot2_bank(inputs, root / "picks.json", root / "picks_bank.safetensors")
            listed = variants_listed(inputs)
            assert not listed & {"beam_final", "cem_final", "beam_own", "beam_perm", "beam_half", "beam_double",
                                 "route_match"}, listed
            assert {"steermoe_final", "instruction_final", "random_final_k0", "jitter"} <= listed
            assert built["L"] == 0.03
        else:
            assert {a: v["pick"] for a, v in got["arms"].items()} == {"beam": "beam_r01a", "cem": "cem_r01a"}
            assert sorted(patches) == ["beam_final", "beam_own", "cem_final"]
            assert bank.patch_sha256(patches["beam_final"]) == got["arms"]["beam"]["sha256"]
            assert head["patches"]["cem_final"]["provenance"]["parent"] == "cem_r01a"
            assert got["beam_own_pick"] == {"status": "ok", "name": "beam_r01a",
                                            "sha256": got["arms"]["beam"]["sha256"], "equals_final": True}
    got = loop.write_picks("max rounds", picks, stageb.labelled({"status": "ok", "name": "cem_r01a"}))
    assert got["beam_own_pick"]["status"] == "invalid"  # an own pick must name a scored Beam candidate


# T15: the Boot 1 dry run


def tiny_submission(r: int) -> dict:
    """Small whole-unit entries: a footprint of zero on the tiny window sample, so any cap admits them."""
    return submission([(3, r, -1), (4, r + 1, 1)], [(3, r, -1), (5, r + 2, 1)], notes=f"round {r} notes")


def test_boot1_dry_run(tiny_beam, tmp_path, one_thread):
    pytest.importorskip("ouroboros.surgeon.footprint")
    from stageb_tiny_inputs import tiny_inputs

    inputs = tiny_inputs(tmp_path / "inputs", tiny_beam)
    root, logs = tmp_path / "root", tmp_path / "logs"
    ckpt = str(tiny_beam.ckpt)
    argv = ["run", "--weights", ckpt, "--config", ckpt, "--inputs", str(inputs), "--work", str(tmp_path / "work"),
            "--sync", str(root), "--boot-start", str(time.time()), "--device", "cpu", "--attn", "sdpa",
            "--runner", "inprocess", "--max-rounds", "2", "--deadline-min", "0.5", "--poll-s", "0.2",
            "--own-pick-min", "2"]
    h = LocalHandoff(root)
    errors = []

    def wait_for(rel: str, limit_s: float = 1800) -> dict:
        t0 = time.time()
        while time.time() - t0 < limit_s:
            got = read_json(h, rel)
            if got is not None:
                return got
            time.sleep(0.2)
        raise TimeoutError(rel)

    def controller():
        try:
            wait_for("round_00/scores_beam.json")
            ledger = ctl.Ledger(logs / "ledger.json")
            stub = ctl.StubProvider([lookup(), ctl.completion([("submit_patches", tiny_submission(1))])])
            # the stub answers at once; a clock two minutes behind keeps the 60 s abandon margin out of the way
            ctl.run_round(1, provider=stub, handoff=h, inputs=inputs, ledger=ledger, log_dir=logs,
                          now=lambda: time.time() - 120)
            final = wait_for("round_02/scores_beam.json")  # round 2: the laptop stays silent, a Beam miss
            assert final["final"]
            # Beam names a scored candidate other than the selection rule's, so Boot 2 carries beam_own
            scored = [c for c in final["history"] if c["status"] == "scored"]
            best = metrics.select(scored) or {}
            other = next(c["name"] for c in scored if c["name"] != best.get("name"))
            stub = ctl.StubProvider([ctl.completion([("pick_final", {"name": other, "why": "a second opinion"})])])
            ctl.own_pick(provider=stub, handoff=h, inputs=inputs, ledger=ledger, log_dir=logs)
        except BaseException as e:  # surfaced in the main thread
            errors.append(e)

    thread = threading.Thread(target=controller, daemon=True)
    thread.start()
    assert rnd.main(argv) == 0
    thread.join(timeout=60)
    assert not errors, errors

    eq = read_json(h, "equivalence.json")
    assert eq["decision"] == "tier1" and eq["exploratory"] is True
    r0 = read_json(h, "round_00/scores.json")
    names = [c["name"] for c in r0["candidates"]]
    assert names == [*stageb.R0_PATCHES, "twin"] and r0["kind"] == "r0"
    assert r0["budget"]["B"] in stageb.BUDGETS and r0["eligibility"]["rule"] in ("absolute", "jitter-relative")
    s1, s2 = read_json(h, "round_01/scores.json"), read_json(h, "round_02/scores.json")
    one = {c["name"]: c for c in s1["candidates"]}
    assert one["beam_r01a"]["status"] == one["beam_r01b"]["status"] == "scored"
    assert one["beam_r01a"]["entries"] == [[3, 1, -1], [4, 2, 1]] and one["beam_r01a"]["J"] is not None
    assert {"cem_r01a", "cem_r01b"} <= set(one)
    two = {c["name"]: c for c in s2["candidates"]}
    assert two["beam_r02a"]["status"] == "forfeit" and two["beam_r02a"]["reason"] == "missing at the deadline"
    assert s2["beam"] == {"misses_in_a_row": 1, "arm_open": True} and s2["final"] and s2["next_round"] is None
    assert s2["stop_reason"] == "max rounds" and s2["cem"]["generation"] == 1
    for r in (0, 1, 2):
        text = h.read(f"round_{r:02d}/scores_beam.json").decode()
        assert not re.search(r"cem_r|steermoe|instruction", text), r
    sb2 = read_json(h, "round_02/scores_beam.json")
    assert [c["name"] for c in sb2["history"]] == ["beam_r01a", "beam_r01b", "beam_r02a", "beam_r02b"]
    own = read_json(h, "own_pick.json")
    assert own["status"] == "ok" and own["name"] in ("beam_r01a", "beam_r01b") and own["exploratory"] is True

    picks = read_json(h, "picks.json")
    assert picks["rounds_completed"] == 2 and picks["stop"] == {"reason": "max rounds",
                                                                 "beam_arm_closed_after_round": None}
    assert picks["beam_own_pick"]["status"] == "ok" and picks["beam_own_pick"]["name"] == own["name"]
    assert picks["beam_own_pick"]["equals_final"] is False
    for arm in ("beam", "cem"):
        want = metrics.select([c for s in (s1, s2) for c in s["candidates"] if c["arm"] == arm])
        assert picks["arms"][arm]["pick"] == (want or {}).get("name")
    for line in (root / "picks.sha256").read_text(encoding="utf-8").splitlines():
        sha, name = line.split("  ")
        assert rnd.sha256_bytes((root / name).read_bytes()) == sha
    order = [(root / f).stat().st_mtime_ns for f in ("picks.json", "picks_bank.safetensors", "picks.sha256",
                                                     "rounds_done.json")]
    assert order == sorted(order)
    done = read_json(h, "rounds_done.json")
    assert (done["status"], done["rounds_completed"], done["stop_reason"]) == ("ok", 2, "max rounds")
    for path in root.rglob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if path.name not in ("index.json", "shards.json"):  # run.py's own G1a files
            assert data.get("exploratory") is True, path
    report = metrics.search_report(root)
    assert report["checks"]["selection"]["beam"]["ok"] and report["checks"]["budget"]["ok"]
    assert report["checks"]["recompute"]["ok"] and report["checks"]["eligibility"]["ok"]

    # A rerun is a no-op; without the picks it replays both rounds from the root (CEM included) and repicks.
    assert rnd.main(argv) == 0
    for name in ("rounds_done.json", "picks.json", "picks.sha256", "picks_bank.safetensors"):
        (root / name).unlink()
    assert rnd.main(argv) == 0
    again = read_json(h, "picks.json")
    assert again["arms"] == picks["arms"] and again["scores_sha256"] == picks["scores_sha256"]

    # Boot 2 builds from these picks (section 11 step 4); an arm that reads none drops out of every variants file.
    stage_b_maps.boot2_bank(inputs, root / "picks.json", root / "picks_bank.safetensors")
    listed = variants_listed(inputs)
    for arm in ("beam", "cem"):
        assert (f"{arm}_final" in listed) == (picks["arms"][arm]["reading"] == "picked"), arm
    assert ("beam_own" in listed) == (picks["arms"]["beam"]["reading"] == "picked")
    none = {**picks, "arms": {**picks["arms"], "cem": {**{k: None for k in picks["arms"]["cem"]}, "reading": "none"}}}
    (tmp_path / "picks_none.json").write_text(json.dumps(none, indent=1) + "\n", encoding="utf-8")
    stage_b_maps.boot2_bank(inputs, tmp_path / "picks_none.json", root / "picks_bank.safetensors")
    assert "cem_final" not in variants_listed(inputs)


# Review fixes: provider failures, the ledger, the run loop, the log-dir binding and the outgoing text


def test_controller_provider_failures(tmp_path, no_sample, monkeypatch):
    sleeps = []
    monkeypatch.setattr(ctl, "_sleep", sleeps.append)

    # LEAK-1: a request lost after it was sent is charged at its worst case and counted; the cap then binds on it.
    timeout = ctl.ProviderError("ReadTimeout: read timed out", sent=True, retryable=True)
    res, stub, *_, logs = run(tmp_path / "a", [timeout, ctl.completion([("submit_patches", GOOD)])])
    ledger = json.loads((logs / "ledger.json").read_text(encoding="utf-8"))
    assert (res.status, res.requests) == ("ok", 2) and ledger["requests"] == 2
    assert ledger["entries"][0]["estimated"] is True and ledger["entries"][0]["completion_tokens"] == ctl.MAX_TOKENS
    assert stub.requests[0]["timeout_s"] is not None  # bounded by the round's time left
    res, stub, *_ = run(tmp_path / "b", [timeout, lookup()], ledger=ctl.Ledger(tmp_path / "b.json", cap_usd=0.05))
    assert (res.forfeit, res.requests, len(stub.requests)) == ("spend cap", 1, 1)

    # LIVE-3: overload is retried with growing pauses until the round's time is gone, not after three tries.
    clock = [time.time()]

    def advance(s):
        sleeps.append(s)
        clock[0] += s

    monkeypatch.setattr(ctl, "_sleep", advance)
    sleeps.clear()
    busy = [ctl.ProviderError("HTTP 529: overloaded", status=529, retryable=True) for _ in range(20)]
    res, stub, h, inputs, logs = run(tmp_path / "c", busy, now=lambda: clock[0], deadline=clock[0] + 600)
    assert res.forfeit == "provider" and len(stub.requests) == 7 and res.requests == 0
    assert sleeps == [15.0, 30.0, 60.0, 120.0, 120.0, 120.0]
    result = json.loads((logs / "round_01" / "result.json").read_text(encoding="utf-8"))
    assert "529" in result["last_provider_error"] and result["live"] is True
    sleeps.clear()
    res, stub, *_ = run(tmp_path / "d", [ctl.ProviderError("HTTP 404: no model", status=404)])
    assert (res.forfeit, len(stub.requests), sleeps) == ("provider", 1, [])  # no retry can help
    res, stub, *_ = run(tmp_path / "e", [RuntimeError("boom")] * 3)
    assert (res.forfeit, len(stub.requests)) == ("provider", 3)  # an unknown failure is counted

    # LIVE-2: a 400 after a lookup drops the echoed reasoning once and resends.
    res, stub, *_ = run(tmp_path / "f", [lookup(), ctl.ProviderError("HTTP 400: bad field", status=400),
                                         ctl.completion([("submit_patches", GOOD)])])
    assert res.status == "ok" and len(stub.requests) == 3
    assert any(m.get("reasoning_content") for m in stub.requests[1]["messages"])
    assert not any("reasoning_content" in m for m in stub.requests[2]["messages"])

    # LIVE-7: 401, 402 and 403 stop the round loudly and leave no beam.json, so a restart can still answer.
    h, inputs, ledger, logs = setup_round(tmp_path / "g")
    with pytest.raises(ctl.ProviderFatal):
        ctl.run_round(1, provider=ctl.StubProvider([ctl.ProviderFatal("HTTP 402: credits")]), handoff=h,
                      inputs=inputs, ledger=ledger, log_dir=logs)
    assert h.read("round_01/beam.json") is None and not (logs / "round_01" / "result.json").exists()

    # LIVE-5: a failed liveness probe forfeits nothing; the chat decides.
    h, inputs, ledger, logs = setup_round(tmp_path / "h")
    res = ctl.run_round(1, provider=ctl.StubProvider([ctl.completion([("submit_patches", GOOD)])], live=False),
                        handoff=h, inputs=inputs, ledger=ledger, log_dir=logs)
    assert res.status == "ok"


def test_ledger_is_shared_and_logs_stay_below_the_round(tmp_path):
    # OPS-6 / LIVE-8: two ledgers on one file (a smoke beside a run) keep every charge.
    a, b = ctl.Ledger(tmp_path / "l.json"), ctl.Ledger(tmp_path / "l.json")
    for _ in range(3):
        a.charge({"prompt_tokens": 1000, "completion_tokens": 1000})
        b.charge({"prompt_tokens": 1000, "completion_tokens": 1000})
    state = json.loads((tmp_path / "l.json").read_text(encoding="utf-8"))
    assert state["requests"] == 6 and state["spent_usd"] == pytest.approx(6 * 1.6e-3)
    assert a.admit(0, 4_000_000) and not a.admit(0, 4_160_000)  # the cap sees both processes' spend
    # LEAK-4: notes of a later round never reach an earlier round's prompt.
    for r in (1, 5):
        (tmp_path / "logs" / f"round_{r:02d}").mkdir(parents=True)
        (tmp_path / "logs" / f"round_{r:02d}" / "result.json").write_text(
            json.dumps({"round": r, "notes": f"note {r}", "candidates": []}), encoding="utf-8")
    assert ctl._own_logs(tmp_path / "logs", below=2)[0] == {1: "note 1"}


def test_outgoing_text_is_checked(tmp_path, no_sample):
    """LEAK-6: the label's reason and held-out names never reach Beam; Beam's own words are not held against it."""
    h, inputs, ledger, logs = setup_round(tmp_path)
    sb = scores_beam(1, history=[obj("beam_r01a", -0.2)])
    assert sb["reason"] == stageb.REASON
    text = ctl.round_message(2, sb, ctl.SurgeonInputs(inputs), n_candidates=2, max_rounds=12, deadline_utc=None,
                             notes={}, extras={})
    assert ctl.check_outgoing(text) == text
    for bad in (stageb.REASON, "the held-out tasks", "Claimed vs Done"):
        with pytest.raises(ValueError, match="nothing was sent"):
            ctl.check_outgoing(text + bad)
    # Beam's own notes may say anything: the round still goes out.
    write_json(h, "round_01/scores_beam.json", sb)
    (logs / "round_01").mkdir(parents=True)
    (logs / "round_01" / "result.json").write_text(json.dumps({"round": 1, "notes": "keep it held out (CvD-ish)",
                                                               "candidates": []}), encoding="utf-8")
    stub = ctl.StubProvider([ctl.completion([("submit_patches", GOOD)])])
    res = ctl.run_round(2, provider=stub, handoff=h, inputs=inputs, ledger=ledger, log_dir=logs)
    assert res.status == "ok" and "held out (CvD-ish)" in stub.requests[0]["messages"][1]["content"]
    assert stageb.REASON not in stub.requests[0]["messages"][1]["content"]
    ctx = tmp_path / "inputs" / "maps" / "surgeon_context.md"
    ctx.write_text("Stage A probe AUROC 0.580 on all tasks.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="banned strings"):
        ctl.SurgeonInputs(tmp_path / "inputs")


class FlakyHandoff(LocalHandoff):
    """Raises on the first `fails` reads, as gcloud does on a dropped connection."""

    def __init__(self, root, fails: int = 1):
        super().__init__(root)
        self.fails = fails

    def read(self, rel):
        if self.fails:
            self.fails -= 1
            raise RuntimeError(f"gcloud storage cat {rel}: Connection reset")
        return super().read(rel)


def run_args(tmp_path: Path, inputs: Path, root: Path, logs: Path) -> SimpleNamespace:
    return SimpleNamespace(handoff=str(root), inputs=str(inputs), log_dir=str(logs), ledger=str(tmp_path / "led.json"),
                           spend_cap=5.0, max_rounds=12, poll_s=0.01, effort="high")


def test_controller_run_recovers_binds_and_locks(tmp_path, no_sample, monkeypatch):
    inputs = fake_inputs(tmp_path / "inputs")
    root, logs = tmp_path / "root", tmp_path / "logs"
    h = FlakyHandoff(root)
    LocalHandoff.write(h, "round_00/scores_beam.json", json.dumps(scores_beam(0)).encode())
    monkeypatch.setattr(ctl, "open_handoff", lambda url: h)
    awake = []
    monkeypatch.setattr(ctl, "keep_awake", lambda: awake.append(1) or True)

    def tick(seconds):  # after round 1 the VM writes the final scores
        if LocalHandoff.read(h, "round_01/beam.json") is not None:
            write_json(h, "round_01/scores_beam.json", scores_beam(1, final=True))

    monkeypatch.setattr(ctl, "_sleep", tick)
    stub = ctl.StubProvider([ctl.completion([("submit_patches", GOOD)])])
    # OPS-2 / LIVE-4: the first read fails, and the controller polls again instead of exiting.
    assert ctl.run(run_args(tmp_path, inputs, root, logs), stub) == 0
    assert awake == [1] and read_json(h, "round_01/beam.json")["status"] == "ok"
    assert read_json(h, "own_pick.json")["status"] == "missing"  # nothing scored, so no request
    binding = json.loads((logs / "handoff.json").read_text(encoding="utf-8"))
    assert binding["exploratory"] is True and binding["round_00_scores_beam_sha256"] == ctl.sha256_bytes(
        h.read("round_00/scores_beam.json"))
    assert not (tmp_path / "controller.lock").exists()  # released at the end

    # LEAK-4: the same log dir against another boot's root refuses to start; so does a log dir of unbound results.
    other = LocalHandoff(tmp_path / "other")
    write_json(other, "round_00/scores_beam.json", scores_beam(0))
    monkeypatch.setattr(ctl, "open_handoff", lambda url: other)
    with pytest.raises(SystemExit) as err:
        ctl.run(run_args(tmp_path, inputs, tmp_path / "other", logs), ctl.StubProvider([]))
    assert err.value.code == ctl.REFUSED
    (logs / "handoff.json").unlink()
    with pytest.raises(SystemExit) as err:
        ctl.run(run_args(tmp_path, inputs, tmp_path / "other", logs), ctl.StubProvider([]))
    assert err.value.code == ctl.REFUSED

    # OPS-6: a live lock refuses a second controller; a stale one is taken over.
    lock = tmp_path / "controller.lock"
    lock.write_text(str(os.getpid()), encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        ctl.run(run_args(tmp_path, inputs, tmp_path / "other", tmp_path / "logs2"), ctl.StubProvider([]))
    assert err.value.code == ctl.REFUSED and lock.exists()
    lock.write_text("999999", encoding="utf-8")
    assert ctl.acquire_lock(lock) == lock and lock.read_text(encoding="utf-8") == str(os.getpid())
    ctl.release_lock(lock)
    assert not lock.exists()


def test_vm_waits_survive_storage_hiccups_and_own_pick_is_bounded(tmp_path, no_sample, monkeypatch):
    inputs = fake_inputs(tmp_path / "inputs")
    loop = rnd.Loop(args_for(tmp_path, inputs, poll_s=0.1, boot_start=time.time() - 6.6 * 3600))

    def broken(rel):
        raise RuntimeError("Connection reset")

    monkeypatch.setattr(loop.h, "read", broken)
    loop.rounds = {0: {"next_deadline_utc": ctl.utc(time.time() + 1)},
                   1: {"candidates": [obj("beam_r01a", -0.2)], "written_utc": ctl.utc()}}
    t0 = time.time()
    assert loop.wait_beam(1) is None  # OPS-2: a failed read waits for the deadline instead of failing the boot
    # OPS-7: 6.6 h into the boot the own-pick wait is already over, so the picks land before the step timeout
    assert loop.wait_own_pick() is None and time.time() - t0 < 10
