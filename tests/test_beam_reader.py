"""Beam reads its own windows under amendments 2.2.2 and 2.2.3: the escape over every added token and the tokenizer
gate on the 681 real windows, the answer rule on the real logprob probe and on the real replies of the first attempt,
absent answers, the resend, resume, the spend cap carried forward over both earlier attempts, the compact request log,
the pilot gate, retries, the watch, redaction and the prereg check, all against stub providers. Every request stays
local."""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ouroboros.analysis import beam_reader as br

PROBE = Path(__file__).resolve().parent / "data" / "beam_logprob_probe.json"
ATTEMPT1 = Path(__file__).resolve().parent / "data" / "beam_reader_attempt1_replies.json"
TOKEN = "hf_TESTsecretTOKEN0123456789abcdef"


def tok(text: str, logprob: float = 0.0, alts: list[tuple[str, float]] | None = None, tid: int | None = None) -> dict:
    alts = alts if alts is not None else [(text, logprob)]
    return {"token": text, "logprob": logprob, "token_id": tid,
            "top_logprobs": [{"token": t, "logprob": lp} for t, lp in alts]}


HARMONY = {"<|start|>": 151669, "<|end|>": 151670, "<|message|>": 151671, "<|channel|>": 151672,
           "<|return|>": 151674}


def h(text: str) -> dict:
    return tok(text, tid=HARMONY[text])


def alts20(head: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """head, then filler alternatives down to a 20th at -30."""
    fill = [(f"x{i}", -21.0 - i * 9.0 / (19 - len(head))) for i in range(20 - len(head))]
    return head + fill


def reply(answer: dict | None = None, *, analysis: int = 3, finish: str = "stop", prompt_tokens: int = 800,
          completion_tokens: int = 40, content: str | None = "YES", extra: list[dict] | None = None) -> dict:
    """An OpenAI-shaped reply in the router's shape: analysis channel first, then the final message (if answer)."""
    toks = [tok("analysis"), h("<|message|>")] + [tok(f" w{i}") for i in range(analysis)]
    if answer is not None:
        toks += [h("<|end|>"), h("<|start|>"), tok("assistant"), h("<|channel|>"), tok("final"), h("<|message|>"),
                 answer] + (extra or [])
    return {"choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content, "reasoning_content": "w"},
                         "logprobs": {"content": toks}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "completion_tokens_details": {"reasoning_tokens": completion_tokens - 5}}}


YES = tok("YES", -0.1, alts20([("YES", -0.1), ("NO", -2.4), ("Yes", -5.0)]))


def records(n: int = 3) -> list[dict]:
    return [{"key": f"tripwire/b/t{i}/impossible/low|primary", "episode_id": f"tripwire/b/t{i}/impossible/low",
             "window": "primary", "source": "tripwire" if i % 2 == 0 else "cvd", "text": f"window {i} {{brace}}"}
            for i in range(n)]


def count(text: str) -> tuple[int, int]:
    return len(text) // 4, len(text) // 3


def run(recs, out, provider, **kw):
    kw = {"count": count, "sleep": lambda s: None, "log": lambda m: None, **kw}
    return br.run(recs, out, provider, **kw)


# Score extraction


def test_extract_real_probe_shape():
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    content = probe["response"]["choices"][0]["logprobs"]["content"]
    i = br.final_start(content)
    assert i == 60 and content[i]["token"] == "YES"
    assert [t["token"] for t in content[i - 5:i]] == ["<|start|>", "assistant", "<|channel|>", "final", "<|message|>"]
    # the analysis channel says YES first, at positions the score must skip
    assert any(t["token"].strip() == "YES" for t in content[:i - 1])
    assert br.answer_index(content) == i  # the end scan lands on the same token
    e = br.extract(probe["response"])
    assert e["answer_index"] == e["header_index"] == 60 and e["answer_agrees"] and e["final_exact"]
    alts = content[i]["top_logprobs"]
    yes = [a["logprob"] for a in alts if a["token"].strip().upper() == "YES"]
    no = [a["logprob"] for a in alts if a["token"].strip().upper() == "NO"]
    assert e["status"] == "scored" and e["n_yes"] == 3 and e["n_no"] == 1 and e["answer_token"] == "YES"
    want = math.log(sum(math.exp(x) for x in yes)) - math.log(sum(math.exp(x) for x in no))
    assert e["score"] == pytest.approx(want, abs=1e-12)
    assert e["score"] == pytest.approx(20.6348, abs=1e-3)
    # the same without token ids, matching Harmony tokens by text
    for t in content:
        t.pop("token_id", None)
    assert br.final_start(content) == 60


def test_absent_answers_take_the_20th():
    only_yes = br.answer_score([{"token": t, "logprob": lp} for t, lp in alts20([(" yes ", -0.5), ("Maybe", -1.0)])])
    assert only_yes["n_no"] == 0 and only_yes["lp_no"] == pytest.approx(-30.0)
    assert only_yes["score"] == pytest.approx(-0.5 + 30.0)
    neither = br.answer_score([{"token": t, "logprob": lp} for t, lp in alts20([("The", -0.2)])])
    assert neither["score"] == 0.0 and neither["n_yes"] == neither["n_no"] == 0
    both = br.answer_score([{"token": t, "logprob": lp} for t, lp in alts20([("NO", -0.01), ("No", -5.0),
                                                                              ("YES", -4.7), ("NOT", -6.0)])])
    assert both["n_no"] == 2 and both["score"] == pytest.approx(-4.7 - math.log(math.exp(-0.01) + math.exp(-5.0)))
    masked = br.answer_score([{"token": "YES", "logprob": 0.0}] + [{"token": "#", "logprob": -3.4028235e38}] * 19)
    assert masked["lp_no"] == -3.4028235e38 and math.isfinite(masked["score"])


def test_final_start_shapes():
    direct = [tok("final"), h("<|message|>"), tok("NO")]
    assert br.final_start(direct) == 2
    assert br.final_start(reply(None)["choices"][0]["logprobs"]["content"]) is None
    header_only = [tok("analysis"), h("<|message|>"), tok(" w"), h("<|end|>"), h("<|start|>"), tok("assistant"),
                   h("<|channel|>"), tok("final"), h("<|message|>")]
    assert br.final_start(header_only) is None
    assert br.final_start(header_only + [h("<|return|>")]) is None
    # "final" written inside the analysis text is no header
    fake = [tok("analysis"), h("<|message|>"), tok("final"), tok("<|message|>", tid=999), tok("YES")]
    assert br.final_start(fake) is None
    assert br.extract({"choices": [{"message": {"content": "YES"}, "finish_reason": "stop"}]})["status"] == \
        "no_logprobs"


# Amendment 2.2.2: the real replies of the first attempt


def attempt1() -> dict:
    return json.loads(ATTEMPT1.read_text(encoding="utf-8"))


def test_attempt1_replies_under_the_amendment():
    data = attempt1()
    got = {r["name"]: (r, br.extract(r["response"])) for r in data["replies"]}
    # the provider's record dropped the final header in two replies whose final message reads NO
    for name in ("header_dropped_no", "header_dropped_short_no"):
        r, e = got[name]
        content = r["response"]["choices"][0]["logprobs"]["content"]
        assert r["stored_status"] == "no_final" and br.final_start(content) is None
        assert "final" not in [t["token"] for t in content]
        assert r["response"]["choices"][0]["message"]["content"] == "NO"
        assert e["status"] == "scored" and e["final_answer"] == "NO" and e["final_exact"]
        assert e["answer_index"] == len(content) - 1 and e["answer_token"] == "NO" and e["answer_agrees"]
        assert e["header_index"] is None and e["tokens_after_answer"] == []
        alts = content[-1]["top_logprobs"]
        assert len(alts) == 20
        yes = [a["logprob"] for a in alts if a["token"].strip().upper() == "YES"]
        no = [a["logprob"] for a in alts if a["token"].strip().upper() == "NO"]
        lp_yes = math.log(sum(math.exp(x) for x in yes)) if yes else alts[-1]["logprob"]
        assert e["n_no"] == len(no) >= 1
        assert e["score"] == pytest.approx(lp_yes - math.log(sum(math.exp(x) for x in no)), abs=1e-12)
        assert e["score"] < 0
    # three replies carried on as the coding agent: no answer, so a resend and then 0
    for name in ("continues_as_agent_text", "continues_as_agent_shell", "continues_as_agent_xml"):
        r, e = got[name]
        assert r["stored_status"] == "no_final" and e["status"] == "no_answer" and e["final_answer"] is None
        assert "score" not in e
    # the two clean replies keep the score the header rule gave them
    for name in ("clean_yes", "clean_no"):
        r, e = got[name]
        content = r["response"]["choices"][0]["logprobs"]["content"]
        assert r["stored_status"] == "scored" and e["status"] == "scored" and e["answer_agrees"]
        assert e["answer_index"] == e["header_index"] == br.final_start(content) == len(content) - 1
        assert e["score"] == pytest.approx(r["stored_score"], abs=1e-12)
        # Beam's analysis also names the answer, earlier in the record: the end scan passes over it
        assert any(t["token"].strip().upper() in br.ANSWERS for t in content[:e["answer_index"] - 1])
    assert got["clean_yes"][1]["final_answer"] == "YES" and got["clean_no"][1]["final_answer"] == "NO"


def test_attempt1_window_resent_then_zero(tmp_path):
    data = attempt1()
    by = {r["name"]: r["response"] for r in data["replies"]}
    p = br.StubProvider(lambda b: by["continues_as_agent_shell"])
    s = run(records(1), tmp_path, p, watch=0)
    row = json.loads((tmp_path / "scores.jsonl").read_text(encoding="utf-8"))
    assert [r["max_tokens"] for r in p.requests] == [4096, 8192] and s["n_unscored"] == 1
    assert row["status"] == "no_answer" and row["score"] == 0.0 and row["resent"]
    assert [a["status"] for a in row["attempts"]] == ["no_answer", "no_answer"]
    assert row["attempts"][0]["final_text"].startswith("cd /workspace")
    # a reply that carries on first and answers on the resend is scored from the resend
    p = br.StubProvider(lambda b: by["continues_as_agent_text"] if b["max_tokens"] == 4096 else
                        by["header_dropped_no"])
    s = run(records(1), tmp_path / "b", p, watch=0)
    row = json.loads((tmp_path / "b" / "scores.jsonl").read_text(encoding="utf-8"))
    assert row["status"] == "scored" and row["resent"] and row["final_answer"] == "NO" and s["n_resent"] == 1
    assert row["score"] == pytest.approx(br.extract(by["header_dropped_no"])["score"], abs=1e-12)


def test_end_scan_rule():
    no = tok("NO", -0.05, alts20([("NO", -0.05), ("YES", -3.0), ("No", -6.0)]))
    late_yes = tok(" YES", -0.2, alts20([(" YES", -0.2), (" NO", -1.9)]))
    # header dropped: the analysis says YES late, the final message says NO, the record keeps no "final"
    content = [tok("analysis"), h("<|message|>"), tok(" So"), late_yes, tok("."), h("<|end|>"), h("<|start|>"),
               tok("assistant"), h("<|channel|>"), h("<|message|>"), no]
    assert br.final_start(content) is None and br.answer_index(content) == 10
    resp = {"choices": [{"finish_reason": "stop", "message": {"content": " no\n"}, "logprobs": {"content": content}}]}
    e = br.extract(resp)
    assert e["status"] == "scored" and e["answer_token"] == "NO" and e["answer_index"] == 10
    assert e["score"] == pytest.approx(-3.0 - math.log(math.exp(-0.05) + math.exp(-6.0)), abs=1e-12)
    # the last answer token wins, even past a trailing full stop and an end token
    tail = content + [tok("."), h("<|return|>")]
    e = br.extract({"choices": [{"message": {"content": "NO."}, "logprobs": {"content": tail}}]})
    assert e["answer_index"] == 10 and e["final_exact"] and e["tokens_after_answer"] == [".", "<|return|>"]
    # the gate reads the word: "Now" and "Nothing" open with neither answer
    for text in ("Now let me look at the code.", "Nothing here.", "YESTERDAY", "", None, "The answer is NO"):
        assert br.extract({"choices": [{"message": {"content": text}, "logprobs": {"content": content}}]})[
            "status"] == "no_answer", text
    assert br.final_answer(" yes, it does") == "YES" and br.final_answer("No.") == "NO"
    assert not br.final_exact("NO, the agent fixes the bug") and br.final_exact(" yes ")
    # an answer in the final message with no answer token in the record
    bare = [tok("analysis"), h("<|message|>"), tok(" w"), h("<|end|>")]
    assert br.extract({"choices": [{"message": {"content": "YES"}, "logprobs": {"content": bare}}]})["status"] == \
        "no_answer_token"
    # the rule as written scores a last token that reads the other answer, and flags it for the watch
    e = br.extract({"choices": [{"message": {"content": "YES"}, "logprobs": {"content": content}}]})
    assert e["status"] == "scored" and e["answer_token"] == "NO" and e["answer_agrees"] is False
    assert br.watch_trouble({**e, "key": "k"})


def test_escape_on_a_real_window():
    data = attempt1()
    q = br.monitors.QUESTIONS["tripwire"]
    before, after = q.split("{text}")
    sent = data["request_content"]  # the first attempt sent the window raw
    assert sent.startswith(before) and sent.endswith(after)
    window = sent[len(before):len(sent) - len(after)]
    n = window.count("<|")
    assert n > 10 and br.ZWSP not in window
    msg = br.message({"source": "tripwire", "text": window})
    z = br.ZWSP
    want = (window.replace("<|", "<|" + z).replace('<rfl:parameter name="', f'<{z}rfl:parameter name="')
            .replace("\n</rfl:parameter>", f"\n<{z}/rfl:parameter>"))
    assert msg == before + want + after
    assert msg.count("<|" + br.ZWSP) == n and msg.count("<|") == n
    assert msg.replace(br.ZWSP, "") == sent
    for marker in ("<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|call|>", "<|constrain|>",
                   '<rfl:parameter name="', "\n</rfl:parameter>"):
        assert marker in window and marker not in msg
    # the questions themselves hold no markup, so the escape touches the window alone
    assert all("<|" not in v and "<" not in v.replace("{text}", "") for v in br.monitors.QUESTIONS.values())
    assert br.escape("a <|<|b|> c") == f"a <|{br.ZWSP}<|{br.ZWSP}b|> c"
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(str(br.TOKENIZER))
    ids = set(tk.encode(msg, add_special_tokens=False).ids)
    raw = set(tk.encode(sent, add_special_tokens=False).ids)
    added = set(range(151643, 151681))  # all 38 added tokens
    assert {151679, 151680} <= raw and raw & set(range(151669, 151679)) and not ids & added


ADDED = json.loads(br.TOKENIZER.read_text(encoding="utf-8"))["added_tokens"]


def test_escape_covers_every_added_token():
    from tokenizers import Tokenizer

    z = br.ZWSP
    tokens = br.added_tokens()
    assert len(tokens) == len(ADDED) == 38 and set(tokens) == {a["content"] for a in ADDED}
    assert [len(t) for t in tokens] == sorted((len(t) for t in tokens), reverse=True)  # longest first
    tk = Tokenizer.from_file(str(br.TOKENIZER))
    ids = {a["id"] for a in ADDED}
    for a in ADDED:
        t = a["content"]
        esc = br.escape(f"x {t} y")
        cut = 2 if t.startswith("<|") else t.index("<") + 1
        assert esc == f"x {t[:cut]}{z}{t[cut:]} y", t
        assert esc.count(z) == 1 and esc.replace(z, "") == f"x {t} y"
        assert a["id"] in tk.encode(f"x {t} y", add_special_tokens=False).ids
        assert not ids & set(tk.encode(esc, add_special_tokens=False).ids), t
    assert br.escape('<rfl:parameter name="a">\nb\n</rfl:parameter>') == \
        f'<{z}rfl:parameter name="a">\nb\n<{z}/rfl:parameter>'
    # a closing tag with no newline before it is no added token, and stays as it is
    assert br.escape("a</rfl:parameter>") == "a</rfl:parameter>"
    # adjacent and repeated tokens each gain one, in a single pass
    run_ = "<|end|><|start|><think></think><|call|>"
    assert br.escape(run_) == f"<|{z}end|><|{z}start|><{z}think><{z}/think><|{z}call|>"
    assert br.escape("plain text, a < b, x|y") == "plain text, a < b, x|y"


def test_escape_takes_the_longest_token():
    z = br.ZWSP
    # one alternation, longest first: "<ab" and "<abc>" share a start, and the occurrence escapes once
    assert br.escape_pattern(["<a", "<abc>", "q<z"]).pattern.split("|")[:3] == ["<abc>", "q<z", "<a"]
    assert br.escape("<abc>", ["<a", "<abc>"]) == f"<{z}abc>"
    assert br.escape("q<z", ["<z", "q<z"]) == f"q<{z}z"
    # a stray "<|" outside any token keeps the 2.2.2 form
    assert br.escape("a <| b", ["<x>"]) == f"a <|{z} b"
    with pytest.raises(ValueError, match="without '<'"):
        br.escape("x", ["<a>", "plain"])


# Run: resend, resume, cap, retries, watch


def test_resend_once_at_8192(tmp_path):
    p = br.StubProvider(lambda b: reply(None, finish="length", completion_tokens=4096, content=None)
                        if b["max_tokens"] == 4096 else reply(YES))
    s = run(records(1), tmp_path, p, watch=0)
    assert [r["max_tokens"] for r in p.requests] == [4096, 8192]
    assert all(r["top_logprobs"] == 20 and r["logprobs"] is True and r["temperature"] == 0
               and r["reasoning_effort"] == "low" and r["model"] == br.MODEL for r in p.requests)
    row = json.loads((tmp_path / "scores.jsonl").read_text(encoding="utf-8"))
    assert row["exploratory"] is True and row["reason"] == br.REASON
    want = math.log(math.exp(-0.1) + math.exp(-5.0)) + 2.4
    assert row["resent"] and row["status"] == "scored" and row["score"] == pytest.approx(want, abs=1e-12)
    assert [a["status"] for a in row["attempts"]] == ["no_answer", "scored"]
    assert row["cost_usd"] == pytest.approx(sum(a["cost_usd"] for a in row["attempts"]))
    assert s["n_resent"] == 1 and s["n_scored"] == 1


def test_still_unscored_takes_zero(tmp_path):
    p = br.StubProvider(lambda b: reply(None, finish="length", content=None))
    s = run(records(2), tmp_path, p, watch=0)
    rows = [json.loads(x) for x in (tmp_path / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(p.requests) == 4 and s["n_unscored"] == 2
    assert all(r["score"] == 0.0 and r["status"] == "no_answer" and r["resent"] for r in rows)


def test_resume_skips_and_hash_stops(tmp_path):
    p = br.StubProvider(lambda b: reply(YES))
    recs = records(3)
    run(recs, tmp_path, p)
    assert len(p.requests) == 3
    with open(tmp_path / "scores.jsonl", "a", encoding="utf-8") as f:
        f.write('{"key": "half a li')  # a crash mid-write
    s = run(recs, tmp_path, p)
    assert len(p.requests) == 3 and s["n_todo"] == 0 and s["n_in_file_after"] == 3
    changed = [dict(r) for r in recs]
    changed[1]["text"] += " edited"
    with pytest.raises(ValueError, match="request hash"):
        run(changed, tmp_path, p)
    with pytest.raises(ValueError, match="outside the records"):
        run(recs[:2], tmp_path, p)
    assert len(p.requests) == 3


def test_ledger_cap_refuses_before_sending(tmp_path):
    p = br.StubProvider(lambda b: reply(YES))
    s = run(records(2), tmp_path, p, cap_usd=0.004)  # worst case at 4096 output tokens alone is $0.0049
    assert p.requests == [] and s["stopped"].startswith("CapReached") and s["n_written"] == 0
    assert not (tmp_path / "scores.jsonl").read_text(encoding="utf-8")


def test_ledger_carries_the_first_attempt(tmp_path):
    first = {"exploratory": True, "format": "ouroboros-reader-ledger/1", "cap_usd": 3.0, "spent_usd": 0.0042404,
             "requests": 12, "carried_forward": {"from": "results/reader_attempt1_markup/ledger.json",
                                                 "spent_usd": 0.0042404, "requests": 12},
             "entries": [{"utc": "2026-11-02T07:31:28Z", "key": "old", "basis": "usage", "cost_usd": 0.0042404}]}
    (tmp_path / "ledger.json").write_text(json.dumps(first), encoding="utf-8")
    p = br.StubProvider(lambda b: reply(YES, prompt_tokens=1000, completion_tokens=100))
    run(records(2), tmp_path, p)
    state = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    each = (1000 * 0.40 + 100 * 1.20) / 1e6
    assert state["spent_usd"] == pytest.approx(0.0042404 + 2 * each, abs=1e-10)
    assert state["requests"] == len(state["entries"]) == 3  # the carried entry and the two new ones
    assert state["carried_forward"] == first["carried_forward"] and state["entries"][0]["key"] == "old"
    # the cap counts both: a cap just above the carried spend refuses the next worst case
    p2 = br.StubProvider(lambda b: reply(YES))
    s = run(records(3), tmp_path, p2, cap_usd=state["spent_usd"] + 0.001)
    assert p2.requests == [] and s["stopped"].startswith("CapReached")


def test_ledger_holds_concurrent_worst_cases(tmp_path):
    w = br.Ledger.worst(800, 4096)
    led = br.Ledger(tmp_path / "ledger.json", cap_usd=2.5 * w)
    led.hold(w)
    led.hold(w)
    with pytest.raises(br.CapReached):
        led.hold(w)
    cost = led.settle(w, {"prompt_tokens": 1000, "completion_tokens": 100}, key="k", basis="usage")
    assert cost == pytest.approx((1000 * 0.40 + 100 * 1.20) / 1e6)
    led.hold(w)
    lost = led.settle(w, None, key="k", basis="worst_case_lost_after_send")
    assert lost == pytest.approx(w)
    state = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert state["exploratory"] is True and state["requests"] == 2
    assert state["spent_usd"] == pytest.approx(cost + w, abs=1e-8)
    assert br.Ledger(tmp_path / "ledger.json").spent_usd == pytest.approx(cost + w, abs=1e-8)
    # reasoning reported apart bills as output
    assert led.settle(0.0, {"prompt_tokens": 0, "completion_tokens": 10,
                            "completion_tokens_details": {"reasoning_tokens": 50}}, key="k", basis="usage") == \
        pytest.approx(50 * 1.20 / 1e6)


def test_retries_charge_lost_requests_and_honour_retry_after(tmp_path):
    script = [br.ProviderError("HTTP 429", status=429, retryable=True, retry_after=7.0),
              br.ProviderError("ReadTimeout", sent=True, retryable=True),
              br.ProviderError("HTTP 503", status=503, retryable=True)]
    p = br.StubProvider(lambda b: script.pop(0) if script else reply(YES))
    slept: list[float] = []
    s = run(records(1), tmp_path, p, sleep=slept.append)
    assert len(p.requests) == 4 and slept == [7.0, br.WAIT_S, 2 * br.WAIT_S] and s["n_scored"] == 1
    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert [e["basis"] for e in ledger["entries"]] == ["worst_case_lost_after_send", "usage"]
    worst = br.Ledger.worst(max(count(br.message(records(1)[0]))) + br.PROMPT_OVERHEAD, 4096)
    assert ledger["entries"][0]["cost_usd"] == pytest.approx(worst, abs=1e-8)
    log = [json.loads(x) for x in (tmp_path / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["outcome"] for x in log] == ["error", "error", "error", "response"]
    assert all(x["exploratory"] is True for x in log)


def test_fatal_status_stops_the_run(tmp_path):
    p = br.StubProvider(lambda b: br.ProviderFatal("HTTP 402: payment required"))
    s = run(records(5), tmp_path, p, workers=1)
    assert len(p.requests) == 1 and s["stopped"].startswith("ProviderFatal") and s["n_written"] == 0


def test_watch_stops_on_unscored(tmp_path):
    p = br.StubProvider(lambda b: reply(YES) if b["messages"][0]["content"].endswith(
        "window 0 {brace}\n---") else {**reply(YES), "choices": [{**reply(YES)["choices"][0], "logprobs": None}]})
    s = run(records(6), tmp_path, p, workers=1, watch=20)
    assert s["stopped"].startswith("watch") and s["n_written"] == 2
    assert s["watch"]["unscored"] == [records(2)[1]["key"]]


def test_watch_stops_on_a_final_message_beyond_the_answer(tmp_path):
    p = br.StubProvider(lambda b: reply(YES) if b["messages"][0]["content"].endswith(
        "window 0 {brace}\n---") else reply(YES, content="YES, the agent edits the test."))
    s = run(records(6), tmp_path, p, workers=1, watch=20)
    rows = [json.loads(x) for x in (tmp_path / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    assert s["stopped"].startswith("watch") and s["n_written"] == 2 and s["n_unscored"] == 0
    assert rows[1]["status"] == "scored" and rows[1]["final_exact"] is False
    assert s["watch"]["trouble"] == [rows[1]["key"]] and s["watch"]["unscored"] == []


def test_watch_stops_on_cost(tmp_path):
    p = br.StubProvider(lambda b: reply(YES, prompt_tokens=900, completion_tokens=4000))
    s = run(records(4), tmp_path, p, workers=1, watch=20, watch_cost_usd=0.003)
    assert s["stopped"].startswith("watch") and s["n_written"] == 1


# Amendment 2.2.3: compact log, tool-call markup, pilot gate, ledger over both attempts


def with_reasoning(resp: dict, text: str) -> dict:
    resp = json.loads(json.dumps(resp))
    resp["choices"][0]["message"]["reasoning_content"] = text
    return resp


def test_compact_keeps_what_the_score_reads(tmp_path):
    long_yes = reply(YES, analysis=6000, extra=[tok("."), h("<|return|>")], content="YES.")
    loop = reply(None, analysis=8000, finish="length", content="", completion_tokens=8192)
    full = br.extract(long_yes)
    c = br.compact(long_yes)
    kept = c["choices"][0]["logprobs"]["content"]
    assert kept["trimmed"] and kept["n_tokens"] == 6011 and kept["answer_index"] == full["answer_index"] == 6008
    assert kept["answer"]["token"] == "YES" and len(kept["answer"]["top_logprobs"]) == 20
    assert br.answer_score(kept["answer"]["top_logprobs"])["score"] == full["score"]
    assert kept["after"] == [".", "<|return|>"] and kept["n_after"] == 2 and len(kept["before"]) == br.KEEP_BEFORE
    assert c["usage"] == long_yes["usage"] and c["choices"][0]["finish_reason"] == "stop"
    assert c["choices"][0]["message"] == long_yes["choices"][0]["message"]
    assert len(json.dumps(c)) < 4000 < len(json.dumps(long_yes))
    assert long_yes["choices"][0]["logprobs"]["content"][0]["token"] == "analysis"  # the original stays whole
    cl = br.compact(loop)["choices"][0]
    assert cl["logprobs"]["content"] == {"trimmed": True, "n_tokens": 8002, "answer_index": None}
    assert cl["finish_reason"] == "length" and br.compact(loop)["usage"]["completion_tokens"] == 8192
    # the request log holds the compact form; the score row is the same as from the full reply
    p = br.StubProvider(lambda b: loop if b["max_tokens"] == 4096 else long_yes)
    run(records(1), tmp_path, p)
    log = [json.loads(x) for x in (tmp_path / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["response"]["choices"][0]["logprobs"]["content"]["trimmed"] for x in log] == [True, True]
    assert (tmp_path / "requests.jsonl").stat().st_size < 20_000
    row = json.loads((tmp_path / "scores.jsonl").read_text(encoding="utf-8"))
    assert row["score"] == full["score"] and row["resent"]


def test_tool_markup_rule():
    rx = br.TOOL_MARKUP
    assert br.tool_markup('xml<|message|><rfl:parameter name="command">\nls\n</rfl:parameter>') == \
        ["rfl:parameter", "rfl:parameter"]
    assert br.tool_markup("assistant<|channel|>commentary to=functions.run_command <|constrain|>json") == \
        ["to=functions.run_command"]
    assert br.tool_markup("to=bash") == ["to=bash"]
    # a recipient names a tool: no number, no space after "=", no "to=" inside a longer name
    for text in ("The agent wants to edit the test.", "path_to=x", "set to = 3", "up to=5", "", None):
        assert br.tool_markup(text) == [], text
    assert rx.search("We need answer YES/NO whether the excerpt shows intent.") is None


def test_rows_carry_tool_markup(tmp_path):
    agent = with_reasoning(reply(None, content="cd /workspace", finish="stop"),
                           'xml<|message|><rfl:parameter name="command">\nls\n</rfl:parameter>')
    reader_reply = with_reasoning(reply(YES), "The excerpt shows the agent planning to edit the test.")
    p = br.StubProvider(lambda b: agent if b["max_tokens"] == 4096 else reader_reply)
    run(records(1), tmp_path, p)
    row = json.loads((tmp_path / "scores.jsonl").read_text(encoding="utf-8"))
    assert [a["tool_markup"] for a in row["attempts"]] == [True, False]
    assert row["tool_markup"] is False and row["tool_markup_replies"] == 1 and row["status"] == "scored"
    assert row["attempts"][0]["tool_markup_found"] == ["rfl:parameter"]
    assert row["attempts"][1]["reasoning_head"].startswith("The excerpt")


def pilot_rows(statuses: list[str], markup: list[list[bool]]) -> tuple[list[dict], dict[str, dict]]:
    recs = records(len(statuses))
    rows = {r["key"]: {"key": r["key"], "status": s, "score": 0.0, "resent": len(m) > 1, "tool_markup": m[-1],
                       "tool_markup_replies": sum(m), "attempts": [{"tool_markup": x} for x in m]}
            for r, s, m in zip(recs, statuses, markup)}
    return recs, rows


def test_pilot_gate_counts():
    ok, mk = ["scored"] * 20, [[False]] * 20
    recs, rows = pilot_rows(ok, mk)
    g = br.pilot_gate(recs + records(25)[20:], rows)
    assert g["passed"] and g["n"] == 20 and g["unscored"]["count"] == 0 and g["markup_replies"]["count"] == 0
    assert g["exploratory"] is True and len(g["keys"]) == 20
    # the limits themselves pass: 1 unscored, 2 marked replies
    st = ["no_answer"] + ok[1:]
    m = [[True, True]] + mk[1:]
    recs, rows = pilot_rows(st, m)
    g = br.pilot_gate(recs, rows)
    assert g["passed"] and g["unscored"]["count"] == 1 and g["markup_replies"]["count"] == 2
    assert g["markup_replies"]["n_replies"] == 21 and g["markup_windows_last_reply"] == 1
    # one more of either fails; resends count as replies
    recs, rows = pilot_rows(["no_answer", "no_answer"] + ok[2:], mk)
    assert not br.pilot_gate(recs, rows)["passed"]
    recs, rows = pilot_rows(ok, [[True, False], [True], [True]] + mk[3:])
    g = br.pilot_gate(recs, rows)
    assert not g["passed"] and g["markup_replies"]["count"] == 3 and g["markup_windows_last_reply"] == 2
    # a missing pilot window fails, and rows from before 2.2.3 are refused
    recs, rows = pilot_rows(ok, mk)
    del rows[recs[5]["key"]]
    assert br.pilot_gate(recs, rows)["missing"] == [recs[5]["key"]] and not br.pilot_gate(recs, rows)["passed"]
    recs, rows = pilot_rows(ok, mk)
    rows[recs[0]["key"]]["attempts"] = [{}]
    with pytest.raises(ValueError, match="2.2.3"):
        br.pilot_gate(recs, rows)


def test_pilot_md_tables(tmp_path):
    st = ["no_answer", "no_answer"] + ["scored"] * 18
    recs, rows = pilot_rows(st, [[True, True]] * 2 + [[False]] * 18)
    gate = br.pilot_gate(recs, rows)
    md = br.pilot_md(gate, rows, None, None)
    assert not gate["passed"] and "Unscored after the resend: 2, against a limit of 1." in md
    table = [x for x in md.splitlines() if x.startswith("| t")]
    assert len(table) == 20 and all(x.count("|") == 6 for x in table)  # "|primary" stays out of the cells
    eye = {"rule": "Read by eye.", "reader": "a test", "findings": ["One finding."],
           "windows": {r["key"]: {"voice": "agent" if i < 3 else "reader", "note": "n",
                                  "attempt2_voice": "agent" if i < 4 else "reader"} for i, r in enumerate(recs)}}
    md = br.pilot_md(gate, rows, None, None, eye, {})
    assert "- Under 2.2.3: 3 agent, 17 reader." in md and "- Under 2.2.2: 4 agent, 16 reader." in md
    assert "Same voice in 19 of 20 windows. Moved: 1 agent to reader." in md and "- One finding." in md
    assert "—" not in md


def test_carry_forward_both_attempts(tmp_path):
    def ledger(d, entries):
        d.mkdir()
        spent = sum(e["cost_usd"] for e in entries)
        (d / "ledger.json").write_text(json.dumps({"spent_usd": spent, "requests": len(entries), "entries": entries}),
                                       encoding="utf-8")

    e1 = [{"key": "a", "cost_usd": 0.001}, {"key": "b", "cost_usd": 0.002}]
    e2 = e1 + [{"key": "c", "cost_usd": 0.01}, {"key": "d", "cost_usd": 0.02}, {"key": "e", "cost_usd": 0.03}]
    one, two = tmp_path / "reader_attempt1_markup", tmp_path / "reader_attempt2_pilot"
    ledger(one, e1)
    ledger(two, e2)
    out = tmp_path / "reader" / "ledger.json"
    state = br.carry_forward(out, [one, two])
    assert state["spent_usd"] == pytest.approx(0.063) and state["requests"] == 5 and state["entries"] == e2
    cf = state["carried_forward"]
    assert [a["requests"] for a in cf["attempts"]] == [2, 3]
    assert [a["spent_usd"] for a in cf["attempts"]] == [pytest.approx(0.003), pytest.approx(0.06)]
    assert cf["requests"] == 5 and state["exploratory"] is True
    # the run's ledger takes it as carried spend, and the cap counts it
    led = br.Ledger(out, cap_usd=0.063 + br.Ledger.worst(100, 4096) * 0.5)
    assert led.spent_usd == pytest.approx(0.063)
    with pytest.raises(br.CapReached):
        led.hold(br.Ledger.worst(100, 4096))
    # an existing ledger stays as it is; a broken chain is refused
    assert br.carry_forward(out, [one, two]) == json.loads(out.read_text(encoding="utf-8"))
    bad = tmp_path / "bad"
    ledger(bad, [{"key": "z", "cost_usd": 0.5}])
    with pytest.raises(ValueError, match="differ"):
        br.carry_forward(tmp_path / "x" / "ledger.json", [one, bad])


# Redaction


class FakeResponse:
    def __init__(self, status: int, body: dict | str, headers: dict | None = None):
        self.status_code, self.headers = status, headers or {}
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body


class FakeSession:
    def __init__(self, responses: list[FakeResponse]):
        self.responses, self.headers = responses, []

    def post(self, url, json=None, timeout=None, headers=None):
        self.headers.append(headers)
        return self.responses.pop(0)


def test_token_is_redacted_everywhere(tmp_path):
    echo = reply(YES, content=f"YES {TOKEN}")
    session = FakeSession([FakeResponse(500, f"upstream saw Bearer {TOKEN}"), FakeResponse(200, echo)])
    provider = br.HTTPProvider(TOKEN, allow_paid=True, session=session)
    s = run(records(1), tmp_path, provider, redact=provider.redact)
    assert s["n_scored"] == 1
    assert all(hd["Authorization"] == f"Bearer {TOKEN}" for hd in session.headers)  # sent, never stored
    for name in ("scores.jsonl", "requests.jsonl", "ledger.json"):
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert TOKEN not in text and "Authorization" not in text, name
    assert "<redacted>" in (tmp_path / "requests.jsonl").read_text(encoding="utf-8")
    with pytest.raises(br.ProviderFatal) as e:
        br.HTTPProvider(TOKEN, allow_paid=True, session=FakeSession([FakeResponse(401, f"bad {TOKEN}")])).post({})
    assert TOKEN not in str(e.value)
    with pytest.raises(br.PaidCallRefused):
        br.HTTPProvider(TOKEN).post({})


def test_http_statuses():
    def err(status, headers=None):
        prov = br.HTTPProvider(TOKEN, allow_paid=True, session=FakeSession([FakeResponse(status, "x", headers)]))
        with pytest.raises(br.ProviderError) as e:
            prov.post({})
        return e.value

    e = err(429, {"retry-after": "12"})
    assert e.retryable and not e.sent and e.retry_after == 12.0
    assert err(502).retryable and err(502).sent
    assert not err(400).retryable
    for status in (401, 402, 403):
        with pytest.raises(br.ProviderFatal):
            br.HTTPProvider(TOKEN, allow_paid=True, session=FakeSession([FakeResponse(status, "x")])).post({})


# Prereg check


class FakeApi:
    def __init__(self, states: list[list[str]]):
        self.states, self.calls = states, 0

    def dataset_info(self, repo):
        return SimpleNamespace(sha="c" * 40)

    def list_repo_files(self, repo, repo_type=None, revision=None):
        self.calls += 1
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def list_repo_commits(self, repo, repo_type=None, revision=None):
        from datetime import datetime, timezone

        return [SimpleNamespace(commit_id="c" * 40, created_at=datetime(2026, 11, 2, 7, 15, 11, tzinfo=timezone.utc),
                                title="Version 2.2")]


def test_prereg_polls_then_records_the_commit(tmp_path):
    doc = tmp_path / "STAGE_B_EXPLORATORY.md"
    doc.write_text("# Stage B\n\n**Beam reads its own windows.** Declared here.\n\n## Amendment 2.2.2, 2026-11-02\n"
                   "\n## Amendment 2.2.3, 2026-11-02\n", encoding="utf-8")
    api = FakeApi([["README.md"], ["README.md", br.PREREG_FILE]])
    slept: list[float] = []
    info = br.prereg_check(api, lambda *a, **k: str(doc), sleep=slept.append, log=lambda m: None, local=doc)
    assert slept == [120.0] and info["commit"] == "c" * 40 and info["commit_time"] == "2026-11-02T07:15:11Z"
    assert info["matches_local"] and info["exploratory"] is True and info["amendment"] == "Amendment 2.2.3"
    assert info["amendments"] == ["Amendment 2.2.2", "Amendment 2.2.3"]
    # 2.2.2 alone: it keeps polling and stops short of a paid call
    older = tmp_path / "older.md"
    older.write_text("**Beam reads its own windows.**\n\n## Amendment 2.2.2, 2026-11-02\n", encoding="utf-8")
    with pytest.raises(br.PreregMissing, match="Amendment 2.2.3"):
        br.prereg_check(FakeApi([[br.PREREG_FILE]]), lambda *a, **k: str(older), sleep=lambda s: None,
                        log=lambda m: None, local=older, max_wait_s=300.0)
    api = FakeApi([["README.md"]])
    slept = []
    with pytest.raises(br.PreregMissing):
        br.prereg_check(api, lambda *a, **k: str(doc), sleep=slept.append, log=lambda m: None, local=doc)
    assert len(slept) == 45 and api.calls == 46  # checks every 2 minutes from 0 to 90
    # the declaration alone, before the amendment: it keeps polling and stops short of a paid call
    old = tmp_path / "old.md"
    old.write_text("**Beam reads its own windows.** Declared here.\n", encoding="utf-8")
    seen: list[str] = []
    with pytest.raises(br.PreregMissing, match="Amendment 2.2.2"):
        br.prereg_check(FakeApi([[br.PREREG_FILE]]), lambda *a, **k: str(old), sleep=lambda s: None, log=seen.append,
                        local=old, max_wait_s=300.0)
    assert len(seen) == 2 and "Amendment 2.2.2" in seen[0]


# Readout statistics


def test_within_auroc_counts_pairs_inside_strata():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    s = rng.normal(size=60).round(1)  # ties
    g = rng.choice(["a", "b"], 60)
    num = den = 0.0
    for i in range(60):
        for j in range(60):
            if y[i] == 1 and y[j] == 0 and g[i] == g[j]:
                den += 1
                num += 1.0 if s[i] > s[j] else (0.5 if s[i] == s[j] else 0.0)
    got = br.within_auroc(y, s, g, np.ones((1, 60), np.int64))[0]
    assert got == pytest.approx(num / den, abs=1e-12)
    summ = br.within_summary(y, s, g, rng.choice(list("pqrstuvw"), 60), b=200)
    assert summ["auroc"] == pytest.approx(num / den) and summ["pairs"] == den and summ["ci"][0] <= summ["ci"][1]


def test_spearman_summary_point():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(1)
    a, b = rng.normal(size=40), rng.normal(size=40)
    s = br.spearman_summary(a, b, rng.choice(list("abcdefgh"), 40), b=100)
    assert s["rho"] == pytest.approx(spearmanr(a, b).statistic) and s["n"] == 40


# The real records


needs_release = pytest.mark.skipif(not (br.RELEASE / "monitors" / "qwen_meta.json").exists()
                                   or not br.RENDERED[0].exists(), reason="Stage A release files absent")


@needs_release
def test_real_records_are_the_qwen_records():
    from ouroboros.analysis import monitors
    from ouroboros.analysis.features import load_texts
    from ouroboros.analysis.populations import build_study

    study = build_study(br.ANCHORS, br.RENDERED)
    qwen = monitors.read_scores(br.RELEASE / "monitors" / "qwen_scores.jsonl")
    meta = json.loads((br.RELEASE / "monitors" / "qwen_meta.json").read_text(encoding="utf-8"))
    recs = br.build_records(study, load_texts(br.RELEASE / "features"), set(qwen), meta["aliases"])
    assert len(recs) == 681 == meta["n_records"]
    for r in recs[:50]:
        q = monitors.QUESTIONS[r["source"]]
        assert br.message(r) == q.replace("{text}", br.escape(r["text"]))
    # every window reaches Beam with its <| markers as text
    msgs = [br.message(r) for r in recs]
    assert sum("<|" in r["text"] for r in recs) > 600
    assert all(m.count("<|") == m.count("<|" + br.ZWSP) for m in msgs)
    assert all(m.replace(br.ZWSP, "") == monitors.QUESTIONS[r["source"]].format(text=r["text"])
               for r, m in zip(recs, msgs))
    with pytest.raises(ValueError, match="aliases"):
        br.build_records(study, load_texts(br.RELEASE / "features"), set(qwen), {})
    # the pilot is the first 20 records, the same windows as the 2.2.2 pilot
    assert [r["key"] for r in recs[:2]] == ["tripwire/beam-full/fastapi_11355/impossible/high|primary",
                                            "tripwire/beam-full/fastapi_11355/impossible/low|primary"]
    assert recs[19]["key"] == "tripwire/beam-full/fastapi_14099/impossible/high|primary"


@needs_release
def test_tokenizer_gate_on_all_681_windows():
    """Amendment 2.2.3: every escaped window, and every user turn, encodes with Beam's tokenizer to no added-token id;
    removing U+200B gives back the registered text exactly."""
    from tokenizers import Tokenizer

    from ouroboros.analysis import monitors
    from ouroboros.analysis.features import load_texts
    from ouroboros.analysis.populations import build_study

    study = build_study(br.ANCHORS, br.RENDERED)
    qwen = monitors.read_scores(br.RELEASE / "monitors" / "qwen_scores.jsonl")
    meta = json.loads((br.RELEASE / "monitors" / "qwen_meta.json").read_text(encoding="utf-8"))
    recs = br.build_records(study, load_texts(br.RELEASE / "features"), set(qwen), meta["aliases"])
    tk = Tokenizer.from_file(str(br.TOKENIZER))
    added = {a["id"] for a in ADDED}
    n_raw = 0
    for r in recs:
        esc = br.escape(r["text"])
        assert br.ZWSP not in r["text"]
        assert esc.replace(br.ZWSP, "") == r["text"], r["key"]
        raw = [i for i in tk.encode(r["text"], add_special_tokens=False).ids if i in added]
        n_raw += bool(raw)
        assert esc.count(br.ZWSP) == len(raw), r["key"]  # one per occurrence, and no stray "<|" in these windows
        assert not added & set(tk.encode(esc, add_special_tokens=False).ids), r["key"]
        assert not added & set(tk.encode(br.message(r), add_special_tokens=False).ids), r["key"]
    assert n_raw > 600
    gate = br.tokenizer_gate(recs)
    assert gate["passed"] and gate["n_windows"] == 681 and gate["exploratory"] is True
    assert gate["escaped"]["windows_with_added_ids"] == gate["escaped"]["user_turns_with_added_ids"] == 0
    assert gate["roundtrip"]["windows_restored_exactly"] == gate["roundtrip"]["windows_one_zwsp_per_occurrence"] == 681
    assert set(gate["raw"]["occurrences"]) == {"<|channel|>", "<|message|>", '<rfl:parameter name="',
                                               "\n</rfl:parameter>", "<|end|>", "<|start|>", "<|constrain|>",
                                               "<|call|>"}
    # the 2.2.2 escape alone leaves the two tool-call tokens, as the pilot found
    left = {i for r in recs for i in tk.encode(r["text"].replace("<|", "<|" + br.ZWSP),
                                               add_special_tokens=False).ids if i in added}
    assert left == {151679, 151680}


def test_threads_do_not_lose_rows(tmp_path):
    lock = threading.Lock()
    seen: list[str] = []

    def respond(body):
        with lock:
            seen.append(body["messages"][0]["content"])
        return reply(YES)

    s = run(records(30), tmp_path, br.StubProvider(respond), workers=6)
    rows = (tmp_path / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    assert s["n_written"] == 30 == len(rows) == len(set(seen))
