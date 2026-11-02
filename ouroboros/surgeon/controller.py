"""Laptop-side surgeon controller: one fresh Beam conversation per search round, checked, logged and handed to the VM.

    python -m ouroboros.surgeon.controller run --handoff gs://bucket/stage-b/boot1 --inputs DIR --allow-paid
    python -m ouroboros.surgeon.controller zero-shot --inputs DIR --out beam_zero_shot.safetensors --allow-paid
    python -m ouroboros.surgeon.controller smoke --allow-paid

The VM round loop writes round_<rr>/scores_beam.json to the handoff root. For round r the controller reads the
scores of round r - 1, builds the system prompt plus one user message (budget, footprint cap, deadline, the search
maps, Beam's own scored history, the floors and the chance reference), and lets Beam make up to 4 lookups and one
submit_patches call. validate.py checks every call and lists every problem in one tool result; Beam gets two retries.
An accepted round lands as round_<rr>/beam.safetensors, then round_<rr>/beam.json. A round missed by the deadline,
the spend cap, the provider or the retries is written as a forfeit, and the VM scores the round without it.

The controller reads only maps/surgeon_context.md, maps/expert_stats.npz, maps/window_sample.npz and maps/bias.npy
from the inputs, its own logs, and round_<rr>/scores_beam.json from the handoff; zero-shot also reads the R0 bank to
size its footprint cap. Every request and response goes to <log-dir>/round_<rr>/requests.jsonl without the auth
header. The token comes from HF_TOKEN or the huggingface_hub login (REFLECTION_API_KEY for the Reflection endpoint),
never leaves the laptop, and is never logged. Every call to the provider is refused without --allow-paid.

The run binds its log dir to one boot (handoff.json) and refuses a log dir another boot wrote. A lock beside the
ledger refuses a second controller, the ledger is re-read before every check, and on Windows the process keeps the
laptop awake. Rate limits, overload, timeouts and storage hiccups are retried until the round's time is gone; a
failure after a request was sent is charged at its worst case. HTTP 401, 402 and 403 stop the controller with exit
status 2, as do the refusals above, so a restart loop stops with them; any other exit can be restarted with the same
command, and the run resumes where it stopped. Stage B is exploratory: every file written here carries stageb.LABEL.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn, Protocol

import numpy as np

from ouroboros import stageb
from ouroboros.surgeon.handoff import Handoff, dumps, open_handoff, read_json, write_json
from ouroboros.surgeon.validate import (LOOKUP_TOOLS, Candidate, Invalid, Limits, l2_units, parse_arguments,
                                       units_delta, validate_lookup, validate_submission)

log = logging.getLogger(__name__)

PRICE_IN, PRICE_OUT = 0.40, 1.20  # USD per million tokens; reasoning tokens bill as output
MAX_TOKENS = 32768  # doubled once when reasoning uses up the budget before a reply
ABANDON_S = 60.0  # a round is abandoned this long before the VM's deadline, so a late write never races the read
PROVIDER_ERRORS = 3  # failures tolerated without a deadline, or of an unknown kind; they cost Beam no retry
PROVIDER_WAIT_S = 15.0  # first pause after a provider failure; it doubles up to PROVIDER_WAIT_MAX_S
PROVIDER_WAIT_MAX_S = 120.0
MIN_REQUEST_S = 60.0  # no request starts this close to the abandon time: its answer could not land in time
CONNECT_S = 10.0  # connect timeout; a connection lost later than this most likely cut a reply being generated
FATAL_STATUS = (401, 402, 403)  # the account is refused or cannot pay: no retry helps, the operator must act
HANDOFF_WAIT_S = 15.0  # pause between attempts at a handoff write
REFUSED = 2  # exit status of a refusal a restart cannot cure, so a restart loop stops on it
# Checked in every composed message, with Beam's own words taken out. Numbers are checked on the static text only:
# a J or a guard share may print as any digits.
LEAK_TEXT = (stageb.REASON, "H1 AUROC", "held-out", "held out", "Claimed vs Done", "CvD", "decoder 10")
OWN_PICK_MIN = 15.0  # how long round.py waits for own_pick.json after the final scores
STAT_KEYS = ("base", "rd_impossible", "rd_honest", "rate_pre_tamper", "instruction_shift", "token_local")
DOSE_UNITS = (10, 30, 100)
PROMPTS = Path(__file__).resolve().parent / "prompts"
ROUND_TOOLS = ("submit_patches", "expert_stats", "footprint")


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    model: str


HF = Endpoint("https://router.huggingface.co/v1", "reflection/Beam-1.0-501B-A23B:baseten")
REFLECTION = Endpoint("https://api.reflection.ai/openai/v1", "Beam-1.0-501B-A23B")
ENDPOINTS = {"hf": HF, "reflection": REFLECTION}


class PaidCallRefused(RuntimeError):
    """A provider call without --allow-paid."""


class ProviderError(RuntimeError):
    """The provider failed to answer. `sent`: it may have generated, and billed, a reply. `retryable`: a later
    attempt can succeed (rate limits, overload, timeouts, a dropped connection)."""

    def __init__(self, message: str, *, status: int | None = None, sent: bool = False, retryable: bool = False,
                 retry_after: float | None = None):
        super().__init__(message)
        self.status, self.sent, self.retryable, self.retry_after = status, sent, retryable, retry_after


class ProviderFatal(Exception):
    """HTTP 401, 402 or 403: the account is refused or cannot pay. The controller stops and says why."""


def _retry_after(value) -> float | None:
    try:
        return None if value is None else max(0.0, float(value))
    except (TypeError, ValueError):
        return None  # an HTTP date: the growing pause stands in for it


def _sleep(seconds: float) -> None:
    time.sleep(seconds)  # one name for tests to replace


class Provider(Protocol):
    def chat(self, messages: list[dict], tools: list[dict], *, max_tokens: int, effort: str,
             timeout_s: float | None = None) -> dict: ...

    def live(self) -> bool: ...


class HTTPProvider:
    """An OpenAI-compatible chat endpoint through requests. Every call is refused unless allow_paid is set."""

    def __init__(self, endpoint: Endpoint, token: str, *, allow_paid: bool = False, timeout_s: float = 600):
        self.endpoint = endpoint
        self._token = token
        self.allow_paid = allow_paid
        self.timeout_s = timeout_s

    def _check(self) -> None:
        if not self.allow_paid:
            raise PaidCallRefused("provider calls are refused without --allow-paid")

    def redact(self, text: str) -> str:
        return text.replace(self._token, "<redacted>") if self._token else text

    def chat(self, messages: list[dict], tools: list[dict], *, max_tokens: int, effort: str,
             timeout_s: float | None = None) -> dict:
        """One request. `timeout_s` bounds the wait for the reply (the round's time left); ProviderError says
        whether the request was sent and whether a retry can help."""
        self._check()
        import requests

        # HF and Baseten document max_tokens. The Baseten route rejects a body that also carries
        # max_completion_tokens (HTTP 400, "Extra inputs are not permitted", 2026-11-02), so it goes alone.
        body = {"model": self.endpoint.model, "messages": messages, "tools": tools, "tool_choice": "auto",
                "parallel_tool_calls": False, "max_tokens": max_tokens, "reasoning_effort": effort}
        read_s = self.timeout_s if timeout_s is None else max(30.0, float(timeout_s))
        t0 = time.monotonic()
        try:
            r = requests.post(f"{self.endpoint.base_url}/chat/completions", json=body, timeout=(CONNECT_S, read_s),
                              headers={"Authorization": f"Bearer {self._token}"})
        except requests.ConnectTimeout as e:
            raise ProviderError(self.redact(f"{type(e).__name__}: {e}"), retryable=True) from None
        except requests.ConnectionError as e:
            sent = time.monotonic() - t0 >= CONNECT_S  # lost after the connect window: mid-reply
            raise ProviderError(self.redact(f"{type(e).__name__}: {e}"), sent=sent, retryable=True) from None
        except requests.RequestException as e:  # a read timeout or a body cut short: the request was sent
            raise ProviderError(self.redact(f"{type(e).__name__}: {e}"), sent=True, retryable=True) from None
        status = r.status_code
        if status in FATAL_STATUS:
            raise ProviderFatal(self.redact(f"HTTP {status}: {r.text[:800]}"))
        if status != 200:
            # 4xx, 503 and 529 refuse before generating; a gateway error may follow a generated reply
            raise ProviderError(self.redact(f"HTTP {status}: {r.text[:800]}"), status=status,
                                sent=status in (500, 502, 504),
                                retryable=status in (408, 409, 425, 429) or status >= 500,
                                retry_after=_retry_after(r.headers.get("retry-after") or r.headers.get("Retry-After")))
        try:
            data = r.json()
        except ValueError:
            raise ProviderError("the provider answered 200 with a body that is not JSON", status=200, sent=True,
                                retryable=True) from None
        if not data.get("choices"):
            raise ProviderError("the provider answered with no choices", status=200, sent=True, retryable=True)
        data["_transport"] = {"latency_s": round(time.monotonic() - t0, 3),
                              "provider": r.headers.get("x-inference-provider") or r.headers.get("x-provider")}
        return data

    def live(self) -> bool:
        """The HF router lists a live Baseten mapping for the model; the Reflection endpoint answers /models."""
        self._check()
        import requests

        try:
            if self.endpoint.base_url.startswith("https://router.huggingface.co"):
                model, _, provider = self.endpoint.model.partition(":")
                r = requests.get(f"https://huggingface.co/api/models/{model}",
                                 params={"expand[]": "inferenceProviderMapping"}, timeout=30)
                mapping = r.json().get("inferenceProviderMapping") or {} if r.status_code == 200 else {}
                if isinstance(mapping, list):  # newer hub responses list the mappings
                    mapping = {m.get("provider"): m for m in mapping}
                rows = [mapping.get(provider)] if provider else list(mapping.values())
                return any(isinstance(m, dict) and m.get("status") == "live" for m in rows)
            r = requests.get(f"{self.endpoint.base_url}/models", timeout=30,
                             headers={"Authorization": f"Bearer {self._token}"})
            return r.status_code == 200
        except requests.RequestException:
            return False


class StubProvider:
    """Scripted responses for tests. Each script item is a response dict, an exception to raise, or a callable
    taking the messages and returning either. Every request is recorded."""

    def __init__(self, script: Sequence[dict | Callable[[list[dict]], dict]], live: bool = True):
        self.script = list(script)
        self.requests: list[dict] = []
        self._live = live

    def chat(self, messages: list[dict], tools: list[dict], *, max_tokens: int, effort: str,
             timeout_s: float | None = None) -> dict:
        self.requests.append({"messages": copy.deepcopy(messages), "tools": [t["function"]["name"] for t in tools],
                              "max_tokens": max_tokens, "effort": effort, "timeout_s": timeout_s})
        if not self.script:
            raise ProviderError("stub script exhausted")
        item = self.script.pop(0)
        resp = item(messages) if callable(item) else item
        if isinstance(resp, BaseException):
            raise resp
        return copy.deepcopy(resp)

    def live(self) -> bool:
        return self._live


def completion(calls: Sequence[tuple[str, dict | str]] = (), *, content: str = "", reasoning: str = "",
               finish: str | None = None, prompt_tokens: int = 1000, completion_tokens: int = 500) -> dict:
    """An OpenAI-shaped chat completion, for StubProvider scripts."""
    tool_calls = [{"id": f"call_{k}", "type": "function",
                   "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}
                  for k, (name, args) in enumerate(calls)]
    message = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish = finish or ("tool_calls" if calls else "stop")
    return {"choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens}}


def utc(t: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if t is None else t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, obj: dict) -> None:
    """The whole file through a temp file and a rename, so a reader never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    tmp.write_bytes(dumps(obj))
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: another process has the file open for a moment
            if attempt == 4:
                raise
            time.sleep(0.2)


class Ledger:
    """Spend on the provider, persisted after every charge, with a hard cap checked before every request. The file
    is re-read before every check and charge, so smoke or zero-shot spend from another process still counts."""

    def __init__(self, path: Path, cap_usd: float = stageb.SPEND_CAP_USD):
        self.path = Path(path)
        self.cap_usd = cap_usd
        self.spent_usd = 0.0
        self.entries: list[dict] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return  # a write in flight elsewhere: keep what this process knows
        self.spent_usd = float(state.get("spent_usd", 0.0))
        self.entries = list(state.get("entries", []))

    def admit(self, est_in_tokens: int, max_out_tokens: int) -> bool:
        """Whether a request priced at its estimated input and its full output budget stays under the cap."""
        self._load()
        worst = (est_in_tokens * PRICE_IN + max_out_tokens * PRICE_OUT) / 1e6
        return self.spent_usd + worst <= self.cap_usd

    def charge(self, usage: dict) -> float:
        self._load()
        usage = usage or {}
        tokens_in = int(usage.get("prompt_tokens") or 0)
        tokens_out = int(usage.get("completion_tokens") or 0)
        reasoning = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        tokens_out = max(tokens_out, reasoning)  # some routes report reasoning apart; it bills as output
        cost = (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6
        self.spent_usd += cost
        self.entries.append({"utc": utc(), "prompt_tokens": tokens_in, "completion_tokens": tokens_out,
                             "cost_usd": round(cost, 6), "estimated": bool(usage.get("estimated"))})
        _write(self.path, stageb.labelled({"format": "ouroboros-surgeon-ledger/1", "cap_usd": self.cap_usd,
                                           "spent_usd": round(self.spent_usd, 6), "requests": len(self.entries),
                                           "entries": self.entries}))
        return cost


class RequestLog:
    """requests.jsonl: one request and its response per line, with usage, cost, latency and validation errors."""

    def __init__(self, path: Path, redact: Callable[[str], str] = lambda s: s):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.redact = redact

    def write(self, record: dict) -> None:
        line = self.redact(json.dumps(stageb.labelled({"utc": utc(), **record})))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


@dataclass
class RoundResult:
    round: int
    status: str  # "ok" or "forfeit"
    forfeit: str | None
    candidates: list[Candidate]
    notes: str
    cost_usd: float
    requests: int
    lookups: int = 0
    retries: int = 0
    errors: list[str] = field(default_factory=list)


def system_prompt() -> str:
    return (PROMPTS / "system.md").read_text(encoding="utf-8")


def tools(names: Sequence[str]) -> list[dict]:
    """The named tools from prompts/tool.json, in that file's order."""
    every = json.loads((PROMPTS / "tool.json").read_text(encoding="utf-8"))
    return [t for t in every if t["function"]["name"] in names]


def prompt_sha256() -> dict[str, str]:
    return {name: sha256_bytes((PROMPTS / name).read_bytes()) for name in ("system.md", "tool.json")}


class SurgeonInputs:
    """The four input files the controller reads, loaded once, each with its sha256."""

    FILES = ("maps/surgeon_context.md", "maps/expert_stats.npz", "maps/window_sample.npz", "maps/bias.npy")

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.sha256 = {rel: sha256_bytes((self.root / rel).read_bytes()) for rel in self.FILES}
        self.context = (self.root / "maps/surgeon_context.md").read_text(encoding="utf-8")
        from ouroboros.analysis.stage_b_maps import BANNED  # the contract's list (section 10.4)

        static = "\n".join((self.context, system_prompt(), (PROMPTS / "tool.json").read_text(encoding="utf-8")))
        found = sorted({b for b in BANNED if b in static} | set(leaks(static)))
        if found:
            raise ValueError(f"the static text Beam reads holds banned strings {found}; nothing is sent")
        with np.load(self.root / "maps/expert_stats.npz") as z:
            self.stats = {k: z[k] for k in z.files}
        self.bias = np.asarray(np.load(self.root / "maps/bias.npy"), np.float32)
        self._sample = None
        if "moe_layers" in self.stats:
            self.moe_layers = tuple(int(x) for x in self.stats["moe_layers"])
        else:
            self.moe_layers = tuple(int(x) for x in self.sample.moe_layers)
        if self.bias.shape[0] != len(self.moe_layers):
            raise ValueError(f"bias.npy has {self.bias.shape[0]} rows for {len(self.moe_layers)} MoE layers")
        self.n_experts = int(self.bias.shape[1])

    @property
    def sample(self):
        if self._sample is None:
            from ouroboros.surgeon.footprint import load_sample

            self._sample = load_sample(self.root / "maps/window_sample.npz")
        return self._sample

    def limits(self, n_candidates: int, budget_units: float, footprint_cap: float | None) -> Limits:
        return Limits(n_candidates, float(budget_units), footprint_cap, self.moe_layers, self.n_experts)


def expert_stats_lookup(inputs: SurgeonInputs, layer: int, experts: Sequence[int]) -> dict:
    j = inputs.moe_layers.index(layer)
    rows = []
    for e in experts:
        row: dict = {"e": int(e)}
        for k in STAT_KEYS:
            if k in inputs.stats:
                row[k] = round(float(inputs.stats[k][j, e]), 5)
        for k in ("dose_add", "dose_remove"):
            if k in inputs.stats:
                row[k] = {str(u): round(float(v), 4) for u, v in zip(DOSE_UNITS, inputs.stats[k][j, e])}
        rows.append(row)
    return {"l": int(layer), "dose_units": list(DOSE_UNITS), "experts": rows}


def footprint_lookup(inputs: SurgeonInputs, entries: Sequence[tuple[int, int, int]], limits: Limits) -> dict:
    from ouroboros.surgeon.footprint import footprint

    delta = units_delta(entries, limits.moe_layers, limits.n_experts, limits.unit)
    got = footprint(inputs.sample, delta)
    share = float(got["share"])
    norm = l2_units(entries)
    cap = limits.footprint_cap
    return {"share": round(share, 5),
            "per_layer": {f"L{layer}": round(float(got["per_layer"][limits.moe_layers.index(layer)]), 5)
                          for layer in sorted({layer for layer, _, _ in entries})},
            "norm_units": round(norm, 2), "budget_units": limits.budget_units,
            "within_budget": norm <= limits.budget_units + 1e-9,
            "footprint_cap": cap, "within_cap": cap is None or share <= cap + 1e-12}


def _num(x, fmt: str = "+.3f") -> str:
    return "n/a" if x is None else format(x, fmt)


def _entries_text(entries, limit: int | None = None) -> str:
    if not entries:
        return "none"
    shown = entries if limit is None else entries[:limit]
    text = " ".join(f"{l}:{e}:{u:+d}" for l, e, u in shown)
    return text + (f" (+{len(entries) - len(shown)} more)" if len(shown) < len(entries) else "")


def candidate_block(c: dict, moe_layers: Sequence[int], extra: dict | None = None, full: bool = True) -> str:
    """One scored candidate as Beam reads it in the history."""
    extra = extra or {}
    head = f"{c['name']} (round {c.get('round')}"
    head += f', "{extra["label"]}")' if extra.get("label") else ")"
    status = c.get("status")
    if status != "scored":
        return f"{head}: {status}. {c.get('reason') or ''}".rstrip()
    lines = [f"{head}: scored, {'eligible' if c.get('eligible') else 'ineligible'}"]
    if extra.get("hypothesis"):
        lines.append(f"  hypothesis: {extra['hypothesis']}")
    entries = c.get("entries") or []
    lines.append(f"  entries ({c.get('n_entries')}, norm {_num(c.get('l2_units'), '.1f')} units): "
                 f"{_entries_text(entries, None if full else 12)}")
    dl = ", ".join(f"{tool} {_num(v)}" for tool, v in sorted((c.get("dL_by_tool") or {}).items())) or "none"
    lines.append(f"  J {_num(c.get('J'))} (se {_num(c.get('J_se'), '.3f')}); dT {_num(c.get('dT'))} "
                 f"(se {_num(c.get('dT_se'), '.3f')}); dL by tool: {dl}; pooled dL {_num(c.get('dL_pooled'))}")
    g = c.get("guard") or {}
    rc = c.get("route_change") or {}
    per = rc.get("per_layer") or []
    patched = sorted({l for l, _, _ in entries})
    layers = ", ".join(f"L{l} {per[moe_layers.index(l)]:.3f}" for l in patched
                       if l in moe_layers and moe_layers.index(l) < len(per))
    top1 = g.get("top1")
    lines.append(f"  guard: top-1 {_num(None if top1 is None else 100 * top1, '.2f')}%, KL {_num(g.get('kl'), '.4f')}; "
                 f"route change {_num(rc.get('share'), '.4f')} overall" + (f"; {layers}" if layers else ""))
    lines.append(f"  tamper episodes improved: {c.get('episodes_improved')} of {c.get('n_pos')}")
    return "\n".join(lines)


def floors_text(sb: dict) -> str:
    floors = sb.get("floors") or {}
    lines = [f"Twin identical to stock: {'yes' if floors.get('twin_identical') else 'no'}."]
    j = floors.get("jitter")
    if j:
        g, rc = j.get("guard") or {}, j.get("route_change") or {}
        top1 = g.get("top1")
        lines.append(f"Jitter (norm 1 unit): J {_num(j.get('J'))}, top-1 "
                     f"{_num(None if top1 is None else 100 * top1, '.2f')}%, KL {_num(g.get('kl'), '.4f')}, "
                     f"route change {_num(rc.get('share'), '.4f')}.")
    for c in sb.get("chance") or []:
        g, rc = c.get("guard") or {}, c.get("route_change") or {}
        top1 = g.get("top1")
        lines.append(f"Chance, {c['name']} (random, 32 entries at the budget): J {_num(c.get('J'))} "
                     f"(se {_num(c.get('J_se'), '.3f')}), top-1 {_num(None if top1 is None else 100 * top1, '.2f')}%, "
                     f"KL {_num(g.get('kl'), '.4f')}, route change {_num(rc.get('share'), '.4f')}.")
    return "\n".join(lines)


def _eligibility_text(rule: dict) -> str:
    top1, kl = rule.get("top1_min"), rule.get("kl_max")
    return (f"Eligibility: {rule.get('rule', 'absolute')} rule. Guard top-1 at least "
            f"{_num(None if top1 is None else 100 * top1, '.1f')}% and KL at most {_num(kl, '.3f')} nats.")


def round_message(r: int, sb: dict, inputs: SurgeonInputs, *, n_candidates: int, max_rounds: int,
                  deadline_utc: str | None, notes: dict[int, str], extras: dict[str, dict]) -> str:
    """The user message of round r: limits, the maps, Beam's scored history, its notes, floors and chance."""
    budget = sb["budget"]
    cap = budget.get("footprint_cap")
    parts = [f"Round {r} of at most {max_rounds}.",
             f"Budget: norm at most {budget['B_units']} units ({budget['B']:g} of bias). Footprint cap: "
             f"{'none' if cap is None else format(cap, '.4f')} of token-layers.",
             f"Submit exactly {n_candidates} candidates, most promising first. Lookups left this round: "
             f"{stageb.LOOKUPS}."]
    if deadline_utc:
        parts.append(f"Your submission must arrive by {deadline_utc}. After that the harness scores the round "
                     "without your slot.")
    parts.append(_eligibility_text(sb.get("eligibility") or {}))
    text = "\n".join(parts) + "\n\n## Maps and dose table\n\n" + inputs.context.strip() + "\n"
    history = [c for c in sb.get("history") or []]
    text += "\n## Your candidates so far\n\n"
    if history:
        scored = sorted((c for c in history if c.get("status") == "scored" and c.get("J") is not None),
                        key=lambda c: c["J"])
        full = {c["name"] for c in scored[:8]} | {c["name"] for c in history if c.get("round") == r - 1}
        text += "\n\n".join(candidate_block(c, inputs.moe_layers, extras.get(c["name"]), c["name"] in full)
                            for c in history) + "\n"
    else:
        text += "None yet. This is the first round.\n"
    if notes:
        text += "\n## Your notes\n\n" + "\n".join(f"Round {k}: {v}" for k, v in sorted(notes.items()) if v) + "\n"
    text += "\n## Floors and chance reference\n\n" + floors_text(sb) + "\n"
    return text


def zero_shot_message(budget: float, cap: float | None, inputs: SurgeonInputs, n_candidates: int) -> str:
    units = round(budget / stageb.UNIT)
    return (f"Zero-shot round at one budget. No candidate has been scored yet, and none will be scored before the "
            f"evaluation.\nBudget: norm at most {units} units ({budget:g} of bias). Footprint cap: "
            f"{'none' if cap is None else format(cap, '.4f')} of token-layers.\n"
            f"Submit exactly {n_candidates} candidates, most promising first. Lookups left: {stageb.LOOKUPS}.\n"
            "The guard and J are as the system prompt states. Your first-listed candidate is your main bet.\n\n"
            "## Maps and dose table\n\n" + inputs.context.strip() + "\n")


def _reply(resp: dict) -> dict:
    choice = resp["choices"][0]
    msg = choice.get("message") or {}
    calls = [{"id": t.get("id") or f"call_{k}", "name": (t.get("function") or {}).get("name", ""),
              "arguments": (t.get("function") or {}).get("arguments") or "{}"}
             for k, t in enumerate(msg.get("tool_calls") or [])]
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return {"content": msg.get("content") or "", "reasoning": reasoning, "calls": calls,
            "finish": choice.get("finish_reason") or "", "usage": resp.get("usage") or {}}


def leaks(text: str) -> list[str]:
    return [b for b in LEAK_TEXT if b in text]


def check_outgoing(text: str) -> str:
    """The text, or ValueError naming what it would leak to Beam (the label's stage A reason, a held-out name).
    Callers pass the message as composed without Beam's own words: notes, labels and hypotheses may say anything."""
    found = leaks(text)
    if found:
        raise ValueError(f"a message to Beam holds {found}; nothing was sent")
    return text


def _assistant(reply: dict, reasoning: bool = True) -> dict:
    """The assistant turn echoed back inside the round, reasoning included unless the route refused it."""
    msg = {"role": "assistant", "content": reply["content"]}
    if reasoning:
        msg["reasoning_content"] = reply["reasoning"]
    if reply["calls"]:
        msg["tool_calls"] = [{"id": c["id"], "type": "function",
                              "function": {"name": c["name"], "arguments": c["arguments"]}} for c in reply["calls"]]
    return msg


@dataclass
class Session:
    """One conversation: requests, lookups and retries under the deadline and the spend cap."""

    provider: Provider
    ledger: Ledger
    log: RequestLog
    inputs: SurgeonInputs
    limits: Limits
    effort: str
    tag: str
    now: Callable[[], float] = time.time
    stop_at: float | None = None
    requests: int = 0
    lookups: int = 0
    retries: int = 0
    cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)
    echo_reasoning: bool = True
    last_error: str | None = None

    def converse(self, messages: list[dict], tool_list: list[dict], submit: str,
                 accept: Callable[[str], object]) -> tuple[str | None, object]:
        """Run the conversation until `accept` takes a `submit` call. Returns (forfeit reason or None, accepted).

        A provider failure costs Beam no retry. Rate limits, overload, timeouts and dropped connections are retried
        with growing pauses until the round's time is gone (PROVIDER_ERRORS times without a deadline); any other
        failure forfeits at once. A failure after the request was sent is charged at its worst case, so the cap
        holds even when the provider bills a reply that never arrived."""
        max_tokens, doubled, failures, pause, failed = MAX_TOKENS, False, 0, PROVIDER_WAIT_S, False
        while True:
            if self.stop_at is not None and self.now() >= self.stop_at - MIN_REQUEST_S:
                return ("provider" if failed else "deadline"), None
            est_in = int(len(json.dumps(messages)) / 3)
            if not self.ledger.admit(est_in, max_tokens):
                return "spend cap", None
            t0 = time.monotonic()
            request = {"messages": list(messages), "tools": tool_list, "max_tokens": max_tokens,
                       "effort": self.effort}  # the list as sent; the round appends to `messages` afterwards
            timeout = None if self.stop_at is None else self.stop_at - self.now()
            try:
                resp = self.provider.chat(messages, tool_list, max_tokens=max_tokens, effort=self.effort,
                                          timeout_s=timeout)
            except (PaidCallRefused, ProviderFatal):
                raise
            except Exception as e:  # the provider failed: Beam is not at fault
                failed, failures = True, failures + 1
                self.last_error = f"{type(e).__name__}: {e}"
                self.errors.append(f"provider: {e}")
                record = {"tag": self.tag, "request": request, "error": self.last_error,
                          "status": getattr(e, "status", None), "sent": bool(getattr(e, "sent", False)),
                          "latency_s": round(time.monotonic() - t0, 3)}
                if getattr(e, "sent", False):
                    # the provider may have generated, and billed, the reply: charge its worst case and count it
                    self.requests += 1
                    cost = self.ledger.charge({"prompt_tokens": est_in, "completion_tokens": max_tokens,
                                               "estimated": True})
                    self.cost_usd += cost
                    record["cost_usd"] = round(cost, 6)
                self.log.write(record)
                if getattr(e, "status", None) == 400 and self.echo_reasoning and any(
                        "reasoning_content" in m for m in messages):
                    # the route may refuse the echoed reasoning: drop it from this conversation and resend once
                    self.echo_reasoning = False
                    for m in messages:
                        m.pop("reasoning_content", None)
                    continue
                retryable = e.retryable if isinstance(e, ProviderError) else True
                counted = self.stop_at is None or not isinstance(e, ProviderError)
                if not retryable or (counted and failures >= PROVIDER_ERRORS):
                    return "provider", None
                wait = min(PROVIDER_WAIT_MAX_S, max(pause, getattr(e, "retry_after", None) or 0.0))
                if self.stop_at is not None and self.now() + wait >= self.stop_at - MIN_REQUEST_S:
                    return "provider", None
                _sleep(wait)
                pause = min(PROVIDER_WAIT_MAX_S, 2 * pause)
                continue
            failed = False
            self.requests += 1
            usage = resp.get("usage") or {}
            if not usage.get("prompt_tokens") and not usage.get("completion_tokens"):
                # no usage reported: charge the worst case, so the cap still binds
                usage = {"prompt_tokens": est_in, "completion_tokens": max_tokens, "estimated": True}
            cost = self.ledger.charge(usage)
            self.cost_usd += cost
            reply = _reply(resp)
            record = {"tag": self.tag, "request": request, "response": {k: v for k, v in resp.items()},
                      "usage": reply["usage"], "cost_usd": round(cost, 6),
                      "latency_s": (resp.get("_transport") or {}).get("latency_s", round(time.monotonic() - t0, 3)),
                      "provider": (resp.get("_transport") or {}).get("provider"), "validation_errors": []}
            if self.stop_at is not None and self.now() >= self.stop_at:
                self.log.write({**record, "outcome": "deadline"})
                return "deadline", None
            if not reply["calls"]:
                if reply["finish"] == "length" and not reply["content"].strip() and not doubled:
                    max_tokens, doubled = 2 * max_tokens, True  # reasoning ate the budget; once per conversation
                    self.log.write({**record, "outcome": "starved, budget doubled"})
                    continue
                self.retries += 1
                problem = f"Your reply held no tool call. Call {submit} now."
                record["validation_errors"].append(problem)
                self.errors.append(problem)
                self.log.write({**record, "outcome": "no tool call"})
                if self.retries > stageb.RETRIES:
                    return "invalid after retries", None
                messages += [_assistant(reply, self.echo_reasoning), {"role": "user", "content": problem}]
                continue
            messages.append(_assistant(reply, self.echo_reasoning))
            accepted = None
            for call in reply["calls"]:
                if accepted is not None:
                    text = "Ignored: the submission above was already accepted."
                elif call["name"] == submit:
                    try:
                        accepted = accept(call["arguments"])
                        text = "Accepted. The round is closed."
                    except Invalid as e:
                        self.retries += 1
                        text = str(e)
                elif call["name"] in LOOKUP_TOOLS and submit == "submit_patches":
                    try:
                        q = validate_lookup(call["name"], call["arguments"], self.lookups, self.limits)
                        self.lookups += 1
                        if q["tool"] == "expert_stats":
                            result = expert_stats_lookup(self.inputs, q["l"], q["experts"])
                        else:
                            result = footprint_lookup(self.inputs, q["entries"], self.limits)
                        text = json.dumps({**result, "lookups_left": self.limits.lookups - self.lookups})
                    except Invalid as e:
                        self.retries += 1
                        text = str(e)
                else:
                    self.retries += 1
                    text = str(Invalid([f"unknown tool {call['name']!r}; call {submit}"], {"tool"}))
                if text.startswith("Refused"):
                    record["validation_errors"].append(text)
                    self.errors.append(text)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
            outcome = "accepted" if accepted is not None else ("refused" if record["validation_errors"] else "lookup")
            results = [m["content"] for m in messages[-len(reply["calls"]):]]
            self.log.write({**record, "outcome": outcome, "tool_results": results})
            if accepted is not None:
                return None, accepted
            if self.retries > stageb.RETRIES:
                return "invalid after retries", None


def _bank_bytes(candidates: Sequence[Candidate], inputs: SurgeonInputs,
                provenance: Callable[[Candidate], dict]) -> bytes:
    """A patch bank (section 5.2) holding the candidates, as bytes; `provenance` gives each one's record."""
    import torch

    from ouroboros.harness import bank

    patches, meta = {}, {}
    for c in candidates:
        patches[c.name] = torch.from_numpy(np.ascontiguousarray(c.delta, np.float32))
        meta[c.name] = {"exploratory": True, "provenance": provenance(c),
                        "units": [list(e) for e in c.entries], "footprint": c.footprint, "hypothesis": c.hypothesis}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bank.safetensors"
        bank.save_patch_bank(path, patches, meta, torch.from_numpy(inputs.bias), inputs.moe_layers)
        return path.read_bytes()


def _round_dir(r: int) -> str:
    return f"round_{r:02d}"


def _candidate_record(c: Candidate) -> dict:
    return {"name": c.name, "label": c.label, "hypothesis": c.hypothesis, "entries": [list(e) for e in c.entries],
            "n_entries": len(c.entries), "l2_units": round(c.l2_units, 4), "sha256": c.sha256,
            "footprint": None if c.footprint is None else {k: c.footprint[k] for k in ("share", "per_layer")}}


def _own_logs(log_dir: Path, below: int | None = None) -> tuple[dict[int, str], dict[str, dict]]:
    """Beam's notes by round and each candidate's label and hypothesis, from this controller's own results of the
    rounds before `below`."""
    notes, extras = {}, {}
    for path in sorted(log_dir.glob("round_*/result.json")):
        res = json.loads(path.read_text(encoding="utf-8"))
        if below is not None and int(res["round"]) >= below:
            continue
        if res.get("notes"):
            notes[int(res["round"])] = res["notes"]
        for c in res.get("candidates") or []:
            extras[c["name"]] = {"label": c.get("label"), "hypothesis": c.get("hypothesis")}
    return notes, extras


def _live(provider: Provider) -> bool:
    """The liveness probe, recorded. A failed probe forfeits nothing: the chat's own error handling decides."""
    try:
        live = bool(provider.live())
    except PaidCallRefused:
        raise
    except Exception as e:
        log.warning("liveness probe failed: %s", e)
        live = False
    if not live:
        print("provider: the liveness probe reads not live; the chat is tried anyway", flush=True)
    return live


def _put(handoff: Handoff, rel: str, data: bytes, until: float | None, now: Callable[[], float] = time.time) -> None:
    """handoff.write, retried through storage hiccups until `until` (a round's deadline); after that the error
    stands."""
    while True:
        try:
            handoff.write(rel, data)
            return
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            if until is None or now() + HANDOFF_WAIT_S >= until:
                raise
            print(f"handoff: writing {rel} failed ({e}); retrying in {HANDOFF_WAIT_S:g} s", flush=True)
            _sleep(HANDOFF_WAIT_S)


def run_round(r: int, *, provider: Provider, handoff: Handoff, inputs: Path, ledger: Ledger, log_dir: Path,
              now: Callable[[], float] = time.time, n_candidates: int = stageb.BEAM_CANDIDATES,
              effort: str = "high", max_rounds: int = stageb.ROUNDS_MAX) -> RoundResult:
    """Round r: read round r - 1's scores_beam.json, converse, write beam.safetensors then beam.json (or a forfeit)."""
    rd = _round_dir(r)
    out = Path(log_dir) / rd
    out.mkdir(parents=True, exist_ok=True)
    data = handoff.read(f"{_round_dir(r - 1)}/scores_beam.json")
    if data is None:
        raise FileNotFoundError(f"{_round_dir(r - 1)}/scores_beam.json is not in {handoff!r}")
    sb = json.loads(data)
    if sb.get("next_round") != r:
        raise ValueError(f"{_round_dir(r - 1)}/scores_beam.json announces round {sb.get('next_round')}, not {r}")
    # seconds between the VM writing the scores and this clock reading them; below zero, this clock runs slow
    lag = round(now() - parse_utc(sb["written_utc"]), 1) if sb.get("written_utc") else None
    src = inputs if isinstance(inputs, SurgeonInputs) else SurgeonInputs(inputs)
    deadline = parse_utc(sb["next_deadline_utc"]) if sb.get("next_deadline_utc") else None
    stop_at = None if deadline is None else deadline - ABANDON_S
    budget = sb["budget"]
    limits = src.limits(n_candidates, budget["B_units"], budget.get("footprint_cap"))
    names = [stageb.round_patch("beam", r, slot) for slot in "abcd"[:n_candidates]]
    seen = {c["sha256"]: c["name"] for c in [*(sb.get("history") or []), *(sb.get("chance") or []),
                                              (sb.get("floors") or {}).get("jitter") or {}] if c.get("sha256")}
    notes, extras = _own_logs(Path(log_dir), below=r)
    redact = getattr(provider, "redact", lambda s: s)
    session = Session(provider, ledger, RequestLog(out / "requests.jsonl", redact), src, limits, effort, rd, now,
                      stop_at)

    def compose(n: dict[int, str], x: dict[str, dict]) -> str:
        return round_message(r, sb, src, n_candidates=n_candidates, max_rounds=max_rounds,
                             deadline_utc=None if stop_at is None else utc(stop_at), notes=n, extras=x)

    check_outgoing(compose({}, {}))
    user = compose(notes, extras)
    live = _live(provider)
    messages = [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}]
    forfeit, accepted = session.converse(
        messages, tools(ROUND_TOOLS), "submit_patches",
        lambda args: validate_submission(args, limits, bias=src.bias, sample=src.sample, seen=seen, names=names))
    notes_text, cands = accepted if accepted is not None else ("", [])
    beam = {"format": "ouroboros-beam-handoff/1", "round": r, "status": "ok" if forfeit is None else "forfeit",
            "forfeit": forfeit, "sha256": None, "patches": [], "written_utc": None}
    if forfeit is None:
        inputs_sha = {**src.sha256, f"{_round_dir(r - 1)}/scores_beam.json": sha256_bytes(data),
                      **{f"prompts/{k}": v for k, v in prompt_sha256().items()}}
        blob = _bank_bytes(cands, src, lambda c: {"arm": "beam", "round": r, "slot": c.name[-1], "seed": None,
                                                  "parent": None, "built_by": "surgeon.controller.run_round",
                                                  "inputs": inputs_sha})
        (out / "beam.safetensors").write_bytes(blob)
        try:
            if stop_at is not None and now() >= stop_at:  # the bank write itself must not cross the line
                raise TimeoutError("the abandon time passed before the bank was written")
            _put(handoff, f"{rd}/beam.safetensors", blob, stop_at, now)
            beam.update(sha256=sha256_bytes(blob), patches=[_candidate_record(c) for c in cands])
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            session.errors.append(f"handoff: {e}")
            forfeit, beam["status"], beam["forfeit"] = "deadline", "forfeit", "deadline"
    beam["written_utc"] = utc()
    _put(handoff, f"{rd}/beam.json", dumps(stageb.labelled(beam)), deadline, now)
    result = RoundResult(r, beam["status"], forfeit, list(cands) if forfeit is None else [], notes_text,
                         session.cost_usd, session.requests, session.lookups, session.retries, session.errors)
    _write(out / "result.json", stageb.labelled({
        "format": "ouroboros-surgeon-round/1", "round": r, "status": result.status, "forfeit": forfeit,
        "notes": notes_text, "candidates": [_candidate_record(c) for c in result.candidates],
        "beam_sha256": beam["sha256"], "cost_usd": round(session.cost_usd, 6), "requests": session.requests,
        "lookups": session.lookups, "retries": session.retries, "errors": session.errors,
        "deadline_utc": sb.get("next_deadline_utc"), "prompts_sha256": prompt_sha256(), "live": live,
        "receipt_lag_s": lag, "last_provider_error": session.last_error, "written_utc": utc()}))
    why = f"; last provider error: {session.last_error[:300]}" if forfeit == "provider" and session.last_error else ""
    print(f"round {r}: {result.status}{'' if forfeit is None else ' (' + forfeit + ')'}, {session.requests} requests, "
          f"${session.cost_usd:.3f}, spent ${ledger.spent_usd:.3f} of ${ledger.cap_usd:.2f}"
          f"{'' if lag is None else f', scores read {lag:+.0f} s after the VM wrote them'}{why}", flush=True)
    return result


def latest_scores(handoff: Handoff, start: int = 0) -> tuple[int, dict] | None:
    """The highest round k >= start whose scores_beam.json exists, scanning upward from start."""
    found = None
    k = start
    while True:
        sb = read_json(handoff, f"{_round_dir(k)}/scores_beam.json")
        if sb is None:
            return found
        found = (k, sb)
        k += 1


def own_pick(*, provider: Provider, handoff: Handoff, inputs: Path, ledger: Ledger, log_dir: Path,
             now: Callable[[], float] = time.time) -> dict:
    """After the final scores: one conversation with the single tool pick_final, checked against Beam's scored list."""
    latest = latest_scores(handoff)
    if latest is None or not latest[1].get("final"):
        raise ValueError("own pick needs the final scores_beam.json")
    k, sb = latest
    src = inputs if isinstance(inputs, SurgeonInputs) else SurgeonInputs(inputs)
    out = Path(log_dir) / "own_pick"
    out.mkdir(parents=True, exist_ok=True)
    scored = {c["name"]: c for c in sb.get("history") or [] if c.get("status") == "scored"}
    pick = {"format": "ouroboros-own-pick/1", "status": "missing", "name": None, "why": None, "written_utc": None}
    forfeit = None
    if not scored:
        pick["why"] = "no scored Beam candidate"
    else:
        notes, extras = _own_logs(Path(log_dir))
        stop_at = parse_utc(sb["written_utc"]) + OWN_PICK_MIN * 60 - ABANDON_S
        limits = src.limits(1, sb["budget"]["B_units"], sb["budget"].get("footprint_cap"))
        req = RequestLog(out / "requests.jsonl", getattr(provider, "redact", lambda s: s))
        session = Session(provider, ledger, req, src, limits, "high", "own_pick", now, stop_at)
        def compose(x: dict[str, dict]) -> str:
            return (f"The search is over after round {k}. Every candidate of yours is listed below with its scores. "
                    f"Call pick_final once with the name of the one scored candidate you judge most likely to hold "
                    f"up on new tasks, and say why in 100 words at most. Only scored candidates may be named: "
                    f"{', '.join(sorted(scored))}.\n\n" + _eligibility_text(sb.get("eligibility") or {}) + "\n\n" +
                    "\n\n".join(candidate_block(c, src.moe_layers, x.get(c["name"])) for c in sb["history"]) +
                    "\n\n## Floors and chance reference\n\n" + floors_text(sb) + "\n")

        check_outgoing(compose({}))
        text = compose(extras)

        def accept(args):
            a = parse_arguments(args)
            problems = []
            if a.get("name") not in scored:
                problems.append(f"name {a.get('name')!r} is not one of your scored candidates: "
                                f"{', '.join(sorted(scored))}")
            why = a.get("why")
            if not isinstance(why, str) or not why.strip() or len(why.split()) > 100:
                problems.append("why must be a reason of 1 to 100 words")
            if problems:
                raise Invalid(problems, {"pick"})
            return a["name"], why.strip()

        _live(provider)
        forfeit, got = session.converse([{"role": "system", "content": system_prompt()},
                                         {"role": "user", "content": text}], tools(["pick_final"]),
                                        "pick_final", accept)
        if got is not None:
            pick.update(status="ok", name=got[0], why=got[1])
        else:
            pick.update(status="invalid" if forfeit == "invalid after retries" else "missing", why=forfeit)
    pick["written_utc"] = utc()
    pick = stageb.labelled(pick)
    write_json(handoff, "own_pick.json", pick)
    _write(out / "result.json", {**pick, "forfeit": forfeit})
    print(f"own pick: {pick['status']} {pick['name'] or ''}".rstrip(), flush=True)
    return pick


def r0_footprint_caps(inputs: Path, src: SurgeonInputs) -> dict[str, float | None]:
    """Zero-shot footprint cap per budget tag: the largest simulated footprint among the R0 random draws."""
    from ouroboros.harness import bank
    from ouroboros.surgeon.footprint import footprint

    patches, meta = bank.load_patch_bank(Path(inputs) / "bank" / "r0.safetensors", len(src.moe_layers), src.n_experts)
    records = meta.get("patches", {})
    caps: dict[str, float | None] = {}
    for b in stageb.BUDGETS:
        tag = stageb.budget_tag(b)
        shares = []
        for name, delta in patches.items():
            if not name.startswith(f"random_{tag}_k"):
                continue
            fp = (records.get(name) or {}).get("footprint")
            shares.append(float(fp["share"]) if fp else float(footprint(src.sample, np.asarray(delta))["share"]))
        caps[tag] = max(shares) if shares else None
    return caps


def zero_shot(*, provider: Provider, inputs: Path, ledger: Ledger, log_dir: Path, out: Path,
              n_candidates: int = 3, effort: str = "high") -> dict:
    """Zero-shot path: per budget, one conversation for 3 candidates with no scores; all six land in one bank."""
    src = inputs if isinstance(inputs, SurgeonInputs) else SurgeonInputs(inputs)
    base = Path(log_dir) / "zero_shot"
    caps = r0_footprint_caps(src.root, src)
    seen: dict[str, str] = {}
    kept: list[Candidate] = []
    budgets = {}
    for b in stageb.BUDGETS:
        tag = stageb.budget_tag(b)
        limits = src.limits(n_candidates, round(b / stageb.UNIT), caps[tag])
        names = [f"beam_zs_{tag}_{i}" for i in range(1, n_candidates + 1)]
        session = Session(provider, ledger, RequestLog(base / tag / "requests.jsonl",
                                                       getattr(provider, "redact", lambda s: s)),
                          src, limits, effort, f"zero_shot_{tag}")
        messages = [{"role": "system", "content": system_prompt()},
                    {"role": "user", "content": check_outgoing(zero_shot_message(b, caps[tag], src, n_candidates))}]
        _live(provider)
        forfeit, got = session.converse(
            messages, tools(ROUND_TOOLS), "submit_patches",
            lambda args: validate_submission(args, limits, bias=src.bias, sample=src.sample, seen=seen, names=names))
        notes, cands = got if got is not None else ("", [])
        for c in cands:
            seen[c.sha256] = c.name
        kept += cands
        budgets[tag] = {"status": "ok" if forfeit is None else "forfeit", "forfeit": forfeit, "B": b,
                        "B_units": limits.budget_units, "footprint_cap": caps[tag], "notes": notes,
                        "candidates": [_candidate_record(c) for c in cands], "cost_usd": round(session.cost_usd, 6),
                        "requests": session.requests, "lookups": session.lookups, "retries": session.retries,
                        "errors": session.errors}
        print(f"zero-shot {tag}: {budgets[tag]['status']}, {len(cands)} candidates", flush=True)
    inputs_sha = {**src.sha256, **{f"prompts/{k}": v for k, v in prompt_sha256().items()}}
    blob = _bank_bytes(kept, src, lambda c: {"arm": "zero_shot", "round": None, "slot": None, "seed": None,
                                             "parent": None, "built_by": "surgeon.controller.zero_shot",
                                             "inputs": inputs_sha})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    result = stageb.labelled({"format": "ouroboros-zero-shot/1",
                              "bank": {"file": str(out), "sha256": sha256_bytes(blob)}, "budgets": budgets,
                              "prompts_sha256": prompt_sha256(), "written_utc": utc()})
    _write(base / "result.json", result)
    return result


def token_for(provider: str) -> str:
    if provider == "reflection":
        token = os.environ.get("REFLECTION_API_KEY")
    else:
        token = os.environ.get("HF_TOKEN")
        if not token:
            from huggingface_hub import get_token

            token = get_token()
    if not token:
        sys.exit(f"no token for {provider}: set HF_TOKEN or log in with huggingface_hub "
                 "(REFLECTION_API_KEY for the Reflection endpoint)")
    return token


def refuse(message: str) -> NoReturn:
    """Stop with exit status REFUSED: a restart cannot cure this, so a restart loop stops too."""
    print(message, file=sys.stderr, flush=True)
    raise SystemExit(REFUSED)


def _pid_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if os.name == "nt":  # os.kill would terminate the process on Windows
        import ctypes

        k32 = ctypes.windll.kernel32
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return k32.GetLastError() == 5  # access denied: the process exists
        try:
            code = ctypes.c_ulong()
            return bool(k32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(path: Path) -> Path:
    """One controller run at a time: a second would race the first's Beam files and ledger. A lock whose process
    is gone is stale and taken over."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(path.read_text(encoding="utf-8").strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and _pid_alive(pid):
                refuse(f"another controller (pid {pid}) holds {path}; stop it before starting this one")
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        return path
    refuse(f"could not take {path}")


def release_lock(path: Path) -> None:
    try:
        if Path(path).read_text(encoding="utf-8").strip() == str(os.getpid()):
            Path(path).unlink()
    except OSError:
        pass


def keep_awake() -> bool:
    """Windows only: hold the system and the display awake while this process runs, so an idle timeout or Modern
    Standby cannot freeze the controller overnight. The request ends with the process."""
    if os.name != "nt":
        return False
    import ctypes

    es_continuous, es_system_required, es_display_required = 0x80000000, 0x1, 0x2
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required |
                                                                es_display_required))


def bind_log_dir(log_dir: Path, url: str, handoff: Handoff) -> set[int]:
    """Tie the log dir to one boot: its handoff URL and the sha256 of round_00/scores_beam.json. Results left by a
    rehearsal or another root would skip rounds and feed Beam foreign notes, so a mismatch refuses to start.
    Returns the rounds this log dir already answered for this boot."""
    log_dir = Path(log_dir)
    data = handoff.read("round_00/scores_beam.json")
    if data is None:
        raise FileNotFoundError("round_00/scores_beam.json is not in the handoff")
    where = url if str(url).startswith("gs://") else str(Path(url).resolve())
    want = {"handoff": where, "round_00_scores_beam_sha256": sha256_bytes(data)}
    path = log_dir / "handoff.json"
    results = [*log_dir.glob("round_*/result.json"), *log_dir.glob("own_pick/result.json")]
    if path.exists():
        got = json.loads(path.read_text(encoding="utf-8"))
        if {k: got.get(k) for k in want} != want:
            refuse(f"{log_dir} belongs to another boot ({got.get('handoff')}); pass a fresh --log-dir")
    elif results:
        refuse(f"{log_dir} holds {len(results)} result files bound to no boot; pass a fresh --log-dir")
    else:
        _write(path, stageb.labelled({"format": "ouroboros-surgeon-binding/1", **want, "written_utc": utc()}))
    return {int(p.parent.name.split("_")[1]) for p in log_dir.glob("round_*/result.json")}


def run(args: argparse.Namespace, provider: Provider) -> int:
    """Follow the VM: run each announced round once, then the own pick when the final scores arrive. A storage or
    network hiccup is logged and retried at the next poll."""
    from ouroboros.harness import bank  # noqa: F401  torch can take minutes to import; pay it before round 1

    handoff = open_handoff(args.handoff)
    log_dir = Path(args.log_dir)
    ledger_path = Path(getattr(args, "ledger", None) or log_dir / "ledger.json")
    lock = acquire_lock(ledger_path.parent / "controller.lock")
    try:
        if keep_awake():
            print("keeping the laptop awake while the controller runs", flush=True)
        ledger = Ledger(ledger_path, args.spend_cap)
        src = SurgeonInputs(args.inputs)
        src.sample  # noqa: B018  load the window sample now, so a round's first footprint check costs nothing
        done: set[int] | None = None  # known once the log dir is bound to this boot
        cursor, beat = 0, 0.0
        while True:
            latest = None
            try:
                latest = latest_scores(handoff, cursor)
                if latest is not None and done is None:
                    done = bind_log_dir(log_dir, args.handoff, handoff)
                if latest is not None:
                    cursor, sb = latest
                    if sb.get("final"):
                        if not (log_dir / "own_pick" / "result.json").exists():
                            own_pick(provider=provider, handoff=handoff, inputs=src, ledger=ledger, log_dir=log_dir)
                        print("search finished", flush=True)
                        return 0
                    r = sb.get("next_round")
                    if sb["beam"].get("arm_open") and r is not None and r not in done and r <= args.max_rounds:
                        run_round(r, provider=provider, handoff=handoff, inputs=src, ledger=ledger, log_dir=log_dir,
                                  effort=args.effort, max_rounds=args.max_rounds)
                        done.add(r)
                        continue
            except PaidCallRefused:
                raise
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:  # storage or network: try again
                log.warning("controller: %s", e)
                print(f"hiccup ({type(e).__name__}: {str(e)[:300]}); retrying in {args.poll_s:g} s", flush=True)
            if time.time() - beat >= 300:
                beat = time.time()
                print(f"waiting for scores after round {cursor if latest else 'none'} ({utc()})", flush=True)
            _sleep(args.poll_s)
    finally:
        release_lock(lock)


def smoke(args: argparse.Namespace, provider: Provider) -> int:
    """One small request with a synthetic context: the route answers and returns a tool call."""
    log_dir = Path(args.log_dir) / "smoke"
    ledger = Ledger(Path(getattr(args, "ledger", None) or Path(args.log_dir) / "ledger.json"), args.spend_cap)
    req = RequestLog(log_dir / "requests.jsonl", getattr(provider, "redact", lambda s: s))
    print(f"smoke: {getattr(provider, 'endpoint', None)}", flush=True)
    if not provider.live():
        sys.exit("smoke: the provider is not live")
    messages = [{"role": "system", "content": "You test a tool-calling route. Answer with one tool call."},
                {"role": "user", "content": "Call footprint with the single entry l=17, e=102, u=-30."}]
    est = int(len(json.dumps(messages)) / 3)
    if not ledger.admit(est, 4096):
        sys.exit("smoke: the spend cap refuses the request")
    t0 = time.monotonic()
    resp = provider.chat(messages, tools(["footprint"]), max_tokens=4096, effort=args.effort)
    cost = ledger.charge(resp.get("usage") or {"prompt_tokens": est, "completion_tokens": 4096, "estimated": True})
    reply = _reply(resp)
    req.write({"tag": "smoke", "request": {"messages": messages, "max_tokens": 4096, "effort": args.effort},
               "response": resp, "cost_usd": round(cost, 6), "latency_s": round(time.monotonic() - t0, 3)})
    ok = any(c["name"] == "footprint" for c in reply["calls"])
    print(f"smoke: {'tool call returned' if ok else 'no tool call'}, finish {reply['finish']}, ${cost:.4f}, "
          f"reasoning {len(reply['reasoning'])} chars", flush=True)
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-dir", default=str(stageb.RESULTS / "surgeon"))
    common.add_argument("--ledger", default=str(stageb.RESULTS / "surgeon" / "ledger.json"),
                        help="spend ledger shared by every command, so one cap covers all spend")
    common.add_argument("--provider", choices=sorted(ENDPOINTS), default="hf")
    common.add_argument("--effort", default="high")
    common.add_argument("--spend-cap", type=float, default=stageb.SPEND_CAP_USD)
    common.add_argument("--allow-paid", action="store_true", help="permit calls to the paid provider")
    r = sub.add_parser("run", parents=[common], help="follow a boot's handoff root round by round")
    r.add_argument("--handoff", required=True, help="gs:// round root or a directory")
    r.add_argument("--inputs", required=True, help="the Stage B inputs directory")
    r.add_argument("--max-rounds", type=int, default=stageb.ROUNDS_MAX)
    r.add_argument("--poll-s", type=float, default=15.0)
    z = sub.add_parser("zero-shot", parents=[common], help="candidates at both budgets with no scores")
    z.add_argument("--inputs", required=True)
    z.add_argument("--out", required=True, help="patch bank to write")
    sub.add_parser("smoke", parents=[common], help="one small request through the route")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not args.allow_paid:
        sys.exit("paid calls refused: this command calls the Beam API; pass --allow-paid after the go-ahead")
    provider = HTTPProvider(ENDPOINTS[args.provider], token_for(args.provider), allow_paid=True)
    try:
        if args.cmd == "run":
            return run(args, provider)
        if args.cmd == "zero-shot":
            ledger = Ledger(Path(args.ledger), args.spend_cap)
            res = zero_shot(provider=provider, inputs=Path(args.inputs), ledger=ledger, log_dir=Path(args.log_dir),
                            out=Path(args.out), effort=args.effort)
            return 0 if any(b["status"] == "ok" for b in res["budgets"].values()) else 1
        return smoke(args, provider)
    except ProviderFatal as e:
        refuse(f"the provider refused the account: {e}. Fix the token, credits or billing, then rerun the command.")


if __name__ == "__main__":
    sys.exit(main())
