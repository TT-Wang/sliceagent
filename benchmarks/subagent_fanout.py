"""Subagent fan-out benchmark (README §4) — sliceagent vs OpenAI Codex, BOTH delegating.

A ColBench-style LLM human-simulator (a staff engineer) EXPLICITLY tells both agents to fan out — one
explorer subagent per module across a 6-module "payments" service — then asks parent-only follow-ups over
a 6-turn session (2 fan-out + 4 follow-up). BOTH agents genuinely delegate: sliceagent via spawn_agent,
Codex via its OWN `exec` collab/spawn_agent primitive. The question is not who CAN delegate — it is what
the ORCHESTRATOR pays to run a fleet.

  * sliceagent runs in-process the way the CLI does: continue_topic SEALS each turn into a bounded digest,
    so the orchestrator's context stays flat no matter how many workers it spawns or how long the session
    runs. Its own child (subagent) tokens are measured via a SEPARATE tap.
  * Codex runs as ONE genuinely-resumed session (`codex exec` then `codex exec resume <thread>`): its
    orchestrator re-carries the whole transcript every turn, so its context grows monotonically. Codex's
    OWN child threads are billed in separate session-rollout files, which this harness recovers from
    ~/.codex/sessions/**/rollout-*-<tid>.jsonl → a true total-vs-total (each agent's parent + own children).

METRICS (measured identically on both sides): per-turn ORCHESTRATOR input tokens (codex: turn.completed
input_tokens; sliceagent: sum of the parent's per-call prompt tokens that turn — both exclude children),
each agent's DELEGATED tokens (its own children), and TRUE TOTAL = orchestrator + own children. A recall
sub-gate is scored but is NOT the headline (it turns on a behavioral re-read choice — see README §4).
Both `gpt-5.5` at matched reasoning effort (same model, so the moat isn't model-confounded).

Run (needs the Codex CLI installed + logged in, and an LLM configured — `sliceagent init` or LLM_API_KEY):
  export LLM_API_KEY=$OPENAI_API_KEY AGENT_MODEL=gpt-5.5 AGENT_REASONING=high CODEX_EFFORT=high
  # (behind a proxy? also export https_proxy/http_proxy/all_proxy for the OpenAI backend)
  ARCH_MODE=humansim PYTHONPATH=src python benchmarks/subagent_fanout.py
  # knobs: ARCH_RUNS=3 (repeat + aggregate) · ARCH_ONLY=slice|codex · ARCH_TURNS=N · CODEX_BIN=codex
"""
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL = os.environ.get("AGENT_MODEL", "gpt-5.5")
CODEX_BIN = os.environ.get("CODEX_BIN", "codex")   # on PATH by default; override for an absolute path
CODEX_EFFORT = os.environ.get("CODEX_EFFORT", "high")
ARCH_RUNS = int(os.environ.get("ARCH_RUNS", "1"))
ARCH_OUT = os.environ.get("ARCH_OUT", "/tmp/subagent_fanout.json")
ONLY = os.environ.get("ARCH_ONLY", "")   # "", "slice", or "codex"


def render_cost_chart(rows, width: int = 40) -> str:
    """Dependency-free ASCII chart of per-turn orchestrator INPUT tokens: sliceagent (flat — re-sealed each
    turn) vs a transcript agent (rising — re-sends the whole history). `rows` need 'turn','peak_in','transcript'."""
    rows = [r for r in rows if r.get("peak_in") is not None and r.get("transcript") is not None]
    if not rows:
        return ""
    mx = max(max(r["peak_in"], r["transcript"]) for r in rows) or 1
    k = lambda v: (f"{v/1000:.0f}k" if v >= 1000 else str(v))   # noqa: E731
    bar = lambda v, ch: ch * max(1, round(width * v / mx))      # noqa: E731
    out = ["per-turn ORCHESTRATOR input — sliceagent (sealed slice) vs codex (full transcript re-sent):", ""]
    for r in rows:
        out.append(f"  t{r['turn']:>2} sliceagent   {bar(r['peak_in'], '▒'):<{width}} {k(r['peak_in'])}")
        out.append(f"      codex      {bar(r['transcript'], '█'):<{width}} {k(r['transcript'])}")
    last = rows[-1]
    out += ["", f"  → by turn {last['turn']}: sliceagent {k(last['peak_in'])} vs codex {k(last['transcript'])} "
            f"({last['transcript'] / max(1, last['peak_in']):.0f}× more) — flat vs linear growth."]
    return "\n".join(out)


class _UsageTap:
    """Wraps an LLMClient and records {prompt,completion,cached} tokens per .complete() call."""
    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def complete(self, messages, tools):
        r = self.inner.complete(messages, tools)
        u = (r.usage or {}) if hasattr(r, "usage") else {}
        self.calls.append({"prompt": u.get("prompt_tokens", 0),
                           "completion": u.get("completion_tokens", 0),
                           "cached": u.get("cached_tokens", 0)})
        return r

    def set_cache_key(self, k):
        if hasattr(self.inner, "set_cache_key"):
            self.inner.set_cache_key(k)
# "seal"   — multi-turn drill-down; the moat is the SEAL (bounded slice re-sealed each turn). Small modules
#            → the model rationally reads inline, so this isolates the seal, not delegation.
# "fanout" — modules ENLARGED + each turn asks for BREADTH across all 6 at once, so fan-out delegation is
#            the rational choice. Isolates the DELEGATION moat: parent-peak stays ≈digests, not material.
ARCH_MODE = os.environ.get("ARCH_MODE", "seal")
ENLARGE_N = int(os.environ.get("ARCH_ENLARGE", "48"))   # helper fns appended per module in fanout mode
# When set, the sliceagent parent is INSTRUCTED to delegate every module read to an explorer subagent
# instead of reading inline. Isolates the delegation MECHANISM's cost profile (bounded parent-peak ≈
# digests) vs the natural-behavior arm where the model reads inline. (codex side is skipped: ARCH_ONLY=slice)
FORCE_DELEGATE = os.environ.get("ARCH_FORCE_DELEGATE", "") not in ("", "0", "off")
_FORCE_SUFFIX = ("\n\nIMPORTANT: do NOT read the module files yourself. Delegate the investigation of EACH "
                 "module to a SEPARATE explorer subagent via spawn_agent(agent=\"explorer\", task=..., "
                 "name=...) — one per module — then synthesize your answer ONLY from their returned digests.")

# ---------------------------------------------------------------------------------------------------
# The planted repo. A tiny payments service; values below are the ground truth the gates check.
# THE INCONSISTENCY: auth enforces a 600s session window, but config declares a 900s token TTL — a real
# bug (config says a token is valid for 900s; auth rejects it after 600s). The recall turn must catch it.
# ---------------------------------------------------------------------------------------------------
GROUND = {"auth_window": "600", "config_ttl": "900", "fallback_rate": "1.0857"}

_AUTH = '''\
"""Request authentication: HMAC-signed session tokens for the payments API."""
import hashlib
import hmac
import time

# A session token is only honored for this many seconds after it is minted. Enforced HERE, at verify
# time — NOT read from config. (Historical: predates the central config module.)
SESSION_WINDOW_SECONDS = 600

_SIGNING_ALGO = hashlib.sha256


def _mac(secret: bytes, payload: str) -> str:
    return hmac.new(secret, payload.encode("utf-8"), _SIGNING_ALGO).hexdigest()


def mint_token(secret: bytes, user_id: str, issued_at: int | None = None) -> str:
    issued_at = int(time.time()) if issued_at is None else issued_at
    payload = f"{user_id}.{issued_at}"
    return f"{payload}.{_mac(secret, payload)}"


def verify_token(secret: bytes, token: str, now: int | None = None) -> bool:
    """Return True iff the signature checks out AND the token is within SESSION_WINDOW_SECONDS of now."""
    now = int(time.time()) if now is None else now
    try:
        user_id, issued_at, sig = token.split(".")
    except ValueError:
        return False
    payload = f"{user_id}.{issued_at}"
    if not hmac.compare_digest(sig, _mac(secret, payload)):
        return False
    age = now - int(issued_at)
    # the window check — a token older than SESSION_WINDOW_SECONDS is rejected even if well-signed
    return 0 <= age <= SESSION_WINDOW_SECONDS


def user_of(token: str) -> str | None:
    parts = token.split(".")
    return parts[0] if len(parts) == 3 else None
'''

_CONFIG = '''\
"""Central configuration constants for the payments service."""

# How long, in seconds, a minted session token is CONSIDERED valid by the platform. NOTE: the auth
# module enforces its own window at verify time; this value is what the rest of the system assumes.
TOKEN_TTL_SECONDS = 900

# Outbound HTTP calls to the rate/settlement providers.
HTTP_TIMEOUT_SECONDS = 30
RETRY_BUDGET = 3                 # max retries per outbound provider call before giving up
RETRY_BACKOFF_SECONDS = 2       # base for exponential backoff: BACKOFF * (2 ** attempt)

# Fees.
FEE_PERCENT = 0.029             # 2.9% of the charged amount
FEE_FIXED_CENTS = 30            # plus a flat 30¢

# Ledger.
LEDGER_FLUSH_EVERY = 50         # buffer this many entries before an fsync

# Webhooks.
WEBHOOK_DEDUPE_WINDOW_SECONDS = 86_400   # remember delivered event ids for a day
'''

_FEES = '''\
"""Fee computation. All money is handled in integer CENTS to avoid float drift; the ONLY rounding is
the single half-up step below."""
import math

from config import FEE_FIXED_CENTS, FEE_PERCENT


def fee_cents(amount_cents: int) -> int:
    """Processing fee for a charge of `amount_cents`.

    fee = amount * FEE_PERCENT + FEE_FIXED_CENTS, then ROUNDED HALF-UP to the nearest whole cent.
    Half-up (not banker's rounding): 0.5 always rounds AWAY from zero, so 2.5 -> 3, 3.5 -> 4.
    """
    raw = amount_cents * FEE_PERCENT + FEE_FIXED_CENTS
    return int(math.floor(raw + 0.5))   # half-up on non-negative amounts


def net_cents(amount_cents: int) -> int:
    """What the merchant actually receives: the charge minus the processing fee."""
    return amount_cents - fee_cents(amount_cents)


def describe(amount_cents: int) -> str:
    f = fee_cents(amount_cents)
    return f"charge={amount_cents}c fee={f}c net={amount_cents - f}c"
'''

_RATES = '''\
"""FX rate lookup with a static fallback when the upstream provider is unreachable."""
from config import HTTP_TIMEOUT_SECONDS, RETRY_BUDGET

# Used ONLY when every provider attempt fails (timeout/5xx). Deliberately conservative; revisited quarterly.
FALLBACK_RATE = 1.0857


class RateProvider:
    def __init__(self, transport):
        self._transport = transport   # a callable(pair) -> float | raises

    def rate(self, pair: str) -> float:
        """Best-effort live rate for e.g. "USD/EUR"; falls back to FALLBACK_RATE after RETRY_BUDGET tries."""
        last_exc = None
        for attempt in range(RETRY_BUDGET):
            try:
                return float(self._transport(pair, timeout=HTTP_TIMEOUT_SECONDS))
            except Exception as e:  # noqa: BLE001 — any transport failure falls through to a retry/fallback
                last_exc = e
        # every attempt failed → static fallback so a charge can still settle
        return FALLBACK_RATE
'''

_WEBHOOK = '''\
"""Inbound webhook handling with at-least-once delivery → idempotent processing by event id."""
import time

from config import WEBHOOK_DEDUPE_WINDOW_SECONDS


class WebhookDeduper:
    """Providers retry deliveries, so the SAME event id can arrive many times. We process each id exactly
    once by remembering ids we have seen within WEBHOOK_DEDUPE_WINDOW_SECONDS."""

    def __init__(self):
        self._seen: dict[str, float] = {}   # event_id -> first-seen unix ts

    def _evict(self, now: float) -> None:
        cutoff = now - WEBHOOK_DEDUPE_WINDOW_SECONDS
        for k in [k for k, ts in self._seen.items() if ts < cutoff]:
            del self._seen[k]

    def is_duplicate(self, event_id: str, now: float | None = None) -> bool:
        """True if this event id was already accepted inside the dedupe window (so the caller SKIPS it)."""
        now = time.time() if now is None else now
        self._evict(now)
        if event_id in self._seen:
            return True
        self._seen[event_id] = now
        return False


def handle(deduper: WebhookDeduper, event: dict) -> str:
    eid = event.get("id", "")
    if not eid or deduper.is_duplicate(eid):
        return "skipped"
    return "processed"
'''

_LEDGER = '''\
"""Append-only charge ledger. Records each settled charge and its fee for reconciliation."""
from dataclasses import dataclass, field

from fees import fee_cents, net_cents


@dataclass
class Entry:
    charge_id: str
    amount_cents: int
    fee_cents: int
    net_cents: int


@dataclass
class Ledger:
    entries: list = field(default_factory=list)

    def record(self, charge_id: str, amount_cents: int) -> Entry:
        """Record a charge end to end: compute the fee (fees.fee_cents), the net (fees.net_cents), append."""
        e = Entry(charge_id, amount_cents, fee_cents(amount_cents), net_cents(amount_cents))
        self.entries.append(e)
        return e

    def total_fees(self) -> int:
        return sum(e.fee_cents for e in self.entries)
'''

MODULES = {"auth.py": _AUTH, "config.py": _CONFIG, "fees.py": _FEES,
           "rates.py": _RATES, "webhook.py": _WEBHOOK, "ledger.py": _LEDGER}


def _enlarge(src: str, name: str, n: int) -> str:
    """Append n realistic, self-contained helper functions so the module is large enough that reading it
    INLINE is costly — making fan-out delegation the rational choice. Deterministic (index-seeded). Purely
    additive: never references or shadows the planted ground-truth code above, so values are unchanged."""
    stem = name[:-3]
    fns = []
    for i in range(n):
        fns.append(f'''

def {stem}_aux_{i}(items, scale={i + 1}):
    """Auxiliary {stem} routine #{i}: fold a sequence with a bounded, index-specific accumulator."""
    total = 0
    for k, v in enumerate(items):
        step = (int(v) + k) * scale - ({i} % 7)
        total = total + step if step > 0 else total - step
    return total % ({101 + i} or 1)''')
    return src + "\n" + "".join(fns) + "\n"


def build_repo(d: str) -> None:
    for name, src in MODULES.items():
        if ARCH_MODE in ("fanout", "humansim"):   # big modules → fan-out is worth it / reading inline is costly
            src = _enlarge(src, name, ENLARGE_N)
        with open(os.path.join(d, name), "w", encoding="utf-8") as f:
            f.write(src)
    with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as f:
        f.write("# payments\n\nA tiny payments service: auth, config, fees, rates, webhook, ledger.\n")


# SEAL mode: one drill-down per turn (naturally multi-file at t4/t5); the LAST turn is the cross-turn
# recall that reconciles turn-1 (auth) with turn-2 (config).
SEAL_DIRECTIVES = [
    "Investigate how auth.py validates a session token. What is the exact session/token window length, "
    "in seconds, after which a well-signed token is rejected? Name the constant and its value.",
    "Look at config.py and list every timeout and retry constant with its exact value (token TTL, HTTP "
    "timeout, retry budget, backoff).",
    "In fees.py, what is the exact fee formula and precisely how is the result rounded (which rounding "
    "rule, and in what direction on a .5)?",
    "Compare rates.py and webhook.py: what fallback exchange rate is used when every provider attempt "
    "fails, and how does the webhook handler avoid processing a duplicate event?",
    "Trace, across ledger.py and fees.py, how a single charge is recorded end to end — which functions "
    "are called and in what order.",
    "Earlier you found the session window in auth.py and the token TTL in config.py. State BOTH values "
    "(in seconds) and say clearly whether the auth session window and the config token TTL are consistent "
    "with each other or not.",
]

# FANOUT mode: every turn demands BREADTH across ALL six (enlarged) modules at once → fan-out is the
# rational choice. Last turn is the same cross-turn recall gate.
FANOUT_DIRECTIVES = [
    "Survey the whole service. Give a ONE-paragraph summary of the purpose and the main public functions "
    "of EACH of the six modules: auth.py, config.py, fees.py, rates.py, webhook.py, ledger.py.",
    "Across ALL six modules, identify every module that imports from config.py and name which config "
    "constant(s) each one uses.",
    "Earlier you surveyed auth.py and config.py. State the auth session window (seconds) and the config "
    "token TTL (seconds), and say clearly whether the two are consistent with each other or not.",
]

# HUMANSIM mode: a ColBench-style LLM human-simulator plays a staff engineer. On the EARLY turns it
# EXPLICITLY commands subagent fan-out (the delegation trigger the model won't self-pull); on the LATER
# turns it asks parent-only FOLLOW-UP questions (no new fan-out) so we can watch the parent's per-turn token
# curve grow over a longer horizon. The SAME persona+agenda drives the sim for BOTH agents (colbench fairness
# invariant). NOTE (corrected 2026-07-10): BOTH agents can fan out — codex `exec` 0.141 has its OWN spawn_agent
# ("collab") primitive. The measured differentiator is the CROSS-TURN SEAL (slice continue_topic → bounded
# digest) vs codex `exec resume` replaying the whole transcript — NOT delegation capability.
HUMAN_SIM_PROMPT = (
    "You are a senior staff engineer pair-working with a coding agent on a payments codebase (modules: "
    "auth.py, config.py, fees.py, rates.py, webhook.py, ledger.py). Speak naturally, first person, ONE short "
    "message (2-4 sentences), like a Slack message to a colleague. Do NOT reveal or guess any specific values "
    "yourself (that is the agent's job to find).\n\n"
    "{mode_instruction}\n\n"
    "THIS STEP'S GOAL: {goal}\n\n"
    "Write your next message to the agent now.")
_FANOUT_INSTR = ("For THIS step you want a lean context, so EXPLICITLY tell the agent to FAN OUT — spin up a "
                 "separate explorer/subagent for EACH module and investigate them in PARALLEL, then synthesize "
                 "from the workers' digests rather than reading everything into one context.")
_FOLLOWUP_INSTR = ("This is a FOLLOW-UP question — do NOT ask for any new fan-out. The agent already "
                   "investigated; just ask it to answer from what its earlier explorers found.")

# Fixed agenda (drives the sim; not shown to the agent verbatim). fanout=True → command delegation;
# fanout=False → parent-only follow-up. The reconcile turn is the correctness gate.
AGENDA = [
    {"goal": "Get a one-paragraph summary of the PURPOSE and main public functions of every one of the six "
             "modules.", "fanout": True, "gate": False},
    {"goal": "Find out which modules import from config.py and which config constant(s) each one uses.",
     "fanout": True, "gate": False},
    {"goal": "Ask which module defines the session/token validity window and what its exact value is (in "
             "seconds).", "fanout": False, "gate": False},
    {"goal": "Ask what the central config's token TTL constant is, in seconds.", "fanout": False, "gate": False},
    {"goal": "Ask them to reconcile the auth session window and the config token TTL: state BOTH values (in "
             "seconds) and whether the two are consistent with each other or not.", "fanout": False, "gate": True},
    {"goal": "Ask for a short plain-English summary of the whole investigation and any risk they noticed.",
     "fanout": False, "gate": False},
]

DIRECTIVES = (["<human-sim-driven>"] * len(AGENDA) if ARCH_MODE == "humansim"
              else FANOUT_DIRECTIVES if ARCH_MODE == "fanout" else SEAL_DIRECTIVES)
N_TURNS = int(os.environ.get("ARCH_TURNS", str(len(DIRECTIVES))))


def next_human_command(sim, turn_idx: int) -> str:
    """One turn of the ColBench-style human simulator: the staff-engineer persona issues the next command for
    AGENDA[turn_idx] — a fan-out order (early turns) or a parent-only follow-up (later turns). The sim is kept
    BLIND to the running dialogue on purpose (validity red-team M3): each agenda item is self-contained, so
    feeding the dialogue back would risk leaking the agent's own discovered values (600/900) into its next
    directive. Blindness also makes turn-k's directive depend only on AGENDA[k]+persona → effectively identical
    across both agents (colbench fairness). Sim tokens are NOT counted in the agent's cost (this is the 'user')."""
    item = AGENDA[turn_idx]
    instr = _FANOUT_INSTR if item.get("fanout") else _FOLLOWUP_INSTR
    p = HUMAN_SIM_PROMPT.replace("{mode_instruction}", instr).replace("{goal}", item["goal"])
    r = sim.complete([{"role": "user", "content": p}], [])
    return (r.content or "").strip() or item["goal"]


def gate(turn_idx: int, answer: str) -> bool | None:
    """Deterministic correctness check (no LLM judge → cheap + reproducible), keyed off the DIRECTIVE text
    so it works in both modes. None = ungated turn."""
    a = answer.lower()
    def _recall_ok() -> bool:
        mismatch = any(w in a for w in ("not consistent", "inconsistent", "differ", "mismatch", "do not",
                                        "don't", "doesn't", "aren't", "not the same", "not equal", "conflict"))
        return (GROUND["auth_window"] in answer) and (GROUND["config_ttl"] in answer) and mismatch
    if ARCH_MODE == "humansim":                                # gate keyed off the fixed agenda item
        item = AGENDA[turn_idx]
        if item["gate"]:
            return _recall_ok()                                # the reconcile turn: both values + mismatch
        g = item["goal"].lower()
        if "validity window" in g:                             # the auth-window recall turn → 600
            return GROUND["auth_window"] in answer
        if "token ttl" in g:                                   # the config-TTL recall turn → 900
            return GROUND["config_ttl"] in answer
        return None                                            # fan-out/summary turns: ungated
    d = DIRECTIVES[turn_idx].lower()
    if "consistent with each other" in d:                      # the cross-turn recall gate
        return _recall_ok()
    if "window length" in d:                                   # the auth drill-down (seal mode t1)
        return GROUND["auth_window"] in answer
    if "list every timeout and retry constant" in d:           # the config drill-down (seal mode t2)
        return GROUND["config_ttl"] in answer
    return None


# ---------------------------------------------------------------------------------------------------
# sliceagent driver — in-process, the CLI's real path: Session + SEAL between turns + DELEGATION.
# ---------------------------------------------------------------------------------------------------
def run_sliceagent(repo: str) -> list[dict]:
    from sliceagent.pfc import slice_sink, record_user, _active
    from sliceagent.seed import make_build_slice
    from sliceagent.text_utils import one_line
    from sliceagent.loop import run_turn
    from sliceagent.tools import LocalToolHost
    from sliceagent.code_index import make_code_index
    from sliceagent.events import ToolResult, make_dispatcher
    from sliceagent.hooks import BudgetHook, CompositeHooks, PermissionHook
    from sliceagent.policy import make_policy
    from sliceagent.llm import OpenAILLM
    from sliceagent.session import Session, make_topic_tools
    from sliceagent.memory import make_memory
    from sliceagent.subagent import SubagentHost, _CaptureLast
    from sliceagent.agents import load_agents
    from sliceagent.hippocampus import (HistoryFS, RosterFS, SubagentFS,
                                        make_episode_sink, make_search_history_tool)

    memory = make_memory()          # durable memem: subagents/ + roster/ + history/ virtual namespaces live
    session = Session(memory)
    sid = session.session_id
    base_tools = LocalToolHost(repo)
    retriever = make_code_index(repo)

    spawn_ctr = {"n": 0}
    child_tap = _UsageTap(OpenAILLM(model=MODEL, timeout=90.0))   # SEPARATE tap → delegated work isolated
    child_tap.set_cache_key(sid + "-child")
    tools = SubagentHost(base_tools, llm=child_tap, retriever=retriever, memory=memory,
                         policy=make_policy("guard"), max_depth=1,
                         agents=load_agents([repo, os.path.join(repo, ".sliceagent")]), session_id=sid)
    for t in make_topic_tools(session):
        base_tools.registry.register(t)
    base_tools._history = HistoryFS(memory, sid)
    base_tools._subagents = SubagentFS(memory, sid)
    base_tools._roster = RosterFS(memory)
    base_tools.registry.register(make_search_history_tool(memory, sid))
    episodic = make_episode_sink(memory, session_id=sid, task_id_fn=lambda: session.active_id or "t",
                                 title_fn=lambda: one_line(session.active().goal, 80) if session.active_id else "")

    parent_tap = _UsageTap(OpenAILLM(model=MODEL, timeout=90.0))
    parent_tap.set_cache_key(sid)
    build = make_build_slice(session, tools, retriever, memory, DIRECTIVES[0], session_id=sid)

    cap = _CaptureLast()
    def spawn_count(e):
        if isinstance(e, ToolResult) and getattr(e, "name", "") in ("spawn_agent", "spawn_explore", "spawn_subagent"):
            spawn_ctr["n"] += 1
    dispatch = make_dispatcher(slice_sink(session), episodic, cap, spawn_count)
    hooks = CompositeHooks(PermissionHook(make_policy("guard"), auto_approve=["*"]), BudgetHook(4_000_000))

    sim = OpenAILLM(model=MODEL, timeout=90.0) if ARCH_MODE == "humansim" else None   # the human simulator (uncounted)
    if sim is not None:
        sim.reasoning = "full"    # identical sim for both agents (colbench fairness); not the agent's effort

    rows = []
    for i in range(N_TURNS):
        if ARCH_MODE == "humansim":
            directive = next_human_command(sim, i)   # a real LLM human issues the explicit fan-out order
        else:
            directive = DIRECTIVES[i] + (_FORCE_SUFFIX if FORCE_DELEGATE else "")
        session.new_topic(directive) if i == 0 else session.continue_topic(directive)   # ← SEAL fires
        record_user(session.active(), directive)
        parent_tap.calls = []
        child_tap.calls = []
        cap.text = ""
        n0 = spawn_ctr["n"]
        t0 = time.time()
        print(f"\n[slice t{i + 1}] {directive[:58]}", end="  ", flush=True)
        try:
            res = run_turn(build_slice=build, llm=parent_tap, tools=tools, dispatch=dispatch, hooks=hooks, max_steps=20)
            stop = res.stop_reason
        except Exception as e:  # noqa: BLE001
            stop = f"err:{type(e).__name__}"
        ans = cap.text
        p_in = sum(c["prompt"] for c in parent_tap.calls)
        p_peak = max((c["prompt"] for c in parent_tap.calls), default=0)
        c_in = sum(c["prompt"] for c in child_tap.calls)
        g = gate(i, ans)
        rows.append({"agent": "sliceagent", "turn": i + 1,
                     "in": p_in,                                   # per-turn parent input (headline, both sides)
                     "parent_peak": p_peak,                        # largest single parent prompt (moat detail)
                     "delegated_in": c_in,                         # all child calls this turn (delegation cost)
                     "out": sum(c["completion"] for c in parent_tap.calls) + sum(c["completion"] for c in child_tap.calls),
                     "cached": sum(c["cached"] for c in parent_tap.calls),
                     "parent_calls": len(parent_tap.calls), "child_calls": len(child_tap.calls),
                     "spawns": spawn_ctr["n"] - n0, "correct": g, "stop": stop,
                     "wall": round(time.time() - t0, 1), "human": directive[:200], "answer": ans[:400]})
        print(f"in={p_in:,} peak={p_peak:,} deleg={c_in:,} spawns={rows[-1]['spawns']} "
              f"correct={g} stop={stop}", flush=True)
    return rows


# ---------------------------------------------------------------------------------------------------
# codex driver — ONE genuinely-resumed session across all turns (real transcript accumulation).
# ---------------------------------------------------------------------------------------------------
def _codex_env() -> dict:
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)          # codex uses the ChatGPT subscription (auth.json), not the API key
    for k in ("https_proxy", "http_proxy", "all_proxy"):
        env[k] = "http://127.0.0.1:7890"     # China proxy for the OpenAI backend
    return env


_CODEX_SESSIONS = os.path.expanduser("~/.codex/sessions")


def _codex_thread_tokens(tid: str) -> dict:
    """codex `turn.completed` usage is PARENT-thread ONLY — spawned children (collab spawn_agent) are billed
    in their OWN rollout files. Recover a child thread's usage from ~/.codex/sessions/**/rollout-*-<tid>.jsonl:
    the LAST `token_count` event's total_token_usage is that thread's cumulative cost (verified 2026-07-10)."""
    import glob
    hits = glob.glob(os.path.join(_CODEX_SESSIONS, "**", f"rollout-*-{tid}.jsonl"), recursive=True)
    if not hits:
        return {"in": 0, "out": 0, "cached": 0, "found": False}
    last = {"in": 0, "out": 0, "cached": 0, "found": True}
    for ln in open(max(hits, key=os.path.getmtime), encoding="utf-8"):
        try:
            o = json.loads(ln)
        except Exception:
            continue
        p = o.get("payload", {})
        if o.get("type") == "event_msg" and p.get("type") == "token_count":
            tu = (p.get("info") or {}).get("total_token_usage") or {}
            if tu:   # cumulative — keep the latest
                last["in"] = tu.get("input_tokens", 0)
                last["out"] = tu.get("output_tokens", 0)
                last["cached"] = tu.get("cached_input_tokens", 0)
    return last


def codex_turn(prompt: str, repo: str, thread_id: str | None) -> tuple[str, dict, str | None]:
    cfg = f'model_reasoning_effort="{CODEX_EFFORT}"'
    if thread_id is None:
        cmd = [CODEX_BIN, "exec", "--json", "--skip-git-repo-check", "--sandbox", "read-only",
               "-m", MODEL, "-c", cfg, prompt]
    else:
        # resume INHERITS the original session's sandbox (read-only) — it rejects a re-passed --sandbox.
        cmd = [CODEX_BIN, "exec", "resume", "--json", "--skip-git-repo-check",
               "-m", MODEL, "-c", cfg, thread_id, prompt]   # resume THIS session (context accumulates)
    try:
        proc = subprocess.run(cmd, cwd=repo, stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, timeout=1200, env=_codex_env())
    except subprocess.TimeoutExpired:
        return "(codex timeout)", {"in": 0, "out": 0, "cached": 0, "child_in": 0, "child_out": 0,
                                   "spawns": 0, "child_missing": 0}, thread_id
    msg, u, tid = "", {"in": 0, "out": 0, "cached": 0, "child_in": 0, "child_out": 0}, thread_id
    child_tids = []
    for ln in proc.stdout.splitlines():
        try:
            o = json.loads(ln)
        except Exception:
            continue
        t = o.get("type")
        if t == "thread.started" and o.get("thread_id"):
            tid = o["thread_id"]
        elif t == "item.completed":
            it = o.get("item", {})
            if it.get("type") == "agent_message" and it.get("text"):
                msg = it["text"]                 # keep the last agent_message = the answer
            elif it.get("type") == "collab_tool_call" and it.get("tool") == "spawn_agent":
                child_tids += it.get("receiver_thread_ids", [])   # children codex spawned THIS turn
        elif t == "turn.completed":
            us = o.get("usage", {})
            u["in"] = us.get("input_tokens", 0)          # PARENT/orchestrator thread ONLY (per-turn)
            u["out"] = us.get("output_tokens", 0)
            u["cached"] = us.get("cached_input_tokens", 0)
    # sum this turn's spawned children from their rollout files → codex's TRUE delegated cost
    child_tids = list(dict.fromkeys(child_tids))   # de-dup, preserve order
    missing = 0
    for ct in child_tids:
        ck = _codex_thread_tokens(ct)
        u["child_in"] += ck["in"]; u["child_out"] += ck["out"]
        if not ck["found"]:
            missing += 1
    u["spawns"] = len(child_tids); u["child_missing"] = missing
    return (msg or "(no message)"), u, tid


def run_codex(repo: str) -> list[dict]:
    from sliceagent.llm import OpenAILLM
    thread_id = None
    sim = None
    if ARCH_MODE == "humansim":
        sim = OpenAILLM(model=MODEL, timeout=90.0); sim.reasoning = "full"   # SAME persona/reasoning as slice side
    rows = []
    for i in range(N_TURNS):
        if ARCH_MODE == "humansim":
            human = next_human_command(sim, i)   # identical human-sim; codex `exec` has no subagents to act on it
            directive = human                    # M2: NO extra suffix — codex gets the SAME message slice gets
        else:
            human = DIRECTIVES[i]
            directive = DIRECTIVES[i] + "\n\nAnswer concisely in prose (no tools needed beyond reading files)."
        t0 = time.time()
        print(f"\n[codex t{i + 1}] {human[:58]}", end="  ", flush=True)
        msg, u, thread_id = codex_turn(directive, repo, thread_id)
        g = gate(i, msg)
        rows.append({"agent": "codex", "turn": i + 1,
                     "in": u["in"],                                # PARENT/orchestrator thread (per-turn)
                     "delegated_in": u.get("child_in", 0),         # codex's OWN spawned children (from rollouts)
                     "spawns": u.get("spawns", 0), "child_missing": u.get("child_missing", 0),
                     "out": u["out"], "cached": u["cached"],
                     "correct": g, "wall": round(time.time() - t0, 1), "thread": thread_id,
                     "human": human[:200], "answer": msg[:400]})
        print(f"parent_in={u['in']:,} deleg={u.get('child_in',0):,} spawns={u.get('spawns',0)} "
              f"correct={g} wall={rows[-1]['wall']}s", flush=True)
    return rows


# ---------------------------------------------------------------------------------------------------
def _score(rows: list[dict]) -> tuple[int, int]:
    gated = [r for r in rows if r["correct"] is not None]
    return sum(1 for r in gated if r["correct"]), len(gated)


def _tot(r):   # true per-turn total = orchestrator + own children
    return r.get("in", 0) + r.get("delegated_in", 0)


def run_once(run_idx: int) -> dict:
    os.environ["SLICEAGENT_VAULT"] = tempfile.mkdtemp(prefix="subfanout-vault-")   # fresh vault per run
    repo = tempfile.mkdtemp(prefix="subfanout-")
    build_repo(repo)
    print(f"\n===== RUN {run_idx + 1}/{ARCH_RUNS} =====  repo: {repo}   model: {MODEL}   "
          f"turns: {N_TURNS}   codex_effort: {CODEX_EFFORT}")
    sa = run_sliceagent(repo) if ONLY != "codex" else []
    cx = run_codex(repo) if ONLY != "slice" else []

    sc = {True: "Y", False: "N", None: "-"}
    print("\n" + "=" * 116)
    print(f"{'turn':>4} | {'slice par':>9} {'peak':>6} {'sl.deleg':>8} {'sl.total':>9} {'sp':>2} {'ok':>2}"
          f" || {'codex par':>9} {'cx.deleg':>8} {'cx.total':>9} {'sp':>2} {'ok':>2}")
    print("-" * 116)
    for i in range(N_TURNS):
        s = sa[i] if i < len(sa) else {}
        c = cx[i] if i < len(cx) else {}
        print(f"{i + 1:>4} | {s.get('in',0):>9,} {s.get('parent_peak',0):>6,} {s.get('delegated_in',0):>8,} "
              f"{_tot(s) if s else 0:>9,} {s.get('spawns',0):>2} {sc[s.get('correct')]:>2}"
              f" || {c.get('in',0):>9,} {c.get('delegated_in',0):>8,} {_tot(c) if c else 0:>9,} "
              f"{c.get('spawns',0):>2} {sc[c.get('correct')]:>2}")
    print("=" * 116)
    if sa and cx:
        print(f"by turn {N_TURNS}:  orchestrator — slice {sa[-1]['in']:,} vs codex {cx[-1]['in']:,} "
              f"({cx[-1]['in']/max(1,sa[-1]['in']):.1f}×)   |   true total — slice {sum(map(_tot,sa)):,} "
              f"vs codex {sum(map(_tot,cx)):,} ({sum(map(_tot,cx))/max(1,sum(map(_tot,sa))):.2f}×)")
        print(render_cost_chart([{"turn": i + 1, "peak_in": sa[i]["in"], "transcript": cx[i]["in"]}
                                 for i in range(N_TURNS)]))
    return {"run": run_idx + 1, "repo": repo, "sliceagent": sa, "codex": cx}


def _agg(runs: list[dict]) -> None:
    """Aggregate the headline metrics across N runs → mean [min–max], the numbers the README §4 cites."""
    def stat(vals):
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals), min(vals), max(vals)) if vals else (0, 0, 0)
    have_sa = all(r["sliceagent"] for r in runs)
    have_cx = all(r["codex"] for r in runs)
    print("\n" + "#" * 80)
    print(f"AGGREGATE over N={len(runs)} runs  (mean [min–max])")
    print("#" * 80)
    if have_sa and have_cx:
        rows = [("orchestrator · by turn N", [r["sliceagent"][-1]["in"] for r in runs],
                                              [r["codex"][-1]["in"] for r in runs]),
                ("orchestrator · per-call peak", [max(x["parent_peak"] for x in r["sliceagent"]) for r in runs],
                                                 [max(x["in"] for x in r["codex"]) for r in runs]),
                ("delegated · own children", [sum(x.get("delegated_in", 0) for x in r["sliceagent"]) for r in runs],
                                             [sum(x.get("delegated_in", 0) for x in r["codex"]) for r in runs]),
                ("TRUE TOTAL · parent+children", [sum(map(_tot, r["sliceagent"])) for r in runs],
                                                 [sum(map(_tot, r["codex"])) for r in runs]),
                ("subagent spawns", [sum(x["spawns"] for x in r["sliceagent"]) for r in runs],
                                    [sum(x["spawns"] for x in r["codex"]) for r in runs])]
        print(f"{'metric':<30} {'sliceagent':>26} {'codex':>26} {'codex/slice':>11}")
        for name, sv, cv in rows:
            sm, slo, shi = stat(sv); cm, clo, chi = stat(cv)
            ratio = cm / sm if sm else 0
            print(f"{name:<30} {f'{sm:,.0f} [{slo:,.0f}-{shi:,.0f}]':>26} "
                  f"{f'{cm:,.0f} [{clo:,.0f}-{chi:,.0f}]':>26} {ratio:>10.2f}×")
    for who in (("sliceagent",) if have_sa else ()) + (("codex",) if have_cx else ()):
        scores = [_score(r[who]) for r in runs]
        print(f"{who} recall sub-gate per run: " + ", ".join(f"{p}/{n}" for p, n in scores)
              + "   (NOT headline — behavioral, see README §4)")


def main() -> None:
    runs = [run_once(i) for i in range(ARCH_RUNS)]
    if ARCH_RUNS > 1:
        _agg(runs)
    json.dump({"model": MODEL, "codex_effort": CODEX_EFFORT, "turns": N_TURNS, "runs": runs},
              open(ARCH_OUT, "w"), indent=2)
    print(f"\nrows ({ARCH_RUNS} run(s)) -> {ARCH_OUT}")


if __name__ == "__main__":
    main()
