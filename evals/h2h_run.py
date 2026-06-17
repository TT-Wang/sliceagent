"""Head-to-head benchmark: memagent (bounded slice) vs Kimi Code (growing transcript), SAME model
(K2.7 Code), SAME fresh workdir, SAME prompt(s), INDEPENDENT verifier.

Metrics (per the ask): token usage, steps, speed, accuracy. Plus per-call input series so we can see
context GROWTH (the thesis): a bounded slice should keep per-call input ~flat while a transcript grows.

Both agents are driven HEADLESS:
  - memagent: in-process run_turn, multi-turn on ONE persistent Slice (continuity = the slice carries
    findings/conversation/edited-files across turns; reset() is called ONCE).
  - Kimi Code: `kimi -p` (turn 1) then `kimi -r <session> -p` (later turns); usage read back from the
    session's wire.jsonl (usage.record entries: inputOther/inputCacheRead/inputCacheCreation/output).

Run (env must point at the SAME model both sides):
  cd ~/Desktop/memagent
  set -a; . "../agent design/.env"; set +a
  export LLM_API_KEY="$MOONSHOT_API_KEY" LLM_BASE_URL="https://api.moonshot.cn/v1" AGENT_MODEL=kimi-k2.7-code
  PYTHONPATH=src .venv/bin/python -m evals.h2h_run            # all scenarios, both agents
  PYTHONPATH=src .venv/bin/python -m evals.h2h_run --scenario s2_largefile_bug --agent kimi
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time

H2H_DIR = os.path.dirname(os.path.abspath(__file__)) + "/h2h"
KIMI_BIN = os.path.expanduser("~/.kimi-code/bin/kimi")
KIMI_HOME = os.path.expanduser("~/.kimi-code")
KIMI_TURN_TIMEOUT = 600  # seconds per headless turn (a turn past this is wedged, not slow)


# ----------------------------------------------------------------------------- scenario loading
def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_scenario(name: str) -> dict:
    d = os.path.join(H2H_DIR, name)
    meta = json.load(open(os.path.join(d, "meta.json")))
    prompts = json.load(open(os.path.join(d, "prompts.json")))
    setup = _load_module(os.path.join(d, "setup.py"), f"{name}_setup").setup
    verify = _load_module(os.path.join(d, "verify.py"), f"{name}_verify").verify
    return {"name": name, "dir": d, "meta": meta, "prompts": prompts, "setup": setup, "verify": verify}


def all_scenarios() -> list[str]:
    return sorted(n for n in os.listdir(H2H_DIR)
                  if os.path.isdir(os.path.join(H2H_DIR, n)) and
                  os.path.exists(os.path.join(H2H_DIR, n, "meta.json")))


# ----------------------------------------------------------------------------- memagent driver
class _UsageTap:
    """Wrap the LLM so every model call's usage AND latency is captured -> per-call series, totals,
    and a STALL-EXCLUDED wall (the flaky Moonshot endpoint occasionally stalls a call to the ~75s
    hard-timeout; those infra stalls would otherwise pollute the wall-time metric)."""
    def __init__(self, inner):
        self.inner = inner
        self.calls = []       # [{prompt, completion, cached}]
        self.latencies = []   # per-call wall seconds (incl. retries)

    def complete(self, messages, tools):
        t0 = time.time()
        r = self.inner.complete(messages, tools)
        self.latencies.append(time.time() - t0)
        u = (r.usage or {}) if hasattr(r, "usage") else {}
        self.calls.append({"prompt": u.get("prompt_tokens", 0),
                           "completion": u.get("completion_tokens", 0),
                           "cached": u.get("cached_tokens", 0)})
        return r


def run_memagent(scn: dict, workdir: str, model: str) -> dict:
    from memagent.slice import Slice, make_build_slice, slice_sink, record_user
    from memagent.loop import run_turn
    from memagent.tools import LocalToolHost
    from memagent.code_index import make_code_index
    from memagent.memory import NullMemory
    from memagent.retriever import NullRetriever
    from memagent.events import make_dispatcher
    from memagent.telemetry import make_telemetry_sink
    from memagent.llm import OpenAILLM

    meta, prompts = scn["meta"], scn["prompts"]
    max_steps = int(meta.get("max_steps_per_turn", 20))

    state = Slice()
    state.reset(prompts[0])
    tools = LocalToolHost(root=workdir)
    retriever = make_code_index(workdir) if meta.get("use_code_index") else NullRetriever()
    tel = make_telemetry_sink()
    dispatch = make_dispatcher(slice_sink(state), tel)
    # Tight per-call timeout so a transient stuck connection ABORTS and retries (memagent has
    # jittered-backoff retry) instead of hanging the whole run for minutes on one wedged socket.
    tap = _UsageTap(OpenAILLM(model=model, timeout=60.0))
    memory = NullMemory()

    per_turn = []
    t0 = time.time()
    err = ""
    try:
        for i, p in enumerate(prompts):
            if i > 0:
                state.goal = p  # new turn's task (do NOT reset — that wipes continuity)
            record_user(state, p)
            build = make_build_slice(state, tools, retriever, memory, p)
            n_before = len(tap.calls)
            res = run_turn(build_slice=build, llm=tap, tools=tools, dispatch=dispatch, max_steps=max_steps)
            calls_turn = tap.calls[n_before:]
            per_turn.append({"turn": i + 1, "steps": res.steps, "stop": res.stop_reason,
                             "in": sum(c["prompt"] for c in calls_turn),
                             "out": sum(c["completion"] for c in calls_turn),
                             "peak_in": max((c["prompt"] for c in calls_turn), default=0)})
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"

    wall = time.time() - t0
    passed, detail = (False, err) if err else scn["verify"](workdir)
    calls = tap.calls
    tm = tel.summary()
    STALL = 65.0  # a call at/above this hit the ~75s hard-timeout backstop = infra stall, not real work
    stalls = sum(1 for x in tap.latencies if x >= STALL)
    wall_clean = sum(x for x in tap.latencies if x < STALL)
    return {
        "agent": "memagent", "scenario": scn["name"], "passed": bool(passed), "detail": str(detail)[:80],
        "steps": len(calls), "wall_s": round(wall, 1),
        "wall_clean": round(wall_clean, 1),  # wall excluding infra stalls (fair vs Kimi's stall-free baseline)
        "stalls": stalls,
        "in_total": sum(c["prompt"] for c in calls), "in_cached": sum(c["cached"] for c in calls),
        "out_total": sum(c["completion"] for c in calls),
        "peak_in": max((c["prompt"] for c in calls), default=0),
        "series_in": [c["prompt"] for c in calls], "per_turn": per_turn,
        "re_reads": tm.get("re_reads", 0), "recalls": tm.get("recalls", 0),
    }


# ----------------------------------------------------------------------------- kimi driver
def _kimi_session_dir(session_id: str) -> str | None:
    idx = os.path.join(KIMI_HOME, "session_index.jsonl")
    if os.path.exists(idx):
        for ln in open(idx):
            try:
                o = json.loads(ln)
            except Exception:
                continue
            if o.get("sessionId") == session_id and os.path.isdir(o.get("sessionDir", "")):
                return o["sessionDir"]
    hits = glob.glob(os.path.join(KIMI_HOME, "sessions", "*", session_id))
    return hits[0] if hits else None


def _kimi_usage(session_dir: str) -> dict:
    """Sum usage.record across ALL agents (main + any subagents) in the session. Each record is one
    model call: usage={inputOther, inputCacheRead, inputCacheCreation, output}."""
    recs = []
    for wf in glob.glob(os.path.join(session_dir, "agents", "*", "wire.jsonl")):
        for ln in open(wf, errors="ignore"):
            try:
                o = json.loads(ln)
            except Exception:
                continue
            if o.get("type") == "usage.record" and isinstance(o.get("usage"), dict):
                recs.append(o)
    recs.sort(key=lambda r: r.get("time", ""))

    def call_in(u):  # total input tokens this call (fresh + cache-read + cache-creation)
        return u.get("inputOther", 0) + u.get("inputCacheRead", 0) + u.get("inputCacheCreation", 0)

    in_total = sum(call_in(r["usage"]) for r in recs)
    in_cached = sum(r["usage"].get("inputCacheRead", 0) for r in recs)
    out_total = sum(r["usage"].get("output", 0) for r in recs)
    peak_in = max((call_in(r["usage"]) for r in recs), default=0)
    model = recs[-1].get("model", "?") if recs else "?"
    return {"calls": len(recs), "in_total": in_total, "in_cached": in_cached, "out_total": out_total,
            "peak_in": peak_in, "series_in": [call_in(r["usage"]) for r in recs], "model": model}


def run_kimi(scn: dict, workdir: str) -> dict:
    prompts = scn["prompts"]
    session_id = None
    per_turn = []
    t0 = time.time()
    err = ""
    for i, p in enumerate(prompts):
        cmd = [KIMI_BIN] + (["-r", session_id] if session_id else []) + \
              ["-p", p, "--output-format", "stream-json"]
        tt = time.time()
        try:
            proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                                  timeout=KIMI_TURN_TIMEOUT)
        except subprocess.TimeoutExpired:
            err = f"turn {i+1} timed out after {KIMI_TURN_TIMEOUT}s"
            per_turn.append({"turn": i + 1, "wall": KIMI_TURN_TIMEOUT, "rc": "timeout"})
            break
        dt = time.time() - tt
        for ln in proc.stdout.splitlines():
            try:
                o = json.loads(ln)
            except Exception:
                continue
            if o.get("type") == "session.resume_hint" or o.get("role") == "meta":
                session_id = o.get("session_id", session_id)
        per_turn.append({"turn": i + 1, "wall": round(dt, 1), "rc": proc.returncode,
                         "stderr_tail": proc.stderr.strip()[-160:]})
        if proc.returncode != 0 and session_id is None:
            err = f"turn {i+1} rc={proc.returncode}: {proc.stderr.strip()[-160:]}"
            break
        if session_id is None:
            err = f"turn {i+1}: no session id in output"
            break
    wall = time.time() - t0

    passed, detail = (False, err) if err else scn["verify"](workdir)
    sess_dir = _kimi_session_dir(session_id) if session_id else None
    u = _kimi_usage(sess_dir) if sess_dir else {"calls": 0, "in_total": 0, "in_cached": 0,
                                                "out_total": 0, "peak_in": 0, "series_in": [], "model": "?"}
    return {
        "agent": "kimi", "scenario": scn["name"], "passed": bool(passed), "detail": str(detail)[:80],
        "steps": u["calls"], "wall_s": round(wall, 1),
        "in_total": u["in_total"], "in_cached": u["in_cached"], "out_total": u["out_total"],
        "peak_in": u["peak_in"], "series_in": u["series_in"], "per_turn": per_turn,
        "kimi_model": u["model"], "session_id": session_id,
    }


# ----------------------------------------------------------------------------- orchestration
def _row(r: dict) -> str:
    tot = r["in_total"] + r["out_total"]
    fresh = r["in_total"] - r["in_cached"]
    cache_pct = 100 * r["in_cached"] / max(r["in_total"], 1)
    wall = r.get("wall_clean", r["wall_s"])
    st = r.get("stalls", 0)
    wtag = f"{wall:>7.1f}s" + (f"(+{st}st)" if st else "       ")
    return (f"{r['scenario']:24} {r['agent']:9} {'PASS' if r['passed'] else 'FAIL':4} "
            f"{r['steps']:>4} {wtag}  tok={tot:>9}  in={r['in_total']:>9} "
            f"(fresh {fresh:>8}) cache={cache_pct:>4.0f}%  out={r['out_total']:>7}  peak={r['peak_in']:>7}  {r['detail'][:30]}")


def header() -> str:
    return (f"{'scenario':24} {'agent':9} {'acc':4} {'step':>4} {'wall':>8}  {'total_tok':>13}  "
            f"{'input':>12} {'(uncached)':>14} {'output':>11}  {'peak_ctx':>8}  detail")


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    # everyone uses the 3.13 venv python (the one running THIS file) for subprocess `python3`
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None, help="one scenario name (default: all)")
    ap.add_argument("--agent", choices=["memagent", "kimi", "both"], default="both")
    ap.add_argument("--model", default=os.environ.get("AGENT_MODEL", "kimi-k2.7-code"))
    ap.add_argument("--out", default=os.path.join(H2H_DIR, "results.json"))
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from memagent.cli import _load_env
    _load_env()

    scenarios = [args.scenario] if args.scenario else all_scenarios()
    agents = ["memagent", "kimi"] if args.agent == "both" else [args.agent]

    # RESUMABLE + INCREMENTAL: load any prior results, skip done cells, persist after EACH cell so an
    # unattended crash/hang recovery never redoes finished work (just relaunch the same command).
    results = []
    if os.path.exists(args.out):
        try:
            results = json.load(open(args.out))
        except Exception:
            results = []
    done = {(r.get("scenario"), r.get("agent")) for r in results}

    print(f"head-to-head · model={args.model} · scenarios={scenarios} · agents={agents} · "
          f"resuming with {len(done)} cell(s) done\n")
    print(header())
    print("-" * 150)

    # Transient endpoint degradation (Moonshot throttling) shows up as timeouts/connection errors, not
    # agent logic failures. Retry a cell that fails PURELY on infra so a passing window isn't burned;
    # a clean PASS or a real (verifier) FAIL breaks immediately and is recorded.
    INFRA = ("timed out", "timeout", "connection", "apierror", "api error", "temporarily",
             "rate limit", "ratelimit", " 429", " 502", " 503", " 504", "overloaded")

    def is_infra_fail(r: dict) -> bool:
        return (not r.get("passed")) and any(m in str(r.get("detail", "")).lower() for m in INFRA)

    for sname in scenarios:
        scn = load_scenario(sname)
        for ag in agents:
            if (sname, ag) in done:
                print(f"{sname:24} {ag:9} SKIP (already in results.json)", flush=True)
                continue
            r = None
            for attempt in range(3):
                workdir = tempfile.mkdtemp(prefix=f"h2h-{sname}-{ag}-")
                scn["setup"](workdir)
                try:
                    r = run_memagent(scn, workdir, args.model) if ag == "memagent" else run_kimi(scn, workdir)
                except Exception as e:  # noqa: BLE001
                    r = {"agent": ag, "scenario": sname, "passed": False, "detail": f"runner: {e}",
                         "steps": 0, "wall_s": 0.0, "in_total": 0, "in_cached": 0, "out_total": 0,
                         "peak_in": 0, "series_in": [], "per_turn": []}
                r["workdir"] = workdir
                if not is_infra_fail(r):
                    break
                print(f"{sname:24} {ag:9} infra-fail attempt {attempt+1}/3 ({r['detail'][:40]}) — retrying",
                      flush=True)
            results.append(r)
            json.dump(results, open(args.out, "w"), indent=2)  # persist immediately (crash-safe)
            print(_row(r), flush=True)

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}  ({len(results)} cells)")


if __name__ == "__main__":
    main()
