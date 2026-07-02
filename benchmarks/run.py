#!/usr/bin/env python3
"""Reproduce the sliceagent multi-turn coding benchmark (README §3).

Drives sliceagent over a scenario's fixed, pre-written turns (a scripted "user" — deterministic, so both
a slice agent and any transcript agent get byte-identical turns), then scores the final repo with the
scenario's own verifier and reports per-turn + total metrics: pass, per-call peak input, tokens
(input/cached/output), wall, steps.

Usage (needs `pip install "sliceagent[tui]"` and an LLM configured — LLM_API_KEY + AGENT_MODEL, or
`sliceagent init`):

    python benchmarks/run.py                         # all three scenarios
    python benchmarks/run.py --scenario s1_longhorizon_debug
    AGENT_REASONING=high python benchmarks/run.py    # match the published run

Note: s2_largefile_bug plants its bug in a copy of CPython 3.13's argparse, so its setup() must run under
Python 3.13 (the agent + verifier are version-independent). Run that one on a 3.13 interpreter.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = os.path.join(HERE, "multiturn_coding")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_scenario(name):
    d = os.path.join(TASKS, name)
    return {
        "name": name,
        "meta": json.load(open(os.path.join(d, "meta.json"))),
        "prompts": json.load(open(os.path.join(d, "prompts.json"))),
        "setup": _load(os.path.join(d, "setup.py"), f"{name}_setup").setup,
        "verify": _load(os.path.join(d, "verify.py"), f"{name}_verify").verify,
    }


class _Tap:
    """Wrap the LLM to capture every call's token usage + latency."""
    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def complete(self, messages, tools):
        t0 = time.time()
        r = self.inner.complete(messages, tools)
        u = (r.usage or {}) if hasattr(r, "usage") else {}
        self.calls.append({"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0),
                           "cached": u.get("cached_tokens", 0), "wall": time.time() - t0})
        return r


def run(scn):
    from sliceagent.code_index import make_code_index
    from sliceagent.events import make_dispatcher
    from sliceagent.llm import OpenAILLM
    from sliceagent.loop import run_turn
    from sliceagent.memory import NullMemory
    from sliceagent.pfc import Slice, record_user, slice_sink
    from sliceagent.retriever import NullRetriever
    from sliceagent.seed import make_build_slice
    from sliceagent.tools import LocalToolHost

    workdir = tempfile.mkdtemp(prefix=f"bench-{scn['name']}-")
    scn["setup"](workdir)
    meta, prompts = scn["meta"], scn["prompts"]
    max_steps = int(meta.get("max_steps_per_turn", 20))

    state = Slice(); state.reset(prompts[0])
    tools = LocalToolHost(root=workdir)
    retriever = make_code_index(workdir) if meta.get("use_code_index") else NullRetriever()
    tap = _Tap(OpenAILLM(model=os.environ.get("AGENT_MODEL", "gpt-5.5"), timeout=60.0))
    if hasattr(tap.inner, "set_cache_key"):
        tap.inner.set_cache_key(os.path.basename(workdir))
    memory = NullMemory()
    dispatch = make_dispatcher(slice_sink(state))

    per_turn = []; t0 = time.time(); err = ""
    try:
        for i, p in enumerate(prompts):
            if i > 0:
                state.goal = p
            record_user(state, p)
            n0 = len(tap.calls)
            run_turn(build_slice=make_build_slice(state, tools, retriever, memory, p),
                     llm=tap, tools=tools, dispatch=dispatch, max_steps=max_steps)
            ct = tap.calls[n0:]
            per_turn.append({"turn": i + 1, "peak_in": max((c["in"] for c in ct), default=0),
                             "in": sum(c["in"] for c in ct), "out": sum(c["out"] for c in ct),
                             "wall": round(sum(c["wall"] for c in ct), 1)})
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"

    passed, detail = (False, err) if err else scn["verify"](workdir)
    calls = tap.calls
    return {
        "scenario": scn["name"], "passed": bool(passed), "detail": str(detail)[:100],
        "steps": len(calls), "wall_s": round(time.time() - t0, 1),
        "peak_in": max((c["in"] for c in calls), default=0),
        "in_total": sum(c["in"] for c in calls), "in_cached": sum(c["cached"] for c in calls),
        "out_total": sum(c["out"] for c in calls), "per_turn": per_turn,
    }


def main():
    ap = argparse.ArgumentParser(description="Run the sliceagent multi-turn coding benchmark.")
    ap.add_argument("--scenario", default=None, help="one scenario name, or omit for all three")
    args = ap.parse_args()
    names = [args.scenario] if args.scenario else sorted(
        n for n in os.listdir(TASKS) if os.path.isdir(os.path.join(TASKS, n)))
    for name in names:
        try:
            r = run(load_scenario(name))
        except Exception as e:  # noqa: BLE001
            print(f"{name}: setup/run error — {type(e).__name__}: {e}"
                  f"{'  (s2 needs Python 3.13 for setup)' if 's2' in name else ''}")
            continue
        print(f"\n{r['scenario']}: {'PASS' if r['passed'] else 'FAIL'}  "
              f"steps={r['steps']} peak_in={r['peak_in']:,} "
              f"tokens={r['in_total'] + r['out_total']:,} (cached {r['in_cached']:,}) wall={r['wall_s']}s"
              f"{'' if r['passed'] else '  · ' + r['detail']}")
        for t in r["per_turn"]:
            print(f"    turn {t['turn']}: peak_in={t['peak_in']:,} in={t['in']:,} out={t['out']:,} wall={t['wall']}s")


if __name__ == "__main__":
    main()
