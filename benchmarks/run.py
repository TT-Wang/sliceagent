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


def _configured_llm():
    """Use the same env-over-config provider resolution promised by the benchmark README."""
    from sliceagent.config import load_config, load_prefs
    from sliceagent.llm import OpenAILLM

    cfg = load_config()
    prefs = load_prefs()
    providers = cfg.providers() or {}
    pinned = prefs.get("provider")
    table = providers.get(pinned) if pinned in providers else None
    if table and table.get("api_key"):
        configured_key, configured_base = table["api_key"], table.get("base_url") or ""
        preferred_model = prefs.get("model") or table.get("model") or cfg.model
    else:
        configured_key, configured_base = cfg.api_key, cfg.base_url
        preferred_model = (None if pinned and pinned not in providers else prefs.get("model")) or cfg.model
    model = os.environ.get("AGENT_MODEL") or preferred_model
    api_key = os.environ.get("LLM_API_KEY") or configured_key
    base_url = os.environ.get("LLM_BASE_URL") or configured_base
    if not model or not api_key:
        raise ValueError("No configured model/key. Run `sliceagent init` or export AGENT_MODEL + LLM_API_KEY.")
    return OpenAILLM(model=model, api_key=api_key, base_url=base_url or None, timeout=60.0)


class _Tap:
    """Wrap the LLM to capture every call's token usage + latency."""
    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def __getattr__(self, name):
        # Instrumentation must be transparent: model identity, context-window hints, retry classification,
        # provider endpoint, and cache hooks are part of the model-runner contract.
        return getattr(self.inner, name)

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
    tap = _Tap(_configured_llm())
    if hasattr(tap.inner, "set_cache_key"):
        tap.inner.set_cache_key(os.path.basename(workdir))
    memory = NullMemory()
    # State reduction is authoritative, not a best-effort observer. A reducer failure must fail the eval.
    dispatch = make_dispatcher(required=(slice_sink(state),))

    per_turn = []; t0 = time.time(); err = ""
    try:
        for i, p in enumerate(prompts):
            record_user(state, p)
            n0 = len(tap.calls)
            result = run_turn(build_slice=make_build_slice(state, tools, retriever, memory, p),
                              llm=tap, tools=tools, dispatch=dispatch, max_steps=max_steps)
            ct = tap.calls[n0:]
            per_turn.append({"turn": i + 1, "peak_in": max((c["in"] for c in ct), default=0),
                             "in": sum(c["in"] for c in ct), "out": sum(c["out"] for c in ct),
                             "wall": round(sum(c["wall"] for c in ct), 1),
                             "stop": result.stop_reason})
            # Match the real host lifecycle: semantic state carries; detailed calls/trajectory counters do not.
            state.seal()
            if result.stop_reason != "end_turn":
                err = f"turn {i + 1} stopped abnormally: {result.stop_reason}"
                break
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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the sliceagent multi-turn coding benchmark.")
    ap.add_argument("--scenario", default=None, help="one scenario name, or omit for all three")
    args = ap.parse_args(argv)
    names = [args.scenario] if args.scenario else sorted(
        n for n in os.listdir(TASKS) if os.path.isdir(os.path.join(TASKS, n)))
    failed = False
    for name in names:
        try:
            r = run(load_scenario(name))
        except Exception as e:  # noqa: BLE001
            print(f"{name}: setup/run error — {type(e).__name__}: {e}")
            failed = True
            continue
        failed = failed or not r["passed"]
        print(f"\n{r['scenario']}: {'PASS' if r['passed'] else 'FAIL'}  "
              f"steps={r['steps']} peak_in={r['peak_in']:,} "
              f"tokens={r['in_total'] + r['out_total']:,} (cached {r['in_cached']:,}) wall={r['wall_s']}s"
              f"{'' if r['passed'] else '  · ' + r['detail']}")
        for t in r["per_turn"]:
            print(f"    turn {t['turn']}: peak_in={t['peak_in']:,} in={t['in']:,} out={t['out']:,} wall={t['wall']}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
