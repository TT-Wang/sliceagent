"""Eval runner: run one case in a fresh temp dir, score it with the case's own
INDEPENDENT verifier, and collect metrics. LLM is injected (real or fake)."""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Callable

from memagent.code_index import make_code_index
from memagent.events import ToolResult, make_dispatcher
from memagent.loop import run_turn
from memagent.memory import NullMemory
from memagent.retriever import NullRetriever
from memagent.slice import Slice, make_build_slice, slice_sink
from memagent.tools import LocalToolHost


@dataclass
class EvalCase:
    name: str
    prompt: str
    setup: Callable[[str], None]          # write seed files into the workdir
    verify: Callable[[str], tuple[bool, str]]  # independent oracle -> (passed, detail)
    max_steps: int = 30
    use_code_index: bool = False          # real-repo cases turn on the ripgrep RELATED CODE tier


@dataclass
class EvalResult:
    name: str
    passed: bool
    detail: str
    steps: int
    tokens: int
    wall_s: float
    tool_calls: int
    re_reads: int = 0   # reconstruction-MISS signal: files re-read soon after (slice didn't carry them)
    recalls: int = 0    # recall_history calls (recovery from the cold cache)


def run_case(case: EvalCase, llm) -> EvalResult:
    workdir = tempfile.mkdtemp(prefix="memeval-")
    case.setup(workdir)

    counter = {"n": 0}

    def count_sink(e):
        if isinstance(e, ToolResult):
            counter["n"] += 1

    from memagent.telemetry import make_telemetry_sink
    tel = make_telemetry_sink()  # reconstruction-quality telemetry (re-reads / recalls), off the moat

    state = Slice()
    state.reset(case.prompt)
    tools = LocalToolHost()
    retriever = make_code_index(workdir) if case.use_code_index else NullRetriever()
    dispatch = make_dispatcher(slice_sink(state), count_sink, tel)  # silent: no terminal sink during eval
    build = make_build_slice(state, tools, retriever, NullMemory(), case.prompt)

    cwd = os.getcwd()
    t0 = time.time()
    detail = ""
    steps = tokens = 0
    try:
        os.chdir(workdir)  # the agent's relative paths (scratch/...) land here
        result = run_turn(build_slice=build, llm=llm, tools=tools, dispatch=dispatch, max_steps=case.max_steps)
        steps = result.steps
        tokens = result.usage.get("prompt_tokens", 0) + result.usage.get("completion_tokens", 0)
    except Exception as e:  # noqa: BLE001
        detail = f"run error: {e}"
    finally:
        os.chdir(cwd)

    passed = False
    if not detail:
        try:
            passed, detail = case.verify(workdir)
        except Exception as e:  # noqa: BLE001
            detail = f"verify error: {e}"

    tm = tel.summary()
    return EvalResult(case.name, passed, detail, steps, tokens, time.time() - t0, counter["n"],
                      re_reads=tm["re_reads"], recalls=tm["recalls"])


def run_eval(cases: list[EvalCase], llm, on_result=None) -> list[EvalResult]:
    """Run each case; if on_result is given it's called as each case COMPLETES (live progress —
    the suite can stream a row per case instead of going dark until the final scorecard)."""
    results = []
    for c in cases:
        r = run_case(c, llm)
        if on_result is not None:
            on_result(r)
        results.append(r)
    return results


def _row(r: EvalResult) -> str:
    return (f"{r.name:22} {'PASS' if r.passed else 'FAIL':4} {r.steps:>5} {r.tokens:>7} "
            f"{r.wall_s:>6.1f} {r.tool_calls:>5} {r.re_reads:>4} {r.recalls:>4}  {r.detail[:26]}")


def print_header() -> None:
    print(f"\n{'case':22} {'pass':4} {'steps':>5} {'tokens':>7} {'wall':>6} {'tools':>5} {'rrd':>4} {'rcl':>4}  detail",
          flush=True)
    print("-" * 96, flush=True)


def print_result_row(r: EvalResult) -> None:
    print(_row(r), flush=True)  # the live per-case callback for run_eval(on_result=...)


def print_footer(results: list[EvalResult]) -> None:
    n = len(results)
    p = sum(1 for r in results if r.passed)
    print("-" * 88, flush=True)
    print(f"PASS {p}/{n}  ·  total tokens {sum(r.tokens for r in results)}  ·  "
          f"total wall {sum(r.wall_s for r in results):.1f}s", flush=True)
