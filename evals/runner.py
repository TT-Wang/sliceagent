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


def run_case(case: EvalCase, llm) -> EvalResult:
    workdir = tempfile.mkdtemp(prefix="memeval-")
    case.setup(workdir)

    counter = {"n": 0}

    def count_sink(e):
        if isinstance(e, ToolResult):
            counter["n"] += 1

    state = Slice()
    state.reset(case.prompt)
    tools = LocalToolHost()
    retriever = make_code_index(workdir) if case.use_code_index else NullRetriever()
    dispatch = make_dispatcher(slice_sink(state), count_sink)  # silent: no terminal sink during eval
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

    return EvalResult(case.name, passed, detail, steps, tokens, time.time() - t0, counter["n"])


def run_eval(cases: list[EvalCase], llm) -> list[EvalResult]:
    return [run_case(c, llm) for c in cases]


def print_scorecard(results: list[EvalResult]) -> None:
    print(f"\n{'case':22} {'pass':4} {'steps':>5} {'tokens':>7} {'wall':>6} {'tools':>5}  detail")
    print("-" * 88)
    for r in results:
        print(f"{r.name:22} {'PASS' if r.passed else 'FAIL':4} {r.steps:>5} {r.tokens:>7} "
              f"{r.wall_s:>6.1f} {r.tool_calls:>5}  {r.detail[:30]}")
    n = len(results)
    p = sum(1 for r in results if r.passed)
    print("-" * 88)
    print(f"PASS {p}/{n}  ·  total tokens {sum(r.tokens for r in results)}  ·  "
          f"total wall {sum(r.wall_s for r in results):.1f}s")
