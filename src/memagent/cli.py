"""memagent CLI — a thin event-sink host over the stateless slice core.

The loop only dispatches events; this host wires the sinks (slice-updater, durable
log, terminal output) and the policy hooks (permission gate, optional Oracle/budget).
Other surfaces (TUI, SDK, channels) are just different sinks over the same core.
"""
from __future__ import annotations

import json
import os
import sys

from .events import (
    ApiRetry,
    AssistantText,
    Event,
    SliceBuilt,
    ToolResult,
    ToolStarted,
    TurnEnd,
    TurnInterrupted,
    make_dispatcher,
)
from .hooks import ALLOW, BudgetHook, CompositeHooks, OracleHook, PermissionHook


def _load_env(path: str = ".env") -> None:
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass


LOG_FILE = "scratch/durable-log.jsonl"


def log_sink(path: str = LOG_FILE):
    def sink(e: Event) -> None:
        rec = None
        if isinstance(e, AssistantText):
            rec = {"role": "assistant", "content": e.content}
        elif isinstance(e, ToolResult):
            rec = {"role": "tool", "name": e.name, "args": e.args, "full": e.output}
        if rec is not None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return sink


def cli_sink(show_slice: bool = False):
    def sink(e: Event) -> None:
        if isinstance(e, SliceBuilt) and show_slice:
            print("\n  ┌─ slice ─────────────")
            print("\n".join("  │ " + ln for ln in e.rendered.splitlines()))
            print("  └─────────────")
        elif isinstance(e, ToolStarted):
            print(f"  → {e.name}({json.dumps(e.args, ensure_ascii=False)})")
        elif isinstance(e, ToolResult):
            print(f"  ← {e.output[:120]}")
        elif isinstance(e, AssistantText):
            print(f"\nAssistant: {e.content}")
        elif isinstance(e, ApiRetry):
            print(f"  …retry #{e.attempt} ({e.error})")
        elif isinstance(e, TurnInterrupted):
            print(f"\n[interrupted: {e.reason}]")
        elif isinstance(e, TurnEnd):
            print(f"  [done: {e.stop_reason} · {e.steps} steps · "
                  f"{e.usage.get('prompt_tokens', 0) + e.usage.get('completion_tokens', 0)} tokens]")
    return sink


def main() -> None:
    _load_env()
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")):
        print("Set OPENAI_API_KEY (or MOONSHOT_API_KEY) — e.g. in a .env file.")
        sys.exit(1)

    from .llm import OpenAILLM
    from .loop import run_turn
    from .memory import make_memory
    from .oracle import CommandOracle
    from .retriever import NullRetriever
    from .slice import Slice, make_build_slice, slice_sink
    from .tools import LocalToolHost

    llm = OpenAILLM()
    tools = LocalToolHost()
    retriever = NullRetriever()
    memory = make_memory()  # memem if available + a vault is configured, else NullMemory
    state = Slice()

    # sinks: update the slice from tool results, persist to disk, print to terminal
    dispatch = make_dispatcher(slice_sink(state), log_sink(), cli_sink(bool(os.environ.get("SHOW_SLICE"))))

    # policy hooks (the seam is always wired; default policy is permissive — P1.5 hardens it)
    hook_list = [PermissionHook(lambda name, args: ALLOW)]
    if os.environ.get("AGENT_VERIFY_CMD"):
        oracle = CommandOracle(os.environ["AGENT_VERIFY_CMD"])
        hook_list.append(OracleHook(oracle, lambda out: setattr(state, "last_error", f"Verification failed:\n{out[:600]}")))
    if os.environ.get("AGENT_MAX_TOKENS"):
        hook_list.append(BudgetHook(int(os.environ["AGENT_MAX_TOKENS"])))
    hooks = CompositeHooks(*hook_list)

    print(f"memagent · slice core (run_turn) · model={llm.model} · memory={type(memory).__name__}")
    print('type a task, or "exit" to quit\n')
    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line in ("exit", "quit"):
            break
        if line:
            state.reset(line)
            build = make_build_slice(state, tools, retriever, memory, line)
            run_turn(build_slice=build, llm=llm, tools=tools, dispatch=dispatch, hooks=hooks)


if __name__ == "__main__":
    main()
