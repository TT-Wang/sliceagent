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
    LessonSaved,
    SliceBuilt,
    ToolResult,
    ToolStarted,
    TurnEnd,
    TurnInterrupted,
    make_dispatcher,
)
from .hooks import BudgetHook, CompositeHooks, OracleHook, PermissionHook


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
        elif isinstance(e, LessonSaved):
            rec = {"role": "lesson", "title": e.title, "content": e.content}
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
        elif isinstance(e, LessonSaved):
            print(f"  💡 learned: {e.title}")
        elif isinstance(e, TurnEnd):
            print(f"  [done: {e.stop_reason} · {e.steps} steps · "
                  f"{e.usage.get('prompt_tokens', 0) + e.usage.get('completion_tokens', 0)} tokens]")
    return sink


def main() -> None:
    _load_env()
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")):
        print("Set OPENAI_API_KEY (or MOONSHOT_API_KEY) — e.g. in a .env file.")
        sys.exit(1)

    from .code_index import make_code_index
    from .llm import OpenAILLM
    from .loop import run_turn
    from .memory import make_memory
    from .mining import make_miner
    from .oracle import CommandOracle
    from .policy import make_policy
    from .slice import Slice, make_build_slice, slice_sink
    from .subagent import SubagentHost
    from .tools import LocalToolHost

    root = os.getcwd()
    policy_mode = os.environ.get("AGENT_POLICY", "guard")  # guard | readonly | allow
    mine_mode = os.environ.get("AGENT_MINE", "deterministic")  # deterministic | llm | off
    sub_depth = int(os.environ.get("AGENT_SUBAGENT_DEPTH", "1"))  # 0 disables delegation
    policy = make_policy(policy_mode)
    llm = OpenAILLM()
    retriever = make_code_index(root)  # ripgrep CodeIndex (RELATED CODE tier); NullRetriever if no rg
    memory = make_memory()  # memem if available + a vault is configured, else NullMemory

    tools = LocalToolHost(root)  # file ops confined to the launch dir; shell via LocalSandbox
    if sub_depth > 0:  # wrap so the model can delegate sub-tasks (summary-only return)
        tools = SubagentHost(tools, llm=llm, retriever=retriever, memory=memory,
                             policy=policy, max_depth=sub_depth, notify=print)
    state = Slice()

    # write side of the memory loop: mine a lesson per successful, error-resolving turn
    miner = None
    if mine_mode not in ("0", "off", "none"):
        miner = make_miner(memory, state, llm=llm, mode=mine_mode,
                           scope=os.path.basename(root) or "default")

    # sinks: update the slice from tool results, mine lessons, persist to disk, print
    sinks = [slice_sink(state)]
    if miner is not None:
        sinks.append(miner)
    sinks += [log_sink(), cli_sink(bool(os.environ.get("SHOW_SLICE")))]
    dispatch = make_dispatcher(*sinks)
    if miner is not None:
        miner.dispatch = dispatch  # late-bind so LessonSaved flows through log + terminal sinks

    # policy hooks (the seam is always wired; default 'guard' blocks catastrophic commands)
    hook_list = [PermissionHook(policy)]
    if os.environ.get("AGENT_VERIFY_CMD"):
        oracle = CommandOracle(os.environ["AGENT_VERIFY_CMD"])
        hook_list.append(OracleHook(oracle, lambda out: setattr(state, "last_error", f"Verification failed:\n{out[:600]}")))
    if os.environ.get("AGENT_MAX_TOKENS"):
        hook_list.append(BudgetHook(int(os.environ["AGENT_MAX_TOKENS"])))
    hooks = CompositeHooks(*hook_list)

    print(f"memagent · slice core (run_turn) · model={llm.model} · policy={policy_mode} · "
          f"code={type(retriever).__name__} · memory={type(memory).__name__} · "
          f"mine={mine_mode if miner is not None else 'off'} · subagents={'on' if sub_depth > 0 else 'off'}")
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
