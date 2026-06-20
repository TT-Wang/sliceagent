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
from .hooks import BudgetHook, CompositeHooks, GuardrailHook, OracleHook, PermissionHook


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
    if not (os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("MOONSHOT_API_KEY")):
        print("Set LLM_API_KEY (and optionally LLM_BASE_URL) — e.g. in a .env file.")
        sys.exit(1)

    from .code_index import make_code_index
    from .config import load_config
    from .episode import make_episode_sink
    from .llm import OpenAILLM
    from .mcp_client import connect_mcp_servers
    from .loop import run_turn
    from .memory import make_memory
    from .mining import make_miner
    from .oracle import CommandOracle
    from .plugins import load_plugins
    from .policy import make_policy
    from .sandbox import make_sandbox
    from .session import Session, make_topic_tools, route_topic
    from .skills import make_skill_manager, make_skill_tool
    from .slice import make_build_slice, one_line, record_user, slice_sink
    from .subagent import SubagentHost
    from .tools import LocalToolHost

    cfg = load_config()  # memagent.toml (user → project), with ENV overriding for one-offs
    root = os.getcwd()
    policy_mode = cfg.policy        # guard | readonly | ask | allow
    mine_mode = cfg.mine           # deterministic | llm | off
    sub_depth = cfg.subagent_depth  # 0 disables delegation
    policy = make_policy(policy_mode)
    llm = OpenAILLM(model=cfg.model)
    retriever = make_code_index(root)  # ripgrep CodeIndex (RELATED CODE tier); NullRetriever if no rg
    memory = make_memory()  # memem if available + a vault is configured, else NullMemory

    sandbox = make_sandbox(cfg.sandbox_backend, image=cfg.sandbox_image, network=cfg.sandbox_network)
    base_tools = LocalToolHost(root, sandbox=sandbox)  # file ops confined to launch dir; shell via sandbox
    for _r in os.environ.get("AGENT_ROOT", "").split(os.pathsep):  # I2: extra dirs the user puts in reach
        if _r.strip():
            base_tools.add_root(_r.strip())
    skills = make_skill_manager(cfg.skills_roots)  # SKILL.md packs (config dirs or defaults)
    # plugins: feed the SAME registry/skills, and contribute MCP servers + hooks (loaded first
    # so plugin skills enter the catalog and plugin MCP servers get connected below)
    plugin_mcp, plugin_hooks = load_plugins(
        base_tools.registry, skills, cfg.plugin_dirs, root=root, config=cfg,
        on_log=lambda m: print(f"  · {m}"))
    skill_tool = make_skill_tool(skills)
    if skill_tool is not None:        # register the `skill` tool into the shared registry
        base_tools.registry.register(skill_tool)
    from .code_grep import make_grep_tool  # guarded ripgrep: the single discovery-on-demand seam
    base_tools.registry.register(make_grep_tool(base_tools))
    # MCP: connect config + plugin-declared servers; tools register into the SAME registry
    mcp_servers, mcp_runtime = connect_mcp_servers(
        base_tools.registry, {**cfg.mcp_servers, **plugin_mcp}, on_log=lambda m: print(f"  · {m}"))
    mcp_tool_count = sum(1 for e in base_tools.registry._tools.values() if e.source == "mcp")
    plugin_tool_count = sum(1 for e in base_tools.registry._tools.values()
                            if e.source.startswith("plugin:"))
    tools = base_tools
    if sub_depth > 0:  # wrap so the model can delegate sub-tasks (summary-only return)
        tools = SubagentHost(base_tools, llm=llm, retriever=retriever, memory=memory,
                             policy=policy, max_depth=sub_depth, notify=print)
    session = Session(memory)        # host-side topic manager (one bounded Slice per topic)
    llm.set_cache_key(session.session_id)   # session-stable prompt-cache routing (cheapest cache lever)
    for t in make_topic_tools(session):   # model can route topics via new_topic / switch_topic
        base_tools.registry.register(t)
    # active-asker MM syscalls: pin (deliberate working-set growth) + view (/proc introspection),
    # bound to the CURRENT active slice so a topic switch retargets them; mechanism is the SwapManager.
    from .slice import _active
    from .swap_tools import make_pin_tool, make_view_tool

    def _get_slice():
        return _active(session)
    if getattr(memory, "is_durable", False):   # model's bounded valve into the cold cache (turns + intra-turn steps)
        from .history import make_history_tool
        base_tools.registry.register(make_history_tool(memory, session.session_id, get_slice=_get_slice))
    base_tools.registry.register(make_pin_tool(_get_slice))
    base_tools.registry.register(make_view_tool(_get_slice))

    # write side of the memory loop: mine a lesson per successful, error-resolving turn
    miner = None
    if mine_mode not in ("0", "off", "none"):
        miner = make_miner(memory, session, llm=llm, mode=mine_mode,
                           scope=os.path.basename(root) or "default")
    # OPT-IN async background-review fork (item 16; OFF unless AGENT_BACKGROUND_REVIEW set).
    # Reads the durable episodic cache off-thread and consolidates incrementally — never
    # touches the slice/loop/prompt. None when disabled, so the default path is unchanged.
    from .background_review import make_background_reviewer
    reviewer = make_background_reviewer(memory, scope=os.path.basename(root) or "default",
                                        on_log=lambda m: print(f"  · {m}"))
    # episodic cache: lossless turn log (None for NullMemory → eval path untouched)
    episodic = make_episode_sink(memory, session_id=session.session_id,
                                 task_id_fn=lambda: session.active_id or "t-none",
                                 title_fn=lambda: one_line(session.active().goal, 80) if session.active_id else "")

    # optional rich TUI (the `tui` extra). Output via Rich, input via prompt_toolkit — temporally
    # separate from the synchronous run_turn, so no patch_stdout/threading. Off when piped (eval).
    _tui = None
    _stats = {"model": llm.model, "policy": policy_mode, "topic": "", "tokens": 0}
    try:
        from . import tui as _tuimod
        if _tuimod.tui_enabled():
            _tui = _tuimod
    except Exception:
        _tui = None
    _console = _tui.Console() if _tui else None

    # wire the ask_user capability to a real prompt when interactive (TUI rich prompt, or plain
    # input); headless/eval keeps the non-interactive default so it never hangs.
    if _tui or sys.stdin.isatty():
        def _ask_user(question, options):
            if _tui:
                return _tui.ask_user(_console, question, options)
            print(f"\n  ❓ {question}")
            for i, o in enumerate(options or [], 1):
                print(f"     {i}. {o}")
            try:
                a = input("  your answer ▸ ").strip()
            except (EOFError, KeyboardInterrupt):
                return "(no answer)"
            if options and a.isdigit() and 1 <= int(a) <= len(options):
                return options[int(a) - 1]
            return a or "(no answer)"
        base_tools.on_ask_user = _ask_user

    # sinks: update the active slice from tool results, mine lessons, cache the turn, persist, print
    sinks = [slice_sink(session)]
    if miner is not None:
        sinks.append(miner)
    if episodic is not None:
        sinks.append(episodic)
    sinks.append(log_sink())
    sinks.append(_tui.make_rich_sink(_console, _stats) if _tui else cli_sink(cfg.show_slice))
    # optional: feed the live web monitor (AGENT_MONITOR=1) — eval path untouched. Writes per-step
    # snapshots to the shared monitor dir; view them in the STANDING server (python -m memagent.monitor),
    # which stays up across sessions and goes idle when none is running.
    if os.environ.get("AGENT_MONITOR"):
        from .monitor import _monitor_dir, make_file_monitor_sink
        sinks.append(make_file_monitor_sink(
            session.session_id,
            context_fn=lambda: {"goal": session.active().goal if session.active_id else "",
                                "topic": session.active_id or ""}))
        print(f"  · slice monitor: writing to {_monitor_dir()} — view at the persistent server "
              "(run: python -m memagent.monitor)")
    dispatch = make_dispatcher(*sinks)
    if miner is not None:
        miner.dispatch = dispatch  # late-bind so LessonSaved flows through log + terminal sinks

    # policy hooks (the seam is always wired; default 'guard' blocks catastrophic commands)
    def _ask(name, args, reason):  # interactive resolver for AGENT_POLICY=ask
        detail = args.get("command") or args.get("path") or args.get("code", "")
        if _tui:  # synchronous mid-run (no pt app live) → a Rich confirm is safe
            return _tui.confirm(_console, name, str(detail), reason)
        if not sys.stdin.isatty():
            return "no"
        ans = input(f"  ⚠ allow {name} {str(detail)[:60]!r}? ({reason}) [y]es/[n]o/[a]lways: ").strip().lower()
        return {"y": "yes", "yes": "yes", "a": "always", "always": "always"}.get(ans, "no")

    def _handle_slash(line):  # TUI navigation palette — wired to existing session ops
        parts = line.split(maxsplit=1)
        cmd, arg = parts[0], (parts[1].strip() if len(parts) > 1 else "")
        if cmd == "/help":
            _console.print("commands: /threads · /switch <id> · /resume <id> · /help · /exit")
        elif cmd == "/threads":
            ts = session.open_threads(include_active=True)
            _console.print("  (no topics yet)" if not ts else
                           "\n".join(f"  [{t.task_id}] {t.title} ({t.status})" for t in ts))
        elif cmd in ("/switch", "/resume"):
            if not arg:
                _console.print(f"  usage: {cmd} <task_id>")
            else:
                try:
                    session.switch_topic(arg)
                    _stats["topic"] = one_line(session.active().goal, 40)
                    _console.print(f"  switched to {arg}")
                except Exception:
                    _console.print(f"  no such topic: {arg}")
        else:
            _console.print(f"  unknown command {cmd} (/help)")
        return True

    hook_list = [PermissionHook(policy, on_ask=_ask if policy_mode == "ask" else None)]
    hook_list.append(GuardrailHook())  # cross-step loop guard (per-turn counters, reset each task)
    if cfg.verify_cmd:
        oracle = CommandOracle(cfg.verify_cmd)
        hook_list.append(OracleHook(oracle, lambda out: setattr(session.active(), "last_error", f"Verification failed:\n{out[:600]}")))
    if cfg.max_tokens:
        hook_list.append(BudgetHook(cfg.max_tokens))
    hook_list.extend(plugin_hooks)  # plugins compose into the same hook chain
    hooks = CompositeHooks(*hook_list)

    info = (f"model={llm.model} · policy={policy_mode} · sandbox={cfg.sandbox_backend} · "
            f"code={type(retriever).__name__} · memory={type(memory).__name__} · "
            f"episodic={'on' if episodic is not None else 'off'} · "
            f"mine={mine_mode if miner is not None else 'off'} · subagents={'on' if sub_depth > 0 else 'off'} · "
            f"skills={len(skills.names())} · mcp_tools={mcp_tool_count} · plugin_tools={plugin_tool_count}")
    _input = _tui.TuiInput(_stats) if _tui else None
    if _tui:
        _tui.banner(_console, info)
    else:
        print("memagent · slice core (run_turn) · " + info)
        print('type a task, or "exit" to quit\n')
    while True:
        if _input is not None:
            line = _input.prompt()
            if line is None:                               # ctrl-d / EOF
                break
            line = line.strip()
        else:
            try:
                line = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
        if line in ("exit", "quit", "/exit"):
            break
        if not line:
            continue
        if _tui and line.startswith("/"):                  # navigation palette (no turn)
            _handle_slash(line)
            continue
        if session.active_id is None:                      # first message bootstraps the first topic
            session.new_topic(line)
        else:                                              # route: continue / new / resume (no junk topic)
            action, tid = route_topic(llm, line, session)
            if action == "new":
                session.new_topic(line)
            elif action == "resume":
                session.switch_topic(tid)
                session.continue_topic(line)
            else:
                session.continue_topic(line)
            if not _tui:                                   # TUI shows the topic in the status bar, not as noise
                print(f"  · topic: {action}{(' ' + tid) if tid else ''}")
        _stats["topic"] = one_line(session.active().goal, 40) if session.active_id else ""
        record_user(session.active(), line)  # short-range continuity: the RECENT CONVERSATION tier
        build = make_build_slice(session, tools, retriever, memory, line, session.session_id)
        # ctrl-c during the turn (incl. while the LLM is thinking) raises KeyboardInterrupt, which
        # run_turn catches → aborts the turn cleanly and returns here to the prompt (then ctrl-d quits).
        result = run_turn(build_slice=build, llm=llm, tools=tools, dispatch=dispatch, hooks=hooks)
        if getattr(memory, "is_durable", False):           # durable checkpoint (no-op under NullMemory)
            from .taskstate import slice_to_task_state
            memory.checkpoint_task(slice_to_task_state(
                session.active(), session.active_id, session_id=session.session_id,
                status="done" if result.stop_reason == "end_turn" else "parked"))
        if reviewer is not None:                            # OPT-IN: critique the turn off-thread
            reviewer.review(session.session_id)

    # session end: consolidate the episodic cache into long-term memory (the cache→memory loop)
    if getattr(memory, "is_durable", False):
        memory.consolidate(session.session_id)
        print("  · consolidated session memory")


if __name__ == "__main__":
    main()
