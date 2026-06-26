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
                    v = v.strip()
                    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                        v = v[1:-1]   # drop surrounding quotes (common .env convention) so the key isn't literal-quoted
                    os.environ.setdefault(k.strip(), v)
    except FileNotFoundError:
        pass


LOG_FILE = "scratch/durable-log.jsonl"
LOG_MAX_BYTES = 5 * 1024 * 1024   # rotate the debug log past this (keep one prior) — Kimi RotatingFileSink


def log_sink(path: str = LOG_FILE):
    from .safety import redact_text   # strip secrets before they hit the on-disk debug log (off the moat)

    def _scrub_args(args: dict) -> dict:          # redact string values (edit_file content, inline tokens)
        return {k: (redact_text(v) if isinstance(v, str) else v) for k, v in (args or {}).items()}

    def sink(e: Event) -> None:
        rec = None
        # REDACT each string field BEFORE serializing (redacting the JSON line itself can corrupt
        # quotes/escapes). A .env read or a token in a command must not land in the log in plaintext
        # — reuses the same safety.redact_text the episodic-persist path uses, so the stores agree.
        if isinstance(e, AssistantText):
            rec = {"role": "assistant", "content": redact_text(e.content)}
        elif isinstance(e, ToolResult):
            rec = {"role": "tool", "name": e.name, "args": _scrub_args(e.args), "full": redact_text(e.output)}
        elif isinstance(e, LessonSaved):
            rec = {"role": "lesson", "title": redact_text(e.title), "content": redact_text(e.content)}
        if rec is not None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            try:                                  # rotate so the debug log can't grow unbounded
                if os.path.getsize(path) > LOG_MAX_BYTES:
                    os.replace(path, path + ".1")
            except OSError:
                pass
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return sink


def _plain_arg(args: dict) -> str:
    """The one informative arg for a plain-mode tool line (path/command/pattern/…), whitespace-collapsed."""
    if not isinstance(args, dict):
        return ""
    for k in ("path", "command", "pattern", "name", "ref", "goal", "task"):
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return " ".join(v.split())[:60]
    v = next((x for x in args.values() if isinstance(x, str) and x.strip()), "")
    return " ".join(v.split())[:60]


def cli_sink(show_slice: bool = False):
    """Plain stdout sink (AGENT_TUI=off / no tui extra / pipes / CI): no color, no spinner — one readable
    line per tool action (✓/✗ + name + primary arg, output for commands/failures) and a delimited answer."""
    _quiet = {"read_file", "list_files"}   # header says enough; their content is noise in a log

    def sink(e: Event) -> None:
        if isinstance(e, SliceBuilt) and show_slice:
            print("\n  ┌─ slice ─────────────")
            print("\n".join("  │ " + ln for ln in e.rendered.splitlines()))
            print("  └─────────────")
        elif isinstance(e, ToolResult):
            mark = "✓" if not e.failing else "✗"
            head = f"  {mark} {e.name} {_plain_arg(e.args)}".rstrip()
            out = " ".join((e.output or "").split())[:140]
            if out and (e.failing or e.name not in _quiet):
                head += f"  — {out}"
            print(head)
        elif isinstance(e, AssistantText):
            if (e.content or "").strip():
                print(f"\n{e.content}\n")
        elif isinstance(e, ApiRetry):
            print(f"  …retry #{e.attempt} ({e.error})")
        elif isinstance(e, TurnInterrupted):
            print(f"\n[interrupted: {e.reason}]")
        elif isinstance(e, LessonSaved):
            print(f"  💡 learned: {e.title}")
        elif isinstance(e, TurnEnd):
            u = e.usage or {}            # usage is non-None from loop.py, but guard like every other sink
            print(f"  [done: {e.stop_reason} · {e.steps} steps · "
                  f"{u.get('prompt_tokens', 0) + u.get('completion_tokens', 0)} tokens]")
    return sink


def _reasoning_note(llm) -> str:
    """One-line hint on whether the chosen reasoning effort will actually take effect, so /model isn't a
    silent no-op (gpt-5 high/max needs /v1/responses; non-reasoning models have no effort knob)."""
    from .model_catalog import capability
    eff = (getattr(llm, "reasoning", "full") or "full").lower()
    if eff == "full":
        return ""
    if not capability(llm.model, getattr(llm, "_base_url", "")).supports_reasoning_effort:
        return f"note: {llm.model} has no reasoning-effort knob — it runs at the provider default."
    return "high/max run WITH tools via /v1/responses." if eff in ("high", "max") else ""


def main() -> None:
    # subcommands (onboarding / discovery) are handled BEFORE any key gate, so `memagent init` runs on a
    # machine with nothing configured yet. A bare `memagent` (or one with non-subcommand args) falls through.
    _argv = sys.argv[1:]
    if _argv and _argv[0] in ("init", "config", "help", "--help", "-h", "version", "--version", "-V"):
        from .onboarding import dispatch as _dispatch
        sys.exit(_dispatch(_argv))

    _load_env()
    # config-persisted key/endpoint (written by `memagent init`) populate the env BEFORE the gate, so a
    # configured user never has to export anything; ENV still wins for one-off overrides.
    from .config import load_config
    cfg = load_config()
    for _env, _val in (("LLM_API_KEY", cfg.api_key), ("LLM_BASE_URL", cfg.base_url)):
        if not os.environ.get(_env) and _val:
            os.environ[_env] = _val
    if not (os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("MOONSHOT_API_KEY")):
        print("No API key found. Run `memagent init` for guided setup, or set LLM_API_KEY (e.g. in a .env file).")
        sys.exit(1)
    # validate enum env vars (warn + use default; never crash) — a typo'd AGENT_POLICY is now visible.
    from .envspec import validate_env
    for _w in validate_env():
        print(f"  · config warning: {_w}")

    from .code_index import make_code_index
    from .episode import make_episode_sink
    from .llm import OpenAILLM
    from .mcp_client import connect_mcp_servers
    from .loop import run_turn
    from .memory import make_memory
    from .oracle import CommandOracle
    from .plugins import load_plugins
    from .policy import CONFIRMS, make_policy, policy_label, resolve_policy_mode
    from .sandbox import make_sandbox
    from .session import Session, make_topic_tools, route
    from .skills import make_skill_manager, make_skill_tool
    from .slice import consolidate_checkpoint, make_build_slice, one_line, record_user, slice_sink
    from .subagent import SubagentHost
    from .tools import LocalToolHost

    # cfg already loaded above (for the config→env key population + the gate)
    root = os.getcwd()
    from .config import load_prefs, save_prefs
    _prefs = load_prefs()
    # mode resolution: explicit env wins, then the saved /mode choice, then config (default teenager).
    canonical = resolve_policy_mode(os.environ.get("AGENT_POLICY") or _prefs.get("policy") or cfg.policy) or "teenager"
    mine_mode = cfg.mine           # deterministic | llm | off
    sub_depth = cfg.subagent_depth  # 0 disables delegation
    # SUBAGENT/base policy never prompts (no human in a spawned turn) → a confirm-mode runs as let-it-go for
    # them (still blocks catastrophic). The MAIN agent's confirming policy is built at the hook below.
    policy = make_policy("letitgo" if CONFIRMS.get(canonical) else canonical)
    # model + reasoning resolution: explicit env wins, then the saved /model choice (prefs), then config.
    _model = os.environ.get("AGENT_MODEL") or _prefs.get("model") or cfg.model
    llm = OpenAILLM(model=_model)
    if _prefs.get("reasoning") and not (os.environ.get("AGENT_REASONING") or os.environ.get("AGENT_THINKING")):
        llm.reasoning = str(_prefs["reasoning"]).lower()   # apply the saved /reasoning choice
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
    # web tools (fetch_url + web_search, DuckDuckGo, no key) — network egress, so gated by AGENT_WEB
    # (default ON). SSRF-guarded + results fenced UNTRUSTED + large pages paged (see web.py).
    if os.environ.get("AGENT_WEB", "1").strip().lower() not in ("0", "off", "false", "no"):
        from .web import make_web_tools
        for _wt in make_web_tools(base_tools):
            base_tools.registry.register(_wt)
    # foreground SKILL writer — the agent-callable tool /learn drives to turn a transcript into a
    # reusable USER-provenance skill (guarded write: validate + threat-scan + redact + atomic).
    from .memory import make_write_skill_tool
    base_tools.registry.register(make_write_skill_tool())
    # MCP: connect config + plugin-declared servers; tools register into the SAME registry
    mcp_servers, mcp_runtime = connect_mcp_servers(
        base_tools.registry, {**cfg.mcp_servers, **plugin_mcp}, on_log=lambda m: print(f"  · {m}"),
        page_out=base_tools._page_out)   # big MCP results → blob + head/tail view, not inlined whole
    mcp_tool_count = sum(1 for e in base_tools.registry._tools.values() if e.source == "mcp")
    plugin_tool_count = sum(1 for e in base_tools.registry._tools.values()
                            if e.source.startswith("plugin:"))
    # subagent activity → ONE dynamic line (Kimi-style), not a line per child tool call. Late-bound: the
    # renderer is set once the rich sink exists (below); plain/headless leaves it None (the spawn tool's
    # result line carries the child's summary, so nothing is lost).
    _sub_render: dict = {"fn": None}
    def _notify_subagent(text):
        fn = _sub_render["fn"]
        if fn is not None:
            fn(text)
    tools = base_tools
    if sub_depth > 0:  # wrap so the model can delegate sub-tasks (summary-only return)
        from .agents import load_agents
        # named-agent registry: built-ins (explorer, general) + user-defined <root>/agents/*.md (Kimi-style)
        agent_roots = list(cfg.skills_roots or []) + [root, os.path.join(root, ".memagent")]
        tools = SubagentHost(base_tools, llm=llm, retriever=retriever, memory=memory,
                             policy=policy, max_depth=sub_depth, notify=_notify_subagent,
                             agents=load_agents(agent_roots))
    session = Session(memory)        # host-side topic manager (one bounded Slice per topic)
    llm.set_cache_key(session.session_id)   # session-stable prompt-cache routing (cheapest cache lever)
    for t in make_topic_tools(session):   # model can route topics via new_topic / switch_topic
        base_tools.registry.register(t)
    # recall_history: the model's bounded valve into the cold cache (paged-out turns of this session).
    if getattr(memory, "is_durable", False):
        from .history import make_history_tool
        base_tools.registry.register(make_history_tool(memory, session.session_id))

    # write side of the memory loop is CACHE-ONLY: distillation runs at session end in
    # memory.consolidate (reads the episodic cache, never the slice). `mine_mode` (off|deterministic|llm)
    # gates that consolidation — see the session-end call below. No per-turn slice-coupled miner.
    # OPT-IN async background-review fork (item 16; OFF unless AGENT_BACKGROUND_REVIEW set).
    # Reads the durable episodic cache off-thread and consolidates incrementally — never
    # touches the slice/loop/prompt. None when disabled, so the default path is unchanged.
    from .background_review import make_background_reviewer
    reviewer = make_background_reviewer(memory, scope=os.path.basename(root) or "default",
                                        on_log=lambda m: print(f"  · {m}"))
    # episodic cache: lossless turn log (None for NullMemory → eval path untouched)
    episodic = make_episode_sink(memory, session_id=session.session_id,
                                 task_id_fn=lambda: session.active_id or "t-none",
                                 title_fn=lambda: one_line(session.active().goal, 80) if session.active_id else "",
                                 # task-outcome signal for consolidation: how many STANDING REQUIREMENTS
                                 # were still open at turn end (0 = none declared OR all met). promote_procedures
                                 # won't mine a "successful workflow" skill from a task that left some unmet.
                                 outcome_fn=lambda: {"requirements_open": sum(
                                     1 for r in session.active().requirements
                                     if isinstance(r, dict) and not r.get("done"))} if session.active_id else {})

    # optional rich TUI (the `tui` extra). Output via Rich, input via prompt_toolkit — temporally
    # separate from the synchronous run_turn, so no patch_stdout/threading. Off when piped (eval).
    _tui = None
    _stats = {"model": llm.model, "policy": policy_label(canonical), "topic": "", "tokens": 0}
    try:
        from . import tui as _tuimod
        if _tuimod.tui_enabled():
            _tui = _tuimod
    except Exception:
        _tui = None
    _console = _tui.make_console() if _tui else None   # themed: no black-bg highlight on inline `code`/paths

    # Decide early whether to use the full-screen Textual TUI. We need this before wiring stdin-
    # dependent callbacks, because Textual owns stdin and synchronous prompts from worker threads
    # would deadlock.
    tui_env = os.environ.get("AGENT_TUI", "").strip().lower()
    force_textual = tui_env in ("1", "on", "true", "yes", "textual")
    disable_textual = tui_env in ("0", "off", "false", "no", "rich")
    # DEFAULT UI = the inline rich+prompt_toolkit REPL: it stays in the NORMAL terminal buffer, so native
    # copy / paste / scrollback work on ANY terminal (incl. macOS Terminal.app), with a pinned composer
    # (patch_stdout, the Python analogue of Ink's <Static>+live-region that Hermes/Claude Code use) and
    # streaming replies. The full-screen Textual app is now OPT-IN (AGENT_TUI=textual): it looks nicer but
    # uses the alternate screen + mouse capture, which break copy/paste/scrollback on stock terminals that
    # lack OSC-52. A wide-audience CLI must work everywhere, so inline is the default. AGENT_TUI=off → plain.
    use_textual = (_tui is not None and not disable_textual and force_textual)
    # AGENT_TUI=live → the always-pinned live composer: the bordered box stays at the bottom EVEN WHILE the
    # agent streams (output prints above it in the normal buffer). Opt-in/experimental; the default REPL
    # (box between turns) is the proven path. Falls back to the REPL if it can't start.
    use_live = (_tui is not None and tui_env == "live")
    if use_textual:
        try:
            from .tui_app import textual_available
            use_textual = textual_available()
        except Exception:
            use_textual = False

    # Build the Textual app BEFORE the dispatcher, so its event sink can BE the sole renderer (the app
    # re-runs run_turn through the same dispatcher it renders from). dispatch + _hooks are injected once
    # they exist (below). Created here (not after dispatch) to avoid the old crossed wiring where the rich
    # sink AND the textual app both received events.
    app = None
    if use_textual:
        from .tui_app import MemagentTui
        from . import recovery as _rec
        app = MemagentTui(
            session=session, tools=tools, retriever=retriever, memory=memory,
            llm=llm, hooks=None, dispatch=None,
            run_turn=run_turn, make_build_slice=make_build_slice,
            record_user=record_user, route_topic=route,
            stats=_stats,
            max_steps=cfg.max_steps,   # else Textual turns are guillotined at run_turn's 40 default
            consolidate=lambda: consolidate_checkpoint(session.active(), compact=False),
            checkpoint=lambda m, s: _rec.record(
                root, goal=(session.active().goal if session.active_id else ""), messages=m, step=s),
            clear_recovery=lambda: _rec.clear(root),
        )

    # Note: in Textual mode, interactive approval and ask_user are wired to the Textual app
    # below (after dispatch is built). Here we only set up the fallback non-Textual prompts.
    if not use_textual and not use_live and (_tui or sys.stdin.isatty()):
        # wire the ask_user capability to a real prompt when interactive (TUI rich prompt, or plain
        # input); headless/eval — AND live mode, where a worker-thread console.input() would contend with
        # the pinned prompt_toolkit app for stdin and HANG — keep the non-interactive default.
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

    # sinks: update the active slice from tool results, cache the turn, persist, print (no per-turn
    # miner — distillation is cache-only at session end via memory.consolidate).
    sinks = [slice_sink(session)]
    if episodic is not None:
        sinks.append(episodic)
    sinks.append(log_sink())
    # optional: the moat-MEASURING cost sink (AGENT_METRICS=1). Accumulates the per-turn FRESH-input
    # curve (should stay flat as the conversation grows) + cache-hit rate + reliability counters; the
    # summary prints at session end. Pure observer — eval/default path untouched.
    metrics = None
    if os.environ.get("AGENT_METRICS"):
        from .metrics import make_metrics_sink
        metrics = make_metrics_sink()
        sinks.append(metrics)
    # EXACTLY ONE renderer is wired: the full-screen Textual app, OR the rich+prompt_toolkit sink, OR the
    # plain stdout sink (headless/eval). Never two — wiring the rich sink alongside Textual made rich print
    # under the alternate screen while the Textual pane was starved of agent events.
    if use_textual:
        sinks.append(MemagentTui.make_sink(app))   # the Textual app IS the renderer (events → its pane)
    elif _tui:
        _rich = _tui.make_rich_sink(_console, _stats)
        sinks.append(_rich)
        llm.set_delta_sink(_rich.on_delta)   # STREAM completions live into the rich TUI spinner (Kimi-style)
        # child agent activity → one dynamic spinner line. NOT in live mode: a rich console Status would
        # fight the pinned prompt_toolkit Application for the screen (garbled output) — let the spawn tool's
        # result line carry the child summary instead, as the plain/headless path does.
        _sub_render["fn"] = None if use_live else _rich.subagent_notify
    else:
        sinks.append(cli_sink(cfg.show_slice))
    # optional: feed the live web monitor (AGENT_MONITOR=1) — eval path untouched. Writes per-step
    # snapshots to the shared monitor dir; view them in the STANDING server (python -m memagent.monitor),
    # which stays up across sessions and goes idle when none is running.
    monitor_sink = None
    if os.environ.get("AGENT_MONITOR"):
        from .monitor import _monitor_dir, make_file_monitor_sink
        monitor_sink = make_file_monitor_sink(
            session.session_id,
            context_fn=lambda: {"goal": session.active().goal if session.active_id else "",
                                "topic": session.active_id or ""})
        sinks.append(monitor_sink)
        print(f"  · slice monitor: writing to {_monitor_dir()} — view at the persistent server "
              "(run: python -m memagent.monitor)")
    dispatch = make_dispatcher(*sinks)
    if app is not None:            # inject the dispatcher the app (created earlier) re-runs turns through
        app._dispatch = dispatch

    # policy hooks (the seam is always wired; default 'guard' blocks catastrophic commands)
    def _ask(name, args, reason):  # interactive resolver for AGENT_POLICY=ask
        detail = args.get("command") or args.get("path") or args.get("code", "")
        if app is not None:  # Textual modal dialog (runs in worker thread, marshals to UI)
            return app.confirm(name, args, reason)
        if use_live:  # the pinned prompt_toolkit app owns stdin → a Rich confirm would hang; deny (safe default)
            return "no"
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
            _console.print("commands: /model · /mode · /learn · /plan · /cost · /threads · /plugins · /mcp · "
                           "/help · /exit\n  (type / for the menu · Esc = undo last turn · "
                           "say \"review my changes\" for code_review · @path pins a file)")
        elif cmd == "/plan":
            s = session.active() if session.active_id else None
            plan = getattr(s, "plan", None) if s else None
            mission = getattr(s, "mission", "") if s else ""
            if mission:
                _console.print(f"  🎯 mission: {mission}")
            if not plan:
                _console.print("  (no active plan — the agent sets one with update_plan on multi-step tasks)")
            else:
                mark = {"done": "✓", "in_progress": "▶", "pending": "○"}
                for it in plan:
                    _console.print(f"  {mark.get(it.get('status'), '○')} {it.get('step', '')}")
        elif cmd == "/cost":
            from .tui import _saved_dollars
            saved = _saved_dollars(_stats)
            spent = _stats.get("cost", 0.0)
            head = (f"  💰 saved ${saved:.4f} vs full-history (@ {_stats.get('model','?')}, cache-aware)"
                    f"  ·  spent ${spent:.4f}" if saved is not None
                    else f"  💰 {_stats.get('saved_cached_tok', 0):,} tokens saved vs full-history (model price unknown)")
            _console.print(head)
            if metrics is None:
                _console.print("  (per-turn curve off — start with AGENT_METRICS=1 to track it)")
            else:
                s = metrics.summary()
                _console.print(f"  per_turn_fresh={s['per_turn_fresh']} avg={s['avg_turn_fresh']} "
                               f"cache_hit={s['cache_hit_rate']} tools={s['tool_calls']} "
                               f"out={s['output']} retries={s['retries']} overflows={s['overflows']}")
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
        elif cmd == "/undo":
            _console.print("  " + base_tools.undo_last())   # revert the last file edit
        elif cmd == "/plugins":
            tools = sorted(e.name for e in base_tools.registry._tools.values()
                           if getattr(e, "source", "") == "plugin")
            _console.print(f"  plugin dirs: {', '.join(cfg.plugin_dirs) or '(none configured)'}")
            _console.print(f"  plugin tools ({len(tools)}): {', '.join(tools) or '(none loaded)'}")
        elif cmd == "/mcp":
            configured = list(cfg.mcp_servers.keys())
            mtools = sorted(e.name for e in base_tools.registry._tools.values()
                            if getattr(e, "source", "") == "mcp")
            if not configured and not mtools:
                _console.print("  no MCP servers configured — add [mcp_servers.<name>] to ~/.memagent/config.toml")
            else:
                _console.print(f"  configured servers: {', '.join(configured) or '(none)'}")
                _console.print(f"  connected tools ({len(mtools)}): {', '.join(mtools) or '(none — check startup logs)'}")
        elif cmd == "/mode":
            _menu_ok = sys.stdin.isatty() and not use_live
            chosen = None
            if not arg and _menu_ok:                       # no arg + interactive → open the picker menu
                from .tui import run_selector
                order = ["babysitter", "teenager", "letitgo"]
                rows = [("baby-sitter", "confirm every edit + command"),
                        ("teenager", "auto edits, confirm commands"),
                        ("let-it-go", "auto-run — still blocks catastrophic moves")]
                cur = next((i for i, k in enumerate(order) if policy_label(k) == _stats.get("policy")), -1)
                pick = run_selector("Permission mode", rows, current=cur)
                chosen = order[pick] if pick is not None else None
            elif arg:                                      # typed: /mode teenager
                chosen = resolve_policy_mode(arg)
                if chosen not in ("babysitter", "teenager", "letitgo"):
                    _console.print("  unknown mode — use: baby-sitter | teenager | let-it-go"); return True
            else:                                          # no arg, non-interactive → just show current
                _console.print(f"  mode: [bold]{_stats.get('policy', '?')}[/]   options: baby-sitter · teenager · let-it-go")
                _console.print("    baby-sitter = confirm every edit + command · teenager = auto edits, confirm "
                               "commands · let-it-go = auto (blocks catastrophic)")
            if chosen:
                eff = chosen if (sys.stdin.isatty() and not use_live) or not CONFIRMS.get(chosen) else "letitgo"
                perm_hook.policy = make_policy(eff)
                perm_hook.on_ask = _ask if CONFIRMS.get(eff) else None
                _stats["policy"] = policy_label(eff)
                save_prefs({"policy": eff})   # remember the choice for next launch
                _console.print(f"  → [bold]{policy_label(eff)}[/]"
                               + ("" if eff == chosen else "  (no interactive prompt here → running as let-it-go)"))
        elif cmd == "/model":
            if not arg and sys.stdin.isatty() and not use_live:   # open the two-tier model→reasoning menu
                from .tui import select_model_reasoning
                choice = select_model_reasoning(llm, cfg)
                if choice:
                    llm.switch(model=choice[0], reasoning=choice[1])
                    _stats["model"] = llm.model
                    save_prefs({"model": llm.model, "reasoning": llm.reasoning})
                    note = _reasoning_note(llm)
                    _console.print(f"  ✓ model → [bold]{llm.model}[/] · reasoning [bold]{llm.reasoning}[/] (saved)"
                                   + (f"\n  {note}" if note else ""))
            elif not arg:
                _console.print(f"  model: [bold]{llm.model}[/]  ·  reasoning: [bold]{llm.reasoning}[/]"
                               f"  ·  net: {getattr(llm, 'proxy_used', 'direct')}")
                known = ("gpt-5.5", "gpt-5", "gpt-5-mini", "o3", "deepseek-chat",
                         "kimi-k2-0905-preview", "claude-sonnet-4-6")
                _console.print("  switch:  /model <name> [fast|full|high|max]")
                _console.print("  known:   " + ", ".join(known))
                provs = cfg.providers()
                if provs:
                    _console.print("  providers (use `config --use <id>` to change endpoint): "
                                   + ", ".join(f"{k}={v.get('model', '?')}" for k, v in provs.items()))
            else:
                name, *rest = arg.split()
                eff = rest[0].lower() if rest else None
                if eff and eff not in ("fast", "full", "high", "max"):
                    _console.print("  effort must be one of: fast | full | high | max"); return True
                llm.switch(model=name, reasoning=eff)
                _stats["model"] = llm.model
                save_prefs({"model": llm.model, "reasoning": llm.reasoning})
                note = _reasoning_note(llm)
                _console.print(f"  ✓ model → [bold]{llm.model}[/]"
                               + (f" · reasoning [bold]{llm.reasoning}[/]" if eff else "")
                               + " (saved)" + (f"\n  {note}" if note else ""))
        elif cmd == "/reasoning":
            if arg.lower() not in ("fast", "full", "high", "max"):
                _console.print("  usage: /reasoning <fast|full|high|max>"
                               "   (full = provider default; high/max use /v1/responses for gpt-5)")
            else:
                llm.switch(reasoning=arg)
                save_prefs({"reasoning": llm.reasoning})
                note = _reasoning_note(llm)
                _console.print(f"  ✓ reasoning → [bold]{llm.reasoning}[/] (saved)" + (f"\n  {note}" if note else ""))
        else:
            _console.print(f"  unknown command {cmd} (/help)")
        return True

    _IMG_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")

    def _expand_mentions(text):
        """@path mentions (Aider/Claude-Code style): pin each EXISTING workspace file referenced as @path into
        the slice's OPEN FILES; an @image is ATTACHED as a vision content part for the next turn when the model
        supports vision (else skipped with a hint). Best-effort; leaves the text intact."""
        if "@" not in text or session.active_id is None:
            return
        import re as _re
        from .model_catalog import capability
        from .slice import touch_file
        vision = capability(llm.model, getattr(llm, "_base_url", "")).supports_vision
        pinned, images, skipped = [], [], []
        for m in _re.findall(r"@([\w./\-]+)", text):
            rel = m.lstrip("/")
            if ".." in rel:               # never let @../ reach outside the workspace (defense-in-depth)
                continue
            if not (rel and os.path.isfile(os.path.join(root, rel))):
                continue
            if rel.lower().endswith(_IMG_EXT):
                if vision:
                    res = base_tools.attach_image(rel)   # only claim "attached" if it actually worked
                    (skipped if isinstance(res, str) and res.startswith("Error") else images).append(rel)
                else:
                    skipped.append(rel)
            else:
                touch_file(session.active(), rel); pinned.append(rel)
        if _tui and _console is not None:
            if pinned:
                _console.print(f"  📎 pinned: {', '.join(pinned)}")
            if images:
                _console.print(f"  🖼  attached image: {', '.join(images)}")
            if skipped:
                _console.print(f"  🖼  skipped (needs a vision-capable AGENT_MODEL): {', '.join(skipped)}")

    # AGENT_AUTO_APPROVE: comma-separated fnmatch globs over the command, pre-approved so safe read-only
    # commands never prompt (e.g. AGENT_AUTO_APPROVE="git status*,git diff*,ls *,cat *").
    _auto = [r.strip() for r in (os.environ.get("AGENT_AUTO_APPROVE") or "").split(",") if r.strip()]
    # MAIN agent's policy: the canonical mode, but a confirm-mode needs a human — with no interactive prompt
    # (headless/piped, or the live composer that owns stdin) fall back to let-it-go (auto, still catastrophic-gated).
    _can_confirm = sys.stdin.isatty() and not use_live
    _eff_mode = canonical if (_can_confirm or not CONFIRMS.get(canonical)) else "letitgo"
    _stats["policy"] = policy_label(_eff_mode)
    perm_hook = PermissionHook(make_policy(_eff_mode),
                               on_ask=_ask if CONFIRMS.get(_eff_mode) else None, auto_approve=_auto)
    hook_list = [perm_hook]
    hook_list.append(GuardrailHook())  # cross-step loop guard (per-turn counters, reset each task)
    if cfg.verify_cmd:
        oracle = CommandOracle(cfg.verify_cmd)
        hook_list.append(OracleHook(oracle, lambda out: setattr(session.active(), "last_error", f"Verification failed:\n{out[:600]}")))
    if cfg.max_tokens:
        hook_list.append(BudgetHook(cfg.max_tokens))
    hook_list.extend(plugin_hooks)  # plugins compose into the same hook chain
    hooks = CompositeHooks(*hook_list)
    if app is not None:
        app._hooks = hooks
        base_tools.on_ask_user = app.ask_user

    info = (f"model={llm.model} · net={getattr(llm, 'proxy_used', 'direct')} · "
            f"policy={policy_mode} · sandbox={cfg.sandbox_backend} · "
            f"code={type(retriever).__name__} · memory={type(memory).__name__} · "
            f"episodic={'on' if episodic is not None else 'off'} · "
            f"mine={mine_mode} · subagents={'on' if sub_depth > 0 else 'off'} · "
            f"skills={len(skills.names())} · mcp_tools={mcp_tool_count} · plugin_tools={plugin_tool_count}")
    # ── choose UI: full-screen Textual (terminal-only) or the original rich+prompt_toolkit REPL ──
    _input = _tui.TuiInput(_stats, root=root) if _tui else None

    def _run_one_turn(text, sink, signal):
        """One turn for the LIVE composer: route (lexical) → build slice → run_turn with a per-turn dispatch
        that feeds the LiveSink. Runs in run_live's worker thread, so the pinned box stays responsive."""
        if session.active_id is None:
            session.new_topic(text)
        else:
            action, tid = route(llm, text, session)
            if action == "new":
                session.new_topic(text)
            elif action == "resume":
                session.switch_topic(tid); session.continue_topic(text, resume=True)
            else:
                session.continue_topic(text)
        _stats["topic"] = one_line(session.active().goal, 40) if session.active_id else ""
        record_user(session.active(), text)
        _expand_mentions(text)            # @path → pin the file into OPEN FILES
        build = make_build_slice(session, tools, retriever, memory, text, session.session_id)
        # live mode must wire the SAME host sinks as the REPL path (episodic cache, durable log, metrics) —
        # not just slice+renderer — else the cache→memory loop and /cost produce NOTHING in live mode.
        _live_sinks = [slice_sink(session)]
        if episodic is not None:
            _live_sinks.append(episodic)
        _live_sinks.append(log_sink())
        if metrics is not None:
            _live_sinks.append(metrics)
        if monitor_sink is not None:        # live mode must feed the web monitor too (was silently omitted)
            _live_sinks.append(monitor_sink)
        _live_sinks.append(sink)
        live_dispatch = make_dispatcher(*_live_sinks)
        llm.set_delta_sink(sink.on_delta)
        import time as _t
        _t0 = _t.monotonic()
        from . import recovery as _rec
        result = run_turn(build_slice=build, llm=llm, tools=tools, dispatch=live_dispatch,
                          hooks=hooks, signal=signal, max_steps=cfg.max_steps,
                          consolidate=lambda: consolidate_checkpoint(session.active(), compact=False),
                          checkpoint=lambda m, s, _g=text: _rec.record(root, goal=_g, messages=m, step=s))
        _rec.clear(root)                                   # clean/parked exit → drop the WAL
        _stats["last_turn_s"] = _t.monotonic() - _t0
        if getattr(memory, "is_durable", False):
            from .taskstate import slice_to_task_state
            memory.checkpoint_task(slice_to_task_state(
                session.active(), session.active_id, session_id=session.session_id,
                status="done" if result.stop_reason == "end_turn" else "parked"))
        if reviewer is not None:
            reviewer.review(session.session_id)

    def _try_live() -> bool:
        """Run the always-pinned live composer; return True if it ran (REPL below is then skipped), False to
        fall back to the REPL on any startup failure — input is never left broken."""
        try:
            _tui.run_live(console=_console, stats=_stats, banner_info=info, root=root,
                          run_one_turn=_run_one_turn, handle_slash=_handle_slash)
            return True
        except Exception as _e:  # noqa: BLE001
            print(f"\n  live UI failed ({type(_e).__name__}: {_e}); using the inline REPL instead.")
            return False

    # crash recovery: a leftover WAL means the last turn in this workspace never reached a clean/parked
    # exit (a hard crash). Surface what was in flight, then clear it (auto-loop-resume is a future step).
    from . import recovery
    _pend = recovery.pending(root)
    if _pend:
        _rg = one_line(_pend.get("goal", ""), 60)
        _rla = one_line(recovery.last_assistant(_pend), 160)
        _note = (f"  ⚠ recovered an interrupted turn (step {_pend.get('step', '?')}): {_rg}"
                 + (f"\n    last said: {_rla}" if _rla else "")
                 + "\n    its progress wasn't saved cleanly — re-send the request to continue.")
        (_console.print(f"[yellow]{_note}[/]") if _console is not None else print(_note))
        recovery.clear(root)

    if use_textual:
        # The Textual app was created earlier so its dialogs could be wired into hooks/tools.
        try:
            app.run()
        except Exception as _e:  # noqa: BLE001 — a Textual runtime failure must not dump a traceback on the
            # default UI; degrade with a clear, actionable message instead.
            print(f"\n  Textual UI failed to run ({type(_e).__name__}: {_e}).\n"
                  "  Rerun with AGENT_TUI=rich for the classic REPL, or AGENT_TUI=off for plain output.")
    elif use_live and _try_live():
        pass                              # the live composer ran the whole session (until ctrl-d/exit)
    else:
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
            if line == "/learn" or line.startswith("/learn "):  # transcript → reusable skill (runs as a turn)
                from .consolidate import build_learn_prompt
                line = build_learn_prompt(line[len("/learn"):].strip())
            elif _tui and line.startswith("/"):                # navigation palette (no turn)
                _handle_slash(line)
                continue
            # INVARIANT: echo the user's line BEFORE any blocking work (esp. route_topic's LLM round-trip),
            # so the message paints the instant Enter is pressed — not ~0.5-2s later. The Textual path
            # enforces the same ordering via its worker thread (tui_app.py: _append_user before _run_user_turn).
            if _tui:                              # anchor the user turn with spacing (fixes cramped layout)
                _tui.user_echo(_console, line)
            if session.active_id is None:                      # first message bootstraps the first topic
                session.new_topic(line)
            else:                                              # route: continue / new / resume (no junk topic)
                # route() is lexical by default (instant, zero round-trips); AGENT_ROUTER=llm restores the
                # classifier (a provider round-trip). Cover it with a 'routing…' spinner so the llm mode has
                # no silent freeze before run_turn's own 'thinking…' spinner (which only starts at SliceBuilt).
                if _tui:
                    with _console.status("[grey50]routing…[/]", spinner="dots"):
                        action, tid = route(llm, line, session)
                else:
                    action, tid = route(llm, line, session)
                if action == "new":
                    session.new_topic(line)
                elif action == "resume":
                    session.switch_topic(tid)
                    session.continue_topic(line, resume=True)
                else:
                    session.continue_topic(line)
                if not _tui:                                   # TUI shows the topic in the status bar, not as noise
                    print(f"  · topic: {action}{(' ' + tid) if tid else ''}")
            _stats["topic"] = one_line(session.active().goal, 40) if session.active_id else ""
            record_user(session.active(), line)  # short-range continuity: the RECENT CONVERSATION tier
            _expand_mentions(line)               # @path → pin the file into OPEN FILES
            build = make_build_slice(session, tools, retriever, memory, line, session.session_id)
            # ctrl-c during the turn (incl. while the LLM is thinking) raises KeyboardInterrupt, which
            # run_turn catches → aborts the turn cleanly and returns here to the prompt (then ctrl-d quits).
            import time as _time
            _t0 = _time.monotonic()
            result = run_turn(build_slice=build, llm=llm, tools=tools, dispatch=dispatch, hooks=hooks,
                              max_steps=cfg.max_steps,
                              consolidate=lambda: consolidate_checkpoint(session.active(), compact=False),
                              checkpoint=lambda m, s, _g=line: recovery.record(root, goal=_g, messages=m, step=s))
            recovery.clear(root)                               # clean/parked exit → drop the WAL
            _stats["last_turn_s"] = _time.monotonic() - _t0   # shown as ⏲ in the status bar
            if getattr(memory, "is_durable", False):           # durable checkpoint (no-op under NullMemory)
                from .taskstate import slice_to_task_state
                memory.checkpoint_task(slice_to_task_state(
                    session.active(), session.active_id, session_id=session.session_id,
                    status="done" if result.stop_reason == "end_turn" else "parked"))
            if reviewer is not None:                            # OPT-IN: critique the turn off-thread
                reviewer.review(session.session_id)

    # session end: tear down background procs / PTY sessions, MCP servers, and consolidate memory. Each
    # step is GUARDED (a failure warns, never crashes the exit) and the MCP shutdown is BOUNDED by a
    # timeout, so a stuck server or index write can never freeze the process on the way out.
    def _safe(label, fn):
        try:
            return fn()
        except Exception as _e:  # noqa: BLE001
            print(f"  · warning: {label} failed ({type(_e).__name__}: {_e})")
            return None

    def _bounded(label, fn, secs=8.0):
        import threading as _th
        t = _th.Thread(target=lambda: _safe(label, fn), daemon=True)
        t.start(); t.join(secs)
        if t.is_alive():
            print(f"  · warning: {label} timed out after {secs:.0f}s — exiting anyway")

    _safe("tool cleanup", base_tools.cleanup)
    if reviewer is not None:           # let an in-flight background review finish (bounded) before consolidating
        _safe("bg-review join", lambda: reviewer.join(timeout=10))
    if mcp_runtime is not None:        # #61/#62: a stuck MCP server must not freeze exit → bounded shutdown
        _bounded("MCP shutdown", mcp_runtime.shutdown)
    # consolidate the episodic cache into long-term memory (the cache→memory loop). mine_mode gates it:
    # off → skip; deterministic → recorded skills; llm → render_skill_llm generalizes (scan-first).
    if getattr(memory, "is_durable", False) and mine_mode not in ("0", "off", "none"):
        st = _safe("memory consolidation",
                   lambda: memory.consolidate(session.session_id, llm=llm, mode=mine_mode)) or {}
        if st.get("lessons") or st.get("skills"):        # report the TRUTH, not a blind 'success'
            print(f"  · consolidated: {st.get('lessons', 0)} lesson(s), {st.get('skills', 0)} skill(s)"
                  + (f", {st['skills_rejected']} rejected" if st.get("skills_rejected") else "")
                  + (f", {st['errors']} error(s)" if st.get("errors") else ""))
        elif st.get("skills_rejected") or st.get("errors"):
            print(f"  · consolidation: {st.get('skills_rejected', 0)} rejected, {st.get('errors', 0)} error(s)")
    _safe("memory close", getattr(memory, "close", lambda: None))   # #33: close the FTS5 index (WAL checkpoint)
    if metrics is not None:                                 # the moat number: per-turn fresh-input curve
        s = metrics.summary()
        print(f"  · metrics: per_turn_fresh={s['per_turn_fresh']} avg={s['avg_turn_fresh']} "
              f"cache_hit={s['cache_hit_rate']} tools={s['tool_calls']}({s['tool_failures']} fail) "
              f"retries={s['retries']} overflows={s['overflows']} errors={s['errors']}")


if __name__ == "__main__":
    main()
