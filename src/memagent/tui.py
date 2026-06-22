"""Optional rich terminal UI (the `tui` extra: rich + prompt_toolkit).

Borrowed periphery — NOT the moat. The loop already decouples rendering via the event
dispatcher; this is just (a) a rich rendering SINK over those events and (b) a prompt_toolkit
input layer. loop.py / slice.py are never touched. The whole module is import-guarded behind the
`tui` extra: core/headless/eval never import rich or prompt_toolkit.

Design (borrowed from Hermes' rich+prompt_toolkit stack and Kimi's TUI UX):
  - SCROLLBACK model: Rich prints finalized output to history; prompt_toolkit owns the input line.
    They are TEMPORALLY separate (output during the synchronous run_turn, input between turns), so
    there is no patch_stdout/threading minefield.
  - tool-call CARDS (spinner -> ✓/✗, primary-arg header, inline diff for edits),
  - a two-line STATUS footer (model · policy · topic · tokens),
  - a SLASH-command palette wired to existing session ops (/new /switch /resume /threads /help /exit),
  - graceful ctrl-c: a SIGINT handler sets run_turn's existing `signal=` Event (loop.py:139) so the
    turn aborts at the next step boundary — no background thread.
"""
from __future__ import annotations

import os

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory

from .events import (AssistantText, ApiRetry, Event, LessonSaved, SliceBuilt, StepEnd,
                     ToolResult, ToolStarted, TurnEnd, TurnInterrupted)

# ── theme (semantic tokens; one place to retheme) ───────────────────────────────────────────
TH = {
    "accent": "bright_cyan", "ok": "green", "fail": "red", "warn": "yellow",
    "dim": "grey50", "tool": "magenta", "add": "green", "del": "red", "user": "bright_cyan",
}

# per-tool emoji + a short verb + which arg is the "primary" one to show in the card header
_TOOL = {
    "read_file":      ("📖", "read",   "path"),
    "edit_file":      ("✏️ ", "write",  "path"),
    "append_to_file": ("➕", "append", "path"),
    "str_replace":    ("✏️ ", "edit",   "path"),
    "list_files":     ("📂", "list",   "path"),
    "run_command":    ("⚡", "run",    "command"),
    "execute_code":   ("🐍", "exec",   "code"),
    "grep":           ("🔍", "grep",   "pattern"),
    "glob":           ("🔍", "glob",   "pattern"),
    "skill":          ("📚", "skill",  "name"),
    "recall_history": ("🕮 ", "recall", "index"),
    "new_topic":      ("🟢", "topic",  "goal"),
    "switch_topic":   ("🔀", "switch", "task_id"),
    "spawn_subagent": ("🤖", "agent",  "task"),
    "spawn_explore":  ("🔭", "explore", "task"),
}


def _shorten(s: str, n: int = 64) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _primary(name: str, args: dict) -> str:
    key = _TOOL.get(name, (None, None, None))[2]
    val = args.get(key) if (key and isinstance(args, dict)) else None
    if val is None and isinstance(args, dict):  # fallback: first non-note string arg
        val = next((v for k, v in args.items() if k != "note" and isinstance(v, str)), "")
    return _shorten(str(val or ""))


def _tool_header(name: str, args: dict) -> str:
    emoji, verb, _ = _TOOL.get(name, ("•", name, None))
    p = _primary(name, args)
    return f"{emoji} {verb} {p}".rstrip()


def _diff(name: str, args: dict):
    """A compact inline diff for a str_replace (old → new). Returns a Rich renderable or None."""
    if name != "str_replace" or not isinstance(args, dict):
        return None
    old, new = args.get("old_string", ""), args.get("new_string", "")
    if not old and not new:
        return None
    lines = []
    for ln in old.splitlines()[:12]:
        lines.append(Text(f"- {ln}", style=TH["del"]))
    for ln in new.splitlines()[:12]:
        lines.append(Text(f"+ {ln}", style=TH["add"]))
    extra = max(0, len(old.splitlines()) - 12) + max(0, len(new.splitlines()) - 12)
    if extra:
        lines.append(Text(f"… {extra} more diff lines", style=TH["dim"]))
    return Group(*lines) if lines else None


# ── the rendering sink (consumes the loop's events) ──────────────────────────────────────────
class RichSink:
    """An event sink that renders the live turn with Rich. Drop-in for cli_sink."""

    def __init__(self, console: Console, stats: dict):
        self.c = console
        self.stats = stats
        self._status = None
        self._stream = ""        # live-streamed assistant text for the CURRENT step (transient; tail shown in spinner)

    def _stop(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _spin(self, label: str) -> None:
        self._stop()
        self._stream = ""        # new step → reset the streamed tail
        self._status = self.c.status(Text(label, style=TH["dim"]), spinner="dots")
        self._status.start()

    def on_delta(self, kind: str, text: str) -> None:
        """Live token sink wired to OpenAILLM.set_delta_sink — turns the static 'thinking…' spinner into a
        LIVE writing indicator (the streamed tail), so a slow turn shows progress instead of freezing. The
        streamed tail is TRANSIENT (it lives only in the spinner label); the final AssistantText renders the
        canonical Markdown once, so there is no double-print. No-op when no spinner is active (e.g. routing)."""
        if self._status is None or kind != "content" or not text:
            return
        self._stream += text
        tail = " ".join(self._stream.split())[-100:]   # last ~100 chars, single line
        self._status.update(Text(f"writing… {tail}", style=TH["dim"]))

    def __call__(self, e: Event) -> None:
        if isinstance(e, SliceBuilt):
            self._spin("thinking…")
        elif isinstance(e, ToolStarted):
            self._spin(f"{_tool_header(e.name, e.args)} …")
        elif isinstance(e, ToolResult):
            self._stop()
            mark = Text("✓", style=TH["ok"]) if not e.failing else Text("✗", style=TH["fail"])
            head = Text.assemble(mark, " ", Text(_tool_header(e.name, e.args), style=TH["tool"]))
            body = [head]
            d = _diff(e.name, e.args)
            if d is not None:
                body.append(d)
            out = _shorten(e.output, 200)
            if out:
                body.append(Text(f"   {out}", style=TH["fail"] if e.failing else TH["dim"]))
            self.c.print(Group(*body))
        elif isinstance(e, AssistantText):
            self._stop()
            if (e.content or "").strip():
                self.c.print(Markdown(e.content))
        elif isinstance(e, ApiRetry):
            self._stop()
            self.c.print(Text(f"  …retry #{e.attempt} ({_shorten(e.error, 60)})", style=TH["warn"]))
        elif isinstance(e, StepEnd):
            u = e.usage or {}
            self.stats["tokens"] = self.stats.get("tokens", 0) + u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
        elif isinstance(e, LessonSaved):
            self.c.print(Text(f"  💡 learned: {_shorten(e.title, 70)}", style=TH["dim"]))
        elif isinstance(e, TurnInterrupted):
            self._stop()
            self.c.print(Text(f"  ⚠ interrupted: {e.message or e.reason}", style=TH["warn"]))
        elif isinstance(e, TurnEnd):
            self._stop()
            tok = (e.usage or {}).get("prompt_tokens", 0) + (e.usage or {}).get("completion_tokens", 0)
            self.c.print(Text(f"  ✓ done · {e.steps} steps · {tok} tokens", style=TH["dim"]))


def make_rich_sink(console: Console, stats: dict) -> RichSink:
    return RichSink(console, stats)


# ── input layer (prompt_toolkit) ─────────────────────────────────────────────────────────────
_SLASH = {
    "/switch":  "switch to a parked topic by id (/switch <id>)",
    "/resume":  "resume a parked topic by id (/resume <id>)",
    "/threads": "list open/parked topics",
    "/help":    "show commands",
    "/exit":    "quit",
}


class _SlashCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for cmd, desc in _SLASH.items():
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text), display_meta=desc)


def _toolbar(stats: dict):
    def render():
        topic = _shorten(stats.get("topic") or "—", 40)
        return HTML(
            f" <b>memagent</b>  model <b>{stats.get('model','?')}</b>"
            f"  policy <b>{stats.get('policy','?')}</b>"
            f"  topic <b>{topic}</b>"
            f"  tokens <b>{stats.get('tokens',0)}</b>"
            f"   <i>/help · ctrl-c abort · ctrl-d quit</i> "
        )
    return render


class TuiInput:
    """prompt_toolkit input with history, slash completion, and the status toolbar."""

    def __init__(self, stats: dict):
        hist_dir = os.path.expanduser("~/.memagent")
        os.makedirs(hist_dir, exist_ok=True)
        self.session = PromptSession(
            history=FileHistory(os.path.join(hist_dir, "history")),
            completer=_SlashCompleter(),
            complete_while_typing=True,
            bottom_toolbar=_toolbar(stats),
        )

    def prompt(self) -> str | None:
        """Return the next line, or None to QUIT. Both ctrl-d (EOF) and ctrl-c (SIGINT) exit cleanly —
        matching the plain-input path (cli.py) so the program is never un-quittable. (Ctrl-c DURING a
        turn is caught earlier by run_turn, which aborts just the turn and returns here; this handles
        ctrl-c at the idle prompt, where the only sensible action is to leave.)"""
        try:
            return self.session.prompt(HTML("<ansicyan><b>You ▸ </b></ansicyan>"))
        except (EOFError, KeyboardInterrupt):   # ctrl-d OR ctrl-c at the prompt → quit
            return None


def confirm(console: Console, name: str, detail: str, reason: str) -> str:
    """Approval prompt used by the permission hook when the TUI is active. Synchronous (no pt app
    is live mid-run), returns 'yes' | 'no' | 'always'."""
    console.print(Text.assemble(
        Text("  ⚠ allow ", style=TH["warn"]), Text(name, style=TH["tool"]),
        Text(f" {_shorten(detail, 60)!r}? ", style=TH["dim"]), Text(f"({reason})", style=TH["dim"])))
    try:
        ans = console.input("    [y]es / [n]o / [a]lways ▸ ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "no"
    return {"y": "yes", "yes": "yes", "a": "always", "always": "always"}.get(ans, "no")


def ask_user(console: Console, question: str, options=None) -> str:
    """The ask_user prompt (the 'come back and ask a follow-up' capability). Synchronous — no pt app
    is live mid-run — so a Rich prompt is safe. Returns the user's answer (a chosen option or free text)."""
    console.print(Text.assemble(Text("  ❓ ", style=TH["accent"]), Text(question, style="bold")))
    if options:
        for i, o in enumerate(options, 1):
            console.print(Text(f"     {i}. {o}", style=TH["dim"]))
        console.print(Text("     (type a number, or your own answer)", style=TH["dim"]))
    try:
        ans = console.input("  your answer ▸ ").strip()
    except (EOFError, KeyboardInterrupt):
        return "(no answer)"
    if options and ans.isdigit() and 1 <= int(ans) <= len(options):
        return options[int(ans) - 1]
    return ans or "(no answer)"


def banner(console: Console, info: str) -> None:
    console.print(Panel(Text(info, style=TH["dim"]),
                        title=Text("memagent · slice core", style=TH["accent"]),
                        border_style=TH["accent"], expand=False))


def tui_enabled() -> bool:
    """On at a TTY unless AGENT_TUI is explicitly off; never on when piped (eval/headless)."""
    flag = os.environ.get("AGENT_TUI", "").strip().lower()
    if flag in ("0", "off", "false", "no"):
        return False
    if flag in ("1", "on", "true", "yes"):
        return True
    try:
        import sys
        return sys.stdout.isatty() and sys.stdin.isatty()
    except Exception:
        return False
