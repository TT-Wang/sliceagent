"""Full-screen Textual TUI for memagent (terminal-only, no web).

This is an opt-in, drop-in sink over the same event-dispatch seam used by cli.py and
monitor.py. The loop core is untouched. If Textual is not installed or the terminal is
piped, memagent falls back to the original rich+prompt_toolkit TUI in tui.py.

Layout:
  ┌─ Header (model · policy · topic · tokens) ──────────────────────────┐
  ├─ Sidebar ─┬─ Plan/Mission panel ─────────────────────────────────────┤
  │ working   │                                                        │
  │ set       ├─ Conversation / tool stream (RichLog) ──────────────────┤
  │           │                                                        │
  │           │                                                        │
  ├───────────┴─ Multi-line input (TextArea) + send hint ──────────────┤
  └─ Footer (shortcuts · status) ──────────────────────────────────────┘

Key bindings:
  enter             send message (ctrl+j inserts a newline; ctrl+enter also sends where the terminal
                    can deliver it)
  ctrl+c            abort current turn (or quit if idle)
  ctrl+d            quit
  ctrl+p            slash-command palette
  ctrl+n            new topic
"""
from __future__ import annotations

import difflib
import os
import subprocess
import threading
from typing import Callable

from textual import events as textual_events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Grid
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
    TextArea,
    Tree,
)
from textual.message import Message
from textual.worker import Worker, WorkerState
from rich import box as _rbox
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from . import events as memagent_events


def _shorten(s: str, n: int = 64) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


# per-tool emoji + verb + primary-arg key
_TOOL_META = {
    "read_file":      ("📖", "read",   "path"),
    "edit_file":      ("✏️", "write",  "path"),
    "append_to_file": ("➕", "append", "path"),
    "str_replace":    ("✏️", "edit",   "path"),
    "list_files":     ("📂", "list",   "path"),
    "run_command":    ("⚡", "run",    "command"),
    "execute_code":   ("🐍", "exec",   "code"),
    "grep":           ("🔍", "grep",   "pattern"),
    "glob":           ("🔍", "glob",   "pattern"),
    "skill":          ("📚", "skill",  "name"),
    "recall_history": ("🕮", "recall", "index"),
    "new_topic":      ("🟢", "topic",  "goal"),
    "switch_topic":   ("🔀", "switch", "task_id"),
    "spawn_subagent": ("🤖", "agent",  "task"),
    "spawn_explore":  ("🔭", "explore", "task"),
    "update_plan":    ("📋", "plan",   "steps"),
    "set_mission":    ("🎯", "mission", "text"),
    "proc_start":     ("▶️", "proc",   "command"),
    "terminal_open":  ("🖥️", "term",   "command"),
}


_PLAN_GLYPH = {"done": "✓", "in_progress": "▶", "pending": "○"}

_TOOL_OUTPUT_FOLD_LINES = 8  # lines before folding long tool output


# ── internal UI messages ─────────────────────────────────────────────────────
class UserSubmit(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class AgentEvent(Message):
    def __init__(self, event: memagent_events.Event) -> None:
        self.event = event
        super().__init__()


class TurnDone(Message):
    def __init__(self) -> None:
        super().__init__()


class PaletteResult(Message):
    def __init__(self, command: str, argument: str) -> None:
        self.command = command
        self.argument = argument
        super().__init__()


class DialogResult(Message):
    def __init__(self, dialog_id: int, value: str) -> None:
        self.dialog_id = dialog_id   # routes the answer back to the exact waiting worker (tools run parallel)
        self.value = value
        super().__init__()


# ── modal screens ────────────────────────────────────────────────────────────
class PaletteScreen(Screen):
    """Fuzzy slash-command palette."""

    CSS = """
    PaletteScreen { align: center middle; }
    #palette { width: 60; height: auto; max-height: 24; border: solid $primary; background: $surface; }
    #pinput { margin: 1 1 0 1; }
    #plist { height: auto; max-height: 16; border: none; }
    """

    COMMANDS = [
        ("/plan", "show current plan + mission"),
        ("/cost", "show per-turn cost metrics"),
        ("/threads", "list open/parked topics"),
        ("/switch", "switch to parked topic (/switch <id>)"),
        ("/resume", "resume parked topic (/resume <id>)"),
        ("/add", "add file to working set (/add <path>)"),
        ("/drop", "remove file from working set (/drop <path>)"),
        ("/diff", "show diff of edited files (/diff [path])"),
        ("/undo", "revert last edit"),
        ("/tokens", "show token usage"),
        ("/context", "show current slice regions"),
        ("/clear", "clear conversation stream"),
        ("/help", "show commands"),
        ("/exit", "quit"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="palette"):
            yield Input(placeholder="type /command or filter…", id="pinput")
            yield ListView(id="plist")

    def on_mount(self) -> None:
        self._populate("")
        self.query_one("#pinput", Input).focus()

    def _populate(self, filter_text: str) -> None:
        lst = self.query_one("#plist", ListView)
        lst.clear()
        filter_text = filter_text.lstrip("/").lower()
        for cmd, desc in self.COMMANDS:
            if not filter_text or filter_text in cmd.lower() or filter_text in desc.lower():
                lst.append(ListItem(Label(f"{cmd} — {desc}"), name=cmd))
        # append() mounts asynchronously — highlight the first item AFTER the refresh, else index=0
        # lands on an empty list and nothing is highlighted.
        self.call_after_refresh(self._highlight_first)

    def _highlight_first(self) -> None:
        lst = self.query_one("#plist", ListView)
        if lst.children:
            lst.index = 0

    def on_input_changed(self, event: Input.Changed) -> None:
        self._populate(event.value)

    def _submit(self) -> None:
        lst = self.query_one("#plist", ListView)
        if lst.highlighted_child is None:
            return
        cmd = lst.highlighted_child.name or ""
        arg = self.query_one("#pinput", Input).value.strip()
        # If user typed "/cmd something", extract argument; otherwise command is bare.
        if arg.startswith(cmd):
            arg = arg[len(cmd):].strip()
        else:
            arg = ""
        self.app.post_message(PaletteResult(cmd, arg))
        self.dismiss()

    def on_list_view_selected(self, _event: ListView.Selected) -> None:
        self._submit()

    def on_key(self, event: textual_events.Key) -> None:
        if event.key == "enter":
            event.stop()
            self._submit()
        elif event.key == "escape":
            event.stop()
            self.dismiss()


class ConfirmScreen(Screen):
    """Modal approval prompt for dangerous tools (AGENT_POLICY=ask)."""

    CSS = """
    ConfirmScreen { align: center middle; }
    #dialog { width: 80; height: auto; max-height: 30; border: solid $warning; background: $surface; padding: 1 2; }
    #reason { color: $warning; }
    #detail { margin: 1 0; max-height: 12; }
    #buttons { height: auto; align: center middle; }
    """

    def __init__(self, name: str, detail: str, reason: str, dialog_id: int) -> None:
        super().__init__()
        self._name = name
        self._detail = detail
        self._reason = reason
        self._dialog_id = dialog_id

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(f"Allow {self._name}?", id="title")
            yield Label(f"Reason: {self._reason}", id="reason")
            yield Static(self._detail, id="detail")
            with Horizontal(id="buttons"):
                yield Button("yes", id="yes", variant="success")
                yield Button("no", id="no", variant="error")
                yield Button("always", id="always")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.app.post_message(DialogResult(self._dialog_id, event.button.id or "no"))
        self.dismiss()

    def on_key(self, event: textual_events.Key) -> None:
        if event.key == "escape":
            event.stop()
            self.app.post_message(DialogResult(self._dialog_id, "no"))
            self.dismiss()


class AskUserScreen(Screen):
    """Modal ask_user prompt with options or free-text input."""

    CSS = """
    AskUserScreen { align: center middle; }
    #dialog { width: 80; height: auto; max-height: 30; border: solid $primary; background: $surface; padding: 1 2; }
    #question { margin-bottom: 1; }
    #options { height: auto; }
    #answer { margin-top: 1; }
    """

    def __init__(self, question: str, options: list[str] | None, dialog_id: int) -> None:
        super().__init__()
        self._question = question
        self._options = options or []
        self._dialog_id = dialog_id

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self._question, id="question")
            if self._options:
                for i, opt in enumerate(self._options, 1):
                    # id is INDEX-based: a Textual id must be [A-Za-z0-9_-] only, but option text can
                    # contain spaces/punctuation ("Explain it") — the label carries the text, the id maps
                    # back to it by position in on_button_pressed.
                    yield Button(f"{i}. {opt}", id=f"opt-{i}")
                yield Input(placeholder="or type your own answer…", id="answer")
            else:
                yield Input(placeholder="your answer…", id="answer")

    def _return(self, value: str) -> None:
        self.app.post_message(DialogResult(self._dialog_id, value))
        self.dismiss()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid.startswith("opt-"):
            try:
                idx = int(bid[4:]) - 1
            except ValueError:
                return
            if 0 <= idx < len(self._options):
                self._return(self._options[idx])

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if self._options and value.isdigit() and 1 <= int(value) <= len(self._options):
            self._return(self._options[int(value) - 1])
        elif value:
            self._return(value)

    def on_key(self, event: textual_events.Key) -> None:
        # escape MUST resolve the dialog — without this the waiting worker thread hangs forever.
        if event.key == "escape":
            event.stop()
            self._return("(no answer)")


class DiffScreen(Screen):
    """Side-by-side / unified diff view for edited files."""

    CSS = """
    DiffScreen { align: center middle; }
    #dialog { width: 100%; height: 100%; border: solid $primary; background: $surface; }
    #title { height: 1; content-align: center middle; }
    #diff { height: 1fr; }
    """

    def __init__(self, path: str, old_text: str, new_text: str) -> None:
        super().__init__()
        self._path = path
        self._old_text = old_text
        self._new_text = new_text

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(f"diff: {self._path}", id="title")
            diff_text = self._unified_diff()
            yield TextArea(diff_text, id="diff", show_line_numbers=False, read_only=True)

    def _unified_diff(self) -> str:
        old_lines = self._old_text.splitlines(keepends=True)
        new_lines = self._new_text.splitlines(keepends=True)
        # Ensure lines end with newline for difflib
        if old_lines and not old_lines[-1].endswith("\n"):
            old_lines[-1] += "\n"
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        return "".join(difflib.unified_diff(
            old_lines, new_lines, fromfile=f"a/{self._path}", tofile=f"b/{self._path}", lineterm="\n"
        ))   # content lines keep their '\n' → control lines (---/+++/@@) must too, else the diff is garbled

    def on_key(self, event: textual_events.Key) -> None:
        if event.key == "escape":
            event.stop()
            self.dismiss()


class InputArea(TextArea):
    """The prompt box. ENTER sends — most terminals cannot deliver Ctrl+Enter as a distinct key, so a
    ctrl+enter binding silently never fires; Enter is the reliable send key. Pasted text keeps its
    newlines, and Ctrl+J inserts a literal newline for the occasional multi-line prompt. Intercepted in
    _on_key (where TextArea would otherwise insert the newline) so it is scoped to THIS widget and never
    leaks into the modal screens. All other editing keys behave normally."""

    async def _on_key(self, event: textual_events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(UserSubmit(self.text))
            return
        if event.key == "ctrl+j":          # explicit newline (Enter is taken by send)
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)


# ── main app ─────────────────────────────────────────────────────────────────
class MemagentTui(App):
    """Terminal user interface for memagent."""

    CSS = """
    Screen { align: center middle; }
    #main { width: 100%; height: 1fr; }
    #sidebar { width: 26%; border: solid $primary; }
    #content { width: 74%; layout: vertical; }
    #plan { height: auto; max-height: 25%; border-bottom: solid $primary; }
    #conversation { height: 1fr; border-bottom: solid $primary; }
    #input-area { height: auto; max-height: 30%; }
    #input { height: 4; }
    #hint { height: 1; content-align: right middle; color: $text-muted; }
    """

    BINDINGS = [
        ("ctrl+d", "quit", "quit"),
        ("ctrl+p", "command_palette", "palette"),
        ("ctrl+n", "new_topic", "new topic"),
        ("ctrl+c", "abort_or_quit", "abort"),
        ("ctrl+enter", "submit_input", "send"),
    ]

    def __init__(
        self,
        *,
        session,
        tools,
        retriever,
        memory,
        llm,
        hooks,
        dispatch: Callable[[memagent_events.Event], None] | None = None,
        run_turn: Callable | None = None,
        make_build_slice: Callable | None = None,
        record_user: Callable | None = None,
        route_topic: Callable | None = None,
        stats: dict | None = None,
        max_steps: int = 60,
        checkpoint: Callable | None = None,
        consolidate: Callable | None = None,
        clear_recovery: Callable | None = None,
    ):
        super().__init__()
        self._session = session
        self._tools = tools
        self._retriever = retriever
        self._memory = memory
        self._llm = llm
        self._hooks = hooks
        self._dispatch = dispatch
        self._run_turn = run_turn
        self._make_build_slice = make_build_slice
        self._record_user = record_user
        self._route_topic = route_topic
        self._max_steps = max_steps          # else run_turn defaults to 40, ignoring cfg.max_steps (60)
        self._checkpoint = checkpoint        # crash-recovery WAL writer (None → no recovery in this UI)
        self._consolidate = consolidate      # overflow-breadcrumb distiller
        self._clear_recovery = clear_recovery
        self._stats = stats or {}
        self._abort_event = threading.Event()
        self._current_worker: Worker | None = None
        # per-dialog routing: tools run on parallel worker threads, so two confirm/ask_user dialogs can be
        # in flight at once. Each gets a unique id → its answer wakes the RIGHT waiting thread (no clobber).
        self._dialogs: dict[int, dict] = {}     # id -> {"event": Event, "result": str}
        self._dialog_seq = 0
        self._dialog_lock = threading.Lock()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield Tree("Working set", id="tree")
            with Vertical(id="content"):
                yield Static(id="plan")
                yield RichLog(id="conversation", highlight=True, wrap=True)
                with Vertical(id="input-area"):
                    yield InputArea(id="input", show_line_numbers=False)
                    yield Static("enter send · ctrl+j newline · ctrl+c abort · ctrl+d quit · ctrl+p palette",
                                 id="hint")
        yield Footer()

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self.title = "memagent"
        self.sub_title = self._stats.get("model", "?")
        self._thread_id = threading.current_thread().ident
        try:   # the shared logo banner (same one the rich CLI shows); skip if unavailable
            from .tui import banner_panel
            info = f"model={self._stats.get('model','?')} · policy={self._stats.get('policy','?')}"
            self._conversation().write(banner_panel(info))
        except Exception:
            pass
        self._update_header()
        self._update_sidebar()
        self._update_plan()
        self.query_one("#input", TextArea).focus()

    # ── event handlers (UI thread) ─────────────────────────────────────────────

    def on_user_submit(self, msg: UserSubmit) -> None:
        text = msg.text.strip()
        if not text:
            return
        if text.startswith("/"):
            self._handle_slash(text)
            return
        self._append_user(text)
        self._run_user_turn(text)

    def on_agent_event(self, msg: AgentEvent) -> None:
        self._render_event(msg.event)

    def on_turn_done(self, _msg: TurnDone) -> None:
        self._current_worker = None
        self._set_input_enabled(True)
        self._update_header()
        self._update_sidebar()
        self._update_plan()

    def on_palette_result(self, msg: PaletteResult) -> None:
        self._handle_slash(f"{msg.command} {msg.argument}".strip())

    def on_dialog_result(self, msg: DialogResult) -> None:
        d = self._dialogs.get(msg.dialog_id)   # route by id → the exact waiting worker
        if d is not None:
            d["result"] = msg.value
            d["event"].set()

    def action_submit_input(self) -> None:
        ta = self.query_one("#input", TextArea)
        self.post_message(UserSubmit(ta.text))

    def action_command_palette(self) -> None:
        self.push_screen(PaletteScreen())

    def action_new_topic(self) -> None:
        w = getattr(self, "_current_worker", None)
        if w is not None and not getattr(w, "is_finished", True):
            self.notify("A turn is running — finish it before starting a new topic.", severity="warning")
            return   # mutating the Session mid-turn races the worker thread
        self._session.new_topic("")
        self._record_user and self._record_user(self._session.active(), "new topic")
        self._refresh_all()
        self._conversation().write("[new topic started]")

    def action_abort_or_quit(self) -> None:
        if self._current_worker is not None and self._current_worker.state in (
            WorkerState.PENDING,
            WorkerState.RUNNING,
        ):
            self._abort_event.set()
            self._conversation().write(Text("⚠ abort requested…", style="yellow"))
        else:
            self.exit()

    # ── slash commands ─────────────────────────────────────────────────────────

    def _handle_slash(self, line: str) -> None:
        parts = line.split(maxsplit=1)
        cmd = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""
        log = self._conversation()
        s = self._session.active() if self._session.active_id else None

        # Commands that mutate the Session/Slice (switch/resume reassign active_id; add/drop edit
        # active_files; diff iterates edited_files) must NOT run mid-turn — they'd race the worker thread
        # (fold events into the wrong slice, or "set changed size during iteration"). Same guard as new_topic.
        # /context and /threads READ in-place-mutated slice/session collections (edited_files set,
        # active_files, tasks) that the worker thread is concurrently changing → "set changed size during
        # iteration"; they join the mutating commands in the mid-turn guard (no lock around the slice).
        w = getattr(self, "_current_worker", None)
        if cmd in ("/switch", "/resume", "/add", "/drop", "/diff", "/context", "/threads") and w is not None and not getattr(w, "is_finished", True):
            log.write(f"a turn is running — wait for it to finish before {cmd}")
            self.query_one("#input", TextArea).text = ""
            return

        if cmd == "/help":
            log.write("commands: /plan /cost /threads /switch /resume /add /drop /diff /undo /tokens /context /clear /exit")
        elif cmd == "/plan":
            self._show_plan_in_stream()
        elif cmd == "/cost":
            log.write("(cost metrics require AGENT_METRICS=1 in the rich TUI; Textual TUI shows tokens in the header)")
        elif cmd == "/threads":
            ts = self._session.open_threads(include_active=True)
            log.write("(no topics yet)" if not ts else
                      "\n".join(f"  [{t.task_id}] {t.title} ({t.status})" for t in ts))
        elif cmd in ("/switch", "/resume"):
            if not arg:
                log.write(f"usage: {cmd} <task_id>")
            else:
                try:
                    self._session.switch_topic(arg)
                    self._refresh_all()
                    log.write(f"switched to {arg}")
                except Exception:
                    log.write(f"no such topic: {arg}")
        elif cmd == "/add":
            if s and arg:
                s.active_files.append(arg)
                self._update_sidebar()
                log.write(f"added {arg} to working set")
        elif cmd == "/drop":
            if s and arg:
                if arg in s.active_files:
                    s.active_files.remove(arg)
                self._update_sidebar()
                log.write(f"dropped {arg} from working set")
        elif cmd == "/diff":
            self._show_diff(arg)
        elif cmd == "/undo":
            log.write("(undo not yet implemented in Textual TUI)")
        elif cmd == "/tokens":
            log.write(f"tokens: {self._stats.get('tokens', 0)} total, fresh: {self._stats.get('fresh', 0)}")
        elif cmd == "/context":
            self._show_context()
        elif cmd == "/clear":
            log.clear()
        elif cmd == "/exit":
            self.exit()
        else:
            log.write(f"unknown command {cmd} (/help)")
        self.query_one("#input", TextArea).text = ""

    def _show_plan_in_stream(self) -> None:
        s = self._session.active() if self._session.active_id else None
        log = self._conversation()
        if s and s.mission:
            log.write(Text.assemble(Text("🎯 ", style="bright_cyan"), Text(s.mission, style="bold")))
        if s and s.plan:
            for p in s.plan:
                g = _PLAN_GLYPH.get(p.get("status"), "○")
                log.write(Text(f"  {g} {p.get('step', '')}"))

    def _show_context(self) -> None:
        s = self._session.active() if self._session.active_id else None
        log = self._conversation()
        if s is None:
            log.write("no active topic")
            return
        log.write(Panel(
            "\n".join([
                f"goal: {s.goal}",
                f"mission: {s.mission}",
                f"active files: {', '.join(s.active_files)}",
                f"edited files: {', '.join(s.edited_files)}",
                f"findings: {len(s.findings)}",
                f"last error: {s.last_error or '(none)'}",
            ]),
            title="context",
            border_style="bright_cyan",
        ))

    def _show_diff(self, path: str) -> None:
        """Show a unified diff for an edited file."""
        s = self._session.active() if self._session.active_id else None
        log = self._conversation()
        if s is None or not s.edited_files:
            log.write("(no edited files to diff)")
            return
        root = self._tools.root() if self._tools else os.getcwd()
        targets = [path] if path else sorted(s.edited_files)
        for p in targets:
            if p not in s.edited_files:
                log.write(f"(not in edited set: {p})")
                continue
            try:
                with open(os.path.join(root, p), encoding="utf-8") as f:
                    new_text = f.read()
            except Exception as e:
                log.write(f"could not read {p}: {e}")
                continue
            # ORIGINAL = the committed version (git HEAD), not edit_anchor (a one-line snippet → garbage
            # diff). Falls back to empty (whole file shown as added) when untracked / not a git repo.
            try:
                r = subprocess.run(["git", "show", f"HEAD:{p}"], cwd=root,
                                   capture_output=True, text=True, timeout=10)
                old_text = r.stdout if r.returncode == 0 else ""
            except Exception:
                old_text = ""
            self.push_screen(DiffScreen(p, old_text, new_text))

    # ── blocking dialogs (called from worker threads) ──────────────────────────

    def _open_dialog(self, default: str):
        """Register a uniquely-id'd dialog and return (id, event). Caller pushes the screen with the id,
        waits on the event, then calls _close_dialog. Thread-safe (parallel tools → concurrent dialogs)."""
        with self._dialog_lock:
            did = self._dialog_seq
            self._dialog_seq += 1
            event = threading.Event()
            self._dialogs[did] = {"event": event, "result": default}
        return did, event

    def _close_dialog(self, did: int, default: str) -> str:
        with self._dialog_lock:
            return self._dialogs.pop(did, {"result": default})["result"]

    def confirm(self, name: str, args: dict, reason: str) -> str:
        """Synchronous approval dialog; blocks the calling worker thread until the user answers."""
        detail = str(args.get("command") or args.get("path") or args.get("code", ""))[:600]
        did, event = self._open_dialog("no")
        self._safe_call(self.push_screen, ConfirmScreen(name, detail, reason, did))
        event.wait(timeout=900)        # safety net: a never-answered dialog can't disable input forever
        return self._close_dialog(did, "no")

    def ask_user(self, question: str, options: list[str] | None = None) -> str:
        """Synchronous ask_user dialog; blocks the calling worker thread until the user answers."""
        did, event = self._open_dialog("(no answer)")
        self._safe_call(self.push_screen, AskUserScreen(question, options, did))
        event.wait(timeout=900)
        return self._close_dialog(did, "(no answer)")

    # ── running a turn ─────────────────────────────────────────────────────────

    def _run_user_turn(self, line: str) -> None:
        # UI thread: do ONLY cheap UI ops, then hand off to the worker IMMEDIATELY so the user's
        # message (already written by _append_user) paints right away. Everything blocking — topic
        # routing (an LLM classifier round-trip!), session mutation, slice build, the turn — runs in
        # the worker. Previously _route_topic ran HERE on the UI thread, so the echo couldn't paint
        # until that hidden call finished: the lag between Enter and the message appearing.
        self._set_input_enabled(False)
        self._abort_event.clear()
        self.query_one("#input", TextArea).text = ""

        def _work():
            try:
                # Route topic exactly like cli.py — now OFF the UI thread.
                if self._session.active_id is None:
                    self._session.new_topic(line)
                else:
                    action, tid = self._route_topic(self._llm, line, self._session)
                    if action == "new":
                        self._session.new_topic(line)
                    elif action == "resume":
                        self._session.switch_topic(tid)
                        self._session.continue_topic(line, resume=True)   # a resume cue must not overwrite the topic goal
                    else:
                        self._session.continue_topic(line)
                self._record_user and self._record_user(self._session.active(), line)
                self._safe_call(self._refresh_all)
                build = self._make_build_slice(
                    self._session, self._tools, self._retriever, self._memory,
                    line, self._session.session_id,
                )
                result = self._run_turn(
                    build_slice=build,
                    llm=self._llm,
                    tools=self._tools,
                    dispatch=self._dispatch,
                    hooks=self._hooks,
                    signal=self._abort_event,
                    max_steps=self._max_steps,        # honor cfg.max_steps (not the 40 default)
                    checkpoint=self._checkpoint,      # write the crash-recovery WAL (was dead in Textual)
                    consolidate=self._consolidate,    # distilled overflow breadcrumb
                )   # run_turn is synchronous (returns TurnResult); no event-loop juggling needed
                if self._clear_recovery is not None:
                    self._clear_recovery()            # clean turn end → drop the WAL (mirror the REPL path)
                if getattr(self._memory, "is_durable", False):
                    from .taskstate import slice_to_task_state
                    self._memory.checkpoint_task(
                        slice_to_task_state(
                            self._session.active(),
                            self._session.active_id,
                            session_id=self._session.session_id,
                            status="done" if result.stop_reason == "end_turn" else "parked",
                        )
                    )
            except Exception as ex:
                self._safe_call(self._conversation().write, Text(f"Error: {ex}", style="red"))
            finally:
                self._safe_call(self.post_message, TurnDone())

        self._current_worker = self.run_worker(_work, thread=True)

    # ── rendering ──────────────────────────────────────────────────────────────

    def _conversation(self) -> RichLog:
        return self.query_one("#conversation", RichLog)

    def _append_user(self, text: str) -> None:
        # ▌ you marker with breathing room (matches the rich UI; no heavy "You" panel)
        log = self._conversation()
        log.write("")
        log.write(Text.assemble(("▌ ", "bold bright_cyan"), ("you  ", "bold bright_cyan"), (text, "bold")))
        log.write("")

    def _assistant_panel(self, content: str) -> Panel:
        """The reply as Markdown in a clean HORIZONTALS box (matches the rich UI's boxed reply)."""
        return Panel(Markdown(content), title="[bold bright_cyan]assistant[/]", title_align="left",
                     border_style="bright_cyan", box=_rbox.HORIZONTALS, padding=(1, 2))

    def _render_event(self, e: memagent_events.Event) -> None:
        log = self._conversation()
        if isinstance(e, memagent_events.SliceBuilt):
            log.write(Text("  thinking…", style="dim"))
        elif isinstance(e, memagent_events.AssistantText):
            if (e.content or "").strip():
                log.write(self._assistant_panel(e.content))
        elif isinstance(e, memagent_events.ToolStarted):
            pass
        elif isinstance(e, memagent_events.ToolResult):
            self._render_tool_result(e)
        elif isinstance(e, memagent_events.ApiRetry):
            log.write(Text(f"…retry #{e.attempt} ({_shorten(e.error, 60)})", style="yellow"))
        elif isinstance(e, memagent_events.LessonSaved):
            log.write(Text(f"💡 learned: {_shorten(e.title, 70)}", style="dim"))
        elif isinstance(e, memagent_events.TurnEnd):
            tok = (e.usage or {}).get("prompt_tokens", 0) + (e.usage or {}).get("completion_tokens", 0)
            log.write(Text(f"✓ done · {e.steps} steps · {tok} tokens", style="dim"))
        elif isinstance(e, memagent_events.TurnInterrupted):
            log.write(Text(f"⚠ interrupted: {e.message or e.reason}", style="yellow"))
        elif isinstance(e, memagent_events.StepEnd):
            u = e.usage or {}
            self._stats["tokens"] = self._stats.get("tokens", 0) + u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
            self._stats["fresh"] = self._stats.get("fresh", 0) + (u.get("input_other", 0) or 0)
            self._update_header()

    def _render_tool_result(self, e: memagent_events.ToolResult) -> None:
        meta = _TOOL_META.get(e.name, ("•", e.name, None))
        emoji, verb, _ = meta
        primary = self._primary_arg(e.name, e.args)
        header = f"{emoji} {verb} {primary}".rstrip()
        mark = "✓" if not e.failing else "✗"
        color = "green" if not e.failing else "red"
        # ┊ gutter + colored mark + magenta tool header (matches the rich UI's tool cards)
        body: list = [Text.assemble(("┊ ", "grey50"), (f"{mark} ", color), (header, "magenta"))]
        diff = self._diff_card(e.name, e.args)
        if diff:
            body.append(diff)
        folded = self._fold_output(e.output, _TOOL_OUTPUT_FOLD_LINES)
        if folded:
            body.append(folded)
        self._conversation().write(Group(*body))

    def _fold_output(self, output: str, max_lines: int) -> Text | None:
        if not output:
            return None
        lines = output.splitlines()
        if len(lines) <= max_lines:
            return Text(f"   {output}", style="dim")
        shown = "\n".join(lines[:max_lines])
        extra = len(lines) - max_lines
        return Text(f"   {shown}\n   … ({extra} more lines; use /diff or read_file to see full output)", style="dim")

    def _primary_arg(self, name: str, args: dict) -> str:
        key = _TOOL_META.get(name, (None, None, None))[2]
        val = args.get(key) if (key and isinstance(args, dict)) else None
        if val is None and isinstance(args, dict):
            val = next((v for k, v in args.items() if k != "note" and isinstance(v, str)), "")
        return _shorten(str(val or ""))

    def _diff_card(self, name: str, args: dict):
        if name != "str_replace" or not isinstance(args, dict):
            return None
        old, new = str(args.get("old_string") or ""), str(args.get("new_string") or "")  # tolerate non-str model args
        if not old and not new:
            return None
        lines = []
        for ln in old.splitlines()[:8]:
            lines.append(Text(f"- {ln}", style="red"))
        for ln in new.splitlines()[:8]:
            lines.append(Text(f"+ {ln}", style="green"))
        extra = max(0, len(old.splitlines()) - 8) + max(0, len(new.splitlines()) - 8)
        if extra:
            lines.append(Text(f"… {extra} more diff lines (use /diff to expand)", style="dim"))
        return Group(*lines) if lines else None

    # ── sidebar / plan / header ────────────────────────────────────────────────

    def _refresh_all(self) -> None:
        self._update_header()
        self._update_sidebar()
        self._update_plan()

    def _update_sidebar(self) -> None:
        s = self._session.active() if self._session.active_id else None
        tree = self.query_one("#tree", Tree)
        tree.clear()
        if s is None:
            tree.root.add_leaf("no active topic")
            return
        if s.active_files:
            n = tree.root.add("active files")
            for p in sorted(s.active_files):
                n.add_leaf(os.path.basename(p) or p)
        if s.edited_files:
            n = tree.root.add("edited")
            for p in sorted(s.edited_files):
                n.add_leaf(os.path.basename(p) or p)
        # ghosts are {"kind","ref"} dicts — extract the FILE refs (sorting/basename on a dict crashed here)
        ghost_files = [g.get("ref", "") for g in (s.ghosts or []) if g.get("kind") == "file"]
        if ghost_files:
            n = tree.root.add("recently evicted")
            for p in sorted(ghost_files)[:10]:
                n.add_leaf(os.path.basename(p) or p)
        tree.root.expand_all()

    def _update_plan(self) -> None:
        s = self._session.active() if self._session.active_id else None
        widget = self.query_one("#plan", Static)
        if s is None:
            widget.update("")
            return
        lines = []
        if s.mission:
            lines.append(Text.assemble(Text("🎯 ", style="bright_cyan"), Text(s.mission, style="bold")))
        if s.requirements:
            lines.append(Text("requirements:", style="underline"))
            for r in s.requirements:
                mark = "✓" if r.get("done") else "○"
                lines.append(Text(f"  {mark} {r.get('text', '')}", style="dim" if r.get("done") else ""))
        if s.plan:
            lines.append(Text("plan:", style="underline"))
            for p in s.plan:
                g = _PLAN_GLYPH.get(p.get("status"), "○")
                lines.append(Text(f"  {g} {p.get('step', '')}"))
        if not lines:
            widget.update("")
        else:
            widget.update(Panel(Group(*lines), title="plan & mission", border_style="bright_cyan"))

    def _update_header(self) -> None:
        # Derive the active topic from the session directly — the Textual path never populates
        # _stats['topic'] (only the REPL/live paths do), so the header otherwise showed a permanent "topic —".
        topic = self._stats.get("topic") or ""
        if not topic:
            try:
                if self._session.active_id:
                    topic = self._session.active().goal or ""
            except Exception:  # noqa: BLE001
                topic = ""
        topic = _shorten(topic or "—", 40)
        self.sub_title = (
            f"{self._stats.get('model','?')} · "
            f"{self._stats.get('policy','?')} · "
            f"tokens {self._stats.get('tokens',0)} (fresh {self._stats.get('fresh',0)}) · "
            f"topic {topic}"
        )

    def _set_input_enabled(self, enabled: bool) -> None:
        ta = self.query_one("#input", TextArea)
        ta.disabled = not enabled
        if enabled:
            ta.focus()

    def _safe_call(self, fn, *args, **kwargs) -> None:
        """Call a function on the UI thread. If already on the UI thread, call directly."""
        if threading.current_thread().ident == self._thread_id:
            fn(*args, **kwargs)
        else:
            self.call_from_thread(fn, *args, **kwargs)

    # ── public sink factory ────────────────────────────────────────────────────

    @classmethod
    def make_sink(cls, app: "MemagentTui") -> Callable[[memagent_events.Event], None]:
        """Returns a sink that marshals loop events into the UI thread."""
        def sink(e: memagent_events.Event) -> None:
            app._safe_call(app.post_message, AgentEvent(e))
        return sink


def textual_available() -> bool:
    try:
        import textual  # noqa: F401
        return True
    except Exception:
        return False
