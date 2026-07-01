"""Optional rich terminal UI (the `tui` extra: rich + prompt_toolkit).

Periphery — NOT the moat. The loop already decouples rendering via the event
dispatcher; this is just (a) a rich rendering SINK over those events and (b) a prompt_toolkit
input layer. loop.py / pfc.py / seed.py are never touched. The whole module is import-guarded behind the
`tui` extra: core/headless/eval never import rich or prompt_toolkit.

Design (a rich + prompt_toolkit terminal UI):
  - SCROLLBACK model: Rich prints finalized output to history; prompt_toolkit owns the input line.
    They are TEMPORALLY separate (output during the synchronous run_turn, input between turns), so
    there is no patch_stdout/threading minefield.
  - tool-call CARDS (spinner -> ✓/✗, primary-arg header, inline diff for edits),
  - a two-line STATUS footer (model · policy · workspace · tokens),
  - a SLASH-command palette wired to existing session ops (/new /switch /resume /threads /help /exit),
  - graceful ctrl-c AND esc: a physical Ctrl-C (SIGINT) aborts a running turn via Python's own
    KeyboardInterrupt delivery; Esc does the SAME thing via `_EscSentinel`, a narrow background thread that
    translates a bare Esc keypress into a real SIGINT (loop.py only checks a `signal=` Event at STEP
    BOUNDARIES, never inside a blocking LLM/tool call, so only a real SIGINT interrupts promptly).
"""
from __future__ import annotations

import os
import threading
import time

from rich import box as _box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

import shutil

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory

from .events import (AssistantText, ApiRetry, Event, LessonSaved, SliceBuilt, StepBegin, StepEnd,
                     ToolResult, ToolStarted, TurnEnd, TurnInterrupted)

# ── theme (semantic tokens; one place to retheme) ───────────────────────────────────────────
TH = {
    "accent": "bright_cyan", "ok": "green", "fail": "red", "warn": "yellow",
    "dim": "grey50", "tool": "magenta", "add": "green", "del": "red", "user": "bright_cyan",
}

# Rich's DEFAULT markdown styles are "bold cyan on black" / "cyan on black" — so any file path or inline
# `code` the model writes in backticks renders with a heavy BLACK-BACKGROUND highlight. Drop the bg
# (foreground-only) so paths read cleanly in a normal terminal. inherit=True keeps every other default.
MD_THEME = Theme({"markdown.code": "cyan", "markdown.code_block": "cyan"}, inherit=True)


def make_console() -> Console:
    """A Rich Console themed so inline `code` / file paths aren't highlighted on a black background."""
    return Console(theme=MD_THEME)

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


# read-only / navigation tools — a long run of these (a review reads + greps a dozen files) is just
# noise as one card each, so the sink COALESCES a consecutive run into ONE compact line. recall_history
# is deliberately NOT here: it's the memory channel and stays its own visible card.
_COALESCE = {"read_file", "list_files", "grep", "glob"}
_READ_VERB = {"read_file": "read", "list_files": "list", "grep": "grep", "glob": "glob"}


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
    old, new = str(args.get("old_string") or ""), str(args.get("new_string") or "")  # tolerate non-str model args
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


_PLAN_GLYPH = {"done": ("✓", "ok"), "in_progress": ("▶", "accent"), "pending": ("○", "dim")}


def _render_plan(steps: list):
    """A live PLAN/TODO checklist panel: '✓ done', '▶ in-progress', '○ pending'.
    Surfaces the model's update_plan tier as first-class UI instead of a generic tool card."""
    lines = []
    for it in steps:
        if not isinstance(it, dict):
            continue
        status = it.get("status", "pending")
        glyph, gstyle = _PLAN_GLYPH.get(status, ("○", "dim"))
        text_style = TH["dim"] if status == "done" else "default"
        lines.append(Text.assemble(Text(f"{glyph} ", style=TH.get(gstyle, gstyle)),
                                   Text(_shorten(str(it.get("step", "")), 80), style=text_style)))
    done = sum(1 for it in steps if isinstance(it, dict) and it.get("status") == "done")
    title = Text(f"plan · {done}/{len(steps)} done", style=TH["accent"])
    return Panel(Group(*lines) if lines else Text("(empty plan)", style=TH["dim"]),
                 title=title, border_style=TH["dim"], expand=False)


# ── the rendering sink (consumes the loop's events) ──────────────────────────────────────────
def _box_width(console: Console) -> int:
    """Bound the response box so long replies read as a column, not edge-to-edge."""
    try:
        w = int(console.width)
    except Exception:
        w = 100
    return max(48, min(w - 2, 100))


def _response_panel(content: str, console: Console) -> Panel:
    """The assistant reply as Rich Markdown in a clean HORIZONTALS box: light
    top/bottom rules, a left-aligned label, generous padding, bounded width — vs bare full-width Markdown."""
    return Panel(
        Markdown(content),
        title=f"[bold {TH['accent']}]assistant[/]",
        title_align="left",
        border_style=TH["accent"],
        box=_box.HORIZONTALS,
        padding=(1, 2),
        width=_box_width(console),
    )


def _render_tool_result(e):
    """The renderable for a ToolResult — SHARED by RichSink (REPL) and LiveSink (live box) so they can't
    drift. The model-curated tiers render as first-class UI (a live PLAN checklist, the MISSION line);
    everything else is a dim '┊'-gutter card: mark · header · optional inline diff · bounded output (shown
    only for action tools / failures — read/list say it all in the header)."""
    if e.name == "update_plan" and not e.failing:
        return _render_plan(e.args.get("steps") or [])
    if e.name == "set_mission" and not e.failing:
        return Text.assemble(Text("  🎯 mission: ", style=TH["accent"]),
                             Text(_shorten(str(e.args.get("text", "")), 80), style="bold"))
    mark = Text("✓", style=TH["ok"]) if not e.failing else Text("✗", style=TH["fail"])
    head = Text.assemble(Text("┊ ", style=TH["dim"]), mark, " ",
                         Text(_tool_header(e.name, e.args), style=TH["tool"]))
    body = [head]
    d = _diff(e.name, e.args)
    if d is not None:
        body.append(Padding(d, (0, 0, 0, 2)))      # indent the diff under the gutter
    if e.failing or e.name not in ("read_file", "list_files"):
        out = _shorten(e.output, 200)
        if out:
            body.append(Text(f"  ┊   {out}", style=TH["fail"] if e.failing else TH["dim"]))
    return Group(*body)


# The RichSink whose spinner / streaming Live currently owns the terminal. A mid-turn console.input()
# (ask_user, confirm) while a Live is active does NOT echo the user's keystrokes — the Live redraws over
# the input line, so the typed answer is invisible. `_pause_active_live()` stops it first; the next event
# restarts a fresh region. Single point so EVERY mid-turn Rich prompt is covered (no per-call-site fix).
_ACTIVE_RICH_SINK = None
# The _EscSentinel (if any) watching for Esc during the current RICH-mode turn — SAME single-choke-point
# idiom as _ACTIVE_RICH_SINK above. Must release the tty raw-mode fd before a confirm()/ask_user() prompt
# does its OWN raw-mode read (_arrow_select), or the two would race for ownership of the same fd.
_ACTIVE_ESC_SENTINEL = None


def _pause_active_live() -> None:
    """Stop the turn spinner AND release the Esc-sentinel's hold on the tty — the ONE choke point every
    mid-turn synchronous read (confirm, ask_user) goes through before touching raw mode itself."""
    s = _ACTIVE_RICH_SINK
    if s is not None:
        try:
            s._stop()
        except Exception:  # noqa: BLE001 — pausing the live UI must never break the prompt
            pass
    sentinel = _ACTIVE_ESC_SENTINEL
    if sentinel is not None:
        try:
            sentinel.pause()
        except Exception:  # noqa: BLE001 — pausing the sentinel must never break the prompt
            pass


def _resume_active_esc_sentinel() -> None:
    """Re-arm the Esc-sentinel after a mid-turn confirm()/ask_user() read finishes — the counterpart to
    the pause() in _pause_active_live(), called once the caller no longer needs raw-mode ownership."""
    sentinel = _ACTIVE_ESC_SENTINEL
    if sentinel is not None:
        try:
            sentinel.resume()
        except Exception:  # noqa: BLE001
            pass


class _EscSentinel:
    """Translates a bare Esc keypress into a real SIGINT during a RICH-mode (non-live) turn, so Esc aborts
    a running turn exactly like Ctrl-C already does. loop.py's `signal=` Event is checked ONLY at STEP
    BOUNDARIES (never inside a blocking llm.complete() or a slow run_command) — an Event-only abort would
    silently NOT interrupt a turn stuck on a slow model call or a hung command. A real SIGINT does: Python
    always delivers it to the MAIN thread regardless of which thread calls os.kill, reaching the exact
    `except KeyboardInterrupt` handlers a physical Ctrl-C already hits — zero new abort logic in loop.py.

    Also RE-IMPLEMENTS physical Ctrl-C detection while active: putting the tty in raw mode (tty.setraw)
    disables ISIG, which is what makes the tty driver auto-generate SIGINT on Ctrl-C in normal (cooked)
    mode — so without this, a real Ctrl-C press would go SILENT for the whole time this sentinel holds
    raw mode. Both \\x1b (Esc) and \\x03 (Ctrl-C/INTR) are handled identically here for that reason.

    Owns the tty raw-mode fd only while ACTIVE; releases it (restores termios) before going idle on
    pause(), so it never races a mid-turn confirm()/ask_user() call for the SAME fd (_arrow_select's own
    comment: raw mode is process-global, main-thread-only — only one owner at a time). Runs entirely on a
    second daemon thread; the turn itself NEVER leaves the main thread, so confirm()'s arrow-key selector
    (Yes/No/Always) is completely unaffected — its own main-thread-only guard is never even exercised.

    Lifetime: created + started immediately before ONE run_turn() call (RICH mode only, never live mode —
    prompt_toolkit already owns all keystrokes there natively), stopped in a `finally` right after — never
    persists between turns, never leaks a thread."""

    def __init__(self):
        self._thread = None
        self._stop_flag = threading.Event()
        self._pause_flag = threading.Event()
        self._paused_ack = threading.Event()
        self._fd = None
        self._raw = False        # True iff THIS sentinel currently holds the fd in raw mode
        self._old_termios = None

    def start(self) -> None:
        """No-op (never spawns a thread) unless this is a real POSIX tty on the MAIN thread — the SAME
        safety gate _arrow_select uses, so a non-tty/headless/eval run is byte-for-byte unaffected."""
        import sys
        global _ACTIVE_ESC_SENTINEL
        if threading.current_thread() is not threading.main_thread():
            return
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return
        try:
            import termios
            self._fd = sys.stdin.fileno()
            termios.tcgetattr(self._fd)   # confirms this really is a controllable tty
        except Exception:  # noqa: BLE001 — not a real terminal / no termios (e.g. Windows) → stay inert
            return
        _ACTIVE_ESC_SENTINEL = self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _enter_raw(self) -> None:
        import termios
        import tty
        try:
            self._old_termios = termios.tcgetattr(self._fd)
            tty.setraw(self._fd)
            # setraw() also clears OPOST (output post-processing), so while this sentinel holds the tty for
            # the whole turn, the reply's '\n' is no longer mapped to '\r\n' — the output staircases to the
            # right and the spinner cascades. Re-enable cooked OUTPUT (OPOST|ONLCR); INPUT stays raw so Esc
            # and Ctrl-C still arrive as bytes (ISIG/ICANON off). This is the fix for the mid-turn garble.
            a = termios.tcgetattr(self._fd)
            a[1] |= (termios.OPOST | termios.ONLCR)   # oflag
            termios.tcsetattr(self._fd, termios.TCSANOW, a)
            termios.tcflush(self._fd, termios.TCIFLUSH)   # drain type-ahead so a stray byte can't false-fire
            self._raw = True
        except Exception:  # noqa: BLE001 — a wedged/vanished tty must never crash the turn
            self._raw = False

    def _exit_raw(self) -> None:
        if not self._raw:
            return
        import termios
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
        except Exception:  # noqa: BLE001
            try:
                termios.tcsetattr(self._fd, termios.TCSANOW, self._old_termios)
            except Exception:  # noqa: BLE001
                pass
        self._raw = False

    def _run(self) -> None:
        import select
        import signal as _signal
        self._enter_raw()
        try:
            while not self._stop_flag.is_set():
                if self._pause_flag.is_set():
                    self._exit_raw()
                    self._paused_ack.set()
                    # sleep in short increments (not a blocking read) so resume()/stop() are noticed
                    # promptly WITHOUT holding raw mode while idle.
                    while self._pause_flag.is_set() and not self._stop_flag.is_set():
                        self._stop_flag.wait(0.05)
                    if self._stop_flag.is_set():
                        return
                    self._enter_raw()
                    continue
                if not self._raw:      # a previous _enter_raw failed (tty went away) → stop, don't spin
                    return
                try:
                    ready, _, _ = select.select([self._fd], [], [], 0.15)
                except Exception:  # noqa: BLE001 — fd gone / select unsupported → stop, never crash the turn
                    return
                if not ready:
                    continue
                try:
                    data = os.read(self._fd, 16)
                except Exception:  # noqa: BLE001
                    return
                if data in (b"\x1b", b"\x03"):   # bare Esc, or Ctrl-C (\x03/INTR). raw mode (tty.setraw)
                    # disables ISIG, so the tty driver's OWN auto-SIGINT-on-Ctrl-C is OFF while this
                    # sentinel holds raw mode (exactly like _arrow_select already has to handle \x03
                    # itself for the same reason, just for a much shorter window) — WITHOUT this, a
                    # physical Ctrl-C would go silent for the whole turn instead of aborting it. An
                    # arrow/CSI sequence also starts with 0x1b but is LONGER (e.g. \x1b[C); a lone
                    # single-byte read of exactly \x1b means nothing followed.
                    try:
                        os.kill(os.getpid(), _signal.SIGINT)
                    except Exception:  # noqa: BLE001
                        pass
                    return              # one-shot per turn, same as a physical Ctrl-C
        finally:
            self._exit_raw()

    def pause(self) -> None:
        """Release the fd BEFORE returning, so the caller's own raw-mode read (confirm/_arrow_select) can
        never race this thread for ownership. Blocks briefly (bounded by the poll granularity) on an ack
        so the caller only proceeds once the fd is provably free."""
        if self._thread is None or not self._thread.is_alive():
            return
        self._paused_ack.clear()
        self._pause_flag.set()
        self._paused_ack.wait(timeout=1.0)   # generous bound; the sentinel acks within ~1 poll tick (~50ms)

    def resume(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        self._pause_flag.clear()

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=0.3)
        global _ACTIVE_ESC_SENTINEL
        if _ACTIVE_ESC_SENTINEL is self:
            _ACTIVE_ESC_SENTINEL = None


def make_esc_sentinel() -> "_EscSentinel":
    return _EscSentinel()


# running turn tally — bucket each completed tool by KIND so the status line shows "how far along" at a glance
_VERB_BUCKET = {"read_file": "read", "list_files": "read", "grep": "read", "glob": "read",
                "edit_file": "edit", "str_replace": "edit", "append_to_file": "edit",
                "run_command": "cmd", "execute_code": "cmd"}


def _fmt_tally(tally: dict) -> str:
    """Compact 'N read · N edit · N cmd · N fail' — only non-zero buckets, in a stable order."""
    return " · ".join(f"{tally[k]} {k}" for k in ("read", "edit", "cmd", "fail") if tally.get(k))


class _LiveStatus:
    """A Rich RichCast whose __rich__ RECOMPUTES the elapsed clocks every frame off the Status Live loop —
    so the timer ticks with NO extra thread (console.status() already redraws ~12×/s to animate the dots).

    HARD RULE: the object handed to console.status() must be THIS instance, never a Text/str — a static
    body freezes the timer. __rich__ emits ONLY Text.assemble segments (never an f-string into markup), so a
    bracketed path/command in the action label can't trigger a MarkupError. It reads its fields off the sink
    under the sink's lock, so a parallel subagent write can't tear a frame; any error degrades to a bare Text
    (a progress indicator must never crash the run)."""

    def __init__(self, sink: "RichSink"):
        self._sink = sink

    def __rich__(self) -> Text:
        s = self._sink
        try:
            with s._lock:
                label, step = (s._subagent or s._label or "working…"), s._step
                a0, t0, tally = s._action_t0, s._turn_t0, dict(s._tally)
            now = time.monotonic()
            a, turn = max(0.0, now - (a0 or now)), max(0.0, now - (t0 or now))
            parts = []
            if step:
                parts.append((f"step {step} · ", TH["dim"]))
            parts.append((str(label), TH["tool"]))          # str() + Text.assemble → never parsed as markup
            parts.append((f" · {a:.0f}s", TH["dim"]))
            tally_str = _fmt_tally(tally)
            parts.append((f"   ·  {tally_str + ' · ' if tally_str else ''}{turn:.0f}s", TH["dim"]))
            return Text.assemble(*parts)
        except Exception:  # noqa: BLE001 — a progress indicator must never break the run
            return Text("working…", style=TH["dim"])


class RichSink:
    """An event sink that renders the live turn with Rich. Drop-in for cli_sink."""

    def __init__(self, console: Console, stats: dict):
        global _ACTIVE_RICH_SINK
        _ACTIVE_RICH_SINK = self        # so ask_user/confirm can pause the live region before reading input
        self.c = console
        self.stats = stats
        self._lock = threading.RLock()   # parallel explorer threads call subagent_notify concurrently; serialize
        #                                  all _status transitions (rich Status is not thread-safe)
        self._status = None
        # AGENT_SPINNER=off disables the animated in-place status spinner (a Rich live region), keeping every
        # other Rich element (reply panel, markdown, tool cards). Default ON everywhere. (The mid-turn garble
        # once blamed on the spinner + Terminal.app was actually the Esc-sentinel clearing OPOST — fixed in
        # _EscSentinel._enter_raw — so the spinner is safe again; this stays as a plain user preference.)
        self._spinner_on = os.environ.get("AGENT_SPINNER", "on").strip().lower() not in ("off", "0", "false", "no")
        self._reads: list = []   # buffered consecutive read-only tool cards (coalesced on the next event)
        # LIVE STATUS fields (read each frame by _LiveStatus.__rich__ under _lock): the current action label,
        # step number, per-action + whole-turn start clocks, running verb tally, and any active subagent line.
        self._body = None        # the _LiveStatus handed to console.status() (None ⇒ no live region)
        self._label = "thinking…"
        self._subagent = None
        self._step = 0
        self._action_t0 = None   # monotonic start of the CURRENT action (resets each step/tool)
        self._turn_t0 = None     # monotonic start of the WHOLE turn (armed on SliceBuilt, survives step churn)
        self._tally: dict = {}   # bucket -> count (read/edit/cmd/fail) for the "how far along" summary

    def _stop(self) -> None:
        """Tear down the live status region, if any, and reset the label to idle. The transient Status already
        erases the visible line on stop; resetting _label/_subagent keeps the sink's resting state honest so a
        stale 'writing…' can never render via a later path that reuses the region without going through _spin."""
        with self._lock:   # serialize vs parallel subagent_notify on _status
            if self._status is not None:
                self._status.stop()
                self._status = None
            self._body = None
            self._label = "thinking…"
            self._subagent = None

    def _spin(self, label: str) -> None:
        """Set the CURRENT action and ensure the ticking status region is live. MUTATES in place when the
        region already exists (no tear-down → no flicker, and the turn clock keeps ticking); only creates the
        Status the first time. On a non-tty, console.status() no-ops its animation (no ANSI, body never
        refreshed) — same degraded behaviour as before, so on_delta's 'a step is active' gate still holds."""
        with self._lock:   # serialize vs parallel subagent_notify on _status
            self._label = label
            self._subagent = None
            self._action_t0 = time.monotonic()
            if self._turn_t0 is None:
                self._turn_t0 = self._action_t0
            if self._spinner_on and self._status is None:   # create once; MUTATE the same region every frame
                self._body = _LiveStatus(self)
                self._status = self.c.status(self._body, spinner="dots")
                self._status.start()

    def subagent_notify(self, text: str) -> None:
        """A child agent's CURRENT activity → the status line's action segment (overwrites in place), so a
        subagent doing 80 reads shows a single updating line, not 80. Called from PARALLEL explorer worker
        threads → guarded by self._lock; each now writes ONE field instead of the non-thread-safe Status.update."""
        try:
            with self._lock:
                self._subagent = text
                if self._action_t0 is None:
                    self._action_t0 = time.monotonic()
                if self._spinner_on and self._status is None:
                    self._body = _LiveStatus(self)
                    self._status = self.c.status(self._body, spinner="dots")
                    self._status.start()
        except Exception:  # noqa: BLE001 — a progress indicator must never break the run
            pass

    def _bump_tally(self, name: str, failing: bool) -> None:
        with self._lock:
            if failing:
                self._tally["fail"] = self._tally.get("fail", 0) + 1
            bucket = _VERB_BUCKET.get(name)
            if bucket:
                self._tally[bucket] = self._tally.get(bucket, 0) + 1

    def on_delta(self, kind: str, text: str) -> None:
        """Live token sink wired to OpenAILLM.set_delta_sink. While a reply streams, the live region just
        flips its label to a calm, FIXED single-line "writing…" (same shape as the "thinking…" spinner, with
        the running clock) — it does NOT render a live preview of the reply text. The full, formatted reply
        prints once when streaming ends, via AssistantText → _response_panel.

        Why no live preview: three separate live reports of stacked "assistant streaming…" panels traced to
        one root cause shared by every variant that showed the GROWING reply (a Markdown rich.live.Live
        panel, then a plain-text bounded panel, then a bounded text tail inside this status line). Any region
        that grows/wraps eventually reaches the bottom of the terminal and forces a scroll, and ANSI
        cursor-up/erase codes cannot un-scroll content already committed to scrollback — so stale frames pile
        up. A fixed one-line indicator has no height to grow and nothing to scroll, which removes the whole
        bug class regardless of terminal size or emulator. No-op until a step is active (nothing streams
        during routing)."""
        if kind != "content" or not text or self._status is None:
            return
        with self._lock:
            self._label = "writing…"   # thinking → writing: the status line reflects the phase (+ its clock)

    def _flush_reads(self) -> None:
        """Emit ONE compact dim line for a buffered run of read-only tools (📖 7 read · 🔍 3 grep · names),
        instead of one card each — so a review that reads a dozen files doesn't bury the window."""
        if not self._reads:
            return
        reads, self._reads = self._reads, []
        from collections import Counter
        cnt = Counter(n for n, _ in reads)
        parts = [f"{_TOOL.get(n, ('•',))[0]} {c} {_READ_VERB.get(n, n)}" for n, c in cnt.items()]
        names = [v for _, v in reads if v]
        tail = ""
        if names:
            tail = "  " + ", ".join(_shorten(x, 30) for x in names[:5]) + (f"  +{len(names) - 5}" if len(names) > 5 else "")
        self.c.print(Text(f"┊ {' · '.join(parts)}{tail}", style=TH["dim"]))

    def __call__(self, e: Event) -> None:
        if isinstance(e, SliceBuilt):
            self._turn_t0 = time.monotonic()              # arm the whole-turn clock (once per turn)
            self._tally = {}
            self._step = 0
            self._spin("thinking…")
            if not self._spinner_on:                       # spinner off → one plain line so it's not silent
                self.c.print(Text("  · thinking…", style=TH["dim"]))
        elif isinstance(e, StepBegin):
            self._step = e.step                           # the step counter the status line shows
            self._spin("thinking…")                       # new step → reset the action clock, keep the turn clock
        elif isinstance(e, ToolStarted):
            self._spin(_tool_header(e.name, e.args))       # the ticking action segment (spinner conveys "…")
        elif isinstance(e, ToolResult):
            self._bump_tally(e.name, e.failing)            # grow the running "how far along" summary
            if e.name in _COALESCE and not e.failing:      # buffer a read-only run → one line on next event; the
                self._reads.append((e.name, _primary(e.name, e.args)))  # status stays LIVE so the timer keeps
                return                                     # ticking through a 12-file read run (no dead air)
            self._flush_reads()                            # a mutating/failing tool ends the read run
            self.c.print(_render_tool_result(e))           # prints ABOVE the live status region
        elif isinstance(e, AssistantText):
            self._stop()
            self._flush_reads()
            if (e.content or "").strip():
                self.c.print(_response_panel(e.content, self.c))
        elif isinstance(e, ApiRetry):
            self._stop()
            self._flush_reads()
            self.c.print(Text(f"  …retry #{e.attempt} ({_shorten(e.error, 60)})", style=TH["warn"]))
        elif isinstance(e, StepEnd):
            u = e.usage or {}
            self.stats["tokens"] = self.stats.get("tokens", 0) + u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
            # FRESH (non-cache-read) input — the moat metric (typed usage from the llm adapter). Shown in
            # the toolbar so the user sees the bounded-slice cost stay flat, not the gross token count.
            self.stats["fresh"] = self.stats.get("fresh", 0) + (u.get("input_other", 0) or 0)
            _accrue_cost(self.stats, u)
        elif isinstance(e, LessonSaved):
            self._flush_reads()
            self.c.print(Text(f"  💡 learned: {_shorten(e.title, 70)}", style=TH["dim"]))
        elif isinstance(e, TurnInterrupted):
            self._stop()
            self._turn_t0 = None                           # disarm the turn clock (next turn re-arms on SliceBuilt)
            self._flush_reads()
            self.c.print(Text(f"  ⚠ interrupted: {e.message or e.reason}", style=TH["warn"]))
        elif isinstance(e, TurnEnd):
            self._stop()
            self._turn_t0 = None
            self._flush_reads()
            tok = (e.usage or {}).get("prompt_tokens", 0) + (e.usage or {}).get("completion_tokens", 0)
            self.c.print(Text(f"  ✓ done · {e.steps} steps · {tok} tokens", style=TH["dim"]))


def make_rich_sink(console: Console, stats: dict) -> RichSink:
    return RichSink(console, stats)


class LiveSink:
    """Event sink for the LIVE composer (AGENT_TUI=live). Static output — tool cards, the reply panel —
    prints ABOVE the pinned box via the Rich console (routed to scrollback by patch_stdout, verified to
    compose: Rich reads sys.stdout dynamically). The transient spinner + streamed tail live in the app's
    STATUS line via a callback, so no Rich Live fights the running prompt_toolkit Application for the screen."""

    def __init__(self, console: Console, stats: dict, set_status):
        self.c = console
        self.stats = stats
        self._set_status = set_status      # callable(str|None) → update the app status line (thread-safe)
        self._stream = ""

    def on_delta(self, kind: str, text: str) -> None:
        if kind != "content" or not text:
            return
        self._stream += text
        self._set_status("writing… " + " ".join(self._stream.split())[-80:])

    def __call__(self, e: Event) -> None:
        if isinstance(e, SliceBuilt):
            self._stream = ""; self._set_status("thinking…")
        elif isinstance(e, ToolStarted):
            self._stream = ""; self._set_status(f"{_tool_header(e.name, e.args)} …")
        elif isinstance(e, ToolResult):
            self.c.print(_render_tool_result(e))      # static card ABOVE the pinned box
            self._set_status("working…")
        elif isinstance(e, AssistantText):
            self._set_status(None)     # reply complete → clear "writing…" BEFORE the panel, not at TurnEnd
            self._stream = ""          # (else the final answer prints above a stale streaming spinner)
            if (e.content or "").strip():
                self.c.print(_response_panel(e.content, self.c))
        elif isinstance(e, ApiRetry):
            self.c.print(Text(f"  …retry #{e.attempt} ({_shorten(e.error, 60)})", style=TH["warn"]))
        elif isinstance(e, StepEnd):
            u = e.usage or {}
            self.stats["tokens"] = self.stats.get("tokens", 0) + u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
            self.stats["fresh"] = self.stats.get("fresh", 0) + (u.get("input_other", 0) or 0)
            _accrue_cost(self.stats, u)
        elif isinstance(e, LessonSaved):
            self.c.print(Text(f"  💡 learned: {_shorten(e.title, 70)}", style=TH["dim"]))
        elif isinstance(e, TurnInterrupted):
            self.c.print(Text(f"  ⚠ interrupted: {e.message or e.reason}", style=TH["warn"]))
        elif isinstance(e, TurnEnd):
            self._set_status(None)


def build_live_app(*, console: Console, stats: dict, root: str | None, run_one_turn, handle_slash=None,
                   pt_input=None, pt_output=None):
    """Build the LIVE composer Application (split out from run_live so a test can drive it with a pipe input).
    Returns (app, state). state = {status, running, signal, last} so a test can inspect what happened.
    run_one_turn(text, sink, signal) executes ONE turn synchronously; it runs in a daemon worker thread so
    the bordered input box stays pinned + responsive WHILE the agent streams output above it."""
    import threading
    from prompt_toolkit.application import Application
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Float, FloatContainer, HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.menus import MultiColumnCompletionsMenu
    from prompt_toolkit.widgets import Frame, TextArea

    state = {"status": "", "running": False, "signal": None, "last": None, "threads": []}
    toolbar = _toolbar(stats)

    def set_status(text):                  # called from the worker thread; invalidate() is thread-safe
        state["status"] = text or ""
        try:
            app.invalidate()
        except Exception:
            pass

    def _status_line():
        if state["running"] or state["status"]:
            return FormattedText([("fg:ansibrightcyan", "  ✶ "),
                                  ("fg:ansibrightblack", _shorten(state["status"] or "working…", 110))])
        return toolbar()                   # idle → the model · policy · topic · tokens bar

    hist_dir = os.path.expanduser("~/.memagent")
    os.makedirs(hist_dir, exist_ok=True)
    ta = TextArea(prompt="❯ ", multiline=False, wrap_lines=True,
                  history=FileHistory(os.path.join(hist_dir, "history")),
                  completer=_InputCompleter(_repo_files(root) if root else None),
                  complete_while_typing=True)
    kb = KeyBindings()

    @kb.add("enter")
    def _(ev):
        if state["running"]:               # one turn at a time — ignore Enter mid-turn
            return
        text = ta.text.strip()
        ta.text = ""
        if not text:
            return
        if text in ("exit", "quit", "/exit"):
            ev.app.exit(); return
        if text == "/learn" or text.startswith("/learn "):   # transcript → reusable skill, runs as a TURN (mirror the REPL)
            from .neocortex import build_learn_prompt
            text = build_learn_prompt(text[len("/learn"):].strip())
        elif text.startswith("/") and handle_slash is not None:
            handle_slash(text); return
        user_echo(console, text)           # echo ABOVE the box (instant), THEN run the turn
        state["last"] = text
        sink = LiveSink(console, stats, set_status)
        sig = threading.Event()
        state["running"] = True; state["signal"] = sig; state["status"] = "thinking…"

        def _work():
            try:
                run_one_turn(text, sink, sig)
            except Exception as exc:       # a turn crash must NOT kill the composer
                console.print(Text(f"  ✗ turn error: {type(exc).__name__}: {exc}", style=TH["fail"]))
            finally:
                state["running"] = False; state["signal"] = None
                set_status(None)
        state["threads"] = [t for t in state["threads"] if t.is_alive()]   # prune finished workers (no unbounded growth)
        th = threading.Thread(target=_work, daemon=True)
        state["threads"].append(th)
        th.start()
        ev.app.invalidate()

    @kb.add("c-c")
    def _(ev):
        if state["running"] and state["signal"] is not None:
            state["signal"].set()          # abort the running turn at the next step boundary
            set_status("interrupting…")
        else:
            ev.app.exit()

    @kb.add("c-d")
    def _(ev):
        ev.app.exit()

    @kb.add("escape")
    def _(ev):                             # mid-turn: abort (same as ctrl-c); idle: clear the line, or undo
        if state["running"]:
            if state["signal"] is not None:
                state["signal"].set()      # abort the running turn at the next step boundary
                set_status("interrupting…")
            return
        if ta.text.strip():
            ta.text = ""
        elif handle_slash is not None:
            handle_slash("/undo")

    app = Application(
        layout=Layout(FloatContainer(
            content=HSplit([Frame(ta, title="message"),
                            Window(FormattedTextControl(_status_line), height=1)]),
            floats=[Float(xcursor=True, ycursor=True,
                          content=MultiColumnCompletionsMenu(min_rows=3, show_meta=True))],
        ), focused_element=ta),
        key_bindings=kb, full_screen=False, mouse_support=False, input=pt_input, output=pt_output)
    return app, state


def run_live(*, console: Console, stats: dict, banner_info: str, root: str | None,
             run_one_turn, handle_slash=None) -> None:
    """The LIVE composer (AGENT_TUI=live): a bordered input box stays pinned at the bottom EVEN WHILE the
    agent streams — output prints above it in the NORMAL terminal buffer (native copy/paste preserved), the
    Python analogue of Ink's <Static>+live-region. ctrl-c aborts a running turn; ctrl-c at idle / ctrl-d quits."""
    from prompt_toolkit.patch_stdout import patch_stdout
    app, _state = build_live_app(console=console, stats=stats, root=root,
                                 run_one_turn=run_one_turn, handle_slash=handle_slash)
    banner(console, banner_info)
    with patch_stdout(raw=True):
        app.run()


# ── modal selectors (second-tier menus) ───────────────────────────────────────────
def run_selector(title, rows, *, current=-1, hint="↑↓ move · Enter select · Esc cancel",
                 pt_input=None, pt_output=None):
    """A modal single-choice list (a prompt_toolkit choice picker). ``rows`` is a list of
    (label, description); returns the chosen INDEX, or None if cancelled. Non-full-screen + transient
    (erases on close), so it overlays the scrollback like the composer. Safe to call ONLY between turns
    (no other pt Application live) — the REPL slash path satisfies that; live mode falls back to typed args."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.widgets import Frame

    if not rows:
        return None
    state = {"cur": current if 0 <= current < len(rows) else 0}

    def render():
        out = []
        for i, (label, desc) in enumerate(rows):
            sel, cur = i == state["cur"], i == current
            prefix = ("✓" if cur else " ") + ("❯" if sel else " ") + " "
            style = "fg:ansibrightcyan bold" if sel else ("fg:ansigreen" if cur else "")
            out.append((style, prefix + str(label) + "\n"))
            if desc:
                out.append(("fg:ansibrightblack", "       " + str(desc) + "\n"))
        return out

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")
    def _(ev):
        state["cur"] = (state["cur"] - 1) % len(rows)

    @kb.add("down")
    @kb.add("c-n")
    def _(ev):
        state["cur"] = (state["cur"] + 1) % len(rows)

    @kb.add("enter")
    def _(ev):
        ev.app.exit(result=state["cur"])

    @kb.add("escape")
    @kb.add("c-c")
    def _(ev):
        ev.app.exit(result=None)

    header = Window(FormattedTextControl(
        lambda: [("fg:ansicyan bold", f" {title}\n"), ("fg:ansibrightblack", f" {hint}")]), height=2)
    body = Frame(HSplit([header, Window(FormattedTextControl(render))]))
    app = Application(layout=Layout(body), key_bindings=kb, full_screen=False, mouse_support=False,
                      erase_when_done=True, input=pt_input, output=pt_output)
    with patch_stdout(raw=True):
        return app.run()


def _model_candidates(llm, cfg):
    """Models to offer in the /model menu: the current model + any configured providers' models + a known
    set, deduped and grouped by inferred provider family. Returns [(model, family)] sorted by family."""
    from .model_catalog import capability
    known = ["gpt-5.5", "gpt-5", "gpt-5-mini", "o3", "deepseek-chat", "kimi-k2-0905-preview", "claude-sonnet-4-6"]
    prov = []
    try:
        for tbl in (cfg.providers() or {}).values():
            m = tbl.get("model") if isinstance(tbl, dict) else None
            if m:
                prov.append(m)
    except Exception:  # noqa: BLE001 — a malformed providers table must not break the menu
        pass
    out, seen = [], set()
    for m in [llm.model] + prov + known:
        if m and m not in seen:
            seen.add(m)
            base = getattr(llm, "_base_url", "") if m == llm.model else ""
            out.append((m, capability(m, base).family))
    out.sort(key=lambda mf: (mf[1], mf[0]))
    return out


def _reasoning_levels(model, base_url):
    """Reasoning levels valid for a model, derived from its capability (provider-aware). Effort-capable
    models (gpt-5/o-series) expose all four; others only fast/full (high/max would degrade to default)."""
    from .model_catalog import capability
    full4 = [("fast", "minimal reasoning — fastest, cheapest"),
             ("full", "provider default reasoning"),
             ("high", "deeper reasoning (effort=high, /v1/responses)"),
             ("max", "deepest reasoning (effort=xhigh)")]
    if capability(model, base_url).supports_reasoning_effort:
        return full4
    return full4[:2]   # fast | full only — the model has no effort knob


def select_model_reasoning(llm, cfg, *, pt_input=None, pt_output=None):
    """Two-tier picker: choose a model (grouped by provider) then its reasoning level (only the levels that
    model supports). Returns (model, reasoning) to apply, or None if the model step was cancelled."""
    cands = _model_candidates(llm, cfg)
    rows = [(m, f"provider: {fam}") for m, fam in cands]
    cur_idx = next((i for i, (m, _) in enumerate(cands) if m == llm.model), -1)
    pick = run_selector("Select model", rows, current=cur_idx,
                        hint="↑↓ move · Enter choose model → reasoning · Esc cancel",
                        pt_input=pt_input, pt_output=pt_output)
    if pick is None:
        return None
    model = cands[pick][0]
    base = getattr(llm, "_base_url", "") if model == llm.model else ""
    levels = _reasoning_levels(model, base)
    lvl_rows = [(name, desc) for name, desc in levels]
    lvl_cur = next((i for i, (n, _) in enumerate(levels) if n == llm.reasoning), -1)
    lpick = run_selector(f"Reasoning for {model}", lvl_rows, current=lvl_cur,
                         hint="↑↓ move · Enter select · Esc keep current",
                         pt_input=pt_input, pt_output=pt_output)
    reasoning = levels[lpick][0] if lpick is not None else llm.reasoning   # Esc on step 2 = keep current
    return (model, reasoning)


# ── input layer (prompt_toolkit) ─────────────────────────────────────────────────────────────
_SLASH = {
    "/model":   "switch model + reasoning — opens a menu (or /model <name> [fast|full|high|max])",
    "/mode":    "permission mode — opens a menu (baby-sitter · teenager · let-it-go)",
    "/cwd":     "switch workspace root (/cwd <path>) — re-roots repo map, file tools & commands",
    "/learn":   "turn what you just did into a reusable SKILL (/learn [name])",
    "/plan":    "show the agent's current PLAN + mission",
    "/cost":    "show $ saved vs full-history + per-turn token metrics",
    "/threads": "list open/parked topics",
    "/plugins": "list loaded plugins + their tools",
    "/mcp":     "list configured MCP servers + connection status",
    "/help":    "show commands  ·  Esc = undo last turn",
    "/exit":    "quit",
}


_COMPLETE_IGNORE = {".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
                    ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build", ".idea", ".vscode"}


def _repo_files(root: str, cap: int = 4000) -> list:
    """A bounded, ignore-pruned list of repo-relative file paths for prompt file-completion
    (let the user tab-complete a filename to reference it). Best-effort; empty on any error."""
    out = []
    try:
        for dp, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in _COMPLETE_IGNORE and not d.startswith(".")]
            for fn in files:
                if fn.startswith("."):
                    continue
                rel = os.path.relpath(os.path.join(dp, fn), root)
                out.append(rel)
                if len(out) >= cap:
                    return out
    except OSError:
        pass
    return out


# Argument suggestions so the menu fills in the VALUE, not just the command. (Suggestions only — you can
# still type any model.) Keep the model list roughly in sync with cli's /model "known" hint.
_REASONING = (("fast", "minimal reasoning, fastest"), ("full", "provider default"),
              ("high", "deeper (gpt-5: /v1/responses)"), ("max", "deepest (gpt-5: xhigh)"))
_KNOWN_MODELS = ("gpt-5.5", "gpt-5", "gpt-5-mini", "o3", "deepseek-chat", "kimi-k2-0905-preview", "claude-sonnet-4-6")
_SLASH_ARGS = {
    "/reasoning": list(_REASONING),
    "/mode": [("baby-sitter", "confirm every edit + command"), ("teenager", "auto edits, confirm commands"),
              ("let-it-go", "auto-run, still blocks catastrophic")],
}


class _InputCompleter(Completer):
    """Slash-command completion at line start (a command palette) + ARGUMENT suggestions for /model,
    /reasoning, /mode + filename completion on an explicit @mention (the same @path syntax cli.py's
    message parser already recognizes for pinning/attaching a file — see the `@([\\w./\\-]+)` scan).
    Matching ANY plain word against the repo file list (the original behavior) popped a
    completion menu on ordinary prose ("please edit util" → suggests util.py) — annoying enough in
    practice that gating it behind the @ the user already has to type to reference a file is strictly
    better: same capability, zero unsolicited popups."""

    def __init__(self, files=None):
        self._files = files or []

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/") and " " not in text:           # slash command palette
            for cmd, desc in _SLASH.items():
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text), display_meta=desc)
            return
        if text.startswith("/"):                               # ARGUMENT of an already-typed slash command
            parts = text.split(" ")
            cmd, cur = parts[0], parts[-1]
            if cmd == "/model":                                # 1st arg = model name, 2nd = reasoning effort
                opts = [(m, "model") for m in _KNOWN_MODELS] if len(parts) == 2 else (
                    list(_REASONING) if len(parts) == 3 else [])
            else:
                opts = _SLASH_ARGS.get(cmd, [])
            for val, meta in opts:
                if val.lower().startswith(cur.lower()):
                    yield Completion(val, start_position=-len(cur), display_meta=meta)
            return
        words = text.split()
        word = words[-1] if (words and not text.endswith(" ")) else ""
        if not word.startswith("@"):                            # file-path completion ONLY on @word
            return
        wl = word[1:].lower()
        starts = [p for p in self._files if os.path.basename(p).lower().startswith(wl)]
        starts_set = set(starts)   # O(1) membership — `p not in starts` (a list) was O(n) per file → O(n²)/keystroke
        subs = [p for p in self._files if wl in p.lower() and p not in starts_set]
        for p in (starts + subs)[:20]:                          # basename-prefix first, then substring
            yield Completion(p, start_position=-(len(word) - 1), display_meta="file")  # keep the "@", replace after it


# rough public list prices, USD per 1M tokens: (input, cached_input, output). Substring-matched on the
# model id; an unknown model shows token counts only (no $). Update as prices change.
def _price(model: str):
    """USD/1M (input, cached, output) for the cost meter — single source is model_catalog.pricing."""
    from .model_catalog import pricing
    return pricing(model)


def _accrue_cost(stats: dict, usage: dict) -> None:
    """Per step: accrue actual $ spend (stats['cost']) AND the MOAT savings in TOKENS (model-independent, so
    a /model switch re-prices them for free — see _saved_dollars).

    Savings model: a full-transcript agent re-reads the WHOLE prior history every step (a growing cache-read)
    while the bounded slice re-reads only its small cached prefix. The saving is that cache-read DIFFERENTIAL;
    the fresh cost of genuinely-new content is the same for both agents, so it cancels. We track the naive
    transcript size (`_transcript_tok`, grown by each step's fresh-input + output) and bank, per step, the
    tokens the naive agent would re-read that the slice didn't (prefix − this step's actual cache-read)."""
    if not usage:
        return
    prefix = stats.get("_transcript_tok", 0)
    actual_cache_read = usage.get("input_cache_read", 0) or 0
    stats["saved_cached_tok"] = stats.get("saved_cached_tok", 0) + max(0, prefix - actual_cache_read)
    stats["_transcript_tok"] = prefix + (usage.get("input_other", 0) or 0) + (usage.get("output", 0) or 0)

    pr = _price(stats.get("model", ""))
    if not pr:
        return
    pin, pcached, pout = pr
    stats["cost"] = stats.get("cost", 0.0) + (
        usage.get("input_other", 0) * pin
        + usage.get("input_cache_read", 0) * pcached
        + usage.get("output", 0) * pout) / 1_000_000


def _saved_dollars(stats: dict):
    """$ the slice saved vs a full-transcript agent, priced at the CURRENT model's cached rate (the rate for
    re-read history). Token-based, so switching /model re-prices the same savings. None if price unknown."""
    pr = _price(stats.get("model", ""))
    if not pr:
        return None
    return stats.get("saved_cached_tok", 0) * pr[1] / 1_000_000


def _toolbar(stats: dict):
    """A pinned status bar (re-rendered each redraw). FormattedText, not HTML — so a workspace
    name containing < & > can never break the markup."""
    _dim, _accent, _val = "fg:ansibrightblack", "fg:ansibrightcyan bold", "fg:ansicyan"
    sep = (_dim, "  │  ")

    def render():
        ws = _shorten(stats.get("workspace") or "—", 28)
        ft = [
            (_accent, " ◆ "), (_val, str(stats.get("model", "?"))),
            sep, ("", str(stats.get("policy", "?"))),
            sep, (_dim, f"📂 {ws}"),
            sep, (_dim, f"Σ {stats.get('tokens', 0)} tok · {stats.get('fresh', 0)} fresh"),
        ]
        # Headline the MOAT number — $ SAVED vs a full-transcript agent (priced at the current model; flips
        # automatically on /model switch). Falls back to token-savings when the model price is unknown.
        saved = _saved_dollars(stats)
        if saved:
            ft += [sep, ("fg:ansigreen bold", f"💰 ${saved:.4f} saved")]
        elif stats.get("saved_cached_tok"):
            ft += [sep, (_dim, f"💰 {stats['saved_cached_tok'] // 1000}k tok saved")]
        t = stats.get("last_turn_s")
        if t is not None:
            ft += [sep, (_dim, f"⏲ {t:.0f}s")]
        ft.append((_dim, "   "))
        return FormattedText(ft)
    return render


def _force_cooked_output() -> None:
    """Re-assert cooked tty OUTPUT (OPOST|ONLCR) so a bare '\\n' is mapped to '\\r\\n'. prompt_toolkit puts the
    terminal in raw mode for its Application; on some terminals (verified: macOS Terminal.app) that output
    mode LEAKS past the box, so the turn's Rich output prints line-feeds with no carriage return and
    staircases to the right. Called after every prompt to guarantee the next turn prints left-aligned.
    No-op off a real tty (tests/pipes/Windows)."""
    try:
        import sys
        import termios
        fd = sys.stdout.fileno()
        a = termios.tcgetattr(fd)
        a[1] |= (termios.OPOST | termios.ONLCR)   # oflag
        termios.tcsetattr(fd, termios.TCSADRAIN, a)
    except Exception:  # noqa: BLE001 — not a real tty → nothing to restore
        pass


class TuiInput:
    """prompt_toolkit input with history, slash/file completion, and the status toolbar.

    The composer is a BORDERED box pinned at the bottom. It's a prompt_toolkit
    Application run full_screen=False with mouse_support=False — so it stays in the NORMAL terminal buffer:
    the conversation above is real scrollback, and native select/copy/paste keep working on EVERY terminal
    (incl. macOS Terminal.app). Degrades to a plain ❯ prompt if the
    framed Application can't run, so input is never broken."""

    def __init__(self, stats: dict, root: str | None = None):
        self.stats = stats
        hist_dir = os.path.expanduser("~/.memagent")
        os.makedirs(hist_dir, exist_ok=True)
        self._history = FileHistory(os.path.join(hist_dir, "history"))
        self._completer = _InputCompleter(_repo_files(root) if root else None)
        # kept for the fallback path (a plain prompt) if the framed Application errors
        self.session = PromptSession(history=self._history, completer=self._completer,
                                     complete_while_typing=True, bottom_toolbar=_toolbar(stats))

    def prompt(self) -> str | None:
        """Return the next line, or None to QUIT (ctrl-d/ctrl-c at the idle prompt both exit cleanly)."""
        try:
            return self._pinned_prompt()
        except (EOFError, KeyboardInterrupt):
            return None
        except Exception:               # any prompt_toolkit Application hiccup → robust plain prompt
            return self._simple_prompt()

    def _build_composer(self, *, pt_input=None, pt_output=None):
        """Build the framed-composer Application + its TextArea. Split out from _pinned_prompt so a test can
        drive it with a pipe input + DummyOutput (verify Enter→submit / ctrl-c→quit without a real tty)."""
        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Float, FloatContainer, HSplit, Layout, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.layout.menus import MultiColumnCompletionsMenu
        from prompt_toolkit.widgets import Frame, TextArea

        ta = TextArea(prompt="❯ ", multiline=False, wrap_lines=True,
                      history=self._history, completer=self._completer, complete_while_typing=True)
        toolbar = _toolbar(self.stats)
        status = Window(FormattedTextControl(lambda: toolbar()), height=1)
        kb = KeyBindings()

        @kb.add("enter")
        def _(ev):
            ev.app.exit(result=ta.text.strip())   # consistent with the live composer (no whitespace-only turns)

        @kb.add("c-j")                  # Ctrl+J → literal newline (Enter is taken by send)
        def _(ev):
            ta.buffer.insert_text("\n")

        @kb.add("c-c")
        @kb.add("c-d")
        def _(ev):
            ev.app.exit(result=None)

        @kb.add("escape")
        def _(ev):                         # Esc clears a half-typed line; Esc on an EMPTY line = undo last turn
            if ta.text.strip():
                ta.text = ""
            else:
                ev.app.exit(result="/undo")

        # Wrap the composer in a FloatContainer with a CompletionsMenu float so the slash-command palette
        # (and file completions) actually RENDER as a dropdown at the cursor — a bare custom Application
        # computes completions but, unlike PromptSession, has no built-in menu to draw them.
        body = FloatContainer(
            content=HSplit([Frame(ta, title="message"), status]),
            floats=[Float(xcursor=True, ycursor=True,
                          content=MultiColumnCompletionsMenu(min_rows=3, show_meta=True))],
        )
        app = Application(
            layout=Layout(body, focused_element=ta),
            key_bindings=kb, full_screen=False, mouse_support=False,
            # erase the bordered box on submit so the composer is TRANSIENT: after Enter the box (and the text
            # the user typed in it) is wiped, and user_echo prints the single "▌ you …" line — no duplication
            # of the message (the box's last frame + the echo). The echo is the persistent scrollback record.
            erase_when_done=True,
            input=pt_input, output=pt_output,
        )
        return app, ta

    def _pinned_prompt(self) -> str | None:
        """The bordered, bottom-pinned composer (a non-full-screen prompt_toolkit Application)."""
        from prompt_toolkit.patch_stdout import patch_stdout
        app, _ta = self._build_composer()
        try:
            with patch_stdout(raw=True):
                return app.run()
        finally:
            _force_cooked_output()   # undo any raw-output leak so the turn's reply isn't staircased right

    def _simple_prompt(self) -> str | None:
        from prompt_toolkit.patch_stdout import patch_stdout
        cols = max(20, shutil.get_terminal_size((80, 24)).columns)
        msg = FormattedText([("fg:ansibrightblack", "─" * cols + "\n"), ("fg:ansicyan bold", "❯ ")])
        try:
            with patch_stdout(raw=True):
                return self.session.prompt(msg)   # PromptSession.prompt has no erase_when_done kwarg → it raised TypeError, crashing the input fallback
        except (EOFError, KeyboardInterrupt):
            return None
        finally:
            _force_cooked_output()   # undo any raw-output leak so the turn's reply isn't staircased right


def _arrow_select(options: list[str], default: int = 0) -> "int | None":
    """Single-line, arrow-key selector: ←/→ (or ↑/↓) move, Enter chooses, Esc/Ctrl-C cancels; the
    first letter of each option is also a hotkey. Returns the chosen index, -1 if cancelled, or None
    if a selector can't SAFELY run (not a TTY, not the main thread, no termios, raw-mode error) so the
    caller falls back to typed input. POSIX only — Windows returns None. termios + ANSI on one line."""
    import sys
    import threading
    # Raw mode is process-global terminal state — only ever drive it from the MAIN thread with nothing
    # else owning the terminal. A worker-thread turn or a live prompt_toolkit app would race and corrupt it.
    if threading.current_thread() is not threading.main_thread():
        return None
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    try:
        import termios
        import tty
    except Exception:  # noqa: BLE001 — non-POSIX → caller falls back to typed input
        return None
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except Exception:  # noqa: BLE001 — not a real terminal
        return None
    idx = default
    hot = {o[:1].lower(): i for i, o in enumerate(options)}   # y/n/a first-letter hotkeys

    def draw() -> None:
        cells = [(f"\x1b[7m {o} \x1b[0m" if i == idx else f"\x1b[2m {o} \x1b[0m")
                 for i, o in enumerate(options)]
        sys.stdout.write("\r\x1b[2K    " + "  ".join(cells) + "   \x1b[2m(←/→ then Enter)\x1b[0m")
        sys.stdout.flush()

    raw_entered = False
    try:
        tty.setraw(fd)
        raw_entered = True
        try:
            termios.tcflush(fd, termios.TCIFLUSH)   # drain the OS input queue (type-ahead / leftover Enter)
        except Exception:  # noqa: BLE001
            pass
        draw()
        while True:
            # Read RAW bytes straight off the fd — NOT sys.stdin.read(): that's a BUFFERED TEXT stream, and
            # its read-ahead buffer (a) hides a leftover Enter that tcflush can't drain and (b) breaks the
            # select()-based escape probe — which is why an arrow press fell through to picking the default.
            # In raw mode one keypress, incl. a 3-byte arrow (\x1b[C), arrives in a SINGLE os.read.
            try:
                data = os.read(fd, 16)
            except OSError:
                idx = -1
                break
            if not data:                                      # EOF (stdin closed) → cancel
                idx = -1
                break
            if data[:1] in (b"\r", b"\n"):                     # Enter → choose the highlighted option
                break
            if data[:1] == b"\x03":                           # Ctrl-C → cancel
                idx = -1
                break
            if data[:1] == b"\x1b":                           # ESC alone, or a CSI/SS3 escape sequence
                if len(data) == 1:                            # bare ESC → cancel
                    idx = -1
                    break
                if data[1:2] in (b"[", b"O"):                 # CSI / SS3 arrows: the direction byte is right
                    arrow = data[2:3]                         # AFTER [ or O (NOT the buffer's last byte — a
                    if arrow in (b"C", b"B"):                 # trailing Enter in the same read would mask it)
                        idx = (idx + 1) % len(options)        # → / ↓
                    elif arrow in (b"D", b"A"):               # ← / ↑
                        idx = (idx - 1) % len(options)
                    draw()
                    if b"\r" in data[3:] or b"\n" in data[3:]:   # arrow + Enter arrived together → also choose
                        break
                continue                                      # unknown/partial escape → keep waiting
            c = data[:1].decode("ascii", "ignore").lower()    # printable → first-letter hotkey (y/n/a)
            if c in hot:
                idx = hot[c]
                draw()
    except Exception:  # noqa: BLE001 — any I/O error → fall back to typed input, never corrupt the turn
        idx = None
    finally:
        if raw_entered:                                       # only restore if setraw actually succeeded
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:  # noqa: BLE001 — wedged terminal: best-effort immediate restore
                try:
                    termios.tcsetattr(fd, termios.TCSANOW, old)
                except Exception:  # noqa: BLE001
                    pass
            try:
                sys.stdout.write("\r\x1b[2K\n")               # wipe the menu line, land on a clean row
                sys.stdout.flush()
            except Exception:  # noqa: BLE001
                pass
    return idx


def confirm(console: Console, name: str, detail: str, reason: str) -> str:
    """Approval prompt used by the permission hook when the TUI is active. Synchronous (no pt app
    is live mid-run), returns 'yes' | 'no' | 'always'. Arrow-key selectable; falls back to a typed
    prompt where no TTY is available."""
    console.print(Text.assemble(
        Text("  ⚠ allow ", style=TH["warn"]), Text(name, style=TH["tool"]),
        Text(f" {_shorten(detail, 60)!r}? ", style=TH["dim"]), Text(f"({reason})", style=TH["dim"])))
    _pause_active_live()        # stop the turn spinner + release the Esc-sentinel's hold on the tty
    try:
        console.file.flush()    # commit the "allow…" line before the selector's raw ANSI writes (no interleave)
    except Exception:  # noqa: BLE001
        pass
    try:
        idx = _arrow_select(["Yes", "No", "Always"], default=0)
        if idx is not None:                                       # selector ran (TTY): -1 cancel → no
            return ("yes", "no", "always")[idx] if idx >= 0 else "no"
        try:                                                      # fallback: \[y] escapes the Rich markup
            ans = console.input(r"    \[y]es / \[n]o / \[a]lways ▸ ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "no"
        return {"y": "yes", "yes": "yes", "a": "always", "always": "always"}.get(ans, "no")
    finally:
        _resume_active_esc_sentinel()   # re-arm Esc watching for the rest of the turn, whichever path returned


def ask_user(console: Console, question: str, options=None) -> str:
    """The ask_user prompt (the 'come back and ask a follow-up' capability). Synchronous — no pt app
    is live mid-run — so a Rich prompt is safe. Returns the user's answer (a chosen option or free text)."""
    console.print(Text.assemble(Text("  ❓ ", style=TH["accent"]), Text(question, style="bold")))
    if options:
        for i, o in enumerate(options, 1):
            console.print(Text(f"     {i}. {o}", style=TH["dim"]))
        console.print(Text("     (type a number, or your own answer)", style=TH["dim"]))
    _pause_active_live()        # stop the turn spinner + release the Esc-sentinel's hold on the tty
    try:
        try:
            ans = console.input("  your answer ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            return "(no answer)"
        if options and ans.isdigit() and 1 <= int(ans) <= len(options):
            return options[int(ans) - 1]
        return ans or "(no answer)"
    finally:
        _resume_active_esc_sentinel()   # re-arm Esc watching for the rest of the turn


# memagent wordmark (figlet "ansi_shadow") + the vertical 3-layer emblem. Identity = the context kernel:
# ▓ slice (L1, hot working set) → ▒ cache (L2, sealed episodes) → ░ memory (L3, distilled lessons); the
# bright→dim gradient is that hot→cold flow. Art hardcoded (no pyfiglet runtime dependency).
_WORDMARK = (
    "███╗   ███╗███████╗███╗   ███╗ █████╗  ██████╗ ███████╗███╗   ██╗████████╗",
    "████╗ ████║██╔════╝████╗ ████║██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝",
    "██╔████╔██║█████╗  ██╔████╔██║███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ",
    "██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ",
    "██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ",
    "╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ",
)
_EMBLEM = (("▓▓", "bright_cyan"), ("▓▓", "bright_cyan"), ("▒▒", "cyan"),
           ("▒▒", "cyan"), ("░░", "grey50"), ("░░", "grey50"))   # 2 rows per layer, beside the wordmark


def user_echo(console: Console, text: str) -> None:
    """Anchor the user's turn with breathing room: a blank line, a colored left-bar 'you' marker with the
    message, then a blank line — so the prompt and the agent's reply don't run together (fixes cramped
    spacing between user input and the response)."""
    console.print()
    console.print(Text.assemble(("▌ ", f"bold {TH['accent']}"), ("you  ", f"bold {TH['accent']}"),
                                (text, "bold")))
    console.print()


def banner_panel(console: Console, info: str) -> Panel:
    """The startup logo: the full ansi_shadow BLOCK wordmark, always (per user preference — never a compact
    fallback). Each art row is no-wrap + crop, so a terminal narrower than the art (~86 cols) clips it
    cleanly on the right instead of wrapping into a staircase; a normal-width window shows it in full.
    `console` is kept in the signature for the callers, though the layout is now width-independent."""
    rows = []
    for i, word in enumerate(_WORDMARK):
        blk, col = _EMBLEM[i]
        t = Text.assemble(("  ", ""), (blk, f"bold {col}"), ("  ", ""), (word, f"bold {col}"))
        t.no_wrap = True
        t.overflow = "crop"          # narrow terminal → clip the art, never wrap it into a staircase
        rows.append(t)
    rows.append(Text(""))
    rows.append(Text("  ▓ slice → ▒ cache → ░ memory   ·   memory-native coding agent", style=TH["dim"]))
    if info:
        rows.append(Text("  " + info, style=TH["dim"]))
    return Panel(Group(*rows), border_style=TH["accent"], box=_box.ROUNDED,
                 title=f"[bold {TH['accent']}]memagent[/]", title_align="left",
                 subtitle="[grey50]/help · ctrl-d to quit[/]", subtitle_align="right", padding=(1, 2))


def banner(console: Console, info: str) -> None:
    console.print(banner_panel(console, info))


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
