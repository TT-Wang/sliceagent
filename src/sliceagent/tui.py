"""Optional rich terminal UI (the `tui` extra: rich + prompt_toolkit).

Periphery — NOT the moat. The loop already decouples rendering via the event
dispatcher; this is just (a) a rich rendering SINK over those events and (b) a prompt_toolkit
input layer. loop.py / pfc.py / seed.py are never touched. The whole module is import-guarded behind the
`tui` extra: core/headless/eval never import rich or prompt_toolkit.

Design (a rich + prompt_toolkit terminal UI):
  - SCROLLBACK model: Rich prints finalized output to history; prompt_toolkit owns the input line.
    They are TEMPORALLY separate (output during the synchronous run_turn, input between turns), so
    there is no patch_stdout/threading minefield.
  - quiet, width-safe activity rails with stronger durable milestones,
  - one truthful live progress row plus a responsive identity footer,
  - one canonical SLASH-command palette shared with CLI help (including session and capability discovery),
  - graceful ctrl-c AND esc: a physical Ctrl-C (SIGINT) aborts a running turn via Python's own
    KeyboardInterrupt delivery; Esc does the SAME thing via `_EscSentinel`, a narrow background thread that
    translates a bare Esc keypress into a real SIGINT (loop.py only checks a `signal=` Event at STEP
    BOUNDARIES, never inside a blocking LLM/tool call, so only a real SIGINT interrupts promptly).
"""
from __future__ import annotations

import os
import shutil
import threading
import time
from collections import Counter
from dataclasses import dataclass

from rich import box as _box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.utils import get_cwidth

from .events import (AssistantText, ApiRetry, Event, LessonSaved, StepBegin, StepEnd,
                     SubagentProgress, ToolResult, ToolStarted, TurnCommitted, TurnEnd,
                     TurnInterrupted, TurnStarted)
from .mentions import completion_path
from .progress import ProgressPhase, TurnProgress
from .receipts import (receipt_completion_label, receipt_has_adverse_lifecycle,
                       receipt_summary_parts)
from .slash import PUBLIC_SLASH_COMMANDS
from .tui_projection import (DELEGATION_TOOLS, QUIET_OUTPUT_TOOLS, AgentResultView,
                             child_incompleteness_label, invocation_id, normalized_tool_status, output_preview,
                             project_agent_result, project_tool_result, safe_terminal_text)

# ── theme (semantic tokens; one place to retheme) ───────────────────────────────────────────
TH = {
    "accent": "bright_cyan", "ok": "green", "fail": "red", "warn": "yellow",
    "dim": "grey50", "gutter": "grey35", "tool": "default", "inspect": "blue",
    "edit": "magenta", "run": "cyan", "verify": "bright_blue",
    "add": "green", "del": "red", "user": "bright_cyan",
}

# Rich's DEFAULT markdown styles are "bold cyan on black" / "cyan on black" — so any file path or inline
# `code` the model writes in backticks renders with a heavy BLACK-BACKGROUND highlight. Drop the bg
# (foreground-only) so paths read cleanly in a normal terminal. inherit=True keeps every other default.
MD_THEME = Theme({"markdown.code": "cyan", "markdown.code_block": "cyan"}, inherit=True)


def make_console() -> Console:
    """A Rich Console themed so inline `code` / file paths aren't highlighted on a black background."""
    return Console(theme=MD_THEME)


def _private_prompt_history() -> FileHistory:
    """Return the shared prompt history after enforcing private on-disk permissions.

    prompt_toolkit creates a missing history file with the process umask (commonly 0644). That file contains
    raw user requests, so pre-create it as 0600 and repair older installs before either composer opens it.
    """
    hist_dir = os.path.expanduser("~/.sliceagent")
    os.makedirs(hist_dir, mode=0o700, exist_ok=True)
    path = os.path.join(hist_dir, "history")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        try:
            os.fchmod(fd, 0o600)
        except (AttributeError, OSError):
            pass
    finally:
        os.close(fd)
    try:
        os.chmod(hist_dir, 0o700)
        os.chmod(path, 0o600)
    except OSError:
        pass
    return FileHistory(path)

# Calm, terminal-stable tool presentation: a short verb + the informative argument.  Avoid emoji here;
# their double-width behavior varies by terminal and makes settled output visibly jitter.
_TOOL = {
    "read_file":      ("read",    "path"),
    "edit_file":      ("write",   "path"),
    "append_to_file": ("append",  "path"),
    "str_replace":    ("edit",    "path"),
    "list_files":     ("list",    "path"),
    "run_command":    ("run",     "command"),
    "execute_code":   ("execute", "code"),
    "grep":           ("search",  "pattern"),
    "glob":           ("find",    "pattern"),
    "skill":          ("skill",   "name"),
    "search_history": ("recall",  "query"),
    "new_topic":      ("task",    "goal"),
    "switch_topic":   ("switch",  "task_id"),
    "spawn_agent":    ("delegate", "task"),
    "spawn_subagent": ("delegate", "task"),
    "spawn_explore":  ("explore",  "task"),
}


# read-only / navigation tools — a long run of these (a review reads + greps a dozen files) is just
# noise as one card each, so the sink COALESCES a consecutive run into ONE compact line. search_history
# is deliberately NOT here: it's the cross-session memory channel and stays its own visible card. (A
# read_file/grep under history/ coalesces like any other read — it IS an ordinary file read now.)
_COALESCE = {"read_file", "list_files", "grep", "glob"}
_READ_VERB = {"read_file": "read", "list_files": "list", "grep": "search", "glob": "find"}


def _shorten(s: str, n: int = 64) -> str:
    s = " ".join(safe_terminal_text(s, multiline=True).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _shorten_cells(value: object, width: int) -> str:
    """Normalize and crop prompt-toolkit text by terminal cells, including CJK and combining text."""
    text = " ".join(safe_terminal_text(value, multiline=True).split())
    width = max(1, int(width))
    if get_cwidth(text) <= width:
        return text
    out, used = [], 0
    for char in text:
        cells = max(0, get_cwidth(char))
        if used + cells > width - 1:
            break
        out.append(char)
        used += cells
    return "".join(out) + "…"


def _crop_cells(text: str, width: int) -> str:
    """Hard cell crop used only as a final safety net for a formatted one-line footer."""
    out, used = [], 0
    for char in text:
        cells = max(0, get_cwidth(char))
        if used + cells > max(0, width):
            break
        out.append(char)
        used += cells
    return "".join(out)


def _primary(name: str, args: dict) -> str:
    key = _TOOL.get(name, (None, None))[1]
    val = args.get(key) if (key and isinstance(args, dict)) else None
    if val is None and isinstance(args, dict):  # fallback: first non-note string arg
        val = next((v for k, v in args.items() if k != "note" and isinstance(v, str)), "")
    return _shorten(str(val or ""))


def _tool_header(name: str, args: dict) -> str:
    verb, _ = _TOOL.get(name, (name.replace("_", " "), None))
    p = _primary(name, args)
    return f"{verb} {p}".rstrip()


def _diff(name: str, args: dict, width: int = 100):
    """A compact inline diff for a str_replace (old → new). Returns a Rich renderable or None."""
    if name != "str_replace" or not isinstance(args, dict):
        return None
    old, new = str(args.get("old_string") or ""), str(args.get("new_string") or "")  # tolerate non-str model args
    if not old and not new:
        return None
    lines = []
    for ln in old.splitlines()[:4]:
        lines.append(_fit_line(Text(
            f"- {safe_terminal_text(ln, multiline=False)}", style=TH["del"],
        ), max(1, width - 2)))
    for ln in new.splitlines()[:4]:
        lines.append(_fit_line(Text(
            f"+ {safe_terminal_text(ln, multiline=False)}", style=TH["add"],
        ), max(1, width - 2)))
    extra = max(0, len(old.splitlines()) - 4) + max(0, len(new.splitlines()) - 4)
    if extra:
        lines.append(_fit_line(Text(f"… {extra} more diff lines", style=TH["dim"]), max(1, width - 2)))
    return Group(*lines) if lines else None


def _render_plan(steps: list, width: int = 100) -> Text:
    """One settled plan summary; ``/plan`` remains the home for the full checklist."""
    steps = steps if isinstance(steps, list) else []
    valid = [it for it in steps if isinstance(it, dict)]
    done = sum(1 for it in valid if it.get("status") == "done")
    current = next((str(it.get("step", "")) for it in valid
                    if it.get("status") == "in_progress" and it.get("step")), "")
    if not current:
        current = next((str(it.get("step", "")) for it in valid
                        if it.get("status") != "done" and it.get("step")), "")
    if not valid:
        tail = "empty"
    elif done == len(valid):
        tail = "complete"
    else:
        tail = _shorten(current or "next step pending", 100)
    return _fit_line(Text.assemble(
        Text("│ ", style=TH["gutter"]),
        Text("plan", style=f"bold {TH['accent']}"),
        Text(f" {done}/{len(valid)} · ", style=TH["dim"]),
        Text(tail, style=TH["dim"] if not valid or done == len(valid) else "default"),
    ), width)


# ── the rendering sink (consumes the loop's events) ──────────────────────────────────────────
def _box_width(console: Console) -> int:
    """Bound the response box so long replies read as a column, not edge-to-edge."""
    try:
        w = int(console.width)
    except Exception:
        w = 100
    return min(max(1, w - 2), 108)


def _line_width(console: Console) -> int:
    """Usable width for settled rows, leaving a small right margin."""
    try:
        return max(1, int(console.width) - 2)
    except Exception:
        return 98


def _response_panel(content: str, console: Console, *, title: str = "assistant",
                    border_style: str | None = None) -> Panel:
    """The assistant reply as Rich Markdown in a clean HORIZONTALS box: light
    top/bottom rules, a left-aligned label, generous padding, bounded width — vs bare full-width Markdown."""
    return Panel(
        Markdown(safe_terminal_text(content, multiline=True)),
        title=Text(title, style=f"bold {border_style or TH['accent']}"),
        title_align="left",
        border_style=border_style or TH["accent"],
        box=_box.HORIZONTALS,
        padding=(1, 2),
        width=_box_width(console),
    )


def _render_milestone(note: str, width: int = 100) -> Text:
    """A model-authored tool note, visibly advisory rather than evidence-backed truth."""
    return _fit_line(Text.assemble(
        Text("│ ", style=TH["gutter"]), Text("agent note · ", style=TH["dim"]),
        Text(_shorten(note, 160), style=TH["dim"]),
    ), width)


def _omitted_lines(count: int) -> str:
    return f"… {count} line{'s' if count != 1 else ''} omitted"


def _render_assistant_update(content: str, width: int = 100) -> Group:
    """Intermediate narration is a light trail entry, not another full answer panel."""
    preview = output_preview(content, max_rows=3, max_chars=240)
    rows = [_fit_line(Text.assemble(
        Text("│ ", style=TH["gutter"]), Text("assistant update", style=TH["dim"]),
    ), width)]
    for index, line in enumerate(preview.lines):
        if preview.hidden_lines and preview.tail_retained and index == len(preview.lines) - 1:
            rows.append(_fit_line(Text.assemble(
                Text("│   ", style=TH["gutter"]),
                Text(_omitted_lines(preview.hidden_lines), style=TH["dim"]),
            ), width))
        rows.append(_fit_line(Text.assemble(
            Text("│   ", style=TH["gutter"]), Text(line, style=TH["dim"]),
        ), width))
    if preview.hidden_lines and not preview.tail_retained:
        rows.append(_fit_line(Text.assemble(
            Text("│   ", style=TH["gutter"]),
            Text(f"… {preview.hidden_lines} more lines", style=TH["dim"]),
        ), width))
    return Group(*rows)


def _render_read_summary(reads: list, width: int = 100) -> Text | None:
    """One terminal-stable line for a settled read wave, shared by both TUI adapters."""
    if not reads:
        return None
    counts = Counter(name for name, _ in reads)
    work = " · ".join(f"{count} {_READ_VERB.get(name, name.replace('_', ' '))}"
                      for name, count in counts.items())
    names = [value for _, value in reads if value]
    tail = ""
    if names:
        tail = "  " + ", ".join(_shorten(value, 30) for value in names[:5])
        if len(names) > 5:
            tail += f"  +{len(names) - 5}"
    return _fit_line(Text.assemble(
        Text("│ ", style=TH["gutter"]), Text(work, style=TH["dim"]), Text(tail),
    ), width)


def _status_chip(status: str, duration_s: float | None) -> str:
    parts = []
    if status == "indeterminate":
        parts.append("state unknown")
    elif status == "steered":
        parts.append("steered")
    elif status == "cancelled":
        parts.append("cancelled")
    if duration_s is not None and duration_s >= 0.05:
        parts.append(_duration(duration_s))
    return " · ".join(parts)


_EVIDENCE_LABEL = {
    "not_assessed": "not assessed",
    "none": "no evidence",
    "navigation_only": "navigation only",
    "content_partial": "content partial",
    "content_retained": "content retained",
}


def _evidence_key(value: object) -> str:
    status = str(value or "not_assessed")
    return status if status in _EVIDENCE_LABEL else "not_assessed"


def _evidence_counts(value: object) -> dict[str, int]:
    try:
        items = value.items() if hasattr(value, "items") else value
        return {
            str(key): max(0, int(count))
            for key, count in (items or ())
            if isinstance(count, int) and not isinstance(count, bool)
        }
    except (TypeError, ValueError):
        return {}


def _evidence_token(status: object, account: object = ()) -> str:
    """Compact typed evidence fact; never infer quality from a child report or activity prose."""
    status = _evidence_key(status)
    counts = _evidence_counts(account)
    content = counts.get("content_success_count", 0)
    retained = counts.get("retained_content_view_count", 0)
    navigation = counts.get("navigation_success_count", 0)
    if status == "content_retained":
        return f"retained {content or retained}" if (content or retained) else "retained"
    if status == "content_partial":
        return f"partial {retained}/{content}" if content else "partial"
    if status == "navigation_only":
        return f"nav {navigation}" if navigation else "navigation"
    if status == "none":
        return "no evidence"
    return "not assessed"


def _evidence_style(status: object) -> str:
    status = _evidence_key(status)
    if status == "content_retained":
        return "ok"
    if status in {"none", "navigation_only", "content_partial"}:
        return "warn"
    return "dim"


def _render_agent_batch(agents: list[AgentResultView], width: int = 100) -> Group | None:
    """One durable, identity-safe summary for a settled delegation wave."""
    if not agents:
        return None
    def display_ordinal(pair) -> int:
        fallback_index, view = pair
        return view.request_ordinal or view.launch_ordinal or (fallback_index + 1)

    ordered = sorted(enumerate(agents), key=lambda pair: (display_ordinal(pair), pair[0]))
    total = len(ordered)
    ready = sum(view.report_ready for _, view in ordered)
    completed_without_report = sum(
        view.status == "succeeded" and not view.report_ready for _, view in ordered
    )
    steered = sum(view.status == "steered" for _, view in ordered)
    timed_out = sum(view.status == "failed" and view.timed_out for _, view in ordered)
    failed = sum(view.status == "failed" and not view.timed_out for _, view in ordered)
    cancelled = sum(view.status == "cancelled" for _, view in ordered)
    indeterminate = sum(view.status == "indeterminate" for _, view in ordered)
    rows = [_fit_line(Text.assemble(
        Text("│ ", style=TH["gutter"]), Text("agents", style=f"bold {TH['accent']}"),
        Text(f" · {ready}/{total} reports ready", style=TH["dim"]),
    ), width)]
    for count, label, style in (
        (completed_without_report, "completed without report", TH["warn"]),
        (steered, "steered", TH["dim"]), (failed, "failed", TH["fail"]),
        (timed_out, "timed out", TH["warn"]),
        (cancelled, "cancelled", TH["dim"]),
        (indeterminate, "state unknown", TH["warn"]),
    ):
        if count:
            rows.append(_fit_line(Text.assemble(
                Text("│   ", style=TH["gutter"]), Text(f"{count} {label}", style=style),
            ), width))
    source_counts = Counter(
        view.source_coverage_status for _, view in ordered
        if view.source_coverage_status in {"source_complete", "source_partial", "source_unsupported"}
    )
    if source_counts:
        source_facts = [
            f"{source_counts[status]} {label}"
            for status, label in (
                ("source_complete", "source complete"),
                ("source_partial", "source partial"),
                ("source_unsupported", "source unsupported"),
            )
            if source_counts[status]
        ]
        rows.append(_fit_line(Text.assemble(
            Text("│   ", style=TH["gutter"]),
            Text("source coverage · " + " · ".join(source_facts), style=TH["warn"]),
        ), width))
    evidence_counts = Counter(_evidence_key(view.evidence_status) for _, view in ordered)
    evidence_facts = []
    for status in (
        "content_retained", "content_partial", "navigation_only", "none", "not_assessed",
    ):
        count = evidence_counts[status]
        if count:
            evidence_facts.append((f"{count} {_EVIDENCE_LABEL[status]}", _evidence_style(status)))
    evidence_line = None
    for fact, style_key in evidence_facts:
        if evidence_line is None:
            evidence_line = Text("│   ", style=TH["gutter"])
            evidence_line.append("evidence · ", style=TH["dim"])
        separator = " · " if evidence_line.plain.rstrip() != "│   evidence ·" else ""
        if separator and get_cwidth(evidence_line.plain + separator + fact) > width:
            rows.append(_fit_line(evidence_line, width))
            evidence_line = Text("│   ", style=TH["gutter"])
            evidence_line.append("evidence · ", style=TH["dim"])
            separator = ""
        if separator:
            evidence_line.append(" · ", style=TH["dim"])
        evidence_line.append(fact, style=TH[style_key])
    if evidence_line is not None:
        rows.append(_fit_line(evidence_line, width))

    # Counts preserve complete lifecycle truth; representative rows stay hard-bounded.
    # Prefer adverse entries, then fill the remaining slots with successful reports.
    # The durable block has a hard 20-row ceiling. Summary rows vary with lifecycle/evidence diversity;
    # reserve three adverse detail rows plus overflow/artifact locators before choosing representatives.
    entry_cap = max(1, min(8, 20 - len(rows) - 5))
    adverse = [pair for pair in ordered if pair[1].status != "succeeded"]
    successful = [pair for pair in ordered if pair[1].status == "succeeded"]
    selected = adverse[:entry_cap]
    selected.extend(successful[: max(0, entry_cap - len(selected))])
    selected.sort(key=lambda pair: (display_ordinal(pair), pair[0]))
    omitted = total - len(selected)
    detail_rows = 0
    for pair in selected:
        _fallback_index, view = pair
        ordinal = display_ordinal(pair)
        if view.status == "succeeded":
            glyph, style = "✓", TH["ok"]
        elif view.status in {"steered", "cancelled"}:
            glyph, style = "↷", TH["dim"]
        elif view.status == "indeterminate" or view.timed_out:
            glyph, style = "!", TH["warn"]
        else:
            glyph, style = "✗", TH["fail"]
        identity = view.kind or "agent"
        if view.name:
            identity += f" {view.name}"
        task = _shorten(view.task or "delegated work", 100)
        suffix = []
        if view.duration_s is not None and view.duration_s >= 0.05:
            suffix.append(_duration(view.duration_s))
        if view.status != "succeeded" and view.terminal_reason:
            suffix.append(view.terminal_reason.replace("_", " "))
        elif view.recovered_from:
            suffix.append("recovered from " + ", ".join(item.replace("_", " ") for item in view.recovered_from))
        if view.status == "succeeded" and not view.report_ready:
            suffix.append(
                "completed · no report"
                if view.report_completion == "absent"
                else "completed · report status unknown"
            )
        else:
            incompleteness = child_incompleteness_label(view.report_completion, view.partial)
            if incompleteness:
                suffix.append(incompleteness)
        row = Text.assemble(
            Text("│   ", style=TH["gutter"]), Text(glyph + " ", style=f"bold {style}"),
            Text(f"{ordinal} {identity}"), Text(" — ", style=TH["dim"]), Text(task),
        )
        if suffix:
            row.append(" · " + " · ".join(suffix), style=TH["dim"] if view.status == "succeeded" else style)
        row.append(
            " · evidence " + _evidence_token(view.evidence_status, view.evidence_account),
            style=TH[_evidence_style(view.evidence_status)],
        )
        if view.source_coverage_status in {
            "source_complete", "source_partial", "source_unsupported",
        }:
            row.append(
                " · " + view.source_coverage_status.replace("_", " "),
                style=TH["warn"],
            )
        rows.append(_fit_line(row, width))
        show_inline_report = view.status == "succeeded" and not view.artifact_id and bool(view.detail)
        show_adverse_detail = view.status != "succeeded" and view.detail and detail_rows < 3
        if show_inline_report or show_adverse_detail:
            rows.append(_fit_line(Text.assemble(
                Text("│     ", style=TH["gutter"]), Text(_shorten(view.detail, 180), style=style),
            ), width))
            if show_adverse_detail:
                detail_rows += 1
    artifact_count = sum(bool(view.artifact_id) for _, view in ordered)
    if omitted:
        rows.append(_fit_line(Text.assemble(
            Text("│   ", style=TH["gutter"]),
            Text(f"… {omitted} more agent{'s' if omitted != 1 else ''}", style=TH["dim"]),
        ), width))
    if total == 1 and agents[0].artifact_id:
        exact = f"artifacts/{agents[0].artifact_id}.md"
        locator = exact if get_cwidth("│   artifact · " + exact) <= width else "artifacts/index.md"
        rows.append(_fit_line(Text.assemble(
            Text("│   ", style=TH["gutter"]), Text("artifact · ", style=TH["dim"]),
            Text(locator, style=TH["dim"]),
        ), width))
    elif artifact_count:
        rows.append(_fit_line(Text.assemble(
            Text("│   ", style=TH["gutter"]), Text("artifacts · ", style=TH["dim"]),
            Text(f"artifacts/index.md · {artifact_count} stored", style=TH["dim"]),
        ), width))
    return Group(*rows)


def _buffer_agent_once(buffer: list[AgentResultView], view: AgentResultView) -> None:
    """ToolResult projection is at-least-once across an interrupted dispatcher boundary."""
    key = view.invocation_id or view.artifact_id
    if key and any((item.invocation_id or item.artifact_id) == key for item in buffer):
        return
    buffer.append(view)


def _accepted_agent_finished_at(snapshot, view: AgentResultView) -> float | None:
    """Return only a terminal timestamp accepted by the semantic reducer.

    Child callbacks are advisory and can arrive late, from an old turn, or with a
    contradictory identity.  Renderers must not time a settled result from such a
    rejected callback; the pre-result snapshot is the single source of truth.
    """
    for item in snapshot.subagents:
        matches_artifact = bool(view.artifact_id and item.agent_id == view.artifact_id)
        matches_invocation = bool(
            view.invocation_id and not item.parent_agent_id
            and item.invocation_id == view.invocation_id
        )
        if (matches_artifact or matches_invocation) and item.phase in {
            "report_ready", "completed", "steered", "failed", "timed_out", "cancelled",
            "indeterminate",
        }:
            return item.finished_at
    return None


def _render_tool_result(e, width: int = 100, *, duration_s: float | None = None):
    """The renderable for a ToolResult — SHARED by RichSink (REPL) and LiveSink (live box) so they can't
    drift. Plan updates render as one settled summary;
    everything else is a dim '│'-gutter rail: header · optional inline diff · bounded output (shown
    only for action tools / failures — read/list say it all in the header)."""
    if e.name in DELEGATION_TOOLS:
        return _render_agent_batch([project_agent_result(e, duration_s=duration_s)], width)
    status = normalized_tool_status(e)
    steered = status == "steered"
    cancelled = status == "cancelled"
    indeterminate = status == "indeterminate"
    safety_stop = cancelled and str(getattr(e, "output", "") or "").startswith("Safety stop:")
    display_failure = status == "failed"
    if e.name == "update_plan" and status == "succeeded":
        args = e.args if isinstance(e.args, dict) else {}
        return _render_plan(args.get("steps") or [], width)
    if safety_stop:
        mark = Text("! ", style=TH["warn"])
    elif indeterminate:
        mark = Text("! ", style=TH["warn"])
    elif steered or cancelled:
        mark = Text("↷ ", style=TH["dim"])
    elif display_failure:
        mark = Text("✗ ", style=TH["fail"])
    else:
        mark = Text()
    head = Text.assemble(
        Text("│ ", style=TH["gutter"]), mark, Text(_tool_header(e.name, e.args)),
    )
    chip = _status_chip(status, duration_s)
    if chip:
        head.append(" · " + chip, style=TH["warn"] if indeterminate else TH["dim"])
    body = [_fit_line(head, width)]
    # A steer proves that no edit ran. Showing the requested patch body here would look like an applied diff.
    d = None if steered else _diff(e.name, e.args, width)
    if d is not None:
        body.append(Padding(d, (0, 0, 0, 2)))      # indent the diff under the gutter
    show_output = status != "succeeded" or e.name not in QUIET_OUTPUT_TOOLS
    if show_output:
        preview = output_preview(e.output, max_rows=5 if status != "succeeded" else 3)
        out_style = (TH["warn"] if safety_stop or indeterminate
                     else TH["dim"] if steered or cancelled
                     else TH["fail"] if display_failure else TH["dim"])
        for index, line in enumerate(preview.lines):
            if preview.hidden_lines and preview.tail_retained and index == len(preview.lines) - 1:
                body.append(_fit_line(Text.assemble(
                    Text("│   ", style=TH["gutter"]),
                    Text(_omitted_lines(preview.hidden_lines), style=TH["dim"]),
                ), width))
            body.append(_fit_line(Text.assemble(
                Text("│   ", style=TH["gutter"]), Text(line, style=out_style),
            ), width))
        if preview.hidden_lines and not preview.tail_retained:
            body.append(_fit_line(Text.assemble(
                Text("│   ", style=TH["gutter"]),
                Text(f"… {preview.hidden_lines} more lines", style=TH["dim"]),
            ), width))
    return Group(*body)


# The RichSink whose spinner / streaming Live currently owns the terminal. A mid-turn console.input()
# (ask_user) while a Live is active does NOT echo the user's keystrokes — the Live redraws over
# the input line, so the typed answer is invisible. `_pause_active_live()` stops it first; the next event
# restarts a fresh region. Single point so EVERY mid-turn Rich prompt is covered (no per-call-site fix).
_ACTIVE_RICH_SINK = None
# The _EscSentinel (if any) watching for Esc during the current RICH-mode turn — SAME single-choke-point
# idiom as _ACTIVE_RICH_SINK above. Must release the tty raw-mode fd before an ask_user prompt
# does its OWN raw-mode read (_arrow_select), or the two would race for ownership of the same fd.
_ACTIVE_ESC_SENTINEL = None


def _pause_active_live() -> None:
    """Stop the turn spinner AND release the Esc-sentinel's hold on the tty — the ONE choke point every
    mid-turn synchronous ask_user read goes through before touching raw mode itself."""
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
    """Re-arm the Esc-sentinel after a mid-turn ask_user read finishes — the counterpart to
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
    pause(), so it never races a mid-turn ask_user call for the SAME fd. Raw mode is process-global, so only
    one owner may touch the descriptor at a time. The sentinel runs entirely on a second daemon thread; the
    turn itself never leaves the main thread.

    Lifetime: created + started immediately before ONE run_turn() call (RICH mode only, never live mode —
    prompt_toolkit already owns all keystrokes there natively), stopped in a `finally` right after — never
    persists between turns, never leaks a thread."""

    def __init__(self):
        self._thread = None
        self._stop_flag = threading.Event()
        self._pause_flag = threading.Event()
        self._paused_ack = threading.Event()
        self._ready_ack = threading.Event()
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
        # Make the input-ownership transition deterministic. In a real foreground tty, cooked Ctrl-C would
        # already generate SIGINT; in PTYs and unusual hosts it may instead be consumed before the daemon has
        # disabled ISIG. A short readiness handshake closes that startup gap.
        self._ready_ack.wait(timeout=0.5)

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
        self._ready_ack.set()
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
                if data == b"\x1b" or b"\x03" in data:  # bare Esc, or Ctrl-C (\x03/INTR). raw mode
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
        """Release the fd BEFORE returning, so the caller's own synchronous input read can
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


def _fmt_tally(tally: dict) -> str:
    """Compact live activity tally — only non-zero buckets, in a stable order."""
    return " · ".join(
        f"{tally[k]} {k}" for k in ("read", "edit", "cmd", "agent", "steer", "fail")
        if tally.get(k)
    )


_PHASE_STYLE = {
    ProgressPhase.IDLE:        ("·", "Idle", "dim"),
    ProgressPhase.PREPARING:   ("◌", "Preparing", "accent"),
    ProgressPhase.THINKING:    ("◌", "Thinking", "accent"),
    ProgressPhase.WRITING:     ("◌", "Writing", "accent"),
    ProgressPhase.INSPECTING:  ("◌", "Inspecting", "inspect"),
    ProgressPhase.EDITING:     ("◌", "Editing", "edit"),
    ProgressPhase.RUNNING:     ("◌", "Running", "run"),
    ProgressPhase.DELEGATING:  ("◌", "Delegating", "accent"),
    ProgressPhase.WAITING:     ("?", "Waiting", "warn"),
    ProgressPhase.RETRYING:    ("↻", "Retrying", "warn"),
    ProgressPhase.COMPACTING:  ("◌", "Fitting context", "warn"),
    ProgressPhase.INTEGRATING: ("◌", "Integrating", "accent"),
    ProgressPhase.VERIFYING:   ("◌", "Checking", "verify"),
    ProgressPhase.FINALIZING:  ("◌", "Finalizing", "accent"),
    ProgressPhase.SAVING:      ("◌", "Saving", "accent"),
    ProgressPhase.COMPLETE:    ("✓", "Saved", "ok"),
    ProgressPhase.INTERRUPTED: ("!", "Interrupted", "warn"),
    ProgressPhase.FAILED:      ("✗", "Failed", "fail"),
}

_DETAIL_PREFIX = {
    ProgressPhase.PREPARING: "building ", ProgressPhase.WRITING: "drafting ",
    ProgressPhase.INSPECTING: "reading ", ProgressPhase.EDITING: "editing ",
    ProgressPhase.RUNNING: "running ", ProgressPhase.DELEGATING: "delegating ",
    ProgressPhase.INTEGRATING: "integrating ", ProgressPhase.SAVING: "saving ",
}


def _duration(seconds: float) -> str:
    total = max(0, int(seconds))
    if total >= 3600:
        return f"{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}"
    return f"{total // 60:02d}:{total % 60:02d}"


def _fit_line(value: Text, width: int) -> Text:
    """Make a Rich Text a literal, one-physical-line terminal row."""
    line = value.copy()
    line.no_wrap = True
    line.overflow = "ellipsis"
    line.truncate(max(1, int(width)), overflow="ellipsis")
    return line


def _progress_focus(snap) -> str:
    """Show only an actionable plan position; never repeat/pin the user's task prompt."""
    if snap.plan.total:
        if snap.plan.current:
            pos = snap.plan.current_index or min(snap.plan.done + 1, snap.plan.total)
            return f"{pos}/{snap.plan.total} {snap.plan.current}"
        return f"{snap.plan.done}/{snap.plan.total} plan complete"
    return ""


def _render_progress(snap, width: int, *, now: float | None = None,
                     include_glyph: bool = True) -> Text:
    """Responsive progress projection: activity always survives; diagnostics appear only when roomy."""
    now = time.monotonic() if now is None else now
    started = snap.started_at if snap.started_at is not None else now
    glyph, label, style_key = _PHASE_STYLE.get(snap.phase, ("◌", "Working", "accent"))
    detail = " ".join(safe_terminal_text(snap.detail or "", multiline=False).split())
    prefix = _DETAIL_PREFIX.get(snap.phase, "")
    if prefix and detail.lower().startswith(prefix):
        detail = detail[len(prefix):]

    elapsed = _duration(now - started)
    diagnostics = []
    if width >= 110:
        if snap.model_pass:
            diagnostics.append(f"pass {snap.model_pass}")
        if snap.provider_attempt > 1:
            diagnostics.append(f"attempt {snap.provider_attempt}")
        tally = _fmt_tally(snap.counts)
        if tally:
            diagnostics.append(tally)
    suffix_plain = f" · {elapsed}" + ((" · " + " · ".join(diagnostics)) if diagnostics else "")
    prefix_plain = f"{glyph} " if include_glyph else ""
    body_width = max(8, int(width) - len(prefix_plain) - len(suffix_plain))

    activity = Text(label, style=f"bold {TH[style_key]}")
    if detail:
        activity.append(" " + detail)
    focus = safe_terminal_text(_progress_focus(snap), multiline=False)
    body = Text()
    if focus and width >= 72 and body_width >= 28:
        focus_width = min(44, max(12, body_width // 2))
        action_width = max(12, body_width - focus_width - 3)
        focus_text = Text(focus, style="bold")
        focus_text.truncate(focus_width, overflow="ellipsis")
        activity.truncate(action_width, overflow="ellipsis")
        body.append_text(focus_text)
        body.append(" · ", style=TH["dim"])
        body.append_text(activity)
    else:
        activity.truncate(body_width, overflow="ellipsis")
        body.append_text(activity)

    rendered = Text()
    if include_glyph:
        rendered.append(glyph + " ", style=f"bold {TH[style_key]}")
    rendered.append_text(body)
    rendered.append(suffix_plain, style=TH["dim"])
    return _fit_line(rendered, width)


@dataclass(frozen=True)
class _AgentMatrixRow:
    display_id: str
    identity: str
    phase: str
    activity: str
    tool_count: int
    elapsed: str
    last_activity: str
    stable_index: int
    source_coverage_status: str = "not_assessed"
    evidence_status: str = "not_assessed"
    evidence_account: tuple[tuple[str, int], ...] = ()
    attempt: int = 0
    max_attempts: int = 0
    retry_delay_s: float = 0.0
    tool_name: str = ""
    terminal_reason: str = ""
    partial: bool = False
    report_completion: str = "unknown"


_AGENT_PHASE_VIEW = {
    "queued": ("◌", "queued", "warn"),
    "starting": ("◌", "starting", "accent"),
    "awaiting_model": ("◌", "model wait", "accent"),
    "model_active": ("◌", "responding", "accent"),
    "reasoning": ("◌", "reasoning", "accent"),
    "writing": ("◌", "writing", "accent"),
    "running_tool": ("◌", "tool", "accent"),
    "retry_wait": ("↻", "retry", "warn"),
    "settling": ("◌", "finalizing", "accent"),
    "running": ("◌", "working", "accent"),
    "report_ready": ("✓", "ready", "ok"),
    "completed": ("✓", "completed", "warn"),
    "steered": ("↷", "steered", "dim"),
    "failed": ("✗", "failed", "fail"),
    "timed_out": ("!", "timed out", "warn"),
    "cancelled": ("↷", "cancelled", "dim"),
    "indeterminate": ("!", "unknown", "warn"),
}
_AGENT_MATRIX_CAP = 8
_ACTIVE_AGENT_MATRIX_PHASES = {
    "queued", "starting", "awaiting_model", "model_active", "reasoning", "writing", "running_tool",
    "retry_wait", "settling", "running",
}


def _pad_cells(value: object, width: int) -> str:
    value = _shorten_cells(value, width)
    return value + (" " * max(0, width - get_cwidth(value)))


def _agent_matrix_rows(snap, *, now: float | None = None) -> list[_AgentMatrixRow]:
    """Project child topology into deterministic tree-preorder rows.

    Event arrival order is not presentation order.  Top-level and nested rows use their
    provider request ordinals when known (``2.1``, ``2.2``), with deterministic fallbacks.
    Cycles/orphans are rendered once as ``?N`` rather than recursing forever.
    """
    now = time.monotonic() if now is None else now
    agents = tuple(getattr(snap, "subagents", ()) or ())
    if not agents:
        return []
    by_id = {item.agent_id: item for item in agents}
    children: dict[str, list] = {}
    roots = []

    def order(item):
        lifecycle_at = (
            item.started_at if item.started_at is not None
            else getattr(item, "queued_at", None)
        )
        return (
            item.request_ordinal or item.launch_ordinal or 1_000_000,
            item.launch_ordinal or 1_000_000,
            lifecycle_at if lifecycle_at is not None else item.updated_at,
            item.agent_id,
        )

    for item in agents:
        if item.parent_agent_id and item.parent_agent_id in by_id and item.parent_agent_id != item.agent_id:
            children.setdefault(item.parent_agent_id, []).append(item)
        else:
            roots.append(item)
    roots.sort(key=order)
    for values in children.values():
        values.sort(key=order)

    rows: list[_AgentMatrixRow] = []
    visited: set[str] = set()

    def number(value, *, as_float: bool = False):
        try:
            converted = float(value or 0.0) if as_float else int(value or 0)
            return max(0.0, converted) if as_float else max(0, converted)
        except (TypeError, ValueError, OverflowError):
            return 0.0 if as_float else 0

    def activity_for(item, phase: str) -> tuple[str, int, int, float, str, str, bool]:
        """Use typed phase fields; never reverse-engineer child state from prose."""
        attempt = number(getattr(item, "attempt", 0))
        max_attempts = number(getattr(item, "max_attempts", 0))
        retry_delay_s = number(getattr(item, "retry_delay_s", 0.0), as_float=True)
        tool_name = safe_terminal_text(getattr(item, "tool_name", ""), multiline=False)
        terminal_reason = safe_terminal_text(
            getattr(item, "terminal_reason", ""), multiline=False,
        )
        partial = bool(getattr(item, "partial", False))
        report_completion = getattr(item, "report_completion", "unknown")
        attempt_text = (
            f"attempt {attempt}/{max_attempts}" if attempt and max_attempts else
            f"attempt {attempt}" if attempt else ""
        )
        if phase == "queued":
            activity = item.detail or "waiting for agent slot"
        elif phase == "starting":
            activity = "starting"
        elif phase == "awaiting_model":
            activity = "awaiting model" + (f" · {attempt_text}" if attempt_text else "")
        elif phase == "model_active":
            activity = "model responding" + (f" · {attempt_text}" if attempt_text else "")
        elif phase == "reasoning":
            activity = "reasoning" + (f" · {attempt_text}" if attempt_text else "")
        elif phase == "writing":
            activity = "writing report" + (f" · {attempt_text}" if attempt_text else "")
        elif phase == "running_tool":
            activity = "running " + (tool_name or "tool")
        elif phase == "retry_wait":
            bits = ["retry wait"]
            if attempt_text:
                bits.append(attempt_text)
            if retry_delay_s:
                bits.append(f"{retry_delay_s:.1f}s")
            activity = " · ".join(bits)
        elif phase == "settling":
            activity = item.detail or "finalizing outcome"
        elif phase == "report_ready":
            activity = "report ready"
        elif phase == "completed":
            activity = (
                "completed · no report"
                if report_completion == "absent"
                else "completed · report status unknown"
            )
        elif phase in {"steered", "failed", "timed_out", "cancelled", "indeterminate"}:
            activity = terminal_reason or item.detail or phase.replace("_", " ")
            incompleteness = child_incompleteness_label(report_completion, partial)
            if incompleteness:
                activity += f" · {incompleteness}"
        else:  # legacy structured callbacks carry only ``running`` + detail
            activity = item.detail or item.objective or "working"
        return activity, attempt, max_attempts, retry_delay_s, tool_name, terminal_reason, partial

    def append(item, display_id: str) -> None:
        if item.agent_id in visited:
            return
        visited.add(item.agent_id)
        phase = item.phase if item.phase in _AGENT_PHASE_VIEW else "indeterminate"
        identity = item.kind or "agent"
        if item.name:
            identity += f" {item.name}"
        activity, attempt, max_attempts, retry_delay_s, tool_name, terminal_reason, partial = \
            activity_for(item, phase)
        ended = item.finished_at if item.finished_at is not None else now
        began = (
            item.started_at if item.started_at is not None
            else getattr(item, "queued_at", None)
        )
        rows.append(_AgentMatrixRow(
            display_id=display_id, identity=identity, phase=phase,
            activity=safe_terminal_text(activity, multiline=False),
            tool_count=number(item.tool_count),
            elapsed=_duration(max(0.0, ended - (began if began is not None else ended))),
            last_activity=_duration(max(0.0, now - item.updated_at)),
            stable_index=len(rows),
            source_coverage_status=getattr(item, "source_coverage_status", "not_assessed"),
            evidence_status=getattr(item, "evidence_status", "not_assessed"),
            evidence_account=getattr(item, "evidence_account", ()),
            attempt=attempt, max_attempts=max_attempts, retry_delay_s=retry_delay_s,
            tool_name=tool_name, terminal_reason=terminal_reason, partial=partial,
            report_completion=getattr(item, "report_completion", "unknown"),
        ))
        used_child_ids: set[int] = set()
        child_fallback = 0
        for child in children.get(item.agent_id, ()):
            wanted = child.request_ordinal or child.launch_ordinal
            if not wanted or wanted in used_child_ids:
                child_fallback += 1
                while child_fallback in used_child_ids:
                    child_fallback += 1
                wanted = child_fallback
            used_child_ids.add(wanted)
            append(child, f"{display_id}.{wanted}")

    fallback = 0
    used_root_ids: set[int] = set()
    for item in roots:
        wanted = item.request_ordinal or item.launch_ordinal
        if not wanted or wanted in used_root_ids:
            fallback += 1
            while fallback in used_root_ids:
                fallback += 1
            wanted = fallback
        used_root_ids.add(wanted)
        append(item, str(wanted))
    for item in sorted((item for item in agents if item.agent_id not in visited), key=order):
        append(item, f"?{len(rows) + 1}")
    return rows


def _select_agent_matrix_rows(rows: list[_AgentMatrixRow], cap: int = _AGENT_MATRIX_CAP):
    """Keep a bounded live surface while preserving the rows needing attention first."""
    if len(rows) <= cap:
        return rows, []
    adverse = [row for row in rows if row.phase in {"failed", "timed_out", "cancelled", "indeterminate"}]
    steered = [row for row in rows if row.phase == "steered"]
    active = [row for row in rows if row.phase in _ACTIVE_AGENT_MATRIX_PHASES]
    completed = [row for row in rows if row.phase == "completed"]
    ready = [row for row in rows if row.phase == "report_ready"]
    selected = []
    for group in (adverse, completed, steered, active, ready):
        for row in group:
            if row not in selected and len(selected) < cap:
                selected.append(row)
    selected.sort(key=lambda row: row.stable_index)
    selected_ids = {id(row) for row in selected}
    return selected, [row for row in rows if id(row) not in selected_ids]


def _agent_matrix_plain_lines(snap, width: int, *, now: float | None = None,
                              cap: int = _AGENT_MATRIX_CAP) -> list[tuple[str, str]]:
    """Return semantic-style/plain-text rows shared by Rich and prompt-toolkit."""
    if cap < 0:
        return []
    rows = _agent_matrix_rows(snap, now=now)
    if not rows:
        return []
    selected, hidden = _select_agent_matrix_rows(rows, cap)
    counts = Counter(row.phase for row in rows)
    summary = []
    for phase, label in (
        ("queued", "queued"), ("starting", "starting"), ("awaiting_model", "model wait"),
        ("model_active", "responding"),
        ("reasoning", "reasoning"), ("writing", "writing"), ("running_tool", "using tool"),
        ("retry_wait", "retrying"), ("settling", "finalizing"), ("running", "working"),
        ("report_ready", "ready"), ("completed", "completed, no report"),
        ("steered", "steered"),
        ("failed", "failed"), ("timed_out", "timed out"),
        ("cancelled", "cancelled"), ("indeterminate", "unknown"),
    ):
        if counts[phase]:
            summary.append(f"{counts[phase]} {label}")
    source_counts = Counter(
        row.source_coverage_status for row in rows
        if row.source_coverage_status in {"source_complete", "source_partial", "source_unsupported"}
    )
    for status, label in (
        ("source_complete", "source complete"),
        ("source_partial", "source partial"),
        ("source_unsupported", "source unsupported"),
    ):
        if source_counts[status]:
            summary.append(f"{source_counts[status]} {label}")
    # Active children have not produced their evidence account yet. Keep the header focused on settled facts;
    # their column still truthfully shows ``not assessed`` until an authoritative ToolResult arrives.
    evidence_counts = Counter(
        _evidence_key(row.evidence_status)
        for row in rows if row.phase not in _ACTIVE_AGENT_MATRIX_PHASES
    )
    evidence_summary = []
    for status, label in (
        ("content_retained", "retained"), ("content_partial", "partial"),
        ("navigation_only", "navigation"), ("none", "no evidence"),
        ("not_assessed", "not assessed"),
    ):
        if evidence_counts[status]:
            evidence_summary.append(f"{evidence_counts[status]} {label}")
    if evidence_summary:
        summary.append("evidence " + "/".join(evidence_summary))
    lines: list[tuple[str, str]] = [("header", _shorten_cells(
        f"  agents {len(rows)} · " + " · ".join(summary), width,
    ))]
    if width < 31 or not selected:
        # Below the minimum viable row width, preserving a partial identity while dropping state/time is a lie.
        # A zero-row height budget uses the same exact aggregate instead of a useless table header/overflow.
        return [(style, _crop_cells(line, width)) for style, line in lines]

    if width >= 100:
        id_w, state_w, evidence_w, tools_w, last_w, time_w = 6, 12, 12, 5, 7, 7
        agent_w = min(26, max(16, width // 5))
        activity_w = max(
            10, width - (
                2 + id_w + agent_w + state_w + evidence_w + tools_w + last_w + time_w + 7
            ),
        )
        lines.append(("dim", "  " + _pad_cells("id", id_w) + " " + _pad_cells("agent", agent_w)
                      + " " + _pad_cells("state", state_w) + " " + _pad_cells("current", activity_w)
                      + " " + _pad_cells("evidence", evidence_w)
                      + " " + _pad_cells("tools", tools_w) + " " + _pad_cells("last", last_w)
                      + " " + _pad_cells("time", time_w)))
        for row in selected:
            glyph, state, style = _AGENT_PHASE_VIEW[row.phase]
            state_cell = f"{glyph} {state}"
            activity = row.activity
            if row.source_coverage_status in {
                "source_complete", "source_partial", "source_unsupported",
            }:
                activity += " · " + row.source_coverage_status.replace("_", " ")
            row_style = _evidence_style(row.evidence_status) if row.phase == "report_ready" else style
            lines.append((row_style, "  " + _pad_cells(row.display_id, id_w) + " "
                          + _pad_cells(row.identity, agent_w) + " " + _pad_cells(state_cell, state_w)
                          + " " + _pad_cells(activity, activity_w) + " "
                          + _pad_cells(_evidence_token(row.evidence_status, row.evidence_account), evidence_w)
                          + " "
                          + _pad_cells(row.tool_count, tools_w) + " "
                          + _pad_cells(row.last_activity, last_w) + " "
                          + _pad_cells(row.elapsed, time_w)))
    elif width >= 72:
        id_w, agent_w, state_w, evidence_w, age_w = 5, 16, 12, 11, 11
        activity_w = max(7, width - (2 + id_w + agent_w + state_w + evidence_w + age_w + 5))
        lines.append(("dim", "  " + _pad_cells("id", id_w) + " " + _pad_cells("agent", agent_w)
                      + " " + _pad_cells("state", state_w) + " " + _pad_cells("current", activity_w)
                      + " " + _pad_cells("evidence", evidence_w)
                      + " " + _pad_cells("last/time", age_w)))
        for row in selected:
            glyph, state, style = _AGENT_PHASE_VIEW[row.phase]
            state_cell = f"{glyph} {state}"
            activity = row.activity
            if row.source_coverage_status in {
                "source_complete", "source_partial", "source_unsupported",
            }:
                activity += " · " + row.source_coverage_status.replace("_", " ")
            activity += f" · {row.tool_count} tools" if row.tool_count else ""
            row_style = _evidence_style(row.evidence_status) if row.phase == "report_ready" else style
            lines.append((row_style, "  " + _pad_cells(row.display_id, id_w) + " "
                          + _pad_cells(row.identity, agent_w) + " " + _pad_cells(state_cell, state_w)
                          + " " + _pad_cells(activity, activity_w) + " "
                          + _pad_cells(_evidence_token(row.evidence_status, row.evidence_account), evidence_w)
                          + " "
                          + _pad_cells(f"{row.last_activity}/{row.elapsed}", age_w)))
    else:
        # Narrow mode keeps identity, state, and elapsed as distinct non-droppable columns.
        id_w, time_w = 5, 5
        if width >= 52:
            state_w = 11
            age_w = 11
            agent_w = min(18, max(9, width // 5))
            activity_w = max(6, width - (2 + id_w + agent_w + state_w + age_w + 4))
            for row in selected:
                glyph, state, style = _AGENT_PHASE_VIEW[row.phase]
                activity = row.activity
                activity += " · e:" + _evidence_token(row.evidence_status, row.evidence_account)
                if row.source_coverage_status in {
                    "source_complete", "source_partial", "source_unsupported",
                }:
                    activity += " · " + row.source_coverage_status.replace("_", " ")
                row_style = _evidence_style(row.evidence_status) if row.phase == "report_ready" else style
                lines.append((row_style, "  " + _pad_cells(row.display_id, id_w) + " "
                              + _pad_cells(row.identity, agent_w) + " "
                              + _pad_cells(f"{glyph} {state}", state_w) + " "
                              + _pad_cells(activity, activity_w) + " "
                              + _pad_cells(f"{row.last_activity}/{row.elapsed}", age_w)))
        else:
            state_w = 10
            agent_w = max(6, width - (2 + id_w + state_w + time_w + 3))
            for row in selected:
                glyph, state, style = _AGENT_PHASE_VIEW[row.phase]
                row_style = _evidence_style(row.evidence_status) if row.phase == "report_ready" else style
                lines.append((row_style, "  " + _pad_cells(row.display_id, id_w) + " "
                              + _pad_cells(row.identity, agent_w) + " "
                              + _pad_cells(f"{glyph} {state}", state_w) + " "
                              + _pad_cells(row.elapsed, time_w)))
    if hidden:
        hidden_counts = Counter(row.phase for row in hidden)
        facts = []
        for phase, label in (("awaiting_model", "model wait"), ("model_active", "responding"),
                             ("reasoning", "reasoning"),
                             ("writing", "writing"), ("running_tool", "using tool"),
                             ("retry_wait", "retrying"), ("settling", "finalizing"),
                             ("running", "working"),
                             ("starting", "starting"), ("queued", "queued"),
                             ("report_ready", "ready"), ("completed", "completed, no report"),
                             ("steered", "steered"),
                             ("failed", "failed"), ("timed_out", "timed out"),
                             ("cancelled", "cancelled"), ("indeterminate", "unknown")):
            if hidden_counts[phase]:
                facts.append(f"{hidden_counts[phase]} {label}")
        lines.append(("dim", _shorten_cells(
            f"  … {len(hidden)} hidden" + ((" · " + " · ".join(facts)) if facts else ""), width,
        )))
    return [(style, _crop_cells(line, width)) for style, line in lines]


def _render_agent_matrix(snap, width: int, *, now: float | None = None) -> Group | None:
    lines = _agent_matrix_plain_lines(snap, width, now=now)
    if not lines:
        return None
    rich_lines = []
    for semantic, line in lines:
        if semantic == "header":
            style = f"bold {TH['accent']}"
        elif semantic == "dim":
            style = TH["dim"]
        else:
            style = TH.get(semantic, TH["tool"])
        rich_lines.append(Text(line, style=style))
    return Group(*rich_lines)


def _render_completion(snap, event: TurnCommitted, width: int, *, now: float | None = None) -> Text | Group:
    now = time.monotonic() if now is None else now
    stop_reason = event.stop_reason or snap.stop_reason or "turn"
    if not event.ok:
        return _fit_line(Text.assemble(
            Text("✗ ", style=f"bold {TH['fail']}"), Text("save failed"),
            Text(" · " + _shorten(event.detail or stop_reason, 120), style=TH["dim"]),
        ), width)
    started = snap.started_at if snap.started_at is not None else now
    label = receipt_completion_label(event.receipt, stop_reason)
    receipt_parts = receipt_summary_parts(event.receipt)
    disposition = str((event.receipt or {}).get("disposition") or "")
    attention = disposition in {"completed_with_warnings", "indeterminate"}
    glyph, style = ("!", TH["warn"]) if attention else ("✓", TH["ok"])
    if receipt_parts and receipt_has_adverse_lifecycle(event.receipt):
        # Adverse lifecycle truth gets self-contained rows. Plan/pass/time are intentionally omitted:
        # terminal cropping must never hide which family failed or was rejected.
        heading = _fit_line(Text.assemble(
            Text(glyph + " ", style=f"bold {style}"), Text(label),
        ), width)
        lifecycle = []
        for part in receipt_parts:
            row = Text("  ", style=TH["dim"])
            row.append(part, style=TH["dim"])
            lifecycle.append(_fit_line(row, width))
        return Group(heading, *lifecycle)
    details = []
    if snap.plan.total:
        details.append(f"plan {snap.plan.done}/{snap.plan.total}")
    if snap.model_pass:
        details.append(f"{snap.model_pass} pass{'es' if snap.model_pass != 1 else ''}")
    if receipt_parts:
        details.extend(receipt_parts)
    elif event.receipt is None:
        tally = _fmt_tally(snap.counts)
        if tally:
            details.append(tally)
    details.append(_duration(now - started))
    return _fit_line(Text.assemble(
        Text(glyph + " ", style=f"bold {style}"), Text(label),
        Text(" · " + " · ".join(details), style=TH["dim"]),
    ), width)


def _interruption_label(reason: str) -> str:
    """Translate runtime stop taxonomy into user language instead of calling every stop an interruption."""
    reason = str(reason or "").lower()
    if reason in ("stuck", "blocked", "overflow"):
        return "stopped"
    if reason in ("max_steps", "max_tokens", "token_budget", "filtered"):
        return "paused"
    if reason == "error":
        return "failed"
    if reason == "indeterminate":
        return "attention needed"
    return "interrupted"


class _LiveStatus:
    """Self-refreshing Rich projection of progress plus the per-child matrix."""

    def __init__(self, sink: "RichSink"):
        self._sink = sink

    def __rich__(self):
        try:
            snap = self._sink.progress.snapshot()
            # Rich Status contributes the animated activity glyph; reserve four cells for it.
            width = max(8, int(getattr(self._sink.c, "width", 100) or 100) - 4)
            progress = _render_progress(snap, width, include_glyph=False)
            matrix = _render_agent_matrix(snap, width)
            return Group(progress, matrix) if matrix is not None else progress
        except Exception:  # noqa: BLE001 — a progress indicator must never break the run
            return Text("working…", style=TH["dim"])


class _ToolTiming:
    """Correlate concurrent starts/results by invocation ID, with a legacy fallback."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._started: dict[str, tuple[str, float]] = {}
        self._legacy_seq = 0

    def clear(self) -> None:
        self._started.clear()
        self._legacy_seq = 0

    def start(self, event: ToolStarted) -> None:
        key = invocation_id(event)
        if not key:
            self._legacy_seq += 1
            key = f"legacy-{self._legacy_seq}"
        self._started[key] = (event.name, self._clock())

    def settle(self, event: ToolResult, *, ended_at: float | None = None) -> float | None:
        key = invocation_id(event)
        if key:
            value = self._started.pop(key, None)
            if value is None:
                return None
            _name, started = value
            return max(0.0, (self._clock() if ended_at is None else ended_at) - started)
        key = next((candidate for candidate, value in reversed(tuple(self._started.items()))
                    if value[0] == event.name), None)
        if key is None:
            return None
        _name, started = self._started.pop(key)
        return max(0.0, (self._clock() if ended_at is None else ended_at) - started)


class _EventSinkCore:
    """Shared event orchestration for the Rich and pinned terminal surfaces.

    The adapters still own how live status is displayed.  Event reduction, result
    de-duplication, buffering, and durable scrollback are deliberately single-owner
    so adding a lifecycle event cannot make the two terminal modes diverge.
    """

    def _init_event_state(self, console: Console, stats: dict, *, await_commit: bool) -> None:
        self.c = console
        self.stats = stats
        self.progress = TurnProgress(await_commit=await_commit)
        self._await_commit = await_commit
        self._lock = threading.RLock()
        self._pending_answer = ""
        self._reads: list = []
        self._agents: list[AgentResultView] = []
        self._timing = _ToolTiming()
        self._presented_result_ids: set[str] = set()

    def _accept_events(self) -> bool:
        return True

    def _before_terminal_event(self, _event: Event) -> None:
        """Adapter hook for tearing down a live status owner before scrollback."""

    def _after_turn_started(self) -> None:
        """Adapter hook for mode-specific per-turn presentation state."""

    def _flush_reads(self) -> None:
        with self._lock:
            if not self._reads:
                return
            reads, self._reads = self._reads, []
            rendered = _render_read_summary(reads, _line_width(self.c))
            if rendered is not None:
                self.c.print(rendered)

    def _flush_agents(self) -> None:
        with self._lock:
            agents, self._agents = self._agents, []
            rendered = _render_agent_batch(agents, _line_width(self.c))
            if rendered is not None:
                self.c.print(rendered)

    def _flush_answer(self, *, title: str = "assistant", border_style: str | None = None) -> None:
        with self._lock:
            content, self._pending_answer = self._pending_answer, ""
            if content.strip():
                self.c.print(_response_panel(content, self.c, title=title, border_style=border_style))

    def subagent_notify(self, update: SubagentProgress | str) -> None:
        try:
            with self._lock:
                if not self._accept_events():
                    return
                self.progress.subagent_activity(update)
                self._sync_status()
        except Exception:  # noqa: BLE001 — presentation must never break a child worker
            pass

    def on_delta(self, kind: str, text: str) -> None:
        with self._lock:
            if not self._accept_events():
                return
            self.progress.on_delta(kind, text)
            self._sync_status()

    def __call__(self, event: Event) -> None:
        with self._lock:
            if not self._accept_events():
                return
            self._handle_event(event)

    def _handle_event(self, event: Event) -> None:
        before = self.progress.snapshot()
        was_active = before.active
        after = self.progress.reduce(event)
        result_view = project_tool_result(event) if isinstance(event, ToolResult) else None
        milestone = bool(
            result_view is not None and result_view.succeeded and after.last_milestone
            and after.last_milestone != before.last_milestone
        )
        if isinstance(event, TurnStarted):
            self._pending_answer = ""
            self._reads = []
            self._agents = []
            self._timing.clear()
            self._presented_result_ids.clear()
            self._after_turn_started()
        elif isinstance(event, ToolStarted):
            self._timing.start(event)
        elif isinstance(event, StepBegin):
            self._pending_answer = ""
        elif isinstance(event, ToolResult):
            assert result_view is not None
            result_id = result_view.invocation_id
            if result_id and result_id in self._presented_result_ids:
                self._sync_status()
                return
            if result_id:
                self._presented_result_ids.add(result_id)
            if result_view.is_delegation:
                self._flush_reads()
                agent = project_agent_result(event)
                finished_at = _accepted_agent_finished_at(before, agent)
                if finished_at is None:
                    self._timing.settle(event)
                    duration_s = None
                else:
                    duration_s = self._timing.settle(event, ended_at=finished_at)
                _buffer_agent_once(
                    self._agents, project_agent_result(event, duration_s=duration_s),
                )
            elif result_view.name in _COALESCE and result_view.succeeded:
                self._flush_agents()
                self._timing.settle(event)
                self._reads.append((event.name, _primary(event.name, event.args)))
            else:
                self._flush_agents()
                self._flush_reads()
                duration_s = self._timing.settle(event)
                self.c.print(_render_tool_result(event, _line_width(self.c), duration_s=duration_s))
            if milestone:
                self._flush_agents()
                self._flush_reads()
                self.c.print(_render_milestone(after.last_milestone, _line_width(self.c)))
        elif isinstance(event, AssistantText):
            self._flush_agents()
            self._flush_reads()
            if (event.content or "").strip():
                if event.final and self._await_commit and was_active:
                    self._pending_answer = event.content
                elif not event.final:
                    self.c.print(_render_assistant_update(event.content, _line_width(self.c)))
                else:
                    self.c.print(_response_panel(event.content, self.c))
        elif isinstance(event, ApiRetry):
            self._flush_agents()
            self._flush_reads()
            delay = f" in {event.delay_s:.1f}s" if event.delay_s > 0 else ""
            self.c.print(_fit_line(Text.assemble(
                Text("│ ", style=TH["gutter"]), Text("↻ ", style=TH["warn"]),
                Text(
                    f"model retry {event.attempt + 1}/{event.max_attempts}{delay}",
                    style=TH["warn"],
                ),
                Text(f" · {_shorten(event.error, 60)}", style=TH["dim"]),
            ), _line_width(self.c)))
        elif isinstance(event, StepEnd):
            _record_usage(self.stats, event.usage or {})
            self._flush_agents()
            if event.stop_reason == "tool_use":
                self._flush_reads()
        elif isinstance(event, LessonSaved):
            self._flush_agents()
            self._flush_reads()
            self.c.print(_fit_line(Text.assemble(
                Text("│ ", style=TH["gutter"]), Text("◆ ", style=TH["accent"]),
                Text("learned · ", style=TH["dim"]), Text(_shorten(event.title, 100)),
            ), _line_width(self.c)))
        elif isinstance(event, TurnInterrupted):
            self._before_terminal_event(event)
            self._flush_agents()
            self._flush_reads()
            self._flush_answer(title="assistant · partial", border_style=TH["warn"])
            self.c.print(_fit_line(Text.assemble(
                Text("! ", style=f"bold {TH['warn']}"),
                Text(_interruption_label(event.reason), style=TH["warn"]),
                Text(f" · {_shorten(event.message or event.reason, 180)}", style=TH["dim"]),
            ), _line_width(self.c)))
        elif isinstance(event, TurnEnd):
            self._flush_agents()
            self._flush_reads()
            if not self._await_commit:
                self._before_terminal_event(event)
                self._flush_answer()
                self.c.print(_fit_line(Text.assemble(
                    Text("✓ ", style=f"bold {TH['ok']}"), Text("turn finished"),
                    Text(
                        f" · {event.steps} pass{'es' if event.steps != 1 else ''}",
                        style=TH["dim"],
                    ),
                ), _line_width(self.c)))
        elif isinstance(event, TurnCommitted):
            self._before_terminal_event(event)
            self._flush_agents()
            self._flush_reads()
            if event.ok:
                self._flush_answer()
            else:
                self._flush_answer(title="assistant · unsaved", border_style=TH["warn"])
            self.c.print(_render_completion(after, event, _line_width(self.c)))
        self._sync_status()


class RichSink(_EventSinkCore):
    """Canonical Rich renderer over one UI-neutral progress reducer."""

    def __init__(self, console: Console, stats: dict, *, await_commit: bool = False):
        global _ACTIVE_RICH_SINK
        _ACTIVE_RICH_SINK = self        # so ask_user can pause the live region before reading input
        self._init_event_state(console, stats, await_commit=await_commit)
        self._status = None
        self._spinner_on = os.environ.get("AGENT_SPINNER", "on").strip().lower() not in ("off", "0", "false", "no")
        self._body = None
        self._last_plain_status = ""

    def _before_terminal_event(self, _event: Event) -> None:
        self._stop()

    def _after_turn_started(self) -> None:
        self._last_plain_status = ""

    def _stop(self) -> None:
        """Tear down only the renderer; the shared semantic state remains intact."""
        with self._lock:
            if self._status is not None:
                self._status.stop()
                self._status = None
            self._body = None

    def _sync_status(self) -> None:
        """Project the latest reducer snapshot without independently interpreting the event."""
        with self._lock:
            snap = self.progress.snapshot()
            if not snap.active:
                self._stop()
                return
            if self._spinner_on and self._status is None:
                self._body = _LiveStatus(self)
                self._status = self.c.status(self._body, spinner="dots")
                self._status.start()
            elif not self._spinner_on:
                line = _render_progress(snap, _line_width(self.c))
                if line.plain != self._last_plain_status:
                    self.c.print(line)
                    self._last_plain_status = line.plain

def make_rich_sink(console: Console, stats: dict, *, await_commit: bool = False) -> RichSink:
    return RichSink(console, stats, await_commit=await_commit)


class LiveSink(_EventSinkCore):
    """Pinned-composer renderer over the same progress reducer as :class:`RichSink`."""

    def __init__(self, console: Console, stats: dict, set_status, *, await_commit: bool = False,
                 set_surface=None):
        self._init_event_state(console, stats, await_commit=await_commit)
        self._set_status = set_status
        self._set_surface = set_surface
        self._retired = False

    def _accept_events(self) -> bool:
        return not self._retired

    def _sync_status(self) -> None:
        with self._lock:
            if self._retired:
                return
            snap = self.progress.snapshot()
            status = _render_progress(snap, _line_width(self.c)).plain if snap.active else None
            if self._set_surface is not None:
                self._set_surface(status, snap if snap.active else None)
            else:
                self._set_status(status)

    def retire(self) -> None:
        """Permanently detach this turn renderer from the shared live composer."""
        with self._lock:
            self._retired = True


class LiveWorkerRetirementError(RuntimeError):
    """The live renderer died while its turn worker still owned mutable runtime state."""


def _live_status_line(status: str, stats: dict, width: int) -> FormattedText:
    """Compose active semantic progress while reserving room for cost orientation."""
    status = status or "◌ Working"
    glyph = status[0] if status[0] in "◌?↻!✗✓" else "◌"
    body = status[1:].lstrip() if status[0] == glyph else status
    glyph_style = {
        "?": "fg:ansiyellow bold", "↻": "fg:ansiyellow bold",
        "!": "fg:ansiyellow bold", "✗": "fg:ansired bold", "✓": "fg:ansigreen bold",
    }.get(glyph, "fg:ansibrightcyan bold")
    width = max(12, int(width or 100))
    tokens = f"{_compact_count(stats.get('tokens', 0))} tok"
    savings = _savings_label(stats)
    meter_cells = get_cwidth(tokens) + get_cwidth(savings) + 6
    if width >= 52 and meter_cells <= width - 18:
        body_width = max(10, width - 5 - meter_cells)
        return FormattedText([
            (glyph_style, f"  {glyph} "),
            ("", _shorten_cells(body, body_width)),
            ("fg:ansibrightblack", " · "),
            ("fg:ansicyan", tokens),
            ("fg:ansibrightblack", " · "),
            ("fg:ansigreen", savings),
        ])
    return FormattedText([
        (glyph_style, f"  {glyph} "),
        ("", _shorten_cells(body, max(8, width - 5))),
    ])


def _live_status_surface(status: str, snap, stats: dict, width: int,
                         *, now: float | None = None, matrix_cap: int = _AGENT_MATRIX_CAP) -> FormattedText:
    """Multi-line prompt-toolkit surface; finalized output still goes to scrollback once."""
    fragments = list(_live_status_line(status, stats, width))
    if snap is None:
        return FormattedText(fragments)
    prompt_style = {
        "header": "fg:ansibrightcyan bold", "dim": "fg:ansibrightblack",
        "accent": "fg:ansibrightcyan", "ok": "fg:ansigreen",
        "fail": "fg:ansired", "warn": "fg:ansiyellow", "tool": "",
    }
    for semantic, line in _agent_matrix_plain_lines(snap, width, now=now, cap=matrix_cap):
        fragments.append(("", "\n"))
        fragments.append((prompt_style.get(semantic, ""), line))
    return FormattedText(fragments)


def build_live_app(*, console: Console, stats: dict, root: str | None, run_one_turn, handle_slash=None,
                   pt_input=None, pt_output=None):
    """Build the LIVE composer Application (split out from run_live so a test can drive it with a pipe input).
    Returns (app, state). state = {status, running, signal, last} so a test can inspect what happened.
    run_one_turn(text, sink, signal) executes ONE turn synchronously; it runs in a daemon worker thread so
    the bordered input box stays pinned + responsive WHILE the agent streams output above it."""
    import threading
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Float, FloatContainer, HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.menus import MultiColumnCompletionsMenu
    from prompt_toolkit.widgets import Frame, TextArea

    state = {"status": "", "status_override": False, "progress": None,
             "running": False, "signal": None, "last": None, "threads": [],
             "input_request": None, "exit_when_idle": False, "closing": False,
             "root": root, "suspended_slash": "", "status_owner": 0, "sink_generation": 0}
    state_lock = threading.RLock()
    app_ref = {"value": None}
    toolbar = _toolbar(stats, lambda: console.width)

    def _matrix_cap(width: int) -> int:
        output = getattr(app_ref.get("value"), "output", None) or pt_output
        try:
            rows = int(output.get_size().rows) if output is not None else 24
        except Exception:
            rows = 24
        # Reserve three rows for the framed composer and one for the primary progress line. Matrix overhead is
        # summary + optional column header + overflow; child rows consume the remainder up to the global cap.
        # A negative cap means even the aggregate is hidden on a four-row emergency terminal.
        matrix_budget = max(0, rows - 4)
        if matrix_budget == 0:
            return -1
        if matrix_budget < 3:
            return 0
        overhead = 1 + (1 if width >= 72 else 0) + 1
        return max(0, min(_AGENT_MATRIX_CAP, matrix_budget - overhead))

    def set_status(text):                  # called from the worker thread; invalidate() is thread-safe
        with state_lock:
            state["status"] = text or ""
            state["status_override"] = True
            # A host message may replace the headline, but active child truth remains visible below it.
            # Clearing this snapshot during Ctrl-C/input made a healthy fan-out appear to vanish until the
            # next child callback happened to repaint it.
        try:
            app.invalidate()
        except Exception:
            pass

    def set_running_status(text, *, owner: int, signal) -> bool:
        """Write a key-handler status only while the same turn still owns the composer.

        Waking a worker can let its ``finally`` retire the owner before the UI handler's next
        statement.  An unowned write in that gap used to pin "Continuing"/"Interrupt requested"
        forever on an idle composer.
        """
        with state_lock:
            if (not state.get("running") or state.get("status_owner") != owner
                    or state.get("signal") is not signal):
                return False
            state["status"] = text or ""
            state["status_override"] = True
        try:
            app.invalidate()
        except Exception:
            pass
        return True

    def _status_line():
        with state_lock:
            running, status, progress = state["running"], state["status"], state["progress"]
            status_override = state["status_override"]
        if running or status:
            width = max(12, int(getattr(console, "width", 100) or 100))
            # Re-project from the frozen semantic snapshot on every prompt-toolkit heartbeat so elapsed
            # time advances even while one child is quiet inside a model call.
            live_status = (status if status_override else _render_progress(progress, width).plain) \
                if progress is not None else status
            return _live_status_surface(
                live_status or "◌ Working", progress, stats, width, matrix_cap=_matrix_cap(width),
            )
        return toolbar()                   # idle → stable product/workspace/model identity

    def _status_height():
        with state_lock:
            running, status, progress = state["running"], state["status"], state["progress"]
        if not (running or status) or progress is None:
            return 1
        width = max(12, int(getattr(console, "width", 100) or 100))
        return 1 + len(_agent_matrix_plain_lines(progress, width, cap=_matrix_cap(width)))

    # Read-only diagnostics used by headless renderer tests; the live UI itself consumes the same callbacks.
    state["render_status"] = _status_line
    state["status_height"] = _status_height

    ta = TextArea(prompt="❯ ", multiline=False, wrap_lines=True,
                  history=_private_prompt_history(),
                  completer=_InputCompleter(_repo_files(root) if root else None),
                  complete_while_typing=True)

    def set_workspace(new_root: str | None) -> None:
        """Refresh only workspace-derived completion; the live Application and model session stay alive."""
        state["root"] = new_root
        completer = _InputCompleter(_repo_files(new_root) if new_root else None)
        ta.completer = completer
        ta.buffer.completer = completer
        try:
            app.invalidate()
        except Exception:
            pass

    state["set_workspace"] = set_workspace

    def _on_ui_thread(callback) -> bool:
        """Run a composer mutation on the UI thread, or abandon it when that UI is retiring.

        A worker must never wait forever for a callback queued just as ``Application.run`` exits. The bounded
        wait is only for this tiny in-memory handoff; it does not impose a deadline on model/tool work.
        """
        if state.get("closing"):
            return False
        loop = getattr(app, "loop", None)
        loop_thread = getattr(app, "_loop_thread", None)
        if loop is None or threading.current_thread() is loop_thread:
            callback()
            return True
        done = threading.Event()
        abandoned = threading.Event()
        error = []

        def apply():
            if abandoned.is_set() or state.get("closing"):
                done.set()
                return
            try:
                callback()
            except Exception as exc:  # noqa: BLE001 - transport back to the requesting worker
                error.append(exc)
            finally:
                done.set()

        try:
            loop.call_soon_threadsafe(apply)
        except (RuntimeError, AttributeError):
            return False
        deadline = time.monotonic() + 2.0
        while not done.wait(0.05):
            try:
                future = getattr(app, "future", None)
                app_done = bool(future is not None and future.done())
            except Exception:
                app_done = False
            try:
                loop_closed = bool(loop.is_closed())
            except Exception:
                loop_closed = False
            if state.get("closing") or app_done or loop_closed or time.monotonic() >= deadline:
                abandoned.set()
                return False
        if error:
            raise error[0]
        return True

    def request_input(question: str, options=None) -> str:
        """Let the worker pause for one answer without competing with the live Application for stdin.

        The existing composer becomes the input surface: the worker publishes a request, Enter resolves it,
        and the turn continues on the same thread. It backs the model's explicit ask_user capability.
        """
        pending = {
            "question": safe_terminal_text(question or "Input needed", multiline=False),
            "options": tuple(safe_terminal_text(option, multiline=False) for option in (options or ())),
            "answer": "",
            "event": threading.Event(),
            "draft": "",
            "accepting": False,
        }

        def activate_prompt():
            # Text typed while the model was working is a draft, never implicit consent. Snapshot and clear it
            # atomically before exposing the request so no keystroke can be lost or mistaken for approval.
            pending["draft"] = ta.text
            ta.text = ""
            state["input_request"] = pending

        if not _on_ui_thread(activate_prompt):
            return ""
        console.print(Text.assemble(
            Text("? ", style=f"bold {TH['warn']}"),
            Text(pending["question"], style="bold"),
        ))
        if pending["options"]:
            console.print(Text(
                "  " + "  ·  ".join(f"{index}. {option}" for index, option in enumerate(pending["options"], 1)),
                style=TH["dim"],
            ))
        set_status("? Input needed · type your answer and press Enter")

        def enable_prompt():
            # Anything typed before the question became visible is still a draft, not an answer. Quarantine it
            # and enable Enter only after the complete question/options/status have rendered.
            early = ta.text
            if early:
                prior = str(pending.get("draft") or "")
                pending["draft"] = (prior + ("\n" if prior else "") + early)
                ta.text = ""
            pending["accepting"] = True

        if not _on_ui_thread(enable_prompt):
            if state.get("input_request") is pending:
                state["input_request"] = None
            pending["event"].set()
            return ""
        while not pending["event"].wait(0.1):
            signal = state.get("signal")
            if signal is not None and signal.is_set():
                pending["answer"] = ""
                pending["event"].set()
        if state.get("input_request") is pending:
            state["input_request"] = None
        draft = str(pending.get("draft") or "")
        if draft:
            def restore_draft():
                if not ta.text:
                    ta.text = draft
            if _on_ui_thread(restore_draft):
                app.invalidate()
        return str(pending.get("answer") or "")

    state["request_input"] = request_input
    kb = KeyBindings()

    @kb.add("enter")
    def _(ev):
        pending = state.get("input_request")
        if pending is not None:
            if not pending.get("accepting"):
                return
            answer = ta.text.strip()
            if not answer:
                return
            ta.text = ""
            options = pending.get("options") or ()
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                answer = options[int(answer) - 1]
            pending["answer"] = answer
            state["input_request"] = None
            with state_lock:
                owner, signal = state.get("status_owner", 0), state.get("signal")
            pending["event"].set()
            set_running_status("◌ Continuing · applying your answer", owner=owner, signal=signal)
            ev.app.invalidate()
            return
        with state_lock:
            if state["running"]:           # one turn at a time — only a pending input request accepts Enter
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
        elif text in {"/config", "/model"} and handle_slash is not None:
            # These commands own an interactive wizard/selector. A prompt_toolkit Application cannot safely
            # nest another one, so ask run_live to retire this idle composer, run the modal at a clean terminal
            # boundary, and resume with the same model/session and current workspace.
            state["suspended_slash"] = text
            ev.app.exit()
            return
        elif text.startswith("/") and handle_slash is not None:
            handle_slash(text)
            return
        user_echo(console, text)           # echo ABOVE the box (instant), THEN run the turn
        state["last"] = text
        with state_lock:
            state["sink_generation"] += 1
            owner = state["sink_generation"]

        def owned_status(value):
            with state_lock:
                if state.get("status_owner") != owner:
                    return
                state["status"] = value or ""
                state["status_override"] = False
                state["progress"] = None
            try:
                app.invalidate()
            except Exception:
                pass

        def owned_surface(value, progress):
            with state_lock:
                if state.get("status_owner") != owner:
                    return
                signal = state.get("signal")
                hold_host_headline = bool(
                    state.get("status_override")
                    and (state.get("input_request") is not None
                         or (signal is not None and signal.is_set()))
                )
                if not hold_host_headline:
                    state["status"] = value or ""
                    state["status_override"] = False
                state["progress"] = progress
            try:
                app.invalidate()
            except Exception:
                pass

        sink = LiveSink(
            console, stats, owned_status, await_commit=True, set_surface=owned_surface,
        )
        sig = threading.Event()
        with state_lock:
            state["status_owner"] = owner
            state["running"] = True
            state["signal"] = sig
            state["status"] = "◌ Preparing · starting turn"
            state["status_override"] = False
            state["progress"] = None

        def _work():
            try:
                run_one_turn(text, sink, sig)
            except Exception as exc:       # a turn crash must NOT kill the composer
                console.print(_fit_line(Text.assemble(
                    Text("✗ ", style=f"bold {TH['fail']}"), Text("turn error", style=TH["fail"]),
                    Text(f" · {type(exc).__name__}: {_shorten(str(exc), 160)}", style=TH["dim"]),
                ), _line_width(console)))
                if getattr(exc, "stop_session", False):
                    try:
                        app.exit()
                    except Exception:
                        pass
            finally:
                sink.retire()
                with state_lock:
                    if state.get("status_owner") == owner:
                        state["status_owner"] = 0
                        state["status"] = ""
                        state["status_override"] = False
                        state["progress"] = None
                    state["signal"] = None
                    # This is deliberately last: Enter cannot start a new owner in
                    # the old sink's clear-status window.
                    state["running"] = False
                try:
                    app.invalidate()
                except Exception:
                    pass
                if state.get("exit_when_idle"):
                    try:
                        app.exit()
                    except Exception:
                        pass
        state["threads"] = [t for t in state["threads"] if t.is_alive()]   # prune finished workers (no unbounded growth)
        th = threading.Thread(target=_work, daemon=True)
        state["threads"].append(th)
        th.start()
        ev.app.invalidate()

    @kb.add("c-j")
    def _(ev):
        # Match the inline composer: Ctrl-J composes a literal newline, while Enter alone submits.  Without
        # this binding prompt_toolkit treats Ctrl-J as an accept key in a single-line TextArea, splitting one
        # intended multiline request into multiple turns.
        if state.get("input_request") is None:
            ta.buffer.insert_text("\n")

    @kb.add("c-c")
    def _(ev):
        with state_lock:
            running, signal = state["running"], state["signal"]
            owner = state.get("status_owner", 0)
        if running and signal is not None:
            signal.set()          # abort the running turn at the next step boundary
            pending = state.get("input_request")
            if pending is not None:
                pending["answer"] = ""
                pending["event"].set()
                state["input_request"] = None
            set_running_status(
                "! Interrupt requested · waiting for current operation", owner=owner, signal=signal,
            )
        else:
            ev.app.exit()

    @kb.add("c-d")
    def _(ev):
        if state["running"]:
            # Never let the application return while its daemon turn still owns workspace/process state.
            # Request cancellation, release a pending ask_user request, then let the worker exit the app from
            # its finally block after the turn has honestly settled.
            state["exit_when_idle"] = True
            with state_lock:
                signal, owner = state["signal"], state.get("status_owner", 0)
            if signal is not None:
                signal.set()
            pending = state.get("input_request")
            if pending is not None:
                pending["answer"] = ""
                pending["event"].set()
                state["input_request"] = None
            set_running_status(
                "! Exit requested · waiting for current operation", owner=owner, signal=signal,
            )
            return
        ev.app.exit()

    @kb.add("escape")
    def _(ev):                             # mid-turn: abort (same as ctrl-c); idle: clear the line, or undo
        with state_lock:
            running, signal = state["running"], state["signal"]
            owner = state.get("status_owner", 0)
        if running:
            if signal is not None:
                signal.set()      # abort the running turn at the next step boundary
                set_running_status(
                    "! Interrupt requested · waiting for current operation", owner=owner, signal=signal,
                )
            return
        if ta.text.strip():
            ta.text = ""
        elif handle_slash is not None:
            handle_slash("/undo")

    app = Application(
        layout=Layout(FloatContainer(
            content=HSplit([Frame(ta),
                            Window(FormattedTextControl(_status_line), height=_status_height,
                                   dont_extend_height=True, wrap_lines=False)]),
            floats=[Float(xcursor=True, ycursor=True,
                          content=MultiColumnCompletionsMenu(min_rows=3, show_meta=True))],
        ), focused_element=ta),
        key_bindings=kb, full_screen=False, mouse_support=False, input=pt_input, output=pt_output,
        min_redraw_interval=0.05, refresh_interval=0.5)
    app_ref["value"] = app
    return app, state


def run_live(*, console: Console, stats: dict, banner_info: str, root: str | None,
             run_one_turn, handle_slash=None, handle_modal_slash=None, on_ready=None,
             worker_retire_timeout: float = 5.0) -> None:
    """The LIVE composer (AGENT_TUI=live): a bordered input box stays pinned at the bottom EVEN WHILE the
    agent streams — output prints above it in the NORMAL terminal buffer (native copy/paste preserved), the
    Python analogue of Ink's <Static>+live-region. ctrl-c aborts a running turn; ctrl-c at idle / ctrl-d quits."""
    from prompt_toolkit.patch_stdout import patch_stdout

    current_root = root
    show_banner = True
    while True:
        app, _state = build_live_app(
            console=console, stats=stats, root=current_root,
            run_one_turn=run_one_turn, handle_slash=handle_slash,
        )
        try:
            # Bridge installation is startup work too. If the callback partially mutates host state and raises,
            # the same finally boundary below still retires it.
            if on_ready is not None:
                on_ready(_state.get("set_workspace"), _state.get("request_input"))
            # Banner rendering is part of live startup. Keep it inside the same retirement boundary as app.run:
            # a terminal/renderer failure here must not leave the host's workspace/input bridges pointing at an
            # Application that never started. Modal resumes do not repaint startup chrome.
            if show_banner:
                banner(console, banner_info)
                show_banner = False
            with patch_stdout(raw=True):
                app.run()
        finally:
            # Any exit path (EOF, renderer exception, startup/fallback failure) must retire the worker before the
            # caller can reuse or clean up its workspace. A daemon turn surviving into the inline fallback would
            # create two concurrent owners of the same session state.
            _state["closing"] = True
            signal = _state.get("signal")
            if signal is not None:
                signal.set()
            pending = _state.get("input_request")
            if pending is not None:
                pending["answer"] = ""
                pending["event"].set()
                _state["input_request"] = None
            deadline = time.monotonic() + max(0.0, float(worker_retire_timeout))
            workers = tuple(_state.get("threads") or ())
            for thread in workers:
                if thread.is_alive():
                    thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if on_ready is not None:
                try:
                    on_ready(None, None)
                except Exception:
                    # Retirement must continue even if the host callback itself is what broke startup.
                    pass
            alive = [thread for thread in workers if thread.is_alive()]
            if alive:
                raise LiveWorkerRetirementError(
                    f"live UI stopped while {len(alive)} turn worker(s) still own runtime state after "
                    f"{max(0.0, float(worker_retire_timeout)):g}s"
                )

        current_root = _state.get("root", current_root)
        suspended = str(_state.get("suspended_slash") or "")
        if not suspended:
            return
        callback = handle_modal_slash or handle_slash
        if callback is not None:
            callback(suspended)


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
    """Models to offer in the /model menu — CONFIGURED providers only (the user's clear-journey rule:
    the switcher shows what you can actually use). Each configured provider (has an api_key)
    contributes its saved model + its wizard suggestions, labeled with the provider id; picking one
    lets the CLI switch model+endpoint+key together. Returns [(model, group_label, provider_id)].
    provider_id None = the current env-configured model (no provider table to rebind to). Fallback
    when NO providers are configured (pure env setup): current model + a small known set."""
    from .model_catalog import capability
    try:
        provs = {pid: t for pid, t in (cfg.providers() or {}).items()
                 if isinstance(t, dict) and t.get("api_key")}
    except Exception:  # noqa: BLE001 — a malformed providers table must not break the menu
        provs = {}
    out, seen = [], set()
    if provs:
        try:
            from .onboarding import MODEL_SUGGESTIONS
        except Exception:  # noqa: BLE001
            MODEL_SUGGESTIONS = {}
        for pid, tbl in provs.items():
            for m in [tbl.get("model")] + list(MODEL_SUGGESTIONS.get(pid, [])):
                if m and (pid, m) not in seen:
                    seen.add((pid, m))
                    out.append((m, pid, pid))
        out.sort(key=lambda t: (t[1], t[0]))
        if all(m != llm.model for m, _, _ in out):   # an env-overridden current model still shows first
            out.insert(0, (llm.model, "current (env)", None))  # not a configured provider — label honestly
    else:
        known = ["gpt-5.5", "gpt-5", "gpt-5-mini", "o3", "deepseek-v4-flash", "deepseek-v4-pro", "kimi-k2-0905-preview",
                 "claude-sonnet-5"]
        for m in [llm.model] + known:
            if m and m not in seen:
                seen.add(m)
                base = getattr(llm, "_base_url", "") if m == llm.model else ""
                out.append((m, capability(m, base).family, None))
        out.sort(key=lambda t: (t[1], t[0]))
    return out


def _reasoning_levels(model, base_url):
    """Reasoning levels valid for a model, derived from its capability (provider-aware). Effort-capable
    models (gpt-5/o-series) expose all four; OpenRouter exposes fast/full/high (its unified reasoning
    object maps max→high, so offering max would be a lie); others only fast/full (high/max would
    degrade to default)."""
    from .model_catalog import capability
    full4 = [("fast", "minimal reasoning — fastest, cheapest"),
             ("full", "provider default reasoning"),
             ("high", "deeper reasoning (effort=high, /v1/responses)"),
             ("max", "deepest reasoning (effort=xhigh)")]
    if "openrouter" in (base_url or "").lower():   # unified reasoning object honors effort WITH tools
        return full4[:2] + [("high", "deeper reasoning (unified reasoning, effort=high)")]
    if capability(model, base_url).supports_reasoning_effort:
        return full4
    return full4[:2]   # fast | full only — the model has no effort knob


def select_model_reasoning(llm, cfg, *, pt_input=None, pt_output=None):
    """Two-tier picker: choose a model (from CONFIGURED providers) then its reasoning level (only the
    levels that model supports). Returns (model, reasoning, provider_id) — provider_id is the configured
    provider to switch endpoint+key to (None = keep the current endpoint) — or None if cancelled."""
    cands = _model_candidates(llm, cfg)
    rows = [(m, f"provider: {grp}") for m, grp, _pid in cands]
    cur_idx = next((i for i, (m, _, _) in enumerate(cands) if m == llm.model), -1)
    pick = run_selector("Select model", rows, current=cur_idx,
                        hint="↑↓ move · Enter choose model → reasoning · Esc cancel",
                        pt_input=pt_input, pt_output=pt_output)
    if pick is None:
        return None
    model, _grp, pid = cands[pick]
    if pid:   # the pick will REBIND to this provider — offer the levels ITS endpoint supports
        base = ((cfg.providers() or {}).get(pid) or {}).get("base_url") or ""
    else:
        base = getattr(llm, "_base_url", "") if model == llm.model else ""
    levels = _reasoning_levels(model, base)
    lvl_rows = [(name, desc) for name, desc in levels]
    lvl_cur = next((i for i, (n, _) in enumerate(levels) if n == llm.reasoning), -1)
    lpick = run_selector(f"Reasoning for {model}", lvl_rows, current=lvl_cur,
                         hint="↑↓ move · Enter select · Esc keep current",
                         pt_input=pt_input, pt_output=pt_output)
    reasoning = levels[lpick][0] if lpick is not None else llm.reasoning   # Esc on step 2 = keep current
    return (model, reasoning, pid)


# ── input layer (prompt_toolkit) ─────────────────────────────────────────────────────────────
_SLASH = PUBLIC_SLASH_COMMANDS


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
_KNOWN_MODELS = ("gpt-5.5", "gpt-5", "gpt-5-mini", "o3", "deepseek-v4-flash", "deepseek-v4-pro", "kimi-k2-0905-preview", "claude-sonnet-4-6")
_SLASH_ARGS = {
    "/reasoning": list(_REASONING),
}


class _InputCompleter(Completer):
    """Slash-command completion at line start (a command palette) + ARGUMENT suggestions for /model and
    /reasoning, plus filename completion on an explicit @mention. Completion and parsing share the grammar
    in mentions.py; paths containing whitespace are emitted as @"quoted paths".
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
        emitted = 0
        for p in starts + subs:                                 # basename-prefix first, then substring
            # Keep the "@", replace after it. completion_path quotes whitespace paths in syntax the CLI
            # parser accepts, so completion can never emit an unattachable filename.
            rendered = completion_path(p)
            if rendered is not None:
                yield Completion(rendered, start_position=-(len(word) - 1), display_meta="file")
                emitted += 1
                if emitted >= 20:
                    return


# rough public list prices, USD per 1M tokens: (input, cached_input, output). Substring-matched on the
# model id; an unknown model shows token counts only (no $). Update as prices change.
def _price(model: str):
    """USD/1M (input, cached, output) for the cost meter — single source is model_catalog.pricing."""
    from .model_catalog import pricing
    return pricing(model)


def _record_usage(stats: dict, usage: dict) -> None:
    """Single accounting seam shared by full turns and the chitchat fast path."""
    if not usage:
        return
    stats["tokens"] = (
        stats.get("tokens", 0)
        + (usage.get("prompt_tokens", 0) or 0)
        + (usage.get("completion_tokens", 0) or 0)
    )
    # "fresh" means input that was not a cache hit. Cache-write/creation tokens are billed input too, even
    # though providers expose them in a separate field, so keep them visible rather than silently dropping them.
    stats["fresh"] = (stats.get("fresh", 0) + (usage.get("input_other", 0) or 0)
                      + (usage.get("input_cache_creation", 0) or 0))
    _accrue_cost(stats, usage)


def _accrue_cost(stats: dict, usage: dict) -> None:
    """Per step: accrue actual $ spend (stats['cost']) AND the MOAT savings in TOKENS (model-independent, so
    a /model switch re-prices them for free — see _saved_dollars).

    Savings model: a full-transcript agent re-reads the WHOLE prior history every step (a growing cache-read)
    while the bounded slice re-reads only its small cached prefix. The saving is that cache-read DIFFERENTIAL;
    the fresh cost of genuinely-new content is the same for both agents, so it cancels. We track the naive
    transcript size (`_transcript_tok`, grown by each step's fresh/cache-write input + output) and bank, per step, the
    tokens the naive agent would re-read that the slice didn't (prefix − this step's actual cache-read)."""
    if not usage:
        return
    prefix = stats.get("_transcript_tok", 0)
    actual_cache_read = usage.get("input_cache_read", 0) or 0
    cache_creation = usage.get("input_cache_creation", 0) or 0
    stats["saved_cached_tok"] = stats.get("saved_cached_tok", 0) + max(0, prefix - actual_cache_read)
    stats["_transcript_tok"] = (prefix + (usage.get("input_other", 0) or 0) + cache_creation
                                + (usage.get("output", 0) or 0))

    pr = _price(stats.get("model", ""))
    if not pr:
        return
    pin, pcached, pout = pr
    # The compact catalog has no fourth cache-write price. Treat creation as ordinary input—the explicit
    # approximation available here—instead of the previous $0 omission.
    stats["cost"] = stats.get("cost", 0.0) + (
        (usage.get("input_other", 0) + cache_creation) * pin
        + usage.get("input_cache_read", 0) * pcached
        + usage.get("output", 0) * pout) / 1_000_000


def _saved_dollars(stats: dict):
    """$ the slice saved vs a full-transcript agent, priced at the CURRENT model's cached rate (the rate for
    re-read history). Token-based, so switching /model re-prices the same savings. None if price unknown."""
    pr = _price(stats.get("model", ""))
    if not pr:
        return None
    return stats.get("saved_cached_tok", 0) * pr[1] / 1_000_000


def _compact_count(value) -> str:
    try:
        number = max(0, int(value or 0))
    except (TypeError, ValueError):
        number = 0
    if number >= 999_500:
        return f"{number / 1_000_000:.1f}M"
    if number >= 1_000:
        return f"{number / 1_000:.1f}k"
    return str(number)


def _savings_label(stats: dict) -> str:
    saved = _saved_dollars(stats)
    if saved is None:
        return f"{_compact_count(stats.get('saved_cached_tok', 0))} tok saved"
    return f"${saved:.4f} saved" if saved < 1 else f"${saved:,.2f} saved"


def _toolbar(stats: dict, width_fn=None):
    """Responsive idle row: identity plus the two numbers users need between every turn."""
    width_fn = width_fn or (lambda: shutil.get_terminal_size((100, 24)).columns)
    dim, accent, value = "fg:ansibrightblack", "fg:ansibrightcyan bold", "fg:ansicyan"
    good = "fg:ansigreen"
    separator = (dim, " · ")

    def render():
        try:
            width = max(12, int(width_fn()))
        except Exception:
            width = 100
        workspace = str(stats.get("workspace") or "—")
        model = str(stats.get("model") or "—")
        tokens = f"{_compact_count(stats.get('tokens', 0))} tok"
        fresh = f"{_compact_count(stats.get('fresh', 0))} fresh"
        savings = _savings_label(stats)
        try:
            elapsed = f"⏲ {_duration(float(stats.get('last_turn_s', 0)))}" \
                if float(stats.get("last_turn_s", 0)) > 0 else ""
        except (TypeError, ValueError):
            elapsed = ""

        if width < 72:
            fields = [
                (accent, " sliceagent"),
                (value, tokens),
                (good, savings),
            ]
        elif width < 110:
            fields = [
                (accent, " sliceagent"),
                ("", _shorten_cells(workspace, 14)),
                (value, tokens),
                (value, fresh),
                (good, savings),
                (value, _shorten_cells(model, 18)),
            ]
        else:
            fields = [
                (accent, " sliceagent"),
                ("", _shorten_cells(workspace, 18)),
                (value, _shorten_cells(model, 22)),
                (value, tokens),
                (value, fresh),
                (good, savings),
            ]
            if elapsed:
                fields.append((dim, elapsed))

        segments = []
        for index, field in enumerate(fields):
            if index:
                segments.append(separator)
            segments.append(field)
        clipped = []
        remaining = width
        for style, content in segments:
            if remaining <= 0:
                break
            piece = _crop_cells(content, remaining)
            if piece:
                clipped.append((style, piece))
                remaining -= get_cwidth(piece)
        return FormattedText(clipped)
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
        self._history = _private_prompt_history()
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

    def set_workspace(self, root: str | None) -> None:
        """Refresh file completion in place after a workspace switch; do not recreate the terminal UI."""
        self._completer = _InputCompleter(_repo_files(root) if root else None)
        self.session.completer = self._completer

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
            content=HSplit([Frame(ta), status]),
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
                if b"\r" in data[1:] or b"\n" in data[1:]:  # hotkey + Enter can share one raw read
                    break
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


def _menu_select(options: list[str], default: int = 0) -> "int | None":
    """VERTICAL arrow-key menu — one option per row: ↑/↓ (or ←/→) move, Enter chooses, Esc/Ctrl-C
    cancels. Exists because the single-line _arrow_select WRAPS with long/many options, and its
    clear-one-line redraw then stacks copies of itself down the screen (live-repro'd in the init
    wizard's 6-entry provider menu). This sibling owns EXACTLY len(options) rows: labels are clamped
    below the terminal width so a row can never wrap, and each redraw walks the cursor up over its
    own rows. Raw-mode note: OPOST is off, so every line break is an explicit \\r\\n (the staircase
    lesson). Same gates + return contract as _arrow_select: index | -1 cancelled | None when a
    selector can't safely run (caller falls back to typed input)."""
    import shutil
    import sys
    import threading
    if threading.current_thread() is not threading.main_thread():
        return None
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    try:
        import termios
        import tty
    except Exception:  # noqa: BLE001 — non-POSIX → typed fallback
        return None
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except Exception:  # noqa: BLE001
        return None
    width = max(20, shutil.get_terminal_size((80, 24)).columns - 8)
    rows = [o if len(o) <= width else o[: width - 1] + "…" for o in options]
    idx = default if 0 <= default < len(rows) else 0

    def draw(first: bool = False) -> None:
        if not first:
            sys.stdout.write(f"\x1b[{len(rows)}A")            # walk back up over OUR rows only
        parts = []
        for i, o in enumerate(rows):
            body = f"\x1b[7m ▸ {o} \x1b[0m" if i == idx else f"\x1b[2m   {o} \x1b[0m"
            parts.append("\r\x1b[2K  " + body)
        sys.stdout.write("\r\n".join(parts) + "\r\n")         # explicit \r\n: raw mode, OPOST off
        sys.stdout.flush()

    raw_entered = False
    try:
        tty.setraw(fd)
        raw_entered = True
        try:
            termios.tcflush(fd, termios.TCIFLUSH)
        except Exception:  # noqa: BLE001
            pass
        draw(first=True)
        while True:
            try:
                data = os.read(fd, 16)                        # raw fd read — see _arrow_select's note
            except OSError:
                idx = -1
                break
            if not data:
                idx = -1
                break
            if data[:1] in (b"\r", b"\n"):                     # Enter → choose
                break
            if data[:1] == b"\x03":                           # Ctrl-C → cancel
                idx = -1
                break
            if data[:1] == b"\x1b":
                if len(data) == 1:                            # bare ESC → cancel
                    idx = -1
                    break
                if data[1:2] in (b"[", b"O"):
                    arrow = data[2:3]
                    if arrow in (b"B", b"C"):                 # ↓ / →
                        idx = (idx + 1) % len(rows)
                    elif arrow in (b"A", b"D"):               # ↑ / ←
                        idx = (idx - 1) % len(rows)
                    draw()
                    if b"\r" in data[3:] or b"\n" in data[3:]:
                        break
                continue
    except Exception:  # noqa: BLE001 — any I/O error → typed fallback, never corrupt the terminal
        idx = None
    finally:
        if raw_entered:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:  # noqa: BLE001
                try:
                    termios.tcsetattr(fd, termios.TCSANOW, old)
                except Exception:  # noqa: BLE001
                    pass
            try:
                sys.stdout.write("\r")                        # cursor already sits below the block
                sys.stdout.flush()
            except Exception:  # noqa: BLE001
                pass
    return idx


def ask_user(console: Console, question: str, options=None) -> str:
    """The ask_user prompt (the 'come back and ask a follow-up' capability). Synchronous — no pt app
    is live mid-run — so a Rich prompt is safe. Returns the user's answer (a chosen option or free text)."""
    console.print(Text.assemble(
        Text("? ", style=f"bold {TH['warn']}"), Text("input needed · ", style=TH["dim"]),
        Text(safe_terminal_text(question, multiline=False), style="bold"),
    ))
    if options:
        for i, o in enumerate(options, 1):
            console.print(Text(
                f"     {i}. {safe_terminal_text(o, multiline=False)}", style=TH["dim"],
            ))
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


# sliceagent wordmark (figlet "ansi_shadow") + the vertical 3-layer emblem. Identity = the context kernel:
# ▓ PFC / Active Work (L1, hot) · ▒ Hippocampal evidence/history (L0, retained) ·
# ░ typed Neocortical knowledge (L2, distilled). The display is a hot→cold view, not a fourth numbering scheme.
# Art hardcoded (no pyfiglet runtime dependency).
_WORDMARK = (
    "███████╗██╗     ██╗ ██████╗███████╗ █████╗  ██████╗ ███████╗███╗   ██╗████████╗",
    "██╔════╝██║     ██║██╔════╝██╔════╝██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝",
    "███████╗██║     ██║██║     █████╗  ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ",
    "╚════██║██║     ██║██║     ██╔══╝  ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ",
    "███████║███████╗██║╚██████╗███████╗██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ",
    "╚══════╝╚══════╝╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ",
)
_EMBLEM = (("▓▓", "bright_cyan"), ("▓▓", "bright_cyan"), ("▒▒", "cyan"),
           ("▒▒", "cyan"), ("░░", "grey50"), ("░░", "grey50"))   # 2 rows per layer, beside the wordmark


def user_echo(console: Console, text: str) -> None:
    """Anchor the user's turn with breathing room: a blank line, a colored left-bar 'you' marker with the
    message, then a blank line — so the prompt and the agent's reply don't run together (fixes cramped
    spacing between user input and the response)."""
    console.print()
    console.print(Text.assemble(("▌ ", f"bold {TH['accent']}"), ("you  ", f"bold {TH['accent']}"),
                                (safe_terminal_text(text, multiline=True), "bold")))
    console.print()


def banner_panel(console: Console, info: str) -> Panel:
    """The startup logo: the full ansi_shadow BLOCK wordmark, always (per user preference — never a compact
    fallback). Each art row is no-wrap + crop, so a terminal narrower than the art (~86 cols) clips it
    cleanly on the right instead of wrapping into a staircase; a normal-width window shows it in full.
    `console` is kept in the signature for the callers, though the layout is now width-independent."""
    # The wordmark itself is 79 cols. Full chrome (2-space row indent + 2-col panel padding + border) needs
    # ~91 cols, so a ~86-col window CROPPED the right edge — the final "t". Shed the indent + horizontal
    # padding as the window narrows so the WHOLE name shows down to ~85 cols (below that it still crops
    # cleanly — no wrap; the emblem stays on every art row). Wide windows keep the roomy framing.
    width = getattr(console, "width", 80) or 80
    indent = "  " if width >= 91 else ""
    hpad = 2 if width >= 91 else 0
    rows = []
    for i, word in enumerate(_WORDMARK):
        blk, col = _EMBLEM[i]
        t = Text.assemble((indent, ""), (blk, f"bold {col}"), ("  ", ""), (word, f"bold {col}"))
        t.no_wrap = True
        t.overflow = "crop"          # narrow terminal → clip the art, never wrap it into a staircase
        rows.append(t)
    rows.append(Text(""))
    rows.append(Text("  ▓ slice → ▒ cache → ░ memory   ·   memory-native coding agent", style=TH["dim"]))
    if info:
        rows.append(Text("  " + safe_terminal_text(info, multiline=False), style=TH["dim"]))
    return Panel(Group(*rows), border_style=TH["accent"], box=_box.ROUNDED,
                 title=f"[bold {TH['accent']}]sliceagent[/]", title_align="left",
                 subtitle="[grey50]/help · ctrl-d to quit[/]", subtitle_align="right", padding=(1, hpad))


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
