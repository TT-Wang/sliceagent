"""The Active Memory Slice — the moat.

No chat history. Each turn the loop rebuilds the user message from these deterministic
tiers + retrieval. Every tier is bounded, and each has its own compaction policy:
  task        -> stable (lives in the system message, cacheable)
  error       -> verbatim, auto-cleared on a clean run
  action tally-> counted; only repeated/failing entries shown (anti-loop)
  recent      -> sliding window of the last K steps
  open files  -> the working set, re-read fresh from ground truth
  related code-> retrieved discovery candidates (fuzzy, agent-correctable)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

K = 4                      # recent steps kept verbatim
MAX_ARTIFACT_CHARS = 1500  # per-file cap (head+tail) — keeps the slice bounded
DISCOVERY_K = 6            # retrieved candidates per turn

SYSTEM_PROMPT = (
    "You are a coding agent driven by an ACTIVE MEMORY SLICE (reconstructed state, not chat history). "
    "Each turn, advance the TASK. OPEN FILES = the live file contents and your GROUND TRUTH; base edits on it, "
    "never on remembered contents. "
    "Editing: edit_file overwrites a whole file (new files only); append_to_file adds; str_replace replaces an "
    "exact snippet copied from OPEN FILES. Test files must import what they test. "
    "If an action is REPEATEDLY FAILING, stop repeating it — read the file, fix the root cause, then re-run. "
    "Work in as FEW turns as possible: each turn make ALL edits you can already determine (batch many tool calls), "
    "then run once to verify. "
    "Never write commentary, explanation, or reasoning as text while working — call tools SILENTLY with empty "
    "message content. Output text ONLY once, as a one-line final summary, and only after the TASK is fully done "
    "and tests pass (then make no tool call)."
)


@dataclass
class Slice:
    goal: str = ""
    recent: list[dict] = field(default_factory=list)        # [{"action", "observation"}]
    action_log: dict[str, dict] = field(default_factory=dict)  # sig -> {"count", "failing", "last"}
    active_files: list[str] = field(default_factory=list)   # LRU of touched paths
    last_error: str = ""

    def reset(self, goal: str) -> None:
        self.goal = goal
        self.recent = []
        self.action_log = {}
        self.active_files = []
        self.last_error = ""


def one_line(s, n: int = 80) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()[:n]


def touch_file(s: Slice, path: str, max_files: int = K) -> None:
    if not path:
        return
    s.active_files = [p for p in s.active_files if p != path]
    s.active_files.append(path)
    if len(s.active_files) > max_files:
        s.active_files = s.active_files[-max_files:]


def action_sig(name: str, args: dict) -> str:
    if name == "run_command":
        return f"run_command `{one_line(args.get('command', ''), 50)}`"
    if name in ("edit_file", "append_to_file", "str_replace", "read_file"):
        return f"{name} {args.get('path', '')}"
    if name == "list_files":
        return f"list_files {args.get('path', '.')}"
    return name


def record_action(s: Slice, name: str, args: dict, out: str) -> None:
    """Fold one tool result into the tiers (deterministic — no LLM)."""
    failing = out.startswith("Error") or out.startswith("Exit code")
    if failing:
        s.last_error = out if len(out) <= 800 else out[:120] + "\n…[trace truncated]…\n" + out[-680:]
    elif name == "run_command":
        s.last_error = ""  # a clean run resolves the current error
    sig = action_sig(name, args)
    prev = s.action_log.get(sig, {"count": 0})
    s.action_log[sig] = {"count": prev["count"] + 1, "failing": failing, "last": one_line(out, 80)}
    s.recent.append({"action": f"{name}({one_line(_args_str(args), 60)})", "observation": one_line(out, 200)})
    s.recent = s.recent[-K:]


def _args_str(args: dict) -> str:
    import json
    try:
        return json.dumps(args, ensure_ascii=False)
    except Exception:
        return str(args)


def render_action_history(action_log: dict) -> str:
    # Only surface what this tier is FOR: repeated or failing actions. One-off successes
    # are already visible in RECENT / OPEN FILES, so listing them just bloats the slice.
    entries = [(sig, a) for sig, a in action_log.items() if a["count"] >= 2 or a["failing"]]
    if not entries:
        return "- (nothing repeated or failing)"
    lines = []
    for sig, a in entries:
        if a["failing"]:
            warn = "  ⚠ REPEATEDLY FAILING — read the file & fix the root cause" if a["count"] >= 3 else "  (failing)"
        else:
            warn = ""
        lines.append(f"- {sig} ×{a['count']}{warn} → {a['last']}")
    return "\n".join(lines)


def render_slice(s: Slice, artifacts: str, discovery: str = "") -> str:
    """Build the volatile user message. The task lives in the system prompt (cacheable)."""
    err = f"# CURRENT ERROR (unresolved — fix this, verbatim)\n{s.last_error}\n\n" if s.last_error else ""
    steps = "\n".join(
        f"{i + 1}. {st['action']}\n     → {st['observation']}" for i, st in enumerate(s.recent[-K:])
    ) or "(none yet — first move)"
    disc = (
        f"\n# RELATED CODE (retrieved candidates — may be incomplete; grep/read to fetch more)\n{discovery}\n"
        if discovery else ""
    )
    parts = [
        err + "# REPEATED/FAILING ACTIONS", render_action_history(s.action_log), "",
        f"# RECENT (last {K})", steps, "",
        "# OPEN FILES (live — your ground truth; edit based on this)", artifacts,
        disc,
        "# NOW: do the next step(s) with tools, or a one-line summary if the task is fully done and tests pass.",
    ]
    return "\n".join(parts)
