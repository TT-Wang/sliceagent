"""The Active Memory Slice — the moat.

No chat history. The host builds the model-visible messages fresh each step via
`make_build_slice` (the reconstruction seam the loop calls). Tool results flow back
into the tiers through `slice_sink` (an event sink) — so the loop stays decoupled
from slice internals and just dispatches events.

Tiers, each with its own compaction policy:
  task        -> stable (system message, cacheable)
  error       -> verbatim, auto-cleared on a clean run
  action tally-> counted; only repeated/failing shown (anti-loop)
  recent      -> sliding window of the last K steps
  open files  -> the working set, re-read fresh from ground truth
  related code-> retrieved discovery candidates (fuzzy, agent-correctable)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .events import Event, ToolResult

K = 4
MAX_ARTIFACT_CHARS = 1500
DISCOVERY_K = 6

# literal paths the model touches via execute_code helpers — so code-as-action reads/edits
# still populate the OPEN FILES working set (they run in the sandbox, bypassing the ToolHost)
_CODE_PATH_RE = re.compile(
    r"\b(?:read_file|write_file|append_file|str_replace)\(\s*['\"]([^'\"]+)['\"]"
)


def paths_in_code(code: str) -> list[str]:
    return _CODE_PATH_RE.findall(code or "")

SYSTEM_PROMPT = (
    "You are a coding agent driven by an ACTIVE MEMORY SLICE (reconstructed state, not chat history). "
    "Each turn, advance the TASK. OPEN FILES = the live file contents and your GROUND TRUTH; base edits on it, "
    "never on remembered contents. "
    "Editing: edit_file overwrites a whole file (new files only); append_to_file adds; str_replace replaces an "
    "exact snippet copied from OPEN FILES. Test files must import what they test. "
    "If an action is REPEATEDLY FAILING, stop repeating it — read the file, fix the root cause, then re-run. "
    "Work in as FEW turns as possible: each turn make ALL edits you can already determine (batch many tool calls), "
    "then run once to verify. For multi-step work, prefer ONE execute_code call (write several files AND run the "
    "test in a single Python script, printing a short result) over many separate tool calls. "
    "Never write commentary, explanation, or reasoning as text while working — call tools SILENTLY with empty "
    "message content. Output text ONLY once, as a one-line final summary, and only after the TASK is fully done "
    "and tests pass (then make no tool call)."
)


@dataclass
class Slice:
    goal: str = ""
    recent: list[dict] = field(default_factory=list)
    action_log: dict[str, dict] = field(default_factory=dict)
    active_files: list[str] = field(default_factory=list)
    last_error: str = ""

    def reset(self, goal: str) -> None:
        self.goal = goal
        self.recent = []
        self.action_log = {}
        self.active_files = []
        self.last_error = ""


def one_line(s, n: int = 80) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()[:n]


def touch_file(s: Slice, path: str) -> None:
    if not path:
        return
    s.active_files = [p for p in s.active_files if p != path]
    s.active_files.append(path)
    if len(s.active_files) > K:
        s.active_files = s.active_files[-K:]


def action_sig(name: str, args: dict) -> str:
    if name == "run_command":
        return f"run_command `{one_line(args.get('command', ''), 50)}`"
    if name == "execute_code":
        return f"execute_code `{one_line(args.get('code', ''), 50)}`"
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
        s.last_error = ""
    sig = action_sig(name, args)
    prev = s.action_log.get(sig, {"count": 0})
    s.action_log[sig] = {"count": prev["count"] + 1, "failing": failing, "last": one_line(out, 80)}
    try:
        astr = json.dumps(args, ensure_ascii=False)
    except Exception:
        astr = str(args)
    s.recent.append({"action": f"{name}({one_line(astr, 60)})", "observation": one_line(out, 200)})
    s.recent = s.recent[-K:]


def render_action_history(action_log: dict) -> str:
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


def build_artifacts(s: Slice, tools) -> str:
    """Re-read the active files FRESH (the working set) — head+tail capped to stay bounded."""
    if not s.active_files:
        return "(no files opened yet)"
    parts = []
    for p in s.active_files:
        try:
            body = tools.read_text(p)
        except Exception:
            parts.append(f"### {p}\n(not created yet)")
            continue
        if len(body) > MAX_ARTIFACT_CHARS:
            shown = body[: MAX_ARTIFACT_CHARS - 500] + "\n…[middle truncated]…\n" + body[-500:]
        else:
            shown = body
        parts.append(f"### {p} ({len(body)} bytes — current contents)\n```\n{shown}\n```")
    return "\n\n".join(parts)


def discovery_query(s: Slice, task: str) -> str:
    """The code-discovery query adapts to the agent's live focus: the task plus the
    current error (which usually names the missing symbol/file) — so the RELATED CODE
    tier tracks what the agent is stuck on, not just the static task."""
    if s.last_error:
        return f"{task}\n{s.last_error[:300]}"
    return task


def build_discovery(s: Slice, retriever, query: str) -> str:
    snippets = retriever.retrieve(query, k=DISCOVERY_K)
    if not snippets:
        return ""
    return "\n\n".join(
        f"### {sn.path} (score {sn.score:.2f})\n```\n{sn.text[:MAX_ARTIFACT_CHARS]}\n```" for sn in snippets
    )


def render_memory(snippets) -> str:
    """Render recalled cross-session lessons (from memem) for the RELEVANT MEMORY tier."""
    if not snippets:
        return ""
    return "\n".join(f"- {one_line(sn.text, 160)}" for sn in snippets)


def render_slice(s: Slice, artifacts: str, discovery: str = "", memory: str = "") -> str:
    err = f"# CURRENT ERROR (unresolved — fix this, verbatim)\n{s.last_error}\n\n" if s.last_error else ""
    mem = f"# RELEVANT MEMORY (lessons from past sessions — apply if useful)\n{memory}\n\n" if memory else ""
    steps = "\n".join(
        f"{i + 1}. {st['action']}\n     → {st['observation']}" for i, st in enumerate(s.recent[-K:])
    ) or "(none yet — first move)"
    disc = (
        f"\n# RELATED CODE (retrieved candidates — may be incomplete; grep/read to fetch more)\n{discovery}\n"
        if discovery else ""
    )
    parts = [
        err + mem + "# REPEATED/FAILING ACTIONS", render_action_history(s.action_log), "",
        f"# RECENT (last {K})", steps, "",
        "# OPEN FILES (live — your ground truth; edit based on this)", artifacts,
        disc,
        "# NOW: do the next step(s) with tools, or a one-line summary if the task is fully done and tests pass.",
    ]
    return "\n".join(parts)


def make_build_slice(s: Slice, tools, retriever, memory, task: str):
    """The reconstruction seam the loop calls each step. Returns [system, user] messages.
    System (instructions + task) is stable/cacheable; the user message is the volatile slice.
    Cross-session memory is recalled ONCE per task (lessons are task-stable); code discovery
    is per-turn (it adapts as the agent works)."""
    system = (
        SYSTEM_PROMPT
        + "\n\n# TASK (your checklist — do the next item that OPEN FILES shows is not done)\n"
        + task
    )
    recalled = render_memory(memory.recall(task)) if memory is not None else ""

    def build() -> list[dict]:
        artifacts = build_artifacts(s, tools)
        discovery = build_discovery(s, retriever, discovery_query(s, task))
        user = render_slice(s, artifacts, discovery, recalled)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    return build


def slice_sink(s: Slice):
    """Event sink that folds tool results back into the tiers (keeps the loop decoupled)."""
    def sink(event: Event) -> None:
        if isinstance(event, ToolResult):
            if event.args.get("path"):
                touch_file(s, event.args["path"])
            elif event.name == "execute_code":
                for p in paths_in_code(event.args.get("code", "")):
                    touch_file(s, p)  # code-as-action reads/edits enter the working set too
            record_action(s, event.name, event.args, event.output)
    return sink
