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
MAX_ARTIFACT_CHARS = 1500  # cap for INCIDENTAL output only (discovery snippets) — never for the working set
DISCOVERY_K = 6
MAX_SKILL_CHARS = 4000   # a loaded skill body is capped before it enters the slice
MAX_ACTIVE_SKILLS = 2    # keep only the most-recently-loaded skills active
# OPEN FILES is NOT size-capped (Markov bounds GROWTH over time; relevance bounds CONTENT).
# A working-set file is shown IN FULL — it's the current relevant state — up to FULL_FILE_LINES,
# which is generous (a ~1000-line file is ~10k tokens, fine in a modern context window). Only a
# PATHOLOGICALLY huge file falls back to its RELEVANT REGION (the agent's focus): a safety valve
# for the context window, never a routine head+tail truncation.
FULL_FILE_LINES = 1200
REGION_LINES = 400

# literal paths the model touches via execute_code helpers — so code-as-action reads/edits
# still populate the OPEN FILES working set (they run in the sandbox, bypassing the ToolHost)
_CODE_PATH_RE = re.compile(
    r"\b(?:read_file|write_file|append_file|str_replace)\(\s*['\"]([^'\"]+)['\"]"
)
# the underlying operations inside an execute_code body — so the anti-loop tally can see
# THROUGH code-as-action (otherwise every script is a unique signature and loops hide)
_CODE_OP_RE = re.compile(
    r"\b(read_file|write_file|append_file|str_replace|list_files|run)\(\s*['\"]?([^'\",)]*)"
)


def paths_in_code(code: str) -> list[str]:
    return _CODE_PATH_RE.findall(code or "")


def code_ops(code: str) -> list[str]:
    """Normalized operation list inside an execute_code body (op + the tail of its literal arg)."""
    out, seen = [], set()
    for op, arg in _CODE_OP_RE.findall(code or ""):
        arg = arg.strip().split("/")[-1][:24]
        sig = f"{op} {arg}".strip()
        if sig not in seen:
            seen.add(sig)
            out.append(sig)
    return out

SYSTEM_PROMPT = (
    "You are a coding agent driven by an ACTIVE MEMORY SLICE (reconstructed state, not chat history). "
    "Each turn, advance the TASK. OPEN FILES = the live file contents and your GROUND TRUTH; base edits on it, "
    "never on remembered contents. "
    "Editing: edit_file overwrites a whole file (new files only); append_to_file adds; str_replace replaces an "
    "exact snippet copied from OPEN FILES. Test files must import what they test. "
    "OPEN FILES shows each file's current contents (a very large file shows the region relevant to your task); "
    "always base edits on what is shown there. "
    "If an action is REPEATEDLY FAILING, stop repeating it — read the file, fix the root cause, then re-run. "
    "Work in as FEW turns as possible: each turn make ALL edits you can already determine (batch many tool calls), "
    "then run once to verify. For multi-step work, prefer ONE execute_code call (write several files AND run the "
    "test in a single Python script, printing a short result) over many separate tool calls. "
    "Verify with the CHEAPEST sufficient check — compile/import the changed module (e.g. python -c 'import pkg') "
    "or run the smallest relevant test. If the environment can't run the tests after ONE attempt (missing deps, "
    "setup errors), do NOT keep installing/retrying — make the minimal correct edit and finish. "
    "Never write commentary, explanation, or reasoning as text while working — call tools SILENTLY with empty "
    "message content. Output text ONLY once, as a one-line final summary, and only after the TASK is fully done "
    "and verified as well as the environment allows (then make no tool call)."
)


@dataclass
class Slice:
    goal: str = ""
    recent: list[dict] = field(default_factory=list)
    action_log: dict[str, dict] = field(default_factory=dict)
    active_files: list[str] = field(default_factory=list)
    last_error: str = ""
    active_skills: list[dict] = field(default_factory=list)  # [{name, body}] loaded SKILLs
    edit_anchor: dict[str, str] = field(default_factory=dict)  # path -> last edit-target text (huge-file focus)

    def reset(self, goal: str) -> None:
        self.goal = goal
        self.recent = []
        self.action_log = {}
        self.active_files = []
        self.last_error = ""
        self.active_skills = []
        self.edit_anchor = {}


def one_line(s, n: int = 80) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()[:n]


def touch_file(s: Slice, path: str) -> None:
    if not path:
        return
    s.active_files = [p for p in s.active_files if p != path]
    s.active_files.append(path)
    if len(s.active_files) > K:
        evicted = s.active_files[:-K]
        s.active_files = s.active_files[-K:]
        for p in evicted:
            s.edit_anchor.pop(p, None)


def action_sig(name: str, args: dict) -> str:
    if name == "run_command":
        return f"run_command `{one_line(args.get('command', ''), 50)}`"
    if name == "execute_code":
        ops = code_ops(args.get("code", ""))
        return "execute_code[" + ", ".join(ops[:4]) + "]" if ops else "execute_code(script)"
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
        if a["failing"] and a["count"] >= 3:
            warn = "  ⚠ REPEATEDLY FAILING — read the file & fix the root cause"
        elif a["count"] >= 3:
            # a non-failing action repeated this much is a soft loop (e.g. a str_replace whose
            # old_string never matches, run via execute_code so it never trips the failing flag)
            warn = "  ⚠ REPEATED with no progress — STOP; change approach (read the full file, make ONE precise edit)"
        elif a["failing"]:
            warn = "  (failing)"
        else:
            warn = ""
        lines.append(f"- {sig} ×{a['count']}{warn} → {a['last']}")
    return "\n".join(lines)


def _focus_line(s: Slice, path: str, lines: list[str]) -> int:
    """For a huge file, the line to center the relevant region on: the agent's last edit
    target if present, else the best relevance match against the task + current error,
    else the top. Relevance selection — not a position guess."""
    anchor = s.edit_anchor.get(path)
    if anchor:
        for i, ln in enumerate(lines, 1):
            if anchor in ln:
                return i
    terms = {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", f"{s.goal} {s.last_error}")}
    if terms:
        best, best_score = 1, 0
        for i, ln in enumerate(lines, 1):
            low = ln.lower()
            score = sum(1 for t in terms if t in low)
            if score > best_score:
                best, best_score = i, score
        if best_score:
            return best
    return 1


def build_artifacts(s: Slice, tools) -> str:
    """Re-read the working-set files FRESH and show them by RELEVANCE, not by a size cap.
    A file is shown IN FULL up to FULL_FILE_LINES; a genuinely huge file is shown as its
    RELEVANT REGION (around the agent's focus) in full — never head+tail. Markov bounds
    growth over time; relevance bounds what's included. No artificial token ceiling."""
    if not s.active_files:
        return "(no files opened yet)"
    parts = []
    for p in s.active_files:
        try:
            body = tools.read_text(p)
        except Exception:
            parts.append(f"### {p}\n(not created yet)")
            continue
        lines = body.splitlines()
        total = len(lines)
        if total <= FULL_FILE_LINES:
            parts.append(f"### {p} ({total} lines — full)\n```\n{body}\n```")
        else:
            focus = _focus_line(s, p, lines)
            half = REGION_LINES // 2
            a = max(1, focus - half)
            b = min(total, a + REGION_LINES - 1)
            a = max(1, b - REGION_LINES + 1)
            region = "\n".join(lines[a - 1:b])
            hdr = (f"### {p} ({total} lines — showing the relevant region, lines {a}-{b}; "
                   f"grep to locate other parts, then edit — a failed str_replace re-aims this region)")
            parts.append(f"{hdr}\n```\n{region}\n```")
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


def add_skill(s: Slice, name: str, body: str) -> None:
    """Fold a loaded SKILL into the ACTIVE SKILL tier so it PERSISTS across turns.

    This is the memagent-specific adaptation: transcript agents inject a skill body as a
    one-shot message that lingers in history; we reconstruct the slice every turn, so a
    skill must live in a tier or it vanishes next turn. Dedup by name, keep the most-recent,
    cap the body."""
    if not name or not body:
        return
    s.active_skills = [sk for sk in s.active_skills if sk["name"] != name]
    s.active_skills.append({"name": name, "body": body[:MAX_SKILL_CHARS]})
    if len(s.active_skills) > MAX_ACTIVE_SKILLS:
        s.active_skills = s.active_skills[-MAX_ACTIVE_SKILLS:]


def render_skills(active_skills: list[dict]) -> str:
    if not active_skills:
        return ""
    return "\n\n".join(f"## SKILL: {sk['name']}\n{sk['body']}" for sk in active_skills)


def render_slice(s: Slice, artifacts: str, discovery: str = "", memory: str = "") -> str:
    err = f"# CURRENT ERROR (unresolved — fix this, verbatim)\n{s.last_error}\n\n" if s.last_error else ""
    skills_body = render_skills(s.active_skills)
    skl = f"# ACTIVE SKILL(S) (loaded instructions — FOLLOW these for the task)\n{skills_body}\n\n" if skills_body else ""
    mem = f"# RELEVANT MEMORY (lessons from past sessions — apply if useful)\n{memory}\n\n" if memory else ""
    steps = "\n".join(
        f"{i + 1}. {st['action']}\n     → {st['observation']}" for i, st in enumerate(s.recent[-K:])
    ) or "(none yet — first move)"
    disc = (
        f"\n# RELATED CODE (retrieved candidates — may be incomplete; grep/read to fetch more)\n{discovery}\n"
        if discovery else ""
    )
    parts = [
        err + skl + mem + "# REPEATED/FAILING ACTIONS", render_action_history(s.action_log), "",
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
            if event.name == "skill" and not event.failing:
                # a loaded skill's body must enter the ACTIVE SKILL tier or it vanishes
                # next turn (no transcript). The skill tool returns the body as its output.
                add_skill(s, event.args.get("name", ""), event.output)
            if event.args.get("path"):
                touch_file(s, event.args["path"])
                # remember the agent's edit target so a HUGE file's region follows it (and a
                # failed str_replace re-aims the region to where the agent meant to edit)
                if event.name == "str_replace" and event.args.get("old_string"):
                    anchor = next((ln.strip() for ln in event.args["old_string"].splitlines() if ln.strip()), "")
                    if anchor:
                        s.edit_anchor[event.args["path"]] = anchor[:80]
            elif event.name == "execute_code":
                for p in paths_in_code(event.args.get("code", "")):
                    touch_file(s, p)  # code-as-action reads/edits enter the working set too
            record_action(s, event.name, event.args, event.output)
    return sink
