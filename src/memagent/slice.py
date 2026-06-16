"""The Active Memory Slice — the moat.

No chat history. The host builds the model-visible messages fresh each step via
`make_build_slice` (the reconstruction seam the loop calls). Tool results flow back
into the tiers through `slice_sink` (an event sink) — so the loop stays decoupled
from slice internals and just dispatches events.

Tiers, each with its own compaction policy:
  task        -> stable (system message, cacheable)
  error       -> verbatim, auto-cleared on a clean run
  findings    -> distilled conclusions the model carries forward (anti-re-derivation)
  action tally-> counted; only repeated/failing shown (anti-loop)
  recent      -> sliding window of the last K steps
  open files  -> the working set, re-read fresh from ground truth
  related code-> retrieved discovery candidates (fuzzy, agent-correctable)

The FINDINGS tier closes the reasoning-model gap in a Markov agent: the slice drops the
transcript, so a reasoning model would RE-DERIVE the situation from scratch every turn
(big completion/reasoning bursts -> slow). We let the model emit ONE short note per turn
(its conclusion + intent) and fold that into a bounded, deduped tier — distilled prior
reasoning it can REUSE, not an unbounded history. Captured for FREE from the assistant
message content (no extra LLM call, which would defeat the latency win).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .events import AssistantText, Event, ToolResult

K = 4  # sliding window for the RECENT (action→observation) tier
# WORKING SET sizing — Markov/north-star: the slice keeps the CHANGE SET (every edited file) plus a
# few recent reads, NOT a fixed last-K of everything touched. Edited files are the relevant state of
# a coding task and are PROTECTED from eviction (the old fixed K=4 could drop a file the current
# multi-file edit still needs); exploratory reads stay capped tight (residue — re-read on demand, so
# they don't bloat a reasoning model's context). Size tracks the CHANGE'S breadth, with a generous
# ceiling as a context-window safety valve — not a per-work cap.
READ_BUDGET = 4    # recent exploratory reads kept (residue)
EDIT_CEILING = 8   # max files in the change set (generous safety valve, not a per-work cap)
MAX_ARTIFACT_CHARS = 1500  # cap for INCIDENTAL output only (discovery snippets) — never for the working set
DISCOVERY_K = 6
DISCOVERY_CHARS = 4000     # cap for the RELATED CODE map (signatures are compact; bounded like every tier)
MAX_FINDINGS = 8         # bounded ring of distilled conclusions (anti-re-derivation; not a transcript)
MAX_FINDING_CHARS = 200  # each finding is ONE compact line — distilled, never narration
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
# the subset that MUTATES a file (vs read_file) — so code-as-action edits join the protected change set
_CODE_EDIT_PATH_RE = re.compile(
    r"\b(?:write_file|append_file|str_replace)\(\s*['\"]([^'\"]+)['\"]"
)
# the underlying operations inside an execute_code body — so the anti-loop tally can see
# THROUGH code-as-action (otherwise every script is a unique signature and loops hide)
_CODE_OP_RE = re.compile(
    r"\b(read_file|write_file|append_file|str_replace|list_files|run)\(\s*['\"]?([^'\",)]*)"
)


def paths_in_code(code: str) -> list[str]:
    return _CODE_PATH_RE.findall(code or "")


def edited_paths_in_code(code: str) -> list[str]:
    return _CODE_EDIT_PATH_RE.findall(code or "")


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

# The STABLE system message (cacheable). Structured into sections; binding rules in <tags> (models obey
# tag-delimited contracts more literally than prose). Tool MECHANICS live in the tool schemas (sent via the
# API's tools= channel) — NOT restated here. Stays LLM-agnostic (no model-family blocks) and task-agnostic
# (no language/tool-specific rules). The volatile per-turn tiers are appended as the user message by render_slice.
SYSTEM_PROMPT = (
    "You are a coding agent. Advance the TASK by taking ACTION with tools — never by describing what you would "
    "do. Write prose only as a final one-line summary once the task is done.\n\n"
    "# YOUR CONTEXT — the ACTIVE MEMORY SLICE\n"
    "You have NO chat history. Each turn you get a freshly reconstructed slice of state, in tiers. Trust them in "
    "this order of AUTHORITY (highest first):\n"
    "1. OPEN FILES — live contents re-read from disk: your GROUND TRUTH. Base every edit on what is shown there, "
    "never on memory. If anything conflicts with OPEN FILES, the file wins. (A huge file shows the region around "
    "your focus; grep to see more.)\n"
    "2. CURRENT ERROR — the unresolved failure to fix.\n"
    "3. WHAT YOU'VE ESTABLISHED — durable facts you concluded on earlier turns. TRUST and build on them; do NOT "
    "re-derive or re-verify them (OPEN FILES still outranks them if they disagree).\n"
    "4. REPEATED/FAILING + RECENT — your recent actions. If an action is REPEATEDLY FAILING, stop repeating it; "
    "read the file and fix the root cause.\n"
    "5. RELATED CODE / RELEVANT MEMORY — fuzzy search candidates and past-session lessons; may be incomplete or "
    "stale — verify against OPEN FILES before relying on them.\n\n"
    "<work>\n"
    "Make the SMALLEST change that resolves the task: only what is necessary, reusing the codebase's existing "
    "helpers and idioms; add no special-cases or defensive logic the task did not ask for. Work in as FEW turns "
    "as possible — batch every edit you can already determine, then verify once. For multi-step work prefer ONE "
    "execute_code script over many separate calls.\n"
    "</work>\n\n"
    "<verification>\n"
    "Verify with the CHEAPEST sufficient check (import/compile/build/lint, or the smallest relevant test). If a "
    "check cannot run after ONE attempt (missing command/deps, setup errors), do NOT keep retrying or repairing "
    "the environment — make the minimal correct edit and stop.\n"
    "</verification>\n\n"
    "<notes>\n"
    "Tool calls take an optional 'note': record a durable FACT you just established (root cause, a confirmed fix, "
    "a ruled-out hypothesis, or that the task is done) — a fact, NOT the action and NOT narration; leave it empty "
    "if nothing new was settled. Notes accumulate into WHAT YOU'VE ESTABLISHED.\n"
    "</notes>\n\n"
    "<stop>\n"
    "When the change is complete and verified as well as the environment allows, write the one-line summary and "
    "make NO tool call. Do not re-run a check you have already passed.\n"
    "</stop>"
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
    findings: list[str] = field(default_factory=list)  # distilled conclusions carried across turns
    edited_files: set = field(default_factory=set)  # the change set — protected from eviction
    since_edit: int = 0  # tool calls since the last successful edit — drives the convergence check

    def reset(self, goal: str) -> None:
        self.goal = goal
        self.recent = []
        self.action_log = {}
        self.active_files = []
        self.last_error = ""
        self.active_skills = []
        self.edit_anchor = {}
        self.findings = []
        self.edited_files = set()
        self.since_edit = 0


def one_line(s, n: int = 80) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()[:n]


def observe(out, n: int = 260) -> str:
    """A one-line observation that PRESERVES THE TAIL. For most command output the decisive part —
    the verdict, the final status, the exception — is at the END, so head-only truncation hides it
    and the agent re-runs to 'see the result'. Task-agnostic: we don't interpret the outcome, we
    just guarantee the end is visible. Keep a little head for context plus the whole tail."""
    o = re.sub(r"\s+", " ", str(out or "")).strip()
    if len(o) <= n:
        return o
    head = n // 4
    return o[:head] + " … " + o[-(n - head - 3):]


def touch_file(s: Slice, path: str, edited: bool = False) -> None:
    """Add/refresh a file in the working set. Membership is by RELEVANCE, not a fixed last-K:
    edited files (the change set) are protected; recent reads are kept up to READ_BUDGET (residue).
    So a multi-file change keeps ALL its files, while exploration stays tight."""
    if not path:
        return
    s.active_files = [p for p in s.active_files if p != path]
    s.active_files.append(path)
    if edited:
        s.edited_files.add(path)
    _prune_working_set(s)


def _prune_working_set(s: Slice) -> None:
    """Keep the change set (every edited file, up to EDIT_CEILING) plus the most-recent READ_BUDGET
    reads. Edited files are NEVER evicted to make room for a read — they're the relevant state of
    the task; reads are residue (re-read on demand)."""
    edited = [p for p in s.active_files if p in s.edited_files][-EDIT_CEILING:]
    reads = [p for p in s.active_files if p not in s.edited_files][-READ_BUDGET:]
    keep = set(edited) | set(reads)
    for p in s.active_files:
        if p not in keep:
            s.edit_anchor.pop(p, None)
            s.edited_files.discard(p)
    s.active_files = [p for p in s.active_files if p in keep]


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
    elif name in ("run_command", "execute_code"):
        s.last_error = ""  # a successful run/script clears the error (both are execution — general)
    sig = action_sig(name, args)
    prev = s.action_log.get(sig, {"count": 0})
    s.action_log[sig] = {"count": prev["count"] + 1, "failing": failing, "last": observe(out, 100)}
    display = {k: v for k, v in args.items() if k != "note"} if isinstance(args, dict) else args
    try:
        astr = json.dumps(display, ensure_ascii=False)
    except Exception:
        astr = str(display)
    s.recent.append({"action": f"{name}({one_line(astr, 60)})", "observation": observe(out, 260)})
    s.recent = s.recent[-K:]


def record_note(s: Slice, text: str) -> None:
    """Fold the model's per-turn note (its distilled conclusion) into the FINDINGS tier.

    The slice carries no transcript, so a reasoning model would otherwise re-derive the
    situation each turn (costly reasoning bursts). This lets it carry its OWN conclusions
    forward — bounded (ring of MAX_FINDINGS) and deduped so it stays distilled, not a log.
    Captured free from assistant content; no extra LLM call."""
    note = one_line(text, MAX_FINDING_CHARS)
    if not note:
        return
    if note in s.findings:          # already established — refresh its recency, don't duplicate
        s.findings.remove(note)
    s.findings.append(note)
    s.findings = s.findings[-MAX_FINDINGS:]


def render_findings(findings: list[str]) -> str:
    if not findings:
        return ""
    return "\n".join(f"- {f}" for f in findings)


STOP_NUDGE_AFTER = 2  # non-edit tool calls since the last edit (with no error) before nudging to converge


def render_convergence(s: Slice) -> str:
    """Convergence pressure against over-verification. Once a change exists and the agent has spent
    several tool calls since its last edit with NO current error, it is re-checking something already
    settled — tell it to finish. General + Markov: purely a function of state (edited? error?
    calls-since-edit), no task/tool/language assumptions. Fires ONLY post-edit and ONLY when nothing
    is broken, so it never cuts off active fixing (a failing check keeps last_error set → no nudge).
    This SHRINKS wasted steps/tokens/time; the model still decides (it may continue for a real edit)."""
    if not s.edited_files or s.last_error or s.since_edit < STOP_NUDGE_AFTER:
        return ""
    strong = "STOP NOW — " if s.since_edit >= STOP_NUDGE_AFTER + 2 else ""
    return (
        f"# CONVERGENCE CHECK\n{strong}you have edited {len(s.edited_files)} file(s) and made "
        f"{s.since_edit} tool calls since your last edit with no error — the change appears complete and "
        f"verified as well as the environment allows. Write the one-line final summary and make NO tool "
        f"call. Continue ONLY to make a SPECIFIC new edit you have identified — do NOT re-read or re-run a "
        f"check you have already passed.\n\n"
    )


# POSIX-general signal that a command is UNAVAILABLE (not that the agent's code is wrong): the
# shell couldn't find/execute it (exit 127 = not found, 126 = not executable). Task-agnostic — no
# tool/language/runner name. Re-running an unavailable command can never succeed.
# Deliberately NOT "no such file" (a path mistake is usually fixable, not an unavailable command).
_CMD_UNAVAILABLE = ("command not found", "[exit 127]", "exit code 127",
                    "[exit 126]", "exit code 126", "not executable", "executable not found")


def render_action_history(action_log: dict) -> str:
    entries = [(sig, a) for sig, a in action_log.items() if a["count"] >= 2 or a["failing"]]
    if not entries:
        return "- (nothing repeated or failing)"
    lines = []
    for sig, a in entries:
        last_low = (a.get("last") or "").lower()
        unavailable = sig.startswith(("run_command", "execute_code")) and any(m in last_low for m in _CMD_UNAVAILABLE)
        if a["failing"] and a["count"] >= 2 and unavailable:
            # the command itself is unavailable here — re-running can't fix it (general: env/tooling
            # gap, not your code). Don't repeat it; finish if the work is done, else change command.
            warn = ("  ⚠ this command is UNAVAILABLE here — re-running can't fix it; if your work is "
                    "complete write the final summary, otherwise use a different command")
        elif a["failing"] and a["count"] >= 3:
            # same command, same failure, repeatedly — re-running won't change the outcome
            warn = ("  ⚠ REPEATEDLY FAILING the same way — re-running won't change it; fix the root cause, "
                    "or if your work is already complete, finish")
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
    """The code-discovery query tracks the agent's CURRENT FOCUS, not just the static task — so on
    a large repo RELATED CODE keeps surfacing what's relevant to the NEXT decision (Markov), not the
    original task terms. Focus = latest finding (the agent's current conclusion/intent) + current
    error (names the missing symbol/file) + the task."""
    parts = [task]
    if s.findings:
        parts.append(s.findings[-1])  # the agent's most recent conclusion = where it is now
    if s.last_error:
        parts.append(s.last_error[:300])
    return "\n".join(parts)


def build_discovery(s: Slice, retriever, query: str) -> str:
    snippets = retriever.retrieve(query, k=DISCOVERY_K)
    if not snippets:
        return ""
    return "\n\n".join(
        f"### {sn.path} (score {sn.score:.2f})\n```\n{sn.text[:DISCOVERY_CHARS]}\n```" for sn in snippets
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


def _active(state):
    """Resolve the current Slice from a Slice or a Session (host-side topic manager)."""
    return state.active() if hasattr(state, "active") else state


MAX_OPEN_THREADS = 6  # OTHER OPEN THREADS tier cap — bounded presentation of parked topics


def render_threads(refs) -> str:
    """Render the bounded OTHER OPEN THREADS index (parked topics the model can resume)."""
    if not refs:
        return ""
    lines = [f"- [{r.task_id}] {r.title} ({r.status})" for r in refs[:MAX_OPEN_THREADS]]
    extra = len(refs) - min(len(refs), MAX_OPEN_THREADS)
    if extra > 0:
        lines.append(f"- …and {extra} more")
    return "\n".join(lines)


def render_slice(s: Slice, artifacts: str, discovery: str = "", memory: str = "", threads: str = "") -> str:
    err = f"# CURRENT ERROR (unresolved — fix this, verbatim)\n{s.last_error}\n\n" if s.last_error else ""
    fnd_body = render_findings(s.findings)
    fnd = (
        "# WHAT YOU'VE ESTABLISHED (your notes from prior turns — BUILD ON these; do NOT re-derive)\n"
        f"{fnd_body}\n\n"
    ) if fnd_body else ""
    skills_body = render_skills(s.active_skills)
    skl = f"# ACTIVE SKILL(S) (loaded instructions — FOLLOW these for the task)\n{skills_body}\n\n" if skills_body else ""
    mem = f"# RELEVANT MEMORY (lessons from past sessions — apply if useful)\n{memory}\n\n" if memory else ""
    thr = (
        "# OTHER OPEN THREADS (parked topics — resume one with switch_topic; do NOT mix them into "
        f"the current task)\n{threads}\n\n"
    ) if threads else ""
    steps = "\n".join(
        f"{i + 1}. {st['action']}\n     → {st['observation']}" for i, st in enumerate(s.recent[-K:])
    ) or "(none yet — first move)"
    disc = (
        f"\n# RELATED CODE (repo map — relevant files & their definitions; read/grep for the actual code)\n{discovery}\n"
        if discovery else ""
    )
    conv = render_convergence(s)
    parts = [
        conv + err + fnd + thr + skl + mem + "# REPEATED/FAILING ACTIONS", render_action_history(s.action_log), "",
        f"# RECENT (last {K})", steps, "",
        "# OPEN FILES (live — your ground truth; edit based on this)", artifacts,
        disc,
        "# NOW: do the next step(s) with tools, or — if your change is complete and verified as well as the "
        "environment allows — write the one-line final summary and make NO tool call.",
    ]
    return "\n".join(parts)


def make_build_slice(state, tools, retriever, memory, task: str):
    """The reconstruction seam the loop calls each step. Returns [system, user] messages.

    `state` is a Slice (single task) OR a Session (host-side topic manager, has .active()). The
    ACTIVE slice is resolved EACH call, so a mid-turn topic switch redirects the next reconstruction.
    System (instructions + the active topic's goal) is stable per topic and cacheable; the user
    message is the volatile slice. Cross-session memory is recalled once per topic-goal (cached);
    code discovery is per-turn (adapts as the agent works)."""
    is_session = hasattr(state, "active")
    cwd = ""
    try:
        cwd = tools.root() if hasattr(tools, "root") else ""
    except Exception:
        cwd = ""
    env_line = (
        f"\n\n# WORKING DIRECTORY\nEvery tool and command already runs INSIDE this workspace: {cwd}\n"
        "Reference files by their path RELATIVE to it (e.g. 'pkg/mod.py', 'test_x.py'). Do NOT use 'cd' "
        "or absolute paths and do NOT hunt for the directory — run_command already starts here."
    ) if cwd else ""
    recall_cache: dict[str, str] = {}

    def _system(goal: str) -> str:
        return (SYSTEM_PROMPT + env_line
                + "\n\n# TASK (your checklist — do the next item that OPEN FILES shows is not done)\n"
                + goal)

    def build() -> list[dict]:
        s = _active(state)
        goal = s.goal or task
        if goal not in recall_cache:
            recall_cache[goal] = render_memory(memory.recall(goal)) if memory is not None else ""
        artifacts = build_artifacts(s, tools)
        discovery = build_discovery(s, retriever, discovery_query(s, goal))
        threads = render_threads(state.open_threads()) if is_session else ""
        user = render_slice(s, artifacts, discovery, recall_cache[goal], threads)
        return [{"role": "system", "content": _system(goal)}, {"role": "user", "content": user}]

    return build


def slice_sink(state):
    """Event sink that folds tool results back into the tiers (keeps the loop decoupled). `state`
    is a Slice or a Session — events fold into the CURRENT active slice (so a topic switch redirects
    subsequent folding)."""
    def sink(event: Event) -> None:
        s = _active(state)
        if isinstance(event, AssistantText):
            # fallback path: models that DO emit message content while working (deepseek and other
            # reasoning models emit empty content during tool calls, so for them this is a no-op —
            # the note arg below is the real capture point)
            record_note(s, event.content)
            return
        if isinstance(event, ToolResult):
            # the model's distilled conclusion rides on the tool call (the note arg) — fold it into
            # the FINDINGS tier so a reasoning model reuses it instead of re-deriving next turn
            record_note(s, event.args.get("note", ""))
            did_edit = False
            if event.name == "skill" and not event.failing:
                # a loaded skill's body must enter the ACTIVE SKILL tier or it vanishes
                # next turn (no transcript). The skill tool returns the body as its output.
                add_skill(s, event.args.get("name", ""), event.output)
            # list_files' "path" is a DIRECTORY to browse, not a working-set file — don't track it
            if event.args.get("path") and event.name != "list_files":
                did_edit = event.name in ("edit_file", "append_to_file", "str_replace") and not event.failing
                touch_file(s, event.args["path"], edited=did_edit)
                # remember the agent's edit target so a HUGE file's region follows it (and a
                # failed str_replace re-aims the region to where the agent meant to edit)
                if event.name == "str_replace" and event.args.get("old_string"):
                    anchor = next((ln.strip() for ln in event.args["old_string"].splitlines() if ln.strip()), "")
                    if anchor:
                        s.edit_anchor[event.args["path"]] = anchor[:80]
            elif event.name == "execute_code":
                code = event.args.get("code", "")
                mutated = set(edited_paths_in_code(code))
                did_edit = bool(mutated) and not event.failing
                for p in paths_in_code(code):
                    touch_file(s, p, edited=(p in mutated))  # code-as-action edits join the change set
            record_action(s, event.name, event.args, event.output)
            # convergence tracking: a real edit resets the spin counter; anything else increments it
            s.since_edit = 0 if did_edit else s.since_edit + 1
    return sink
