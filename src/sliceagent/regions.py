"""Typed-region renderers — per-kind views over the EXISTING Slice dataclass fields.

The slice is an address space of TYPED REGIONS (open files, ghosts, conversation, skills,
threads, …); each region knows how to render itself and to SUPPRESS itself when empty.
seed.py's render_slice is the layout pass that orders these region renderers into the one
user string (the moat); the renderers themselves live here.

This module is a pure rendering/metadata layer: it reads Slice fields (pfc.py) and low-level
helpers (safety.wrap_untrusted, the working-set bounds OWNED by swap.py) but imports NOTHING
from pfc.py/seed.py — they import FROM here (one direction), so there is no import cycle.
"""
from __future__ import annotations

import os
import re

from .safety import wrap_untrusted
from .text_utils import normalize_ws, one_line

MANIFEST_TURNS = 50      # PAGED-OUT HISTORY manifest window — bounded locator count (the moat: constant
# size regardless of session length; content is paged in on demand, never accumulated into the slice).
MAX_OPEN_THREADS = 6  # OTHER OPEN THREADS tier cap — bounded presentation of parked topics
MAX_FINDINGS = 8         # bounded ring of distilled conclusions (anti-re-derivation; not a transcript)
MAX_FINDING_CHARS = 300  # each finding is ONE compact line — distilled, never narration (causal tail matters)
MAX_REQUIREMENTS = 20    # bounded STANDING REQUIREMENTS contract (count) — the moat's no-unbounded-growth
MAX_REQ_CHARS = 300      # each requirement is ONE compact line (contracts — a long signature must survive)
MAX_PLAN_ITEMS = 20      # bounded PLAN (TodoWrite) — same no-unbounded-growth rule as requirements
MAX_PLAN_CHARS = 300     # each plan step is ONE compact line (multi-file scope must survive)
_PLAN_MARK = {"done": "x", "in_progress": "~", "pending": " "}

MAX_REPORT_CHARS = 280   # OPEN USER REPORT — one compact verbatim line (bounded; never a transcript)
MAX_ACTION_LOG = 24      # bounded anti-loop tally (no-transcript: the action_log can't grow per-topic forever)
MAX_ACTION_SHOWN = 12    # cap on REPEATED/FAILING entries rendered (highest-signal first)

# Working-set view caps (the OPEN FILES region). A working-set file is shown IN FULL up to
# FULL_FILE_LINES; only a PATHOLOGICALLY huge file collapses to its RELEVANT REGION (REGION_LINES).
# Co-located here because they parameterize the OPEN FILES region renderer (build_artifacts in
# seed.py imports them from here — one direction). DISCOVERY_K is the RELATED CODE region's k.
FULL_FILE_LINES = 1200
REGION_LINES = 400
DISCOVERY_K = 6
MAX_CONVERSATION = 4     # RECENT CONVERSATION ring — last N user<->assistant exchanges (short-range continuity)
CONVO_MSG_CHARS = 800    # per-message GIST cap in the conversation tier (count-bounded by MAX_CONVERSATION;
# the cache holds the full text and recall pages it back, so this is a display-gist size, not the only copy)


# ── PER-REGION RENDER: UNCAPPED-BY-RELEVANCE ──────────────────────────────────
# _NO_CAP — the "no render cap" sentinel. OPEN FILES / YOUR NOTES are bounded by RELEVANCE
# (record_note dedup/retire), never by an arbitrary size cap — bound ≠ size, the slice shows all that's
# relevant. The only hard limit is the physical context window, handled by the loop's overflow path
# (drop the oldest accumulated exchange), not by truncating a tier.
_NO_CAP = 1_000_000


# `one_line` is re-exported from text_utils (single definition — pfc.py/seed.py/neocortex.py import the
# real definition directly). Kept importable from regions too for the existing call sites here.


def render_cache_manifest(refs) -> str:
    """PAGED-OUT HISTORY body: one locator line per earlier turn of THIS session (NOT in the slice),
    each ending with the EXACT read_file call to page it back — so reaching back is copy-paste, not a
    blind guess. This is the TRIGGER the dead recall channel was missing: a cache the model can't see is
    a cache it never calls (the read-side analogue of REPO MAP advertising file paths). The turns are
    read-only VIRTUAL files under history/ — the model reaches for read_file far more readily than a
    bespoke recall tool (measured 2026-07-06). ``refs`` are locator-only PageRefs from
    PageTable._episodes_thissession (ONE read seam); this is pure formatting. MOAT: locators only —
    turn/title/breadcrumb, never content; the turn's body is served on demand from the bounded seal."""
    if not refs:
        return ""
    lines = []
    for r in refs:
        if r.handle == "…older":
            lines.append(f"- {r.preview}")          # the "+N earlier" tail (no single-turn call)
        else:
            lines.append(f'- {r.preview}  → read_file("history/turn-{r.handle}.md")')
    return "\n".join(lines)


def render_focus(focus, extra_roots, *, home: str = "", workspace: str = "") -> str:
    """CURRENT PROJECT body: the dir the agent is actively working in, when it has moved beyond the boundary
    root. Surfaces the auto-granted file-tool reach + the moved relative-path base (otherwise INVISIBLE →
    the model stays in the start-dir frame and can't resolve 'the project' / a bare filename to where the
    work actually is, then re-asks or cold-searches — the hunter 'index.ts' miss). The boundary (the floor)
    never moves; this is the frame on top of it. Self-suppresses for the common single-project case."""
    def short(p: str) -> str:
        return ("~" + p[len(home):]) if home and p.startswith(home) else p
    roots = [r for r in (extra_roots or []) if r and r != workspace]
    if not roots and not (focus and focus != workspace):
        return ""
    lines = []
    if focus and focus != workspace:
        lines.append(
            f"You are now working in `{short(focus)}`. Bare relative paths resolve HERE, and your file "
            f"tools — read_file, list_files, grep, edit_file — act here. Resolve a bare filename or "
            f"\"the project\"/\"it\" against THIS and the RECENT CONVERSATION first; do NOT fall back to a "
            f"boundary-wide search or re-ask when the referent is already clear from recent work.")
    others = [short(r) for r in roots if r != focus]
    if others:
        lines.append("Also within your boundary (reachable by file tools): " + ", ".join(f"`{o}`" for o in others) + ".")
    return "\n".join(lines)


def render_skills(active_skills: list[dict]) -> str:
    if not active_skills:
        return wrap_untrusted("", kind="skill")
    joined = "\n\n".join(f"## SKILL: {sk['name']}\n{sk['body']}" for sk in active_skills)
    return wrap_untrusted(joined, kind="skill")


def render_threads(refs) -> str:
    """Render the bounded OTHER OPEN THREADS index (parked topics the model can resume)."""
    if not refs:
        return ""
    lines = [f"- [{r.task_id}] {r.title} ({r.status})" for r in refs[:MAX_OPEN_THREADS]]
    extra = len(refs) - min(len(refs), MAX_OPEN_THREADS)
    if extra > 0:
        lines.append(f"- …and {extra} more")
    return "\n".join(lines)


def render_user_log(s) -> str:
    """USER INSTRUCTIONS tier: EVERY user message this topic, verbatim, in order — the authoritative
    record of what was asked. The user's stated intent/constraints are IRREDUCIBLE ground truth (they
    cannot be re-derived from disk or from an archived turn's gist), so unlike the RECENT CONVERSATION
    ring this is uncapped and never truncated. Excludes the in-progress turn (it renders separately as
    CURRENT REQUEST at the salient tail). Empty until there is at least one PRIOR user message."""
    log = getattr(s, "user_log", None)
    if not log:
        return ""
    prior = log[:-1]   # the last entry is the current turn's message == the CURRENT REQUEST
    if not prior:
        return ""
    return "\n".join(f"- [turn {e['turn']}] {e['text']}" for e in prior)


def render_conversation(s) -> str:
    """The RECENT CONVERSATION tier: the last few COMPLETED user<->assistant exchanges (the in-progress
    one is excluded — its user message is the current task). Ends with a pointer to recall the rest."""
    prior = [e for e in s.conversation[:-1] if e.get("user")]
    if not prior:
        return ""
    lines = []
    for e in prior:
        lines.append(f"- user: {e['user']}")
        if e.get("assistant"):
            lines.append(f"  you:  {e['assistant']}")
            if e.get("truncated"):
                # the gist above is a CUT of a longer reply — point at the history/ files so the model pages
                # the FULL text back instead of confabulating detail past the cut. This reply is one of the
                # turns listed in PAGED-OUT HISTORY below (each with its read_file("history/turn-N.md") call).
                lines.append("        ⋯ (shortened to a gist — read the FULL reply from its turn file in "
                             "PAGED-OUT HISTORY below, read_file(\"history/index.md\") to find it, before "
                             "answering about its specifics; do NOT guess past the cut)")
    older = s.turns - len(prior) - 1  # turns beyond the ring (minus the current in-progress turn)
    tail = (f"\n(+{older} earlier turn(s) this session not shown — they're listed in PAGED-OUT HISTORY "
            "below; read_file(\"history/turn-N.md\") to view any)") if older > 0 else ""
    return "\n".join(lines) + tail


# I1 PROVENANCE — narration filter. A FINDING must be a durable FACT, never the model's running
# narration. Notes that merely announce intent ("Let me run it", "I'll check the file", "Now I'll
# edit X", "Next, …") carry no established fact: folding them made FINDINGS read like a transcript
# and let "**Done — built it**" ratchet as an ESTABLISHED truth (F1/C3/G5). Task-agnostic + cheap:
# pure lexical, no LLM. Matched at the START of the note (the leading clause sets its kind).
_NARRATION_RE = re.compile(
    r"^\s*(?:ok(?:ay)?[,. ]+)?(?:"
    r"(?:let'?s|let me|let us|i['’]?ll|i will|i['’]?m going to|i am going to|now i|now let|"
    r"then i|i need to|i should|going to|gonna|i plan to)\b"
    r"|(?:next|first|then)\b[,. ]"  # leading sequencing adverbs ("Next, …", "First …")
    r")",
    re.I,
)
# A note that ASSERTS completion ("done", "all set", "task complete", "finished") is a CLAIM, not an
# observation — durable ONLY if a real tool RESULT backed it (see slice_sink). Detected so the source
# can be DOWNGRADED to "claim" (rendered "unverified — confirm against OPEN FILES"), never silently
# promoted to an established truth. Task-agnostic lexical signal; no LLM.
_DONE_CLAIM_RE = re.compile(
    r"\b(?:done|all set|all done|complete(?:d|ly)?|finished|it works|works now|ready to use|"
    r"task (?:is )?(?:done|complete)|already (?:done|complete|built|implemented)|"
    r"successfully (?:built|created|implemented|added|completed))\b",
    re.I,
)


def is_done_claim(text: str) -> bool:
    """True when `text` asserts the work is finished — a claim that needs an observation to be durable."""
    return bool(_DONE_CLAIM_RE.search(text or ""))


# RECALL-ON-CUT marker (see memory: recall-ring-truncation-gap). A silent one_line() cut with no signal
# reads as "this is the whole thing" — the model then RE-DERIVES the missing part from scratch instead of
# recalling it, and a re-derived answer usually does NOT match the original (confabulation, not correction).
# Found live TWICE via two independent cut sites (a bug-hunt reply cut in the RECENT CONVERSATION ring, then
# again cut here in FINDINGS/OPEN USER REPORT) — any NEW site that bounds model- or user-authored text with
# one_line() should go through this helper rather than a bare one_line() call.
_RECALL_ON_CUT_MARK = ' [cut — PARTIAL; see PAGED-OUT HISTORY (history/ files) or search_history("...") for the rest, don\'t guess]'


def _cut_with_recall_marker(text: str, cap: int) -> str:
    """one_line(text, cap), but if the cut actually removed content, replace the tail with a marker
    naming the cut + the two general recall paths (the history/ files listed in PAGED-OUT HISTORY, and
    search_history for content across sessions) + an explicit don't-guess instruction."""
    was_cut = len(one_line(text, cap + 1)) > cap
    if not was_cut:
        return one_line(text, cap)
    return one_line(text, max(0, cap - len(_RECALL_ON_CUT_MARK))) + _RECALL_ON_CUT_MARK


def record_note(s, text: str, source: str = "tool-note") -> bool:
    """Fold the model's per-turn note (a distilled FACT it established) into the FINDINGS tier.
    Returns True iff a GENUINELY NEW finding was added (not narration, not a dedup refresh) — the
    convergence check uses this so 'actively learning' doesn't count as 'spinning' (review #5).

    The slice carries no transcript, so a reasoning model would otherwise re-derive the
    situation each turn (costly reasoning bursts). This lets it carry its OWN conclusions
    forward — bounded (ring of MAX_FINDINGS) and deduped so it stays distilled, not a log.

    I1 PROVENANCE: a finding is a FACT FROM THE WORLD, never raw narration. Notes that announce
    intent ("Let me…", "I'll…") are dropped — they're transcript, not established state. `source`
    tags where the fact came from ("observed" > "tool-note" > "claim"); a completion ("done") note
    is downgraded to "claim" unless the caller passed an observed source, so it can't ratchet into
    an ESTABLISHED truth. No extra LLM call — pure lexical, captured from the note arg on a real call.

    RECALL BRIDGE: a long AssistantText reply (e.g. a multi-item bug-hunt report) folds in here as a
    "claim" — a hard per-item cut to MAX_FINDING_CHARS, since findings must stay compact. See
    _cut_with_recall_marker: without a signal, a later "what were those bugs" sees ONLY the surviving
    fragment and re-derives the rest from the code instead of recalling it — a confirmed fabrication."""
    note = _cut_with_recall_marker(text, MAX_FINDING_CHARS)
    if not note:
        return False
    if _NARRATION_RE.match(note):   # pure intent/narration — carries no durable fact
        return False
    # a "done" claim is durable only if an observation backed it; otherwise it's a hypothesis
    if source != "observed" and is_done_claim(note):
        source = "claim"
    is_new = note not in s.findings  # genuinely new knowledge vs a refresh of an existing finding
    if not is_new:                   # already established — refresh its recency, don't duplicate
        s.findings.remove(note)
    s.findings.append(note)
    # BOUNDED = SEAL THE LOOP, not cut within it: findings are NOT truncated or retired inside a loop —
    # every distinct conclusion the loop established stays whole (any within-loop cut harms the LLM). The
    # only reduction is exact-duplicate dedup above (same fact refreshed, no information lost). The bound
    # is the loop-boundary SEAL (TurnEnd archive + a fresh next loop), never a within-section filter.
    s.finding_source[note] = source
    # keep the source map bounded to the LIVE finding set (no unbounded growth across turns)
    live = set(s.findings)
    for k in [k for k in s.finding_source if k not in live]:
        del s.finding_source[k]
    return is_new


# I1 PROVENANCE — per-source trust framing. The slice's #1 ground truth is OPEN FILES (disk);
# FINDINGS are the model's own prior notes, which must be VERIFIED, never blindly reused. We never
# render model-sourced text as "do not re-derive" (that authored the "already done" ratchet).
_SOURCE_TAG = {
    "observed": "",                          # backed by a tool result — trust, but OPEN FILES still wins
    "tool-note": " (your note — verify against OPEN FILES)",
    "claim": " (UNVERIFIED claim — confirm against OPEN FILES/a tool result before relying on it)",
}


def render_findings(findings: list[str], sources: dict | None = None) -> str:
    if not findings:
        return ""
    sources = sources or {}
    return "\n".join(f"- {f}{_SOURCE_TAG.get(sources.get(f, 'tool-note'), '')}" for f in findings)


def render_world(world: dict) -> str:
    """The agent's durable WORLD MODEL — a maintained key→value scratchpad (maze map, inventory,
    system state, plan). Long/multiline values render as their own block; short ones as bullets.
    No cap (bound = the seal, not a cut): the whole maintained state renders into each turn's seed."""
    if not world:
        return ""
    parts = []
    for k, v in world.items():
        v = str(v)
        if "\n" in v or len(v) > 80:
            parts.append(f"## {k}\n{v}")
        else:
            parts.append(f"- {k}: {v}")
    return "\n".join(parts)


def render_requirements(requirements: list[dict]) -> str:
    """The STANDING REQUIREMENTS contract body: the constraints that must hold when the task is DONE.
    Self-suppresses when empty (a greeting/question has no contract → no bytes, no binding spec — the
    structural kill for the 'first message becomes the spec' bug). Append-order + status-flip-in-place
    (open '- [ ]', satisfied '- [x] … (done)') so a change touches only its own line and unrelated
    turns stay byte-identical (warm STABLE prefix). Bounded by MAX_REQUIREMENTS (folded in slice_sink)."""
    if not requirements:
        return ""
    return "\n".join(f"- [{'x' if r.get('done') else ' '}] {r.get('text', '')}" + (" (done)" if r.get("done") else "")
                     for r in requirements)


def render_plan(plan: list[dict]) -> str:
    """The PLAN tier body: the model's ordered execution steps with live status (todo list).
    Numbered + status-marked ('[~]' in-progress, '[x]' done, '[ ]' pending). Self-suppresses when empty.
    Bounded by MAX_PLAN_ITEMS (folded in slice_sink). Volatile WORKING state — distinct from STANDING
    REQUIREMENTS (acceptance criteria): this is the step sequence and the agent's live progress through it."""
    if not plan:
        return ""
    return "\n".join(f"{i}. [{_PLAN_MARK.get(it.get('status'), ' ')}] {it.get('step', '')}"
                     for i, it in enumerate(plan, 1))


# ── ANTI-LOOP / RECENT / CURRENT ERROR ────────────────────────────────────────
# the underlying operations inside an execute_code body — so the anti-loop tally can see
# THROUGH code-as-action (otherwise every script is a unique signature and loops hide)
_CODE_OP_RE = re.compile(
    r"\b(read_file|write_file|append_file|str_replace|list_files|run)\(\s*['\"]?([^'\",)]*)"
)


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


def observe(out, n: int = 260) -> str:
    """A one-line observation that PRESERVES THE TAIL. For most command output the decisive part —
    the verdict, the final status, the exception — is at the END, so head-only truncation hides it
    and the agent re-runs to 'see the result'. Task-agnostic: we don't interpret the outcome, we
    just guarantee the end is visible. Keep a little head for context plus the whole tail."""
    o = normalize_ws(out)
    if len(o) <= n:
        return o
    if n < 8:                            # too small to split head+sep+tail; a plain head-cut is the bound
        return o[:n]                     # (else tail = n-head-3 <= 0 and o[-0:] returns the WHOLE string)
    head = n // 4
    tail = n - head - 3                  # 3 = len(" … "); head + sep + tail == n
    return o[:head] + " … " + o[-tail:]


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


def record_action(s, name: str, args: dict, out: str, failing: bool | None = None) -> None:
    """Fold one tool result into the action tally + error/exploration state (deterministic — no LLM).

    `failing` is the AUTHORITATIVE flag from the tool layer (ToolText.ok / event.failing); the loop
    passes it. The prose heuristic is a back-compat fallback only — relying on it misclassified a grep/
    log line that legitimately starts with "Error" as a failure (corrupting last_error/anti-loop)."""
    s.turn_actions = getattr(s, "turn_actions", 0) + 1   # per-turn exploration counter (finding-independent)
    if failing is None:
        failing = out.startswith("Error") or out.startswith("Exit code")
    if failing:
        s.last_error = out if len(out) <= 3000 else out[:2000] + "\n…[trace truncated]…\n" + out[-900:]
    elif name in ("run_command", "execute_code"):
        s.last_error = ""  # a successful run/script clears the error (both are execution — general)
    sig = action_sig(name, args)
    prev = s.action_log.get(sig, {"count": 0})
    s.action_log[sig] = {"count": prev["count"] + 1, "failing": failing, "last": observe(out, 100)}
    if len(s.action_log) > MAX_ACTION_LOG:
        # bounded like every tier (no-transcript): evict lowest-signal first — oldest one-shot,
        # non-failing entries — so failing/repeated ones (the anti-loop signal) survive longest.
        for k in [k for k, a in s.action_log.items() if a["count"] < 2 and not a["failing"]]:
            if len(s.action_log) <= MAX_ACTION_LOG:
                break
            del s.action_log[k]
        while len(s.action_log) > MAX_ACTION_LOG:
            del s.action_log[next(iter(s.action_log))]


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
    entries.sort(key=lambda e: (e[1]["failing"], e[1]["count"]), reverse=True)  # highest-signal first
    extra = max(0, len(entries) - MAX_ACTION_SHOWN)
    entries = entries[:MAX_ACTION_SHOWN]
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
    if extra:
        lines.append(f"- …and {extra} more repeated/failing (omitted)")
    return "\n".join(lines)


# ── CONVERGENCE ───────────────────────────────────────────────────────────────
STOP_NUDGE_AFTER = 2  # non-edit tool calls since the last edit (with no error) before nudging to converge
READONLY_NUDGE_AFTER = 4  # read-only tool calls with NO edit at all before nudging to answer/act
EXPLORE_NUDGE_AFTER = 5  # tool calls in ONE turn with no edit before nudging to ANSWER or ask_user — keyed on
# turn_actions (finding-INDEPENDENT), so a read-heavy Q&A that records a note each step still converges
CLOSURE_MAX_SHOWN = 3   # max dangling-dependent locators in one CLOSURE block (bounds tokens; symbol-aware
# staleness keeps the set tiny + self-extinguishing, so no window cap is needed to prevent a cascade)


def render_closure(s) -> str:
    """CHANGE-SET CLOSURE — the PRECISE half of 'verify before done'. After an edit settles, name the
    dependents whose code STILL references a symbol your edit removed or moved: a dangling call-site a
    coordinated change must fix (re-observation-reach = action-reach). SYMBOL-AWARE (SwapManager.prefetch
    computes stale_deps from the code graph — a SENSORY CORTEX derived view, re-derived on file change,
    not a persisted store: pre-edit defs - current defs, intersected with each
    dependent's current ref tokens), so it is SILENT on feature-adds (nothing removed → never inflates a
    non-refactor task) and on already-fixed sites (their tokens no longer name the symbol). Locator-only,
    advisory, self-extinguishing (kept to UNOPENED stale deps), bounded; empty on a no-graph host. It does
    NOT police a WRONG edit at an already-reached site (e.g. s3's depth bug) — that needs behavioral verify."""
    if os.environ.get("SLICEAGENT_NO_CLOSURE"):    # safety kill-switch for the gated rollout
        return ""
    stale = getattr(s, "stale_deps", None) or set()
    # SYMBOL-AWARE: stale_deps (computed in SwapManager.prefetch) are the dependents whose CURRENT tokens
    # still reference a symbol the edit removed/moved — silent on feature-adds (nothing removed). Keep only
    # the UNOPENED ones so the nudge self-extinguishes the instant the model opens the site to fix/confirm
    # (precise + terminating: no cascade on tasks whose callers don't need changing). Capped small.
    if not stale or s.last_error or s.since_edit < STOP_NUDGE_AFTER:
        return ""
    active = set(s.active_files)
    unclosed = sorted(p for p in stale if p not in s.edited_files and p not in active)[:CLOSURE_MAX_SHOWN]
    if not unclosed:
        return ""
    edited = ", ".join(sorted(s.edited_files)[:4])
    body = "\n".join(f"- {p} — still references a symbol you changed/moved in {edited}; open it and update "
                     f"it (or confirm it's correct)" for p in unclosed)
    return ("# CHANGE-SET CLOSURE\nyour edit removed or moved a symbol these files still reference — a "
            "coordinated change must fix every call-site, not just the definition:\n" + body + "\ndo NOT "
            "declare done until each is updated or confirmed.\n\n")


def render_convergence(s) -> str:
    """Convergence pressure against over-verification. Once a change exists and the agent has spent
    several tool calls since its last edit with NO current error, it is re-checking something already
    settled — tell it to finish. General + Markov: purely a function of state (edited? error?
    calls-since-edit), no task/tool/language assumptions. Fires ONLY post-edit and ONLY when nothing
    is broken, so it never cuts off active fixing (a failing check keeps last_error set → no nudge).
    This SHRINKS wasted steps/tokens/time; the model still decides (it may continue for a real edit)."""
    if not s.edited_files:
        # EXPLORER children are SUPPOSED to do many read-only calls (their deliverable is the
        # investigation); the read-only nudge below is for the TOP-LEVEL agent over-exploring instead
        # of answering the user, and it was cutting delegated reviews short BEFORE the key (large) files
        # were read. max_steps bounds an explorer, not this nudge.
        if getattr(s, "explore_mode", False):
            return ""
        # READ-ONLY spin: many tool calls, nothing changed. Edit-gated convergence never fires here,
        # so a trivial/answer-only task (greeting, "show the path", "summarize") over-explores. Nudge
        # it to answer/act. General + Markov (edits vs non-edits, no task-type); dormant once anything
        # is edited (→ the post-edit path below), so real edit-tasks are unaffected.
        ta = getattr(s, "turn_actions", 0)
        if not s.last_error and ta >= EXPLORE_NUDGE_AFTER:
            strong = "STOP exploring NOW — " if ta >= EXPLORE_NUDGE_AFTER + 3 else ""
            return (
                f"# CONVERGENCE CHECK\n{strong}you've made {ta} tool calls this turn and edited nothing. Decide "
                f"NOW — stop exploring (do NOT re-read what you've seen). If the task needs a CODE CHANGE, make "
                f"your best-effort minimal edit immediately: never finish, and never run out of steps, having "
                f"edited nothing — an empty result is a failure, a best-effort patch is not. If the task only "
                f"needs an ANSWER, answer the user now (cite OPEN FILES) and make NO tool call; if it is "
                f"genuinely ambiguous, call ask_user with ONE concise question.\n\n")
        return ""
    if s.last_error or s.since_edit < STOP_NUDGE_AFTER:
        return ""
    if render_closure(s):           # an unreached dependent outranks the done-nudge (targeted > frequency):
        return ""                   # show CLOSURE instead of STOP so the model finishes the refactor first
    strong = "STOP NOW — " if s.since_edit >= STOP_NUDGE_AFTER + 2 else ""
    return (
        f"# CONVERGENCE CHECK\n{strong}you have edited {len(s.edited_files)} file(s) and made "
        f"{s.since_edit} tool calls since your last edit with no error — the change appears complete and "
        f"verified as well as the environment allows. Write your final summary and make NO tool "
        f"call. Continue ONLY to make a SPECIFIC new edit you have identified — do NOT re-read or re-run a "
        f"check you have already passed.\n\n"
    )


# ── OPEN USER REPORT ──────────────────────────────────────────────────────────
# I3 — OPEN USER REPORT capture heuristic. A user follow-up that looks like a FAILURE REPORT ("it
# can't play", "it doesn't work", "still broken", "cd: no such file") is the user pushing back on a
# (possibly false) "done" — the dialectic a Markov snapshot loses. We carry it as a blocker the model
# must verify against the REAL artifact before re-claiming done (it drove the "already done" ratchet:
# F1's user-pushback half). Task-agnostic + LLM-agnostic: pure lexical, no command/tool parsing, no
# model call. Two signals: (a) negation/failure phrasing about the work, (b) a literal error/diagnostic
# pasted from a terminal (a shell/runtime error string the user is reporting back).
_USER_REPORT_RE = re.compile(
    r"(?:"
    # explicit failure/negation about the artifact
    r"\b(?:doesn'?t|does not|don'?t|do not|won'?t|will not|can'?t|cannot|can ?not)\b\s*"
    r"(?:\w+\s+){0,3}?(?:work|works|run|runs|play|plays|load|loads|open|opens|start|starts|build|builds|compile|compiles)\b"
    r"|\b(?:not|isn'?t|aren'?t|wasn'?t)\s+(?:\w+\s+){0,2}?(?:work|working|run|running|play|playing|load|loading|right|correct)\b"
    r"|\b(?:still\s+)?(?:broken|failing|fails|failed|crash(?:es|ed|ing)?|errored|buggy|not working)\b"  # bare 'error'/'bug' dropped (dev vocabulary, not a report); re-admitted with context below
    r"|\b(?:it|this|that)\s+(?:still\s+)?(?:doesn'?t|does not|won'?t|can'?t|cannot)\b"
    # a pasted terminal/runtime diagnostic the user is reporting
    r"|\b(?:no such file|command not found|traceback|exception|permission denied|"
    r"syntaxerror|nameerror|typeerror|modulenotfound|exit code|segmentation fault)\b"
    r"|:\s*no such file or directory\b"
    # phrasings the first pass missed: hangs / no-output, red|failing tests/build,
    # "didn't fix it", "same error still", HTTP 4xx/5xx in a failure context, ModuleNotFoundError.
    r"|\b(?:hang(?:s|ing|ed)?|frozen|freeze(?:s|ing)?|stuck)\b"
    r"|\bnothing (?:happen(?:s|ed)?|shows?|showed|loads?|loaded|renders?|rendered)\b"
    r"|\b(?:tests?|the build|build|ci|pipeline)\b(?:\s+\w+){0,2}?\s+(?:are|is|still|now)?\s*(?:red|failing|fail|broken)\b"
    r"|\b(?:failing|red|broken)\s+(?:tests?|build|ci)\b"
    r"|\bdid(?:n'?t|\s+not)\b(?:\s+\w+){0,2}?\s*fix\b"
    r"|\b(?:still|same)\b(?:\s+\w+){0,3}?\s+(?:error|issue|problem|bug|failure|failing|broken)\b"
    r"|\bhttp\s*[45]\d\d\b|\b[45]\d\d\s+(?:error|not found|internal server)\b"  # dropped bare 'status'/'response' (feature-spec phrasing, e.g. 'return a 404 status')
    r"|\b(?:return(?:s|ed|ing)?|get(?:s|ting)?|got|throw(?:s|n|ing)?|give[sn]?)\s+(?:an?\s+)?(?:http\s*)?[45]\d\d\b"
    r"|\bmodulenotfounderror\b"
    r")",
    re.I,
)


def is_user_report(text: str) -> bool:
    """True when a user message looks like a FAILURE REPORT about prior work — captured as an OPEN
    USER REPORT blocker. Conservative + task-agnostic (pure lexical); a normal directive that merely
    contains 'add'/'fix' is NOT a report unless it carries an explicit failure/negation signal."""
    return bool(_USER_REPORT_RE.search(text or ""))


def capture_user_report(s, message: str) -> bool:
    """If `message` looks like a failure report, store it (verbatim, bounded) as the OPEN USER REPORT
    blocker on the slice and return True. A NEWER report replaces an older one (most-recent wins,
    inherently bounded). Returns False (and leaves any prior report intact) for a non-report message —
    so a benign follow-up does NOT clear a still-open report.

    The CAPTURING turn also shows the message in full via CURRENT REQUEST (no cap there); the risk is a
    LATER turn, where this bounded field is the only surviving copy — see _cut_with_recall_marker."""
    if not is_user_report(message):
        return False
    s.open_report = _cut_with_recall_marker(message, MAX_REPORT_CHARS)
    return True


# ── REGION_ORDER — the slice layout, region-by-region ─────────────────────────
# The slice is an address space of TYPED REGIONS. REGION_ORDER encodes their EXACT render order and
# the stable/volatile split that governs prompt-cache locality. A prefix cache matches only up to the
# first byte that differs from the previous request, so the STABLE BULK (OPEN FILES, RELATED CODE,
# skills, memory, conversation — byte-identical across the common read-only / reasoning steps) LEADS,
# and the VOLATILE tier (findings, action tally, RECENT, error, convergence — changes most steps) is
# the recency-salient TAIL: the immediate state and the high-authority blocker/error sit right above
# NOW. Each region renders its OWN framed fragment (header + body + spacing) and SUPPRESSES itself
# when empty (returns ''); render_regions joins the fragments. This replaces render_slice's
# hand-ordered parts[] list — the iteration MUST equal the old concatenation byte-for-byte.
#
# `slot` groups fragments into the original parts[] elements (fragments in the same slot are
# concatenated, in REGION_ORDER order, into one "\n".join part); the slot sequence + blank-line
# glue is fixed in render_regions. `tier` documents the stable/volatile split.
STABLE, VOLATILE = "stable", "volatile"


# Each region is (name, tier, render(ctx)->framed-fragment, slot). The renderer OWNS its header
# literal + spacing and SUPPRESSES itself (returns '') when empty. `tier` documents the
# stable-bulk/volatile-tail split (prompt-cache locality). `slot` maps the fragment onto the former
# CURRENT REQUEST (the live user ask) and the NOW footer render OUTSIDE the <context> envelope in
# slice.build() — NOT as REGION_ORDER entries. The envelope marks "reference STATE"; the live INSTRUCTION must
# frame it from OUTSIDE, at both ends (primacy + recency), with NOW as the outermost tail. ONE `goal` source
# feeds both request copies (no primacy/recency divergence).
_CURRENT_REQUEST_HDR = ("# CURRENT REQUEST (what the user is asking for RIGHT NOW — your PRIMARY instruction; "
                        "address THIS)\n")
_NOW_FOOTER = ("# NOW: address the CURRENT REQUEST above. If it asks a QUESTION or for an explanation, answer "
               "it directly (read/grep to ground the answer if useful — you need NOT edit); if it asks for a "
               "CHANGE, make it with tools based on OPEN FILES; once the request is fully handled and verified "
               "as well as the environment allows, write your final summary and make NO tool call.")


def render_current_request(goal: str) -> str:
    """The live user ask, rendered OUTSIDE the context fence (used at BOTH primacy and recency from
    one source). Empty goal → '' (no header)."""
    g = (goal or "").strip()
    return f"{_CURRENT_REQUEST_HDR}{g}\n\n" if g else ""


def render_now(hints: str = "") -> str:
    """The intent-aware NOW footer — the OUTERMOST tail (after the fence closes), so the final instruction
    reads as an instruction, not as 'context'. `hints` = pre-framed SUBDIRECTORY CONTEXT prefix (may be '')."""
    return (hints or "") + _NOW_FOOTER


# parts[] grouping: fragments sharing a slot are concatenated, in order, into one "\n".join part —
# so the iteration equals the old hand-ordered concatenation BYTE-FOR-BYTE. (Provenance framing for
# # YOUR NOTES / the # OPEN USER REPORT blocker / the # REPEATED-FAILING header all live in the
# literals below — relocated verbatim from render_slice, not duplicated.)
REGION_ORDER = (
    # ──────────── TIER 1 · INTENT — what the user wants (the contract). STABLE, slot-0: leads the cache prefix. ────────────
    # STANDING REQUIREMENTS — the live contract that must hold when the task is DONE: a model-curated set
    # of constraints (exact signature, output format, stated rule, an added requirement), maintained in-band
    # via require/requirement_done/drop_requirement. NOT the frozen first message — EMPTY by default, so a
    # greeting/question renders nothing (the structural kill for the 'first message = binding spec' bug).
    # STABLE/slot-0 but write-RARELY (changes only on a require/drop/done event) → the prefix stays cache-warm.
    ("requirements",   STABLE,   lambda c: (f"# STANDING REQUIREMENTS (the contract that must HOLD when the task is done — honor each EXACTLY; '[x]' = already satisfied)\n{render_requirements(c['s'].requirements)}\n\n" if getattr(c['s'], 'requirements', None) else ""), 0),
    # USER INSTRUCTIONS — every prior user message this session, VERBATIM + uncapped. The user's stated
    # intent/constraints are irreducible ground truth (unre-derivable from disk or an archived gist), so
    # they are the one part of the conversation never compacted. STABLE/slot-0: these bytes never change
    # turn-to-turn, so they EXTEND the cacheable prefix. (T1 buried-detail fix — a constraint stated once
    # early survives to a much later turn; 'later supersedes earlier' handles corrections.)
    ("user_log",       STABLE,   lambda c: (f"# USER INSTRUCTIONS (every request you've been given this session, verbatim — the authoritative record of what was asked; honor every still-applicable one, and a LATER statement supersedes an earlier one it contradicts)\n{render_user_log(c['s'])}\n\n" if render_user_log(c['s']) else ""), 0),
    # ──────────── TIER 2 · GROUND TRUTH — the world, re-derived from durable stores each turn. ────────────
    ("open_files",     STABLE,   lambda c: "# OPEN FILES (live — your ground truth; edit based on this. Lines are numbered for citation/reference; the leading number is NOT part of the file — never include it in a str_replace old_string)\n" + c["artifacts"], 0),
    ("related_code",   STABLE,   lambda c: (f"\n# RELATED CODE (repo map — relevant files & their definitions; read/grep for the actual code)\n{c['discovery']}\n" if c["discovery"] else ""), 1),
    # REPO MAP moved to the BYTE-STABLE system prefix (make_build_slice) so it's a prompt-cache PREFIX
    # shared across every turn + subagent, instead of full-price in the volatile user slice. (Region removed.)
    ("skills",         STABLE,   lambda c: (f"# ACTIVE SKILL(S) (loaded instructions — FOLLOW these for the task)\n{render_skills(c['s'].active_skills)}\n\n" if render_skills(c["s"].active_skills) else ""), 2),
    ("memory",         STABLE,   lambda c: (f"# RELEVANT MEMORY (lessons from past sessions — apply if useful)\n{c['memory']}\n\n" if c["memory"] else ""), 2),
    # ──────────── TIER 3 · MY STATE — what the agent has established / is doing. ────────────
    ("conversation",   STABLE,   lambda c: (f"# RECENT CONVERSATION (the last few exchanges this session — for continuity; older turns are paged out — see PAGED-OUT HISTORY below for the read_file(\"history/turn-N.md\") call to fetch each)\n{render_conversation(c['s'])}\n\n" if render_conversation(c["s"]) else ""), 2),
    ("findings",       VOLATILE, lambda c: (f"# YOUR NOTES FROM PRIOR TOOL CALLS (established facts to REUSE — don't re-derive these; OPEN FILES stays the ground truth for current file contents. Per-note tags mark trust: no tag = observed, '(your note)' = your summary, '(UNVERIFIED claim)' = not yet confirmed)\n{render_findings(c['s'].findings[-c['max_findings']:], c['s'].finding_source)}\n\n" if render_findings(c["s"].findings[-c["max_findings"]:], c["s"].finding_source) else ""), 3),
    ("plan",           VOLATILE, lambda c: (f"# PLAN (your ordered steps & live progress — keep exactly ONE step in_progress; '[~]'=in progress, '[x]'=done, '[ ]'=pending; update with update_plan)\n{render_plan(c['s'].plan)}\n\n" if getattr(c['s'], 'plan', None) else ""), 3),
    ("world",          VOLATILE, lambda c: (f"# WORLD MODEL (durable task state YOU maintain — your map / inventory / progress; update with world_set, it persists across turns until the task changes)\n{render_world(c['s'].world)}\n\n" if c['s'].world else ""), 3),
    # ──────────── TIER 4 · RECALL — paged out of the slice; fetched on demand. ────────────
    ("threads",        VOLATILE, lambda c: (f"# OTHER OPEN THREADS (parked topics — resume one with switch_topic; do NOT mix them into the current task)\n{c['threads']}\n\n" if c["threads"] else ""), 3),
    # PAGED-OUT HISTORY — the cache MANIFEST: earlier turns of THIS session that are NOT in the slice,
    # each with the exact read_file("history/turn-N.md") call to page it back (they're read-only virtual
    # files under history/). Sits beside GHOST INDEX (same "it's paged out, here's the one call to get it"
    # idiom) so the model has a SEEN target to read; an unseen cache is the dead channel. Locators only.
    ("cache_manifest", VOLATILE, lambda c: (f"\n# PAGED-OUT HISTORY (your OWN earlier turns this session — your memory of what you did, kept as read-only files under history/ and NOT in the slice; read any back with the call shown, read_file(\"history/index.md\") for the full list, or search_history(\"keywords\") across sessions)\n{c['cache_manifest']}\n" if c.get("cache_manifest") else ""), 3),
    # ──────────── TIER 5 · STEERING & LIVE STATE — what's wrong / where things stand (VOLATILE, high-authority tail). ────────────
    # # REPEATED/FAILING ACTIONS header (always present; body says "(nothing…)" when empty) closes slot 3.
    ("action_header",  VOLATILE, lambda c: "# REPEATED/FAILING ACTIONS", 3),
    ("action_history", VOLATILE, lambda c: render_action_history(c["s"].action_log), 4),  # body — own part
    # (CURRENT REQUEST renders OUTSIDE the fence in build() — see render_current_request above — not here.)
    # REPO STATE — the LIVE world-state region (SENSORY CORTEX — a derived view, tier A): current branch
    # + changed-file set, re-probed every build (not the session-start snapshot, and never persisted).
    # High-authority current-state ground truth, so it rides in the salient tail just above the blocker/
    # error. Suppresses itself when not a repo.
    # CURRENT PROJECT — where the agent is working RIGHT NOW (the frame on top of the immutable boundary):
    # the moved relative-path base + auto-granted file-tool reach, otherwise invisible. Rides the salient
    # tail so a follow-up's referent resolves HERE. Self-suppresses for the single-project case.
    ("focus",          VOLATILE, lambda c: (f"# CURRENT PROJECT (where you are working RIGHT NOW — bare relative paths resolve here and your file tools reach here)\n{c['focus']}\n\n" if c.get("focus") else ""), 6),
    ("worktree",       VOLATILE, lambda c: (f"# REPO STATE (LIVE — current branch & changed files, re-read THIS turn; this is the up-to-date git state — trust it over any session-start project facts)\n{c['worktree']}\n\n" if c.get("worktree") else ""), 6),
    # OPEN USER REPORT rides ABOVE the error (a stale "done" note can't outrank a user's BROKEN report);
    # both are the highest-authority, freshest tail right above NOW.
    ("user_report",    VOLATILE, lambda c: (f"# OPEN USER REPORT (the user reports this is BROKEN — treat it as an UNRESOLVED blocker; do NOT claim it is done or already working until you have VERIFIED the fix against the real artifact, e.g. run/open it and observe success)\n{c['s'].open_report}\n\n" if c["s"].open_report else ""), 6),
    ("error",          VOLATILE, lambda c: (f"# CURRENT ERROR (unresolved — fix this, verbatim)\n{c['s'].last_error}\n\n" if c["s"].last_error else ""), 6),
    ("closure",        VOLATILE, lambda c: render_closure(c["s"]), 6),
    ("convergence",    VOLATILE, lambda c: render_convergence(c["s"]), 6),
    # (NOW footer renders OUTSIDE the fence as the outermost tail in build() — see render_now above — not here.)
)


def render_regions(ctx: dict) -> str:
    """Iterate REGION_ORDER, render each typed region into its framed fragment, and assemble the ONE
    user string (the moat). Each region suppresses itself when empty; the slot grouping + the blank-line
    separator between the action tally (slot 4) and the high-authority tail (slot 6) keeps the stable bulk
    leading for prompt-cache locality and the volatile salient tail trailing. `ctx` carries the Slice + the
    pre-rendered passthroughs (artifacts / discovery / memory / threads) + the max_findings cap."""
    slots: dict[int, str] = {}
    for _name, _tier, render, slot in REGION_ORDER:
        slots[slot] = slots.get(slot, "") + render(ctx)
    if not slots:
        return ""
    # #17: assemble by iterating ALL slot positions rather than a hand-synced literal index list — that
    # list KeyError'd if a leading slot was empty and SILENTLY DROPPED any region added at a gap slot
    # (e.g. 5). Slot 5 stays the reserved blank separator between the stable bulk (≤4, cache-leading) and
    # the volatile high-authority tail (≥6); an empty slot renders as "" (a blank line), as before.
    return "\n".join(slots.get(i, "") for i in range(max(slots) + 1))
