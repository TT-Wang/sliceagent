"""Consolidation — the cache→long-term step (MEMORY-SPEC step 4).

Reads the lossless episodic cache for a session and PROMOTES durable knowledge, ROUTED BY TYPE
(EverOS pattern: facts→memory, procedures→skills):
  - FACTS: a CORRECTIVE episode (pitfall hit, then ended clean) → a declarative "Pitfall/Resolution"
    lesson, deduped, secrets excluded, FREQUENCY-WEIGHTED (a recurring pitfall ranks first).
  - PROCEDURES: a SMOOTH successful multi-step workflow → a reusable SKILL.md (Kimi format), deduped
    by action-shape and frequency-weighted (repeated workflows first — EverOS "repeated patterns
    become skills"); capped to avoid skill spam.
Both `promote_episodes` and `promote_procedures` are pure (no I/O, no LLM) → testable offline.
`MememMemory.consolidate` wires them to the cache + remember()/skill files. The deterministic skill
body is a RECORDED procedure; LLM-distillation (generalizing the steps) is the clean upgrade at
`render_skill`. Cross-session frequency is handled separately by retrieval-feedback (bump_access).
"""
from __future__ import annotations

import logging
import os
import re
from collections import Counter

from .finding_types import badge, classify_finding
from .mining import is_self_inflicted   # the cache-distill path now owns self-inflicted filtering
from .safety import redact_text, scan_for_threats   # F1: scan recorded material BEFORE the LLM call + the return
from .skill_provenance import AUTO, frontmatter_line
from .slice import one_line

_log = logging.getLogger("memagent.consolidate")

PROC_MIN_ACTIONS = 3     # a workflow worth a skill = at least this many meaningful actions
MAX_PROCEDURES = 3       # cap skills promoted per session (avoid spam)
_SKILL_OPS = frozenset(("edit_file", "str_replace", "append_to_file", "write_file",
                        "run_command", "read_file", "execute_code"))

_SECRET_RE = re.compile(r"(api[_-]?key|secret|token|password|credential|bearer)\s*[:=]", re.I)
_EXT_TAG = {".py": "python", ".js": "javascript", ".ts": "typescript", ".go": "go", ".rs": "rust",
            ".java": "java", ".rb": "ruby", ".c": "c", ".cpp": "cpp", ".sh": "shell"}


def _tags(files) -> str:
    tags = {"memagent"}
    for f in files:
        t = _EXT_TAG.get(os.path.splitext(f)[1])
        if t:
            tags.add(t)
    return ",".join(sorted(tags))


def _is_secret(text: str) -> bool:
    return bool(_SECRET_RE.search(text or ""))


def _by_task(records: list[dict]) -> list[list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r.get("task_id", ""), []).append(r)
    return [sorted(g, key=lambda r: r.get("turn", 0)) for g in groups.values()]


def promote_episodes(records: list[dict]) -> list[dict]:
    """FACTS: one declarative lesson per CORRECTIVE episode (hit an error, then ended clean). Deduped
    by pitfall signature, secrets excluded, FREQUENCY-WEIGHTED (a pitfall that recurred across the
    session ranks first and is tagged recurring). Pure. Returns {title, content, tags, kind, freq}."""
    # pass 1: collect the corrective pitfall of each task + count signatures (frequency, #4)
    cand = []
    for recs in _by_task(records):
        rmeta = [r.get("record", {}) for r in recs]
        had_fail = any(m.get("meta", {}).get("failing") for m in rmeta)
        last = rmeta[-1].get("meta", {})
        if not (had_fail and last.get("stop_reason") == "end_turn" and not last.get("failing")):
            continue
        # all failing observations across the task's turns, oldest→newest; pick the LAST NON-self-
        # inflicted one. A turn whose only failures are the agent hitting its OWN sandbox teaches
        # nothing, but a real error AFTER a self-inflicted one must still be mined (the removed live
        # miner's D2 behaviour, now owned by this cache path). A step's observation counts as a failure
        # when its action carried failing=True (the STRUCTURED signal — catches ToolResult(ok=False)
        # whose text lacks an "Error"/"Exit code" prefix) OR the text matches the prefix (back-compat).
        fails = []
        for m in rmeta:
            for st in m.get("steps", []):
                # Decide PER OBSERVATION using its paired action's failing flag (a step can mix a failing
                # call with successful ones — a whole-step `step_failed` flag would tag a SUCCESS line as
                # the durable pitfall). Pair by index; fall back to the prose prefix when the observation
                # has no paired action (non-tool steps record observations with action=[]).
                actions = st.get("action", [])
                for i, o in enumerate(st.get("observation", [])):
                    a = actions[i] if i < len(actions) else None
                    failing = isinstance(a, dict) and a.get("failing")
                    if isinstance(o, str) and (failing or o.startswith("Error") or o.startswith("Exit code")):
                        fails.append(o)
        pitfall = next((o for o in reversed(fails) if not is_self_inflicted(o)), "")
        if not pitfall or _is_secret(pitfall):
            continue
        note = next((m.get("note") for m in reversed(rmeta) if m.get("note")), "")
        files = sorted({f for m in rmeta for f in m.get("meta", {}).get("files", [])})
        cand.append({"sig": one_line(pitfall, 100).lower(), "pitfall": pitfall, "note": note, "files": files})
    freq = Counter(c["sig"] for c in cand)
    # pass 2: one lesson per unique pitfall, frequency-first
    out, seen = [], set()
    for c in sorted(cand, key=lambda c: freq[c["sig"]], reverse=True):
        if c["sig"] in seen:
            continue
        seen.add(c["sig"])
        n = freq[c["sig"]]
        recurring = f" [recurred {n}×]" if n > 1 else ""
        content = (f"Pitfall: {one_line(c['pitfall'], 200)}{recurring}\n"
                   f"Resolution: {one_line(c['note'], 200) or 'resolved'} "
                   f"(files: {', '.join(c['files']) or 'n/a'})")
        # typed finding (item 14a): a corrective-and-cleared episode is a RESOLVED question by
        # construction; a note that reads as a dead end / decision overrides via classify_finding.
        ftype = classify_finding(c["note"], edited=bool(c["files"]), had_error=True, resolved=True)
        title = badge(ftype) + "Lesson: " + (one_line(c["note"], 60) or one_line(c["pitfall"], 60))
        out.append({"title": title, "content": content, "tags": _tags(c["files"]),
                    "kind": "fact", "freq": n, "finding_type": ftype, "files": c["files"]})  # files: R1 tag
    return out


# ── B2: /learn — turn the session transcript into a reusable USER skill (Hermes pattern) ──────────
_LEARN_STANDARDS = """\
Author the skill to this standard:
- name: lowercase-hyphenated, no spaces.
- description: ONE sentence, <=60 chars, stating the CAPABILITY (not the implementation); no marketing words.
- body sections, in order (omit one only if genuinely empty):
  ## When to use   - concrete trigger phrases / situations.
  ## Process       - numbered, GENERALIZED steps (declarative, NOT this session's verbatim commands).
  ## Pitfalls      - gotchas / things that look broken but aren't (or 'none known').
  ## Verification  - one check that proves it worked.
- Reference memagent's own tools by name (read_file, grep, edit_file, run_command) - not raw shell utilities.
- Be tight (~60-150 lines). Prefer exact signatures/commands/paths from the source; NEVER invent flags or APIs.
- Do not write a skill that only points at other skills."""


def build_learn_prompt(user_request: str = "") -> str:
    """B2 (Hermes /learn pattern): build ONE prompt that has the LIVE agent distill a reusable skill from
    the source the user named and save it via the `write_skill` tool. No separate distill engine + no new
    LLM seam (llm-agnostic, works on any backend); the agent reads THIS session from the CACHE via
    recall_history (never the slice), honoring the cache-only-distill invariant."""
    req = (user_request or "").strip() or ("the workflow we just went through in this session - review the "
                                            "steps taken and distill them into a reusable skill")
    return (
        "[/learn] Distill a REUSABLE SKILL from the source below and save it with the write_skill tool.\n\n"
        f"WHAT TO LEARN FROM:\n{req}\n\n"
        "Do this:\n"
        "1. Gather the material with the tools you already have: recall_history (review THIS session's "
        "earlier turns - the lossless cache), read_file / grep for files, the recent conversation for "
        "'what we just did'. If the scope is ambiguous, make a reasonable choice and note it; do not stall.\n"
        "2. Call write_skill ONCE with a name, a <=60-char description, and the body. After it succeeds, "
        "tell the user the skill name and a one-line summary of what it captured.\n\n"
        f"{_LEARN_STANDARDS}"
    )


_GOAL_STOP = frozenset("the a an to of in for and or with on at by add fix make build update create".split())


def _goal_tokens(s: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if t not in _GOAL_STOP and len(t) > 2}


def _near_dup_goal(a: str, b: str, thresh: float = 0.6) -> bool:
    """R3 (EverOS cluster-before-promote, lexical): two procedure goals are the same INTENT if their
    content-token sets overlap heavily (Jaccard). No embedder/memem dependency — task-agnostic."""
    ta, tb = _goal_tokens(a), _goal_tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= thresh


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:40] or "procedure"


def _op_hint(action: dict) -> str:
    a = action.get("args")
    a = a if isinstance(a, dict) else {}   # a truthy non-dict (list/str) would crash a.get(); skip it, don't abort the whole batch
    tgt = a.get("path") or a.get("command") or ""
    if not tgt and a.get("code"):
        tgt = next((ln.strip() for ln in str(a["code"]).splitlines() if ln.strip()), "")
    return action.get("name", "?") + (f" — {one_line(tgt, 50)}" if tgt else "")


def promote_procedures(records: list[dict], *, min_actions: int = PROC_MIN_ACTIONS,
                       cap: int = MAX_PROCEDURES) -> list[dict]:
    """PROCEDURES: a SMOOTH successful multi-step workflow → a reusable skill. Only NON-corrective
    tasks (corrective ones become facts) that ended clean with ≥min_actions meaningful actions of ≥2
    distinct kinds. Deduped by action-shape and FREQUENCY-WEIGHTED (repeated workflows first, the
    EverOS rule), capped. Pure. Returns {kind, name, description, steps, files, freq, tags}."""
    cand = []
    for recs in _by_task(records):
        rmeta = [r.get("record", {}) for r in recs]
        if any(m.get("meta", {}).get("failing") for m in rmeta):
            continue                                   # corrective → a fact, not a procedure
        last = rmeta[-1].get("meta", {})
        if last.get("stop_reason") != "end_turn":
            continue                                   # smooth SUCCESS only
        if last.get("requirements_open", 0) > 0:
            continue                                   # the task DECLARED standing requirements and left
                                                       # some unmet at its final turn → an INCOMPLETE task.
                                                       # A skill claims a workflow that WORKS, so don't mine
                                                       # one from unfinished work (task-outcome gate, #3).
                                                       # Absent/0 (no contract, or all met) → not suppressed.
        actions = [a for m in rmeta for st in m.get("steps", []) for a in st.get("action", [])
                   if not a.get("failing") and a.get("name") in _SKILL_OPS]
        names = [a.get("name") for a in actions]
        if len(actions) < min_actions or len(set(names)) < 2:
            continue                                   # a real multi-step workflow, not one action
        goal = next((m.get("title") for m in rmeta if m.get("title")), "") or "procedure"
        if _is_secret(goal):
            continue
        files = sorted({f for m in rmeta for f in m.get("meta", {}).get("files", [])})
        cand.append({"shape": "→".join(names), "goal": goal,
                     "steps": [_op_hint(a) for a in actions][:12], "files": files})
    sig_freq = Counter(c["shape"] for c in cand)
    out, seen, kept_goals, used_names = [], set(), [], set()
    for c in sorted(cand, key=lambda c: sig_freq[c["shape"]], reverse=True):
        if c["shape"] in seen:
            continue
        if any(_near_dup_goal(c["goal"], g) for g in kept_goals):   # R3: collapse same-INTENT workflows
            continue                                                # (keep the higher-freq one, sorted first)
        seen.add(c["shape"]); kept_goals.append(c["goal"])
        # distinct goals can slugify identically (the on-disk skill name) → disambiguate so one doesn't
        # overwrite the other's SKILL.md (data loss) and the stats don't overcount.
        name = base = _slug(c["goal"]); _i = 2
        while name in used_names:
            name = f"{base}-{_i}"; _i += 1
        used_names.add(name)
        out.append({"kind": "procedure", "name": name, "description": one_line(c["goal"], 80),
                    "steps": c["steps"], "files": c["files"], "freq": sig_freq[c["shape"]],
                    "tags": _tags(c["files"])})
        if len(out) >= cap:
            break
    dropped = len(sig_freq) - len(out)          # F3: surface the silent cap instead of dropping quietly
    if dropped > 0:
        _log.debug("promote_procedures: capped at %d; dropped %d lower-frequency procedure(s)", cap, dropped)
    return out


def render_skill(proc: dict, *, origin: str = AUTO) -> str:
    """A procedure → a SKILL.md (Kimi format: name/description frontmatter + When-to-use/Process).
    DETERMINISTIC = a RECORDED procedure (the steps verbatim). `render_skill_llm` is the LLM-generalized
    upgrade. Stamps a `provenance:` field (item 13: AUTO=consolidation, USER=foreground /learn) so a
    curator prunes only auto skills."""
    steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(proc.get("steps", []))) or "(no steps)"
    n = proc.get("freq", 1)
    prov = f"Observed from {n} successful run(s) this session." if n > 1 else \
        "Observed from a successful run."
    return (f"---\nname: {proc['name']}\ndescription: {proc['description']}\n"
            f"{frontmatter_line(origin)}\n---\n\n"
            f"# {proc['description']}\n\n{prov}\n\n"
            f"## When to use\n{proc['description']}\n\n"
            f"## Process (observed)\n{steps}\n\n"
            f"## Files\n{', '.join(proc.get('files', [])) or 'n/a'}\n")


# ── B1: LLM-generalized skill body (the upgrade render_skill was stubbed for) ──────────────────────
_GENERALIZE_SYS = (
    "You turn a RECORDED procedure (the exact steps an agent took) into a REUSABLE skill body for a "
    "future agent. Generalize: state when to use it, the process as declarative steps (not this session's "
    "verbatim commands), and 1-3 pitfalls. Markdown only, no preamble. Output EXACTLY these sections and "
    "nothing else:\n## When to use\n<concrete trigger phrases>\n## Process\n<numbered, generalized steps>\n"
    "## Pitfalls\n<gotchas, or 'none known'>"
)


def render_skill_llm(proc: dict, llm, *, origin: str = AUTO) -> str:
    """B1 — render a skill whose BODY is LLM-generalized from the recorded steps (the upgrade the
    deterministic `render_skill` was stubbed for). The frontmatter stays DETERMINISTIC (name/description/
    provenance) so the SkillManager always parses it. F1: scan the recorded material BEFORE the LLM call —
    never send a secret to distillation. Falls back to the deterministic `render_skill` on no-llm, a
    threat hit, or any LLM failure, so this is a safe drop-in."""
    if llm is None:
        return render_skill(proc, origin=origin)
    steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(proc.get("steps", []))) or "(no steps)"
    material = f"Goal: {proc.get('description', '')}\nFiles: {', '.join(proc.get('files', []))}\nSteps:\n{steps}"
    if _is_secret(material) or scan_for_threats(material, scope="strict"):
        return render_skill(proc, origin=origin)            # F1: tainted → deterministic, no LLM
    try:
        resp = llm.complete([{"role": "system", "content": _GENERALIZE_SYS},
                             {"role": "user", "content": material}], [])
        body = (resp.content or "").strip()
    except Exception:  # noqa: BLE001 — any LLM failure falls back, never breaks consolidation
        body = ""
    body = redact_text(body)                                 # F1 (return path): never emit a leaked secret…
    if scan_for_threats(body, scope="strict"):              # …or an injection the LLM may have produced
        return render_skill(proc, origin=origin)
    if "## When to use" not in body or "## Process" not in body:
        return render_skill(proc, origin=origin)            # invalid/empty → deterministic fallback
    n = proc.get("freq", 1)
    prov = f"Generalized from {n} successful run(s)." if n > 1 else "Generalized from a successful run."
    return (f"---\nname: {proc['name']}\ndescription: {proc['description']}\n"
            f"{frontmatter_line(origin)}\n---\n\n# {proc['description']}\n\n{prov}\n\n{body}\n")
