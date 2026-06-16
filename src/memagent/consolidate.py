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

import os
import re
from collections import Counter

from .finding_types import badge, classify_finding
from .slice import one_line

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


def _failing_obs(record: dict) -> str:
    for st in record.get("steps", []):
        for o in st.get("observation", []):
            if isinstance(o, str) and (o.startswith("Error") or o.startswith("Exit code")):
                return o
    return ""


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
        pitfall = next((p for m in reversed(rmeta) if (p := _failing_obs(m))), "")
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
                    "kind": "fact", "freq": n, "finding_type": ftype})
    return out


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:40] or "procedure"


def _op_hint(action: dict) -> str:
    a = action.get("args", {}) or {}
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
    out, seen = [], set()
    for c in sorted(cand, key=lambda c: sig_freq[c["shape"]], reverse=True):
        if c["shape"] in seen:
            continue
        seen.add(c["shape"])
        out.append({"kind": "procedure", "name": _slug(c["goal"]), "description": one_line(c["goal"], 80),
                    "steps": c["steps"], "files": c["files"], "freq": sig_freq[c["shape"]],
                    "tags": _tags(c["files"])})
        if len(out) >= cap:
            break
    return out


def render_skill(proc: dict) -> str:
    """A procedure → a SKILL.md (Kimi format: name/description frontmatter + When-to-use/Process).
    Deterministic = a RECORDED procedure; the LLM-distill upgrade (generalizing the steps) slots in
    here without changing callers. Stamps a `provenance:` frontmatter field (item 13) marking the
    skill consolidation-AUTHORED, so a future curator prunes ONLY auto skills, never user skills."""
    from .skill_provenance import AUTO, frontmatter_line
    steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(proc.get("steps", []))) or "(no steps)"
    n = proc.get("freq", 1)
    prov = f"Auto-distilled from {n} successful run(s) this session." if n > 1 else \
        "Auto-distilled from a successful run."
    return (f"---\nname: {proc['name']}\ndescription: {proc['description']}\n"
            f"{frontmatter_line(AUTO)}\n---\n\n"
            f"# {proc['description']}\n\n{prov}\n\n"
            f"## When to use\n{proc['description']}\n\n"
            f"## Process (observed)\n{steps}\n\n"
            f"## Files\n{', '.join(proc.get('files', [])) or 'n/a'}\n")
