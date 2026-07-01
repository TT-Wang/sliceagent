"""Memory implementations — the state VAULT (task resumability) that MememMemory/NullMemory share,
plus the two brain-region MIXINS that give MememMemory its HIPPOCAMPUS (hippocampus.py) and
NEOCORTEX (neocortex.py) behavior. This file owns only what's left once those two concerns are
factored out: the skill-writer utilities (shared by /learn and consolidation), the task-state
markdown (de)serialization, and `checkpoint_task`/`load_task`/`list_session_tasks` — task resume is
neither episodic recall nor a distilled lesson, so it stays here rather than forcing it into either
mixin.

memem is the plug for cross-session lessons (via NeocortexMixin): its in-process hybrid retrieval
feeds the RELEVANT MEMORY tier and `memory_save` stores lessons. memem stays behind the `Memory`
interface — the moat never imports it — and we degrade to NullMemory when memem/its vault is absent.

`is_durable` is the structural marker: NullMemory sets it False, so hosts skip cache/checkpoint
wiring and evals stay deterministic. The vault root is decoupled from memem's STATE dir
(`MEMEM_DIR` = db/logs) — the cache is memagent-owned (`MEMAGENT_VAULT`).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading

from .hippocampus import HippocampusMixin
from .interfaces import Snippet, TaskRef, TaskState
from .neocortex import NeocortexMixin
from .safety import redact_text, scan_for_threats   # persist-guards: block-on-write + redact-on-persist
from .text_utils import now_iso as _now_iso


def _write_atomic(path: str, text: str) -> None:
    """#39: write text atomically (temp in the same dir + os.replace) so a crash mid-write can't corrupt
    a task file or the session index — the original stays intact and the rename is atomic on POSIX."""
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _safe_vault_id(x: str) -> str | None:
    """A task_id / session_id is model- and user-controllable (switch_topic, /resume) and is joined into a
    vault path, so reject anything that could traverse out (`..`, separators, nul). Returns the id or None."""
    x = (x or "").strip()
    if not x or not re.fullmatch(r"[A-Za-z0-9._-]+", x) or ".." in x:
        return None
    return x


def _vault_root() -> str:
    """memagent-owned vault root. Prefers a dedicated var; then a document-vault var; NEVER
    MEMEM_DIR (that is memem's state/db dir, not a vault). Falls back to ~/.memagent/vault."""
    for k in ("MEMAGENT_VAULT", "MEMAGENT_CACHE_DIR",
              "MEMEM_OBSIDIAN_VAULT", "CORTEX_OBSIDIAN_VAULT", "MEMEM_VAULT", "CORTEX_VAULT"):
        v = os.environ.get(k)
        if v:
            return os.path.expanduser(v)
    return os.path.join(os.path.expanduser("~"), ".memagent", "vault")


def _skills_dir() -> str:
    """Where consolidation writes promoted-procedure SKILL.md packs — a dir the SkillManager scans
    (default ~/.memagent/skills, so skills are discovered next session). MEMAGENT_SKILLS_DIR overrides."""
    return os.path.expanduser(os.environ.get("MEMAGENT_SKILLS_DIR")
                              or os.path.join("~", ".memagent", "skills"))


def write_skill_file(name: str, body: str, *, skills_dir: str | None = None) -> str | None:
    """Persist ONE SKILL.md to the skills dir, the single guarded writer shared by auto-consolidation
    and the foreground /learn tool. Validates the frontmatter, BLOCKS on a threat scan (a poisoned skill
    re-injects unscanned every session), REDACTS any secret before it lands on disk, and writes
    atomically. Returns the path written, or None if rejected. Never raises."""
    try:
        name = re.sub(r"[^a-z0-9._-]+", "-", (name or "").strip().lower()).strip("-").strip(".")[:64] or "skill"  # strip(".") rejects '.'/'..' dir escape
        if not body.lstrip().startswith("---") or "name:" not in body[:200]:
            return None                                  # not a valid SKILL.md (frontmatter required)
        if scan_for_threats(body, scope="strict"):       # (a) BLOCK on write — poisoned skill
            return None
        d = os.path.join(skills_dir or _skills_dir(), name)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "SKILL.md")
        # _write_atomic uses a per-writer mkstemp temp (not a fixed `path + ".tmp"`), so two concurrent
        # skill writes can't clobber each other's temp and corrupt SKILL.md — each rename is isolated.
        _write_atomic(path, redact_text(body))           # (c) redact any secret before persisting
        return path
    except Exception:  # noqa: BLE001 — a skill-write failure must never break the caller
        return None


def make_write_skill_tool():
    """The FOREGROUND skill writer (the tool /learn drives) — the agent-callable writer memagent lacked.
    The agent supplies name/description/body; WE own the frontmatter (provenance: user — never
    auto-pruned) and the guarded write (validate + threat-scan + redact + atomic), so a model can't forge
    AUTO provenance or smuggle an unscanned skill onto disk."""
    from .registry import ToolEntry
    from .skill_provenance import USER, frontmatter_line

    def handler(args: dict) -> str:
        name = re.sub(r"[^a-z0-9._-]+", "-", (args.get("name") or "").strip().lower()).strip("-").strip(".")[:64]  # strip(".") rejects '.'/'..' dir escape
        desc = (args.get("description") or "").strip().replace("\n", " ")[:120]
        body = (args.get("body") or "").strip()
        if not name or not desc or not body:
            return "write_skill: need a name, a description, and a body."
        md = f"---\nname: {name}\ndescription: {desc}\n{frontmatter_line(USER)}\n---\n\n{body}\n"
        path = write_skill_file(name, md)
        if not path:
            return "write_skill: rejected (invalid frontmatter, empty, or flagged by the security scan)."
        return f"Skill saved to {path} (provenance: user — it will load next session)."

    schema = {"type": "function", "function": {
        "name": "write_skill",
        "description": ("Save a REUSABLE skill (SKILL.md) authored by you, so a FUTURE session can load and "
                        "reuse it. Provide a lowercase-hyphenated `name`, a <=60-char `description` of the "
                        "capability, and the markdown `body` (## When to use / ## Process / ## Pitfalls / "
                        "## Verification). This is how /learn turns what you just did into a durable skill."),
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "lowercase-hyphenated skill name (no spaces)"},
            "description": {"type": "string", "description": "one sentence, <=60 chars, the capability"},
            "body": {"type": "string", "description": "the skill body markdown (sections as above)"},
        }, "required": ["name", "description", "body"]}}}
    return ToolEntry(name="write_skill", schema=schema, handler=handler, source="builtin")


# --- task-state markdown (de)serialization — pure module fns (no memem) -------------------

def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Parse a leading `---\\n...\\n---` block of flat `key: value` scalars; return (fm, body)."""
    fm: dict = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for ln in text[3:end].strip("\n").splitlines():
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    fm[k.strip()] = v.strip().strip('"')
            return fm, text[end + 4:].lstrip("\n")
    return fm, text


_BODY_HDR_ESC = "⁣"  # invisible separator: prefix a VERBATIM line that begins with '## ' so
                          # _read_sections doesn't mistake it for a section header (model-written markdown
                          # in goal/mission/last_error/resolution otherwise truncates/misroutes on resume).


def _esc_body(t: str) -> str:
    # escape a line that starts with '## ' OR already starts with the sentinel (so verbatim content that
    # natively begins with the sentinel round-trips exactly — _unesc peels exactly one layer).
    if not t or ("## " not in t and _BODY_HDR_ESC not in t):
        return t
    return "\n".join(_BODY_HDR_ESC + ln if (ln.startswith("## ") or ln.startswith(_BODY_HDR_ESC)) else ln
                     for ln in t.split("\n"))


def _unesc_body(t: str) -> str:
    if not t or _BODY_HDR_ESC not in t:
        return t
    return "\n".join(ln[1:] if ln.startswith(_BODY_HDR_ESC) else ln for ln in t.split("\n"))


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _read_sections(body: str) -> dict:
    """Split a body into {lower-header: verbatim text} by '## ' headers (preserves multi-line)."""
    out, cur, buf = {}, None, []
    for ln in body.splitlines():
        if ln.startswith("## "):
            if cur is not None:
                out[cur] = "\n".join(buf).strip("\n")
            cur, buf = ln[3:].strip().lower(), []
        elif cur is not None:
            buf.append(ln)
    if cur is not None:
        out[cur] = "\n".join(buf).strip("\n")
    return out


def _bullets(text: str) -> list[str]:
    out = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if s.startswith("- "):
            out.append(s[2:].strip())
    return out


def _render_task_md(task: TaskState, *, created: str, updated: str) -> str:
    # #37: frontmatter is one flat `key: value` per line — a newline in a value would spill onto a line
    # the parser drops (truncating the value). Collapse newlines to spaces for the scalar fields.
    def _fm(v):
        return str(v).replace("\r", " ").replace("\n", " ")
    fm = [
        "---", "type: task-state", "v: 1",
        f"session_id: {task.session_id}", f"task_id: {task.task_id}",
        f"title: {_fm(task.title)}", f"status: {_fm(task.status)}",
        f"created: {created}", f"updated: {updated}",
        f"since_edit: {task.since_edit}",
        f"links: {','.join(task.links)}", f"tags: {_fm(task.tags)}", "---",
    ]
    body = [
        "## Goal", _esc_body(task.goal),
        "## Findings", "\n".join(f"- {f}" for f in task.findings),
        # provenance per finding (JSON bullet, like World) — else cross-session resume drops it and a
        # 'claim'-tier finding silently reads back at the higher 'tool-note' trust tier.
        "## Finding sources", "\n".join(f"- {json.dumps([k, v], ensure_ascii=False)}"
                                        for k, v in task.finding_source.items()),
        # carried slice tiers — JSON-per-bullet so dict items round-trip EXACTLY (no markdown-escape
        # hazard). Without these, resuming a task silently dropped the standing contract / todo / north-
        # star / world model (data loss). Mission is a single verbatim line like Status.
        "## Requirements", "\n".join(f"- {json.dumps(r, ensure_ascii=False)}" for r in task.requirements),
        "## Plan", "\n".join(f"- {json.dumps(p, ensure_ascii=False)}" for p in task.plan),
        "## Mission", _esc_body(task.mission),
        "## Open report", _esc_body(getattr(task, "open_report", "")),
        "## World", "\n".join(f"- {json.dumps([k, v], ensure_ascii=False)}" for k, v in task.world.items()),
        "## Working set", "\n".join(f"- {p}" for p in task.active_files),
        "## Edited", "\n".join(f"- {p}" for p in sorted(task.edited_files)),
        # anchor is TAB-separated (a path never contains TAB; anchors may contain ' :: ' etc.)
        "## Anchors", "\n".join(f"- {p}\t{a}" for p, a in task.edit_anchor.items()),
        "## Status", _esc_body(task.last_error),   # verbatim, may be empty/multi-line
        "## Resolution", _esc_body(task.resolution),
    ]
    return "\n".join(fm) + "\n" + "\n".join(body) + "\n"


def _parse_task_md(path: str) -> TaskState | None:
    with open(path, encoding="utf-8") as f:
        fm, body = _split_frontmatter(f.read())
    sec = _read_sections(body)
    anchors: dict = {}
    for b in _bullets(sec.get("anchors", "")):
        if "\t" in b:
            p, a = b.split("\t", 1)
            anchors[p.strip()] = a
    def _json_bullets(key):
        out = []
        for b in _bullets(sec.get(key, "")):
            b = b.strip()
            if not b:
                continue
            try:
                out.append(json.loads(b))
            except Exception:  # a corrupt line must not break resume
                pass
        return out

    world = {}
    for kv in _json_bullets("world"):
        if isinstance(kv, list) and len(kv) == 2 and isinstance(kv[0], str):   # non-str key is unhashable → skip the bullet, not the whole task
            world[kv[0]] = kv[1]
    return TaskState(
        task_id=fm.get("task_id", ""), session_id=fm.get("session_id", ""),
        title=fm.get("title", ""), status=fm.get("status", "active"),
        goal=_unesc_body(sec.get("goal", "")),
        findings=_bullets(sec.get("findings", "")),
        finding_source={kv[0]: kv[1] for kv in _json_bullets("finding sources")
                        if isinstance(kv, list) and len(kv) == 2 and isinstance(kv[0], str)},
        requirements=[r for r in _json_bullets("requirements") if isinstance(r, dict)],
        plan=[p for p in _json_bullets("plan") if isinstance(p, dict)],
        mission=_unesc_body(sec.get("mission", "")),
        open_report=_unesc_body(sec.get("open report", "")),
        world=world,
        active_files=_bullets(sec.get("working set", "")),
        edited_files=_bullets(sec.get("edited", "")),
        edit_anchor=anchors,
        last_error=_unesc_body(sec.get("status", "")),
        since_edit=_safe_int(fm.get("since_edit"), 0),   # corrupt counter → 0, don't abort the whole load
        links=[x for x in fm.get("links", "").split(",") if x],
        tags=fm.get("tags", ""),
        resolution=_unesc_body(sec.get("resolution", "")),
    )


def _upsert_session_index(vault: str, task: TaskState, updated: str) -> None:
    """Maintain ONE bounded index file per session (so list_session_tasks reads it, not a glob)."""
    d = os.path.join(vault, "sessions")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{task.session_id}.md")
    rows: dict = {}  # task_id -> row text (without leading "- ")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            _, body = _split_frontmatter(f.read())
        for b in _bullets(_read_sections(body).get("tasks", "")):
            rows[b.split(" · ", 1)[0].strip()] = b
    title = redact_text((task.title or "").replace("\n", " "))   # model-derived → redact before persisting
    # title LAST — but OMIT the trailing " · title" when empty, else _bullets strips the trailing field
    # and the row parses to 3 parts and is silently dropped from the session index.
    rows[task.task_id] = f"{task.task_id} · {task.status} · {updated}" + (f" · {title}" if title else "")
    lines = ["---", "type: session", f"session_id: {task.session_id}", "---", "## Tasks"]
    lines += [f"- {r}" for r in rows.values()]
    _write_atomic(path, "\n".join(lines) + "\n")


def _parse_session_index(path: str) -> list[TaskRef]:
    with open(path, encoding="utf-8") as f:
        _, body = _split_frontmatter(f.read())
    out: list[TaskRef] = []
    for b in _bullets(_read_sections(body).get("tasks", "")):
        parts = b.split(" · ", 3)  # task_id · status · updated · title  (title optional / may contain ' · ')
        if len(parts) >= 3:
            tid, status, updated = parts[0], parts[1], parts[2]
            title = parts[3] if len(parts) == 4 else ""
            out.append(TaskRef(task_id=tid.strip(), title=title.strip(),
                               status=status.strip(), updated=updated.strip()))
    return out


# --- implementations ----------------------------------------------------------------------

class NullMemory:
    """No durable memory (the default until a vault is configured). A TRUE no-op — every method is
    inert (no I/O, no clock), so the eval path is deterministic and adds nothing to the slice."""

    is_durable = False

    def recall(self, query: str, k: int = 6, paths: list[str] | None = None) -> list[Snippet]:
        return []

    def remember(self, content: str, *, title: str = "", scope: str = "default", tags: str = "",
                 paths: list[str] | None = None) -> None:
        return None

    def append_episode(self, session_id: str, task_id: str, turn: int, record: dict) -> None:
        return None

    def read_episodes(self, session_id: str, *, limit: int | None = None) -> list[dict]:
        return []

    def episode_manifest(self, session_id: str, k: int) -> tuple[list[dict], int]:
        return [], 0

    def search_episodes(self, query: str, *, limit: int = 5,
                        exclude_session: str | None = None,
                        only_session: str | None = None) -> list[dict]:
        return []

    def checkpoint_task(self, task: TaskState) -> None:
        return None

    def load_task(self, task_id: str) -> TaskState | None:
        return None

    def list_session_tasks(self, session_id: str) -> list[TaskRef]:
        return []

    def mark_used(self, memory_id: str) -> None:
        return None

    def consolidate(self, session_id: str, *, llm=None, mode: str = "deterministic") -> dict:
        return {"lessons": 0, "skills": 0, "skills_rejected": 0, "errors": 0}

    def close(self) -> None:
        return None


class MememMemory(HippocampusMixin, NeocortexMixin):
    """Adapter over memem (lessons, via NeocortexMixin) + the on-disk episodic cache (via
    HippocampusMixin) + the state vault (task resume, below). Construction fails fast if memem
    isn't importable. The vault is memagent-owned (_vault_root), decoupled from memem's state dir."""

    is_durable = True

    def __init__(self) -> None:
        import memem.retrieve  # noqa: F401  — fail fast if memem is absent
        self._vault = _vault_root()
        self._scope = os.path.basename(os.getcwd()) or "default"   # same-project soft bonus on recall
        self._idx_lock = threading.Lock()   # serialize the lazy FTS-index open across parallel explorers

    # --- task state / resume ---
    def checkpoint_task(self, task: TaskState) -> None:
        try:
            d = os.path.join(self._vault, "tasks")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{task.task_id}.md")
            created = _now_iso()
            if os.path.exists(path):  # preserve the original created on update
                with open(path, encoding="utf-8") as f:
                    fm, _ = _split_frontmatter(f.read())
                created = fm.get("created") or created
            updated = _now_iso()
            # redact the WHOLE rendered task state before it lands on disk — title/goal/findings/last_error/
            # resolution/mission/world are all model/tool-derived and may carry secrets (mirrors the episodic
            # cache redaction). Redact-the-output is future-proof: new fields are covered automatically.
            _write_atomic(path, redact_text(_render_task_md(task, created=created, updated=updated)))
            _upsert_session_index(self._vault, task, updated)
        except Exception:
            pass

    def load_task(self, task_id: str) -> TaskState | None:
        tid = _safe_vault_id(task_id)
        if tid is None:
            return None   # reject path-traversal in a model/user-controlled id
        try:
            path = os.path.join(self._vault, "tasks", f"{tid}.md")
            return _parse_task_md(path) if os.path.exists(path) else None
        except Exception:
            return None

    def list_session_tasks(self, session_id: str) -> list[TaskRef]:
        sid = _safe_vault_id(session_id)
        if sid is None:
            return []
        try:
            path = os.path.join(self._vault, "sessions", f"{sid}.md")
            return _parse_session_index(path) if os.path.exists(path) else []
        except Exception:
            return []


def make_memory(prefer_memem: bool = True):
    """Return MememMemory if memem is importable, else NullMemory (graceful)."""
    if prefer_memem:
        try:
            return MememMemory()
        except Exception:
            pass
    return NullMemory()
