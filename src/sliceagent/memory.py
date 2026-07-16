"""Compatibility facade over SliceAgent's native evidence/work/knowledge stores.

Production construction now returns :class:`LocalMemory`: native Hippo/history, task compatibility views,
and typed SQLite knowledge stay available without Memem.  ``MememMemory`` remains as a compatibility adapter
for embeddings/tests that instantiate the old class directly; it is no longer the switch that enables L0/L1.

The broad ``Memory`` surface is retained while callers migrate to the narrower EvidenceArchive,
WorkRepository, and KnowledgeRepository contracts.  Physical state remains private and model-facing access is
through ContextFS, never raw vault paths.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass

from .hippocampus import HippocampusMixin
from .interfaces import Snippet, TaskRef, TaskState
from .knowledge import (FeedbackEvent, FeedbackKind, KnowledgeFreshness, KnowledgeKind,
                        KnowledgeConflictError, KnowledgeQuery, KnowledgeRecord, KnowledgeRepository,
                        KnowledgeScope, KnowledgeSensitivity, KnowledgeSourceRef,
                        KnowledgeStatus)
from .knowledge_index import MememKnowledgeIndex, NativeKnowledgeIndex, NullKnowledgeIndex
from .neocortex import NeocortexMixin
from .private_state import is_private_state_path, private_dir, private_file
from .safety import redact_text, scan_for_threats   # persist-guards: block-on-write + redact-on-persist
from .text_utils import now_iso as _now_iso


def _write_atomic(path: str, text: str, *, private: bool = True) -> None:
    """#39: write text atomically (temp in the same dir + os.replace) so a crash mid-write can't corrupt
    a task file or the session index — the original stays intact and the rename is atomic on POSIX.

    Vault records are private by default. Explicit project/shared skill roots opt out: a new file is
    published as 0644 and a rewrite preserves the existing file mode, so atomic replacement does not
    silently turn a collaborator-readable skill into owner-only state.
    """
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            if not private:
                try:
                    mode = os.stat(path, follow_symlinks=False).st_mode & 0o777
                except OSError:
                    mode = 0o644
                try:
                    os.fchmod(f.fileno(), mode)
                except (AttributeError, OSError):
                    pass
        os.replace(tmp, path)
        if private:
            private_file(path)
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
    """sliceagent-owned vault root. Prefers a dedicated var; then a document-vault var; NEVER
    MEMEM_DIR (that is memem's state/db dir, not a vault). Falls back to ~/.sliceagent/vault."""
    for k in ("SLICEAGENT_VAULT", "SLICEAGENT_CACHE_DIR",
              "MEMEM_OBSIDIAN_VAULT", "CORTEX_OBSIDIAN_VAULT", "MEMEM_VAULT", "CORTEX_VAULT"):
        v = os.environ.get(k)
        if v:
            return os.path.expanduser(v)
    return os.path.join(os.path.expanduser("~"), ".sliceagent", "vault")


def _skills_dir() -> str:
    """Return the adjacent capability store for promoted ``SKILL.md`` packs.

    Skills are executable assets discovered by ``SkillManager``, not L2 records or another memory layer.
    ``SLICEAGENT_SKILLS_DIR`` overrides the default ``~/.sliceagent/skills`` location.
    """
    return os.path.expanduser(os.environ.get("SLICEAGENT_SKILLS_DIR")
                              or os.path.join("~", ".sliceagent", "skills"))


def _knowledge_db_path() -> str:
    base = os.environ.get("SLICEAGENT_CACHE_DIR") or os.path.join("~", ".sliceagent")
    return os.path.realpath(os.path.expanduser(
        os.environ.get("SLICEAGENT_KNOWLEDGE_DB")
        or os.path.join(base, "knowledge", "knowledge.db")
    ))


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
        base = skills_dir or _skills_dir()
        d = os.path.join(base, name)
        if is_private_state_path(base):  # personal state is private; an explicit shared/project dir stays shared
            private_dir(base)
            private_dir(d)
        else:
            os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "SKILL.md")
        # _write_atomic uses a per-writer mkstemp temp (not a fixed `path + ".tmp"`), so two concurrent
        # skill writes can't clobber each other's temp and corrupt SKILL.md — each rename is isolated.
        _write_atomic(path, redact_text(body), private=is_private_state_path(base))
        # (c) redact any secret before persisting. Personal skills are 0600; an explicitly shared/project
        # skill remains collaborator-readable instead of inheriting mkstemp's owner-only mode.
        return path
    except Exception:  # noqa: BLE001 — a skill-write failure must never break the caller
        return None


def make_write_skill_tool():
    """The FOREGROUND skill writer (the tool /learn drives) — the agent-callable writer sliceagent lacked.
    The agent supplies name/description/body; WE own the frontmatter (provenance: user — never
    auto-pruned) and the guarded write (validate + threat-scan + redact + atomic), so a model can't forge
    AUTO provenance or smuggle an unscanned skill onto disk."""
    from .registry import ToolEntry
    from .skill_provenance import USER, frontmatter_line

    def handler(args: dict) -> str:
        from .execution import ToolStatus
        from .registry import ToolText

        name = re.sub(r"[^a-z0-9._-]+", "-", (args.get("name") or "").strip().lower()).strip("-").strip(".")[:64]  # strip(".") rejects '.'/'..' dir escape
        desc = (args.get("description") or "").strip().replace("\n", " ")[:120]
        body = (args.get("body") or "").strip()
        if not name or not desc or not body:
            return ToolText("write_skill: need a name, a description, and a body.",
                            status=ToolStatus.FAILED)
        md = f"---\nname: {name}\ndescription: {desc}\n{frontmatter_line(USER)}\n---\n\n{body}\n"
        if scan_for_threats(md, scope="strict"):
            return ToolText("write_skill: rejected by the security scan.",
                            status=ToolStatus.FAILED)
        path = write_skill_file(name, md)
        if not path:
            return ToolText(
                "write_skill: the write did not finish cleanly; inspect the destination before retrying.",
                status=ToolStatus.INDETERMINATE,
            )
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
                          # in goal/last_error/resolution otherwise truncates/misroutes on resume).


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
        "---", "type: task-state", f"v: {getattr(task, 'schema_version', 2)}",
        f"session_id: {task.session_id}", f"task_id: {task.task_id}",
        f"title: {_fm(task.title)}", f"status: {_fm(task.status)}",
        f"created: {created}", f"updated: {updated}",
        f"since_edit: {task.since_edit}",
        f"workspace_epoch: {max(0, int(getattr(task, 'workspace_epoch', 0) or 0))}",
        f"intent_next_id: {getattr(task, 'intent_next_id', 1)}",
        f"objective_status: {_fm(getattr(task, 'objective_status', 'active'))}",
        f"links: {','.join(task.links)}", f"tags: {_fm(task.tags)}", "---",
    ]
    body = [
        "## Goal", _esc_body(task.goal),
        "## Goal source", _esc_body(getattr(task, "goal_source", "")),
        "## Current request", _esc_body(getattr(task, "current_request", "") or task.goal),
        # The graph is JSON-per-bullet like typed intent.  Source text is not copied here; its immutable
        # event/artifact locator and digest remain the authority.
        "## Active work", "\n".join(
            f"- {json.dumps(r, ensure_ascii=False)}" for r in getattr(task, "active_work", [])
        ),
        # v2 authoritative intent records. JSON bullets preserve exact clauses, status, provenance and ranges.
        "## Intent", "\n".join(f"- {json.dumps(r, ensure_ascii=False)}"
                                  for r in getattr(task, "intent_entries", [])),
        "## Findings", "\n".join(f"- {f}" for f in task.findings),
        # provenance per finding (JSON bullet, like World) — else cross-session resume drops it and a
        # 'claim'-tier finding silently reads back at the higher 'tool-note' trust tier.
        "## Finding sources", "\n".join(f"- {json.dumps([k, v], ensure_ascii=False)}"
                                        for k, v in task.finding_source.items()),
        # carried slice tiers — JSON-per-bullet so dict items round-trip EXACTLY (no markdown-escape
        # hazard). Without these, resuming a task silently dropped the standing contract / todo /
        # world model (data loss).
        # Derived v1 view for one compatibility window. v2 readers always prefer Intent above.
        "## Requirements", "\n".join(f"- {json.dumps(r, ensure_ascii=False)}" for r in task.requirements),
        "## Plan", "\n".join(f"- {json.dumps(p, ensure_ascii=False)}" for p in task.plan),
        "## Progress signals", "\n".join(f"- {json.dumps(p, ensure_ascii=False)}"
                                           for p in getattr(task, "progress_signals", [])),
        "## Deliverable requirement", (
            f"- {json.dumps(getattr(task, 'deliverable_requirement'), ensure_ascii=False)}"
            if getattr(task, "deliverable_requirement", None) else ""
        ),
        "## Open report", _esc_body(getattr(task, "open_report", "")),
        "## Execution uncertainty", _esc_body(getattr(task, "reconciliation_required", "")),
        "## Uncertainty targets", "\n".join(
            f"- {json.dumps(target, ensure_ascii=False)}"
            for target in getattr(task, "reconciliation_targets", [])
        ),
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
        task_id=fm.get("task_id", ""), schema_version=_safe_int(fm.get("v"), 1),
        session_id=fm.get("session_id", ""),
        title=fm.get("title", ""), status=fm.get("status", "active"),
        goal=_unesc_body(sec.get("goal", "")),
        goal_source=_unesc_body(sec.get("goal source", "")),
        objective_status=fm.get("objective_status", "active"),
        current_request=_unesc_body(sec.get("current request", "")),
        workspace_epoch=max(0, _safe_int(fm.get("workspace_epoch"), 0)),
        active_work=[r for r in _json_bullets("active work") if isinstance(r, dict)],
        intent_entries=[r for r in _json_bullets("intent") if isinstance(r, dict)],
        intent_next_id=_safe_int(fm.get("intent_next_id"), 1),
        findings=_bullets(sec.get("findings", "")),
        finding_source={kv[0]: kv[1] for kv in _json_bullets("finding sources")
                        if isinstance(kv, list) and len(kv) == 2 and isinstance(kv[0], str)},
        requirements=[r for r in _json_bullets("requirements") if isinstance(r, dict)],
        plan=[p for p in _json_bullets("plan") if isinstance(p, dict)],
        progress_signals=[p for p in _json_bullets("progress signals") if isinstance(p, dict)],
        deliverable_requirement=next((
            item for item in _json_bullets("deliverable requirement")
            if isinstance(item, dict)
        ), None),
        open_report=_unesc_body(sec.get("open report", "")),
        reconciliation_required=_unesc_body(
            sec.get("execution uncertainty", sec.get("reconciliation required", ""))
        ),
        reconciliation_targets=[
            target for target in (
                _json_bullets("uncertainty targets") or _json_bullets("reconciliation targets")
            ) if isinstance(target, str)
        ],
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
    private_dir(d)
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

    def append_subagent_artifact(self, session_id: str, artifact: dict) -> str:
        return ""   # no vault → not archived; run_subagent falls back to the inline digest

    def read_subagent_artifacts(self, session_id: str) -> list[dict]:
        return []

    def index_subagent_artifact(self, session_id: str, handle: str, artifact: dict) -> None:
        return None   # no FTS index without a vault

    # roster (standing specialists) — no vault → no durable workforce; named spawns run as temps
    def roster_get(self, name: str):
        return None

    def roster_hire(self, name: str, kind: str) -> dict:
        return {}

    def roster_list(self) -> list[dict]:
        return []

    def roster_recent(self, k: int) -> tuple[list[dict], int]:
        return [], 0

    def roster_append_job(self, name: str, artifact: dict) -> str:
        return ""

    def roster_read_jobs(self, name: str) -> list[dict]:
        return []

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


@dataclass(frozen=True)
class _MemoryScopeState:
    """One atomically replaceable PROJECT/revision binding."""

    project_id: str
    workspace_id: str
    workspace_root: str
    resource_revision: str
    label: str


class LocalMemory(HippocampusMixin):
    """Always-on native evidence/work compatibility plus typed L2 knowledge.

    Memem is an optional semantic index over canonical typed L2 records; construction never depends on it.
    Project identity is set by the live workspace runtime through :meth:`set_scope` and remains a hard
    predicate in both Memem and the canonical repository. ``_scope`` is only a human-facing legacy label.
    """

    is_durable = True

    def __init__(self, *, prefer_memem: bool = True) -> None:
        self._vault = _vault_root()
        self._vault_error = ""
        try:
            private_dir(self._vault)
        except Exception as exc:  # legacy mirrors may degrade without taking down canonical L0/L1
            self._vault_error = type(exc).__name__
        self._scope_binding = _MemoryScopeState(
            project_id="project-unscoped", workspace_id="", workspace_root="",
            resource_revision="", label=os.path.basename(os.getcwd()) or "default",
        )
        self._user_id = os.environ.get("SLICEAGENT_USER_ID", "local-user") or "local-user"
        self._agent_id = os.environ.get("SLICEAGENT_AGENT_ID", "sliceagent") or "sliceagent"
        self._idx_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._last_consolidation_cache: dict[str, dict] = {}
        self._compatibility_writes: dict[str, dict[str, object]] = {
            channel: {"attempts": 0, "succeeded": 0, "failed": 0, "last_error": ""}
            for channel in ("episodic_mirror", "legacy_fts", "task_projection", "session_projection")
        }
        self._knowledge: KnowledgeRepository | None = None
        self._knowledge_error = ""
        self._native_operation_error = ""
        self._native_knowledge_index = NullKnowledgeIndex()
        self._memem_knowledge_index: MememKnowledgeIndex | None = None
        try:
            self._knowledge = KnowledgeRepository(_knowledge_db_path())
            self._native_knowledge_index = NativeKnowledgeIndex(self._knowledge)
            self._knowledge_index = self._native_knowledge_index
        except Exception as exc:
            # Native L2 is independent of canonical events/artifacts and Active Work. Preserve those layers,
            # expose the exact failure class through health/ContextFS, and allow repair on a later restart.
            self._knowledge_error = type(exc).__name__
            self._knowledge_index = NullKnowledgeIndex()
        self._memem_available = False
        self._memem_state = "disabled"
        self._memem_detail = "not installed"
        if prefer_memem and self._knowledge is not None:
            backend = MememKnowledgeIndex(
                self._knowledge, fallback=self._native_knowledge_index,
            )
            self._memem_knowledge_index = backend
            if backend.is_active:
                self._memem_available = True
                self._knowledge_index = backend
                try:
                    # One authoritative rebuild closes the old mirror gap: all
                    # canonical records are projected by stable id before Memem
                    # becomes the live semantic search path.
                    backend.rebuild()
                    self._mark_memem("healthy", "canonical L2 projection synchronized")
                except Exception as exc:
                    # Search remains usable and has an explicit whole-query
                    # native fallback; health exposes the failed synchronization.
                    self._mark_memem("degraded", f"projection sync failed ({type(exc).__name__})")
            else:
                detail = backend.health().get("error") or "structured index protocol unavailable"
                self._memem_detail = f"not available ({detail})"

    def _mark_memem(self, state: str, detail: str) -> None:
        self._memem_state = str(state)
        self._memem_detail = str(detail)

    @property
    def _project_id(self) -> str:
        return self._scope_binding.project_id

    @property
    def _workspace_id(self) -> str:
        return self._scope_binding.workspace_id

    @property
    def _workspace_root(self) -> str:
        return self._scope_binding.workspace_root

    @property
    def _resource_revision(self) -> str:
        return self._scope_binding.resource_revision

    @property
    def _scope(self) -> str:
        binding = getattr(self, "_scope_binding", None)
        if binding is not None:
            return binding.label
        # Pre-refactor embedding hosts sometimes construct the deprecated
        # MememMemory adapter without LocalMemory.__init__. Their label is not
        # project authority; retain it only for the legacy direct-Memem API.
        return str(getattr(self, "_legacy_scope_label", "default"))

    @_scope.setter
    def _scope(self, value: str) -> None:
        binding = getattr(self, "_scope_binding", None)
        if binding is None:
            self._legacy_scope_label = str(value or "default")
            return
        self._scope_binding = _MemoryScopeState(
            project_id=binding.project_id,
            workspace_id=binding.workspace_id,
            workspace_root=binding.workspace_root,
            resource_revision=binding.resource_revision,
            label=str(value or binding.label),
        )

    def set_scope(
        self, *, project_id: str, workspace_id: str = "", label: str = "",
        workspace_root: str = "", resource_revision: str = "",
    ) -> None:
        current = self._scope_binding
        # Normalize every potentially-failing input before the single pointer
        # replacement. A concurrent recall sees all of A or all of B, never B's
        # project id with A's dependency root/revision.
        next_scope = _MemoryScopeState(
            project_id=str(project_id) if project_id else current.project_id,
            workspace_id=str(workspace_id or ""),
            workspace_root=os.path.realpath(workspace_root) if workspace_root else "",
            resource_revision=str(resource_revision or ""),
            label=str(label) if label else current.label,
        )
        self._scope_binding = next_scope

    @property
    def knowledge_repository(self) -> KnowledgeRepository:
        if self._knowledge is None:
            raise KnowledgeConflictError(
                f"native knowledge repository is unavailable ({self._knowledge_error or 'unknown error'})",
            )
        return self._knowledge

    @property
    def evidence_archive(self):
        return self

    @property
    def work_repository(self):
        return self

    def knowledge_health(self) -> dict:
        if self._knowledge is None:
            native = {
                "active": False, "backend": "native-unavailable",
                "error": self._knowledge_error or "unknown",
            }
        else:
            try:
                native = self._native_knowledge_index.health()
            except Exception as exc:  # noqa: BLE001 — status is diagnostic and never load-bearing
                native = {"active": False, "backend": "native", "error": type(exc).__name__}
            if self._native_operation_error:
                native = {
                    **native,
                    "active": False,
                    "state": "degraded",
                    "error": self._native_operation_error,
                }
        if self._memem_knowledge_index is not None and self._memem_knowledge_index.is_active:
            memem = self._memem_knowledge_index.health()
            # LocalMemory tracks startup/index lifecycle in language useful to
            # ContextFS while the backend provides structured counters.
            backend_degraded = memem.get("state") == "degraded"
            local_state = getattr(self, "_memem_state", "disabled")
            state = "degraded" if backend_degraded or local_state == "degraded" else "healthy"
            detail = getattr(self, "_memem_detail", "status not reported")
            if backend_degraded and memem.get("error"):
                detail = f"latest backend operation failed ({memem['error']})"
            memem = {
                **memem,
                "active": True,
                "healthy": state == "healthy",
                "state": state,
                "detail": detail,
            }
        else:
            memem = {
                "active": getattr(self, "_memem_state", "disabled") == "healthy",
                "available": bool(getattr(self, "_memem_available", False)),
                "backend": "memem-legacy-compat" if getattr(self, "_memem_available", False) else "none",
                "state": getattr(self, "_memem_state", "unknown"),
                "detail": getattr(self, "_memem_detail", "status not reported"),
            }
        return {
            "native": native,
            "memem": memem,
        }

    def _record_compatibility_write(
        self, channel: str, *, succeeded: bool, error: BaseException | None = None,
    ) -> None:
        """Record best-effort mirror health without making it canonical state."""
        # ``MememMemory`` remains a pre-1.0 compatibility adapter and some
        # embedding hosts construct it through ``__new__`` before assigning a
        # legacy vault. Keep those narrow writers functional even when the new
        # LocalMemory initializer did not run; production instances already
        # own both fields from ``__init__``.
        lock = getattr(self, "_status_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._status_lock = lock
        writes = getattr(self, "_compatibility_writes", None)
        if not isinstance(writes, dict):
            writes = {}
            self._compatibility_writes = writes
        with lock:
            item = writes.setdefault(
                channel, {"attempts": 0, "succeeded": 0, "failed": 0, "last_error": ""},
            )
            item["attempts"] = int(item["attempts"]) + 1
            key = "succeeded" if succeeded else "failed"
            item[key] = int(item[key]) + 1
            if error is not None:
                item["last_error"] = type(error).__name__

    def _put_knowledge(self, record: KnowledgeRecord) -> KnowledgeRecord:
        """Commit canonical meaning, then refresh the optional semantic index."""
        stored = self.knowledge_repository.put(record)
        if self._memem_knowledge_index is not None and self._memem_knowledge_index.is_active:
            try:
                self._memem_knowledge_index.index((stored,))
                self._mark_memem("healthy", "latest canonical L2 projection completed")
            except Exception as exc:
                # Canonical persistence already succeeded. Do not lie or roll it
                # back; expose the projection lag and let search fall back.
                self._mark_memem("degraded", f"latest projection failed ({type(exc).__name__})")
        return stored

    def _query(
        self, text: str = "", *, limit: int = 100, statuses=None, kinds=(),
        paths_context=(),
    ):
        return KnowledgeQuery(
            text=text, user_id=self._user_id, project_id=self._project_id,
            agent_id=self._agent_id, limit=limit,
            statuses=tuple(statuses) if statuses is not None else (KnowledgeStatus.ACTIVE,),
            kinds=tuple(kinds),
            paths_context=tuple(paths_context or ()),
        )

    def knowledge_records(self, *, include_candidates: bool = False, limit: int = 100) -> list[KnowledgeRecord]:
        statuses = [KnowledgeStatus.ACTIVE]
        if include_candidates:
            statuses.extend((KnowledgeStatus.CANDIDATE, KnowledgeStatus.LEGACY_UNPROVENANCED))
        return self.knowledge_repository.query(self._query(limit=limit, statuses=statuses))

    def knowledge_counts(self) -> dict[str, int]:
        return self.knowledge_repository.count_by_axis(self._query())

    @staticmethod
    def _count_legacy_files(path: str, *, suffix: str = "") -> int | None:
        """Count one compatibility directory exactly, without reading record contents."""
        try:
            with os.scandir(path) as entries:
                return sum(
                    1 for entry in entries
                    if entry.is_file(follow_symlinks=False)
                    and (not suffix or entry.name.endswith(suffix))
                )
        except FileNotFoundError:
            return 0
        except OSError:
            return None

    def _legacy_inventory(self) -> dict[str, int | None]:
        """Host-owned compatibility inventory; counts are never treated as L2 records."""
        inventory: dict[str, int | None] = {
            "task_projection_files": self._count_legacy_files(
                os.path.join(self._vault, "tasks"), suffix=".md",
            ),
            "session_projection_files": self._count_legacy_files(
                os.path.join(self._vault, "sessions"), suffix=".md",
            ),
            "episodic_session_files": self._count_legacy_files(
                os.path.join(self._vault, "episodic"), suffix=".jsonl",
            ),
            "subagent_archive_files": self._count_legacy_files(
                os.path.join(self._vault, "subagents"), suffix=".jsonl",
            ),
        }
        try:
            roster_root = os.path.join(self._vault, "roster")
            with os.scandir(roster_root) as entries:
                inventory["roster_profile_files"] = sum(
                    1 for entry in entries
                    if entry.is_dir(follow_symlinks=False)
                    and os.path.isfile(os.path.join(entry.path, "profile.json"))
                )
        except FileNotFoundError:
            inventory["roster_profile_files"] = 0
        except OSError:
            inventory["roster_profile_files"] = None

        # The FTS sidecar may contain both turn and child-discovery rows. Name
        # the metric generically instead of misclassifying every row as an
        # episode or promoting the sidecar into a fourth memory layer.
        try:
            import sqlite3
            from .search_index import default_index_path

            index_path = default_index_path()
            if not os.path.isfile(index_path):
                inventory["legacy_search_rows"] = 0
            else:
                connection = sqlite3.connect(
                    f"file:{index_path}?mode=ro", uri=True, timeout=0.2,
                )
                try:
                    row = connection.execute("SELECT COUNT(*) FROM episodes").fetchone()
                    inventory["legacy_search_rows"] = int(row[0]) if row is not None else None
                finally:
                    connection.close()
        except Exception:
            inventory["legacy_search_rows"] = None
        return inventory

    def _runtime_status(self, key: str) -> tuple[dict | None, str]:
        if self._knowledge is None:
            return None, self._knowledge_error or "native knowledge unavailable"
        try:
            return self._knowledge.get_runtime_metadata(key), ""
        except Exception as exc:
            return None, type(exc).__name__

    def _compatibility_health(self) -> dict:
        """Structured current-process health for best-effort legacy writers."""
        with self._status_lock:
            channels = {name: dict(values) for name, values in self._compatibility_writes.items()}
        attempts = sum(int(item.get("attempts", 0)) for item in channels.values())
        failures = sum(int(item.get("failed", 0)) for item in channels.values())
        for item in channels.values():
            if int(item.get("failed", 0)):
                item["state"] = "degraded"
            elif int(item.get("attempts", 0)):
                item["state"] = "healthy"
            else:
                item["state"] = "not_observed"
        return {
            "state": "degraded" if failures else ("healthy" if attempts else "not_observed"),
            "scope": "current process; compatibility only; never canonical authority",
            "attempts": attempts,
            "failed": failures,
            "channels": channels,
        }

    def _compatibility_retirement_gate(self, health: dict) -> dict:
        """Refuse retirement until explicit parity and fallback proofs exist.

        A migration evaluator may publish the four equivalence/read gates plus
        ``compatibility_writes`` in runtime metadata.  Runtime failures always
        override a stored pass.  Nothing here deletes data automatically.
        """
        proof, proof_error = self._runtime_status("compatibility-retirement:global")
        proof = proof or {}
        names = (
            "canonical_l0_equivalence",
            "canonical_l1_equivalence",
            "canonical_l2_equivalence",
            "legacy_read_fallback",
        )
        gates = {
            name: "passed" if str(proof.get(name, "")).lower() == "passed" else "unproven"
            for name in names
        }
        if int(health.get("failed", 0)):
            gates["compatibility_writes"] = "failed"
        elif int(health.get("attempts", 0)):
            gates["compatibility_writes"] = "passed"
        else:
            gates["compatibility_writes"] = (
                "passed" if str(proof.get("compatibility_writes", "")).lower() == "passed"
                else "unproven"
            )
        ready = all(value == "passed" for value in gates.values())
        detail = (
            "retirement permitted by explicit equivalence/read-fallback proof; deletion remains a separate action"
            if ready else
            "retained until L0/L1/L2 equivalence, legacy read fallback, and compatibility-write health all pass"
        )
        if proof_error:
            detail += f" (proof status unavailable: {proof_error})"
        return {
            "state": "ready" if ready else "blocked",
            "ready": ready,
            "automatic_deletion": False,
            "gates": gates,
            "detail": detail,
        }

    def memory_status(self) -> dict:
        """Canonical self-inspection status with no private path disclosure.

        This surface distinguishes storage/index health from corpus state. It
        also reports the retained compatibility layout separately from typed L2
        so legacy telemetry cannot be narrated as a knowledge-migration backlog.
        """
        inventory = self._legacy_inventory()
        compatibility_health = self._compatibility_health()
        inventory_values = tuple(inventory.values())
        known_legacy = sum(value for value in inventory_values if isinstance(value, int))
        inventory_unknown = any(value is None for value in inventory_values)
        transition, transition_error = self._runtime_status("compatibility-layout:global")
        if transition is None:
            if transition_error:
                transition = {
                    "state": "unknown",
                    "detail": f"compatibility-layout status unavailable ({transition_error})",
                }
            elif known_legacy:
                transition = {
                    "state": "retained",
                    "detail": (
                        "legacy compatibility stores are intentionally retained; no bulk L2 migration is defined"
                    ),
                }
            elif inventory_unknown:
                transition = {
                    "state": "unknown",
                    "detail": "legacy compatibility inventory is incomplete",
                }
            else:
                transition = {
                    "state": "absent",
                    "detail": "no legacy compatibility stores were found",
                }
        with self._status_lock:
            cached = self._last_consolidation_cache.get(self._project_id)
        persisted_consolidation, consolidation_error = self._runtime_status(
            f"consolidation:{self._project_id}",
        )
        historical_outputs = 0
        if cached is None and persisted_consolidation is None and not consolidation_error:
            try:
                historical_outputs = self.knowledge_repository.count_by_source_namespace(
                    self._query(limit=1), "legacy-episodic-session",
                )
            except Exception:
                historical_outputs = 0
        consolidation = (
            cached
            or persisted_consolidation
            or ({
                "state": "historical_output_present",
                "detail": (
                    f"{historical_outputs} current-scope typed record(s) cite legacy episodic input; "
                    "exact run lifecycle predates status tracking"
                ),
            } if historical_outputs else None)
            or ({
                "state": "unknown",
                "detail": f"consolidation record unavailable ({consolidation_error})",
            } if consolidation_error else {
                "state": "not_recorded",
                "detail": "no consolidation attempt is recorded for this project",
            })
        )
        return {
            "legacy_inventory": inventory,
            "legacy_inventory_scope": "global compatibility store; not typed project knowledge",
            "compatibility_health": compatibility_health,
            "compatibility_transition": transition,
            "retirement_gate": self._compatibility_retirement_gate(compatibility_health),
            "last_consolidation": consolidation,
        }

    def read_project_episodes(self, session_id: str, *, project_id: str) -> list[dict]:
        """Read one project's legacy episode mirror from an app-wide session.

        New mirror records carry the stable project identity.  If an older session
        has no tagged rows at all, retain the pre-refactor behavior so its history
        can still be consolidated once under the caller's explicit scope.
        """
        rows = self.read_episodes(session_id)

        def row_project(row: object) -> str:
            if not isinstance(row, dict):
                return ""
            record = row.get("record")
            meta = record.get("meta") if isinstance(record, dict) else None
            return str(meta.get("project_id") or "") if isinstance(meta, dict) else ""

        tagged = [row for row in rows if row_project(row)]
        if not tagged:
            return rows
        identity = str(project_id or "")
        return [row for row in tagged if row_project(row) == identity]

    @staticmethod
    def _record_title(record: KnowledgeRecord) -> str:
        title = record.metadata.get("title") if hasattr(record.metadata, "get") else ""
        # This value is embedded in ContextFS markdown indexes. Keep it one-line and bounded so a stored title
        # cannot manufacture another record row or a fake locator in the model-facing read surface.
        surface = " ".join(str(title or record.content.splitlines()[0]).split())
        return surface[:80] or record.id

    @staticmethod
    def _record_relevant_to_query(record: KnowledgeRecord, terms: list[str]) -> bool:
        """Admission floor over the record's bounded retrieval representation.

        Memem deliberately ranks ``primary_index`` plus ``cues`` instead of the
        complete historical value.  Rechecking only ``record.content`` here
        would discard a valid semantic hit whenever its concise cues carry the
        user's wording (and would make the optional backend appear broken).
        Keep this second gate, but apply it to the same bounded representation:
        title/primary abstraction, cues, paths, applicability, and canonical
        content.  Lifecycle and revision validity remain separate gates.
        """
        if not terms:
            return False
        metadata = record.metadata if hasattr(record.metadata, "get") else {}

        def text_values(value: object) -> list[str]:
            if isinstance(value, str):
                return [value]
            if isinstance(value, (list, tuple)):
                return [str(item) for item in value if str(item).strip()]
            return []

        representation = [
            *text_values(metadata.get("title")),
            *text_values(metadata.get("primary_index")),
            *text_values(metadata.get("cues")),
            *text_values(metadata.get("paths")),
            record.applicability,
            record.content,
        ]
        searchable = "\n".join(representation).casefold()
        matched = sum(
            (
                bool(re.search(r"\b" + re.escape(term.casefold()) + r"\b", searchable))
                if term.isascii() else term.casefold() in searchable
            )
            for term in dict.fromkeys(terms)
        )
        # One discriminating term is sufficient for a short query. Longer requests need two independent
        # anchors so a record sharing only "entry", "service", or another broad word is not auto-injected.
        required = 1 if len(set(term.casefold() for term in terms)) <= 2 else 2
        return matched >= required

    def _record_auto_admissible(self, record: KnowledgeRecord) -> bool:
        """Lifecycle/revision gate for automatic seed push, not explicit recall.

        No wall-clock decay is applied. USER preferences and reusable CRAFT
        procedures remain standing until an explicit lifecycle/freshness change.
        Project diagnostics are different: closed reports and observations whose
        declared dependency revision drifted stay available by locator/search but
        cannot silently narrate the current workspace.
        """
        if record.freshness is KnowledgeFreshness.STALE:
            return False
        if record.scopes.project_id is None:
            return True

        metadata = record.metadata
        role = str(metadata.get("memory_role") or metadata.get("record_type") or "").strip().lower()
        if not role and record.applicability.strip().lower() == "corrective engineering work":
            role = "diagnostic_issue"
        if role in {"diagnostic", "diagnostic_issue", "bug_report", "corrective_issue"}:
            issue_state = str(metadata.get("issue_state") or "").strip().lower()
            # A completed failure→fix report is evidence/history and an explicit
            # search lead, not a standing claim that the old bug is still open.
            if issue_state not in {"open", "unresolved", "current"}:
                return False

        declared_revisions = {
            str(ref.resource_revision) for ref in record.source_refs if ref.resource_revision
        }
        metadata_revision = str(metadata.get("resource_revision") or "").strip()
        if metadata_revision:
            declared_revisions.add(metadata_revision)
        if self._resource_revision and declared_revisions and declared_revisions != {self._resource_revision}:
            return False

        workspace_revision = metadata.get("workspace_revision")
        if workspace_revision and self._workspace_root:
            try:
                from collections.abc import Mapping
                from .workspace_revision import WorkspaceRevision

                def thaw(value):
                    if isinstance(value, Mapping):
                        return {str(key): thaw(child) for key, child in value.items()}
                    if isinstance(value, tuple):
                        return [thaw(child) for child in value]
                    return value

                revision = WorkspaceRevision.from_dict(thaw(workspace_revision))
                if os.path.realpath(revision.root) != self._workspace_root or not revision.is_current():
                    return False
            except Exception:
                # Revision-bound knowledge fails closed to explicit pull.
                return False
        return True

    def recall(self, query: str, k: int = 6, paths: list[str] | None = None) -> list[Snippet]:
        """Compile a bounded, typed knowledge push for the current request.

        USER preferences are standing collaboration constraints and receive a tiny always-present budget.
        PROJECT and CRAFT records must win a scoped lexical search for the live request. Every item stays a
        labelled lead with a canonical locator; current user text and fresh observation still outrank it.
        """
        out: list[Snippet] = []
        try:
            from .code_index import _terms
            query_terms = _terms(query)
            # code discovery intentionally extracts ASCII identifiers. Knowledge requests can be multilingual,
            # so retain non-ASCII word runs as additional admission anchors instead of turning those requests
            # into an accidental "never recall project knowledge" rule.
            seen_terms = {term.casefold() for term in query_terms}
            for term in re.findall(r"\w+", query or "", flags=re.UNICODE):
                if term.isascii() or term.casefold() in seen_terms:
                    continue
                query_terms.append(term)
                seen_terms.add(term.casefold())
            records = self.knowledge_repository.query(self._query(
                limit=100, kinds=(KnowledgeKind.PREFERENCE,),
            ))
            standing_user = [
                record for record in records
                if record.scopes.user_id is not None
                and record.kind is KnowledgeKind.PREFERENCE
                and self._record_auto_admissible(record)
            ][:min(2, k)]
            seen_ids = set()
            for record in standing_user:
                out.append(Snippet(
                    path=f"@sliceagent/memory/records/{record.id}.md",
                    text=("[USER knowledge preference — sourced lead; CURRENT REQUEST overrides] "
                          + record.content),
                    score=100.0,
                ))
                seen_ids.add(record.id)
            search_limit = min(100, max(k * 6, 24))
            for hit in self._knowledge_index.search(self._query(
                query, limit=search_limit, paths_context=paths or (),
            )):
                record = hit.record
                if record.id in seen_ids:
                    continue
                if not self._record_auto_admissible(record):
                    continue
                if not self._record_relevant_to_query(record, query_terms):
                    continue
                if record.scopes.project_id is not None:
                    label = "PROJECT knowledge — verify against fresh workspace state"
                elif record.scopes.agent_id is not None:
                    label = "CRAFT knowledge — sourced reusable lead"
                else:
                    continue
                out.append(Snippet(
                    path=f"@sliceagent/memory/records/{record.id}.md",
                    text=f"[{label}] {record.content}", score=float(hit.score),
                ))
                seen_ids.add(record.id)
                if len(out) >= k:
                    break
            self._native_operation_error = ""
        except Exception as exc:
            # The seed builder treats an absent native result as no push. ContextFS health remains the explicit
            # diagnostic surface and never converts this operational failure into a claim of zero records.
            self._native_operation_error = type(exc).__name__
            out = []
        # Memem results now enter through ``KnowledgeIndex.search`` above and
        # are resolved to canonical record ids before this admission gate.  Do
        # not append the old unprovenanced vault tail: that path created a
        # second authority and let full historical reports bypass typed L2.
        return out[:k]

    def seed_recall(self, query: str, k: int = 6, paths: list[str] | None = None) -> list[Snippet]:
        """Typed standing/relevance push seam used by the dependency-first seed compiler."""
        return self.recall(query, k=k, paths=paths)

    def _scope_for(self, scope: str) -> KnowledgeScope:
        normalized = str(scope or "").strip().lower()
        if normalized == "user":
            return KnowledgeScope(user_id=self._user_id)
        if normalized in ("craft", "agent"):
            return KnowledgeScope(agent_id=self._agent_id)
        return KnowledgeScope(project_id=self._project_id)

    def remember(self, content: str, *, title: str = "", scope: str = "default", tags: str = "",
                 paths: list[str] | None = None) -> None:
        """Record an unsourced/manual input as a typed candidate."""
        self._remember_scoped(
            content, scopes=self._scope_for(scope),
            title=title, tags=tags, paths=paths,
        )

    def remember_for_project(
        self, content: str, *, project_id: str, title: str = "", tags: str = "",
        paths: list[str] | None = None,
    ) -> None:
        """Write a project candidate against an immutable scope binding.

        Background consolidation can overlap a live workspace handoff.  It must
        therefore carry the project identity captured by its owning workspace
        instead of consulting the mutable foreground scope at write time.
        """
        identity = str(project_id or "").strip()
        if not identity:
            raise ValueError("project_id must be non-empty")
        self._remember_scoped(
            content, scopes=KnowledgeScope(project_id=identity),
            title=title, tags=tags, paths=paths,
        )

    def _remember_scoped(
        self, content: str, *, scopes: KnowledgeScope,
        title: str = "", tags: str = "", paths: list[str] | None = None,
    ) -> None:
        if scan_for_threats(f"{title}\n{content}", scope="strict"):
            return
        content, title = redact_text(content), redact_text(title)
        try:
            record = KnowledgeRecord.create(
                kind=KnowledgeKind.FACT, scopes=scopes, content=content,
                applicability="manual or legacy memory ingestion", status=KnowledgeStatus.CANDIDATE,
                authority="unverified", proof_family="unavailable",
                sensitivity=KnowledgeSensitivity.PRIVATE,
                metadata={"title": title, "tags": tags, "paths": list(paths or ())},
            )
            self._put_knowledge(record)
        except Exception:
            pass

    def _record_consolidation_status(
        self, *, project_id: str, attempted_at: str, source_episode_count: int,
        mode: str, stats: dict[str, int],
    ) -> None:
        outputs = int(stats.get("lessons", 0)) + int(stats.get("skills", 0))
        errors = int(stats.get("errors", 0))
        rejected = int(stats.get("skills_rejected", 0))
        if errors:
            state = "partial" if outputs else "failed"
        elif rejected:
            state = "completed_with_rejections"
        elif outputs:
            state = "completed"
        else:
            state = "no_eligible_output"
        payload = {
            "state": state,
            "attempted_at": attempted_at,
            "source_episode_count": max(0, int(source_episode_count)),
            "mode": str(mode or "deterministic"),
            "lessons": int(stats.get("lessons", 0)),
            "skills": int(stats.get("skills", 0)),
            "skills_rejected": rejected,
            "errors": errors,
        }
        with self._status_lock:
            self._last_consolidation_cache[project_id] = dict(payload)
        if self._knowledge is None:
            return
        try:
            self._knowledge.set_runtime_metadata(
                f"consolidation:{project_id}", payload, updated_at=attempted_at,
            )
        except Exception:
            pass  # diagnostics cannot turn a completed consolidation into a foreground failure

    def mark_used(self, memory_id: str) -> None:
        """Record exposure separately; it deliberately does not strengthen rank or lifecycle."""
        if not memory_id:
            return
        try:
            self.knowledge_repository.feedback(FeedbackEvent(
                record_id=memory_id, kind=FeedbackKind.SERVED,
                metadata={"surface": "legacy-memory-interface"},
            ))
        except Exception:
            pass

    def consolidate(self, session_id: str, *, llm=None, mode: str = "deterministic") -> dict:
        """Promote cache-derived lessons into typed native L2 and procedures into reviewed skill assets."""
        return self.consolidate_for_project(
            session_id, project_id=self._project_id, workspace_id=self._workspace_id,
            llm=llm, mode=mode,
        )

    def consolidate_for_project(
        self, session_id: str, *, project_id: str, workspace_id: str = "",
        llm=None, mode: str = "deterministic",
    ) -> dict:
        """Consolidate a session using the project identity captured when it ran."""
        from .neocortex import promote_episodes, promote_procedures, render_skill, render_skill_llm
        stats = {"lessons": 0, "skills": 0, "skills_rejected": 0, "errors": 0}
        attempted_at = _now_iso()
        source_episode_count = 0
        bound_project_id = str(project_id or "").strip()
        if not bound_project_id:
            stats["errors"] += 1
            return stats
        try:
            episodes = self.read_project_episodes(session_id, project_id=bound_project_id)
            source_episode_count = len(episodes)
            if not episodes:
                self._record_consolidation_status(
                    project_id=bound_project_id, attempted_at=attempted_at,
                    source_episode_count=0, mode=mode, stats=stats,
                )
                return stats
            source_text = json.dumps(episodes, ensure_ascii=False, sort_keys=True, default=str)
            source_ref = KnowledgeSourceRef.bind_text(
                "legacy-episodic-session", session_id, source_text,
                observer="sliceagent-host", observed_at=_now_iso(),
                project_id=bound_project_id, workspace_id=str(workspace_id or "") or None,
            )
            for lesson in promote_episodes(episodes):
                try:
                    # Consolidation may be retried during shutdown/recovery.  Bind identity to the exact
                    # source digest, project, and derived meaning so replay is idempotent instead of creating
                    # another active fact (or another Memem mirror) on every retry.
                    identity = json.dumps({
                        "source": source_ref.digest,
                        "project_id": bound_project_id,
                        "agent_id": self._agent_id,
                        "kind": KnowledgeKind.FACT.value,
                        "title": lesson["title"],
                        "content": lesson["content"],
                    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    record_id = "knowledge-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
                    if self.knowledge_repository.get(record_id) is not None:
                        continue
                    record = KnowledgeRecord.create(
                        record_id=record_id,
                        kind=KnowledgeKind.FACT,
                        scopes=KnowledgeScope(project_id=bound_project_id, agent_id=self._agent_id),
                        content=lesson["content"], applicability="corrective engineering work",
                        source_refs=(source_ref,), authority="derived",
                        proof_family="execution_outcome", observed_at=_now_iso(),
                        freshness=KnowledgeFreshness.CURRENT, status=KnowledgeStatus.ACTIVE,
                        sensitivity=KnowledgeSensitivity.PRIVATE,
                        metadata={"title": lesson["title"], "tags": lesson["tags"],
                                  "paths": lesson.get("files") or [], "frequency": lesson.get("freq", 1),
                                  "memory_role": "diagnostic_issue", "issue_state": "resolved"},
                    )
                    self._put_knowledge(record)
                    stats["lessons"] += 1
                except Exception:
                    stats["errors"] += 1
            skills_dir = _skills_dir()
            for procedure in promote_procedures(episodes):
                try:
                    body = render_skill_llm(procedure, llm) if mode == "llm" else render_skill(procedure)
                    if write_skill_file(procedure["name"], body, skills_dir=skills_dir):
                        stats["skills"] += 1
                    else:
                        stats["skills_rejected"] += 1
                except Exception:
                    stats["errors"] += 1
        except Exception:
            stats["errors"] += 1
        self._record_consolidation_status(
            project_id=bound_project_id, attempted_at=attempted_at,
            source_episode_count=source_episode_count, mode=mode, stats=stats,
        )
        return stats

    # --- task state / resume (legacy L1 compatibility projection) ---
    def checkpoint_task(self, task: TaskState) -> None:
        report = self._record_compatibility_write
        try:
            private_dir(self._vault)
            d = os.path.join(self._vault, "tasks")
            private_dir(d)
            path = os.path.join(d, f"{task.task_id}.md")
            created = _now_iso()
            if os.path.exists(path):  # preserve the original created on update
                with open(path, encoding="utf-8") as f:
                    fm, _ = _split_frontmatter(f.read())
                created = fm.get("created") or created
            updated = _now_iso()
            # redact the WHOLE rendered task state before it lands on disk — title/goal/findings/last_error/
            # resolution/world are all model/tool-derived and may carry secrets (mirrors the episodic
            # cache redaction). Redact-the-output is future-proof: new fields are covered automatically.
            _write_atomic(path, redact_text(_render_task_md(task, created=created, updated=updated)))
        except Exception as exc:
            report("task_projection", succeeded=False, error=exc)
            return
        report("task_projection", succeeded=True)
        try:
            _upsert_session_index(self._vault, task, updated)
        except Exception as exc:
            report("session_projection", succeeded=False, error=exc)
            return
        report("session_projection", succeeded=True)

    def load_task(self, task_id: str) -> TaskState | None:
        tid = _safe_vault_id(task_id)
        if tid is None:
            return None   # reject path-traversal in a model/user-controlled id
        try:
            path = os.path.join(self._vault, "tasks", f"{tid}.md")
            if os.path.exists(path):
                private_file(path)
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

    def close(self) -> None:
        try:
            HippocampusMixin.close(self)
        finally:
            try:
                if self._knowledge is not None:
                    self._knowledge.close()
            except Exception:
                pass


class MememMemory(LocalMemory, NeocortexMixin):
    """Deprecated compatibility adapter preserving the pre-refactor direct-Memem behavior.

    Production uses :class:`LocalMemory`; keeping this class avoids breaking embedding hosts while the broad
    protocol is retired.  In particular, old tests and explicit callers still observe Memem access feedback.
    """

    def __init__(self) -> None:
        import memem.retrieve  # noqa: F401
        super().__init__(prefer_memem=True)

    def recall(self, query: str, k: int = 6, paths: list[str] | None = None) -> list[Snippet]:
        return NeocortexMixin.recall(self, query, k=k, paths=paths)

    def remember(self, content: str, *, title: str = "", scope: str = "default", tags: str = "",
                 paths: list[str] | None = None) -> None:
        return NeocortexMixin.remember(
            self, content, title=title, scope=scope, tags=tags, paths=paths,
        )

    def mark_used(self, memory_id: str) -> None:
        return NeocortexMixin.mark_used(self, memory_id)

    def consolidate(self, session_id: str, *, llm=None, mode: str = "deterministic") -> dict:
        return NeocortexMixin.consolidate(self, session_id, llm=llm, mode=mode)


def make_memory(prefer_memem: bool = True):
    """Return the native runtime; optional Memem never controls canonical L0, L1, or native L2."""
    return LocalMemory(prefer_memem=prefer_memem)
