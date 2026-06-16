"""Memory implementations — the RELEVANT MEMORY tier (lessons) + the durable STATE VAULT.

memem is the plug for cross-session lessons: its in-process hybrid retrieval feeds the tier and
`memory_save` stores lessons. On top of that, this module is the **state vault** (MEMORY-SPEC):
an episodic cache (lossless turn log) and resumable task-state records, all on disk under a
memagent-owned vault. memem stays behind the `Memory` interface — the moat never imports it — and
we degrade to NullMemory when memem/its vault is absent.

`is_durable` is the structural marker: NullMemory sets it False, so hosts skip cache/checkpoint
wiring and evals stay deterministic. The vault root is decoupled from memem's STATE dir
(`MEMEM_DIR` = db/logs) — the cache is memagent-owned (`MEMAGENT_VAULT`).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .interfaces import Snippet, TaskRef, TaskState

_MAX_RECORD_VALUE_BYTES = 256 * 1024  # per-value disk safety valve (one pathological output)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _vault_root() -> str:
    """memagent-owned vault root. Prefers a dedicated var; then a document-vault var; NEVER
    MEMEM_DIR (that is memem's state/db dir, not a vault). Falls back to ~/.memagent/vault."""
    for k in ("MEMAGENT_VAULT", "MEMAGENT_CACHE_DIR",
              "MEMEM_OBSIDIAN_VAULT", "CORTEX_OBSIDIAN_VAULT", "MEMEM_VAULT", "CORTEX_VAULT"):
        v = os.environ.get(k)
        if v:
            return os.path.expanduser(v)
    return os.path.join(os.path.expanduser("~"), ".memagent", "vault")


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
    fm = [
        "---", "type: task-state", "v: 1",
        f"session_id: {task.session_id}", f"task_id: {task.task_id}",
        f"title: {task.title}", f"status: {task.status}",
        f"created: {created}", f"updated: {updated}",
        f"since_edit: {task.since_edit}",
        f"links: {','.join(task.links)}", f"tags: {task.tags}", "---",
    ]
    body = [
        "## Goal", task.goal,
        "## Findings", "\n".join(f"- {f}" for f in task.findings),
        "## Working set", "\n".join(f"- {p}" for p in task.active_files),
        "## Edited", "\n".join(f"- {p}" for p in sorted(task.edited_files)),
        # anchor is TAB-separated (a path never contains TAB; anchors may contain ' :: ' etc.)
        "## Anchors", "\n".join(f"- {p}\t{a}" for p, a in task.edit_anchor.items()),
        "## Status", task.last_error,            # verbatim, may be empty/multi-line
        "## Resolution", task.resolution,
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
    return TaskState(
        task_id=fm.get("task_id", ""), session_id=fm.get("session_id", ""),
        title=fm.get("title", ""), status=fm.get("status", "active"),
        goal=sec.get("goal", ""),
        findings=_bullets(sec.get("findings", "")),
        active_files=_bullets(sec.get("working set", "")),
        edited_files=_bullets(sec.get("edited", "")),
        edit_anchor=anchors,
        last_error=sec.get("status", ""),
        since_edit=int(fm.get("since_edit", "0") or 0),
        links=[x for x in fm.get("links", "").split(",") if x],
        tags=fm.get("tags", ""),
        resolution=sec.get("resolution", ""),
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
    title = (task.title or "").replace("\n", " ")
    rows[task.task_id] = f"{task.task_id} · {task.status} · {updated} · {title}"  # title LAST
    lines = ["---", "type: session", f"session_id: {task.session_id}", "---", "## Tasks"]
    lines += [f"- {r}" for r in rows.values()]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _parse_session_index(path: str) -> list[TaskRef]:
    with open(path, encoding="utf-8") as f:
        _, body = _split_frontmatter(f.read())
    out: list[TaskRef] = []
    for b in _bullets(_read_sections(body).get("tasks", "")):
        parts = b.split(" · ", 3)  # task_id · status · updated · title  (title may contain ' · ')
        if len(parts) == 4:
            tid, status, updated, title = parts
            out.append(TaskRef(task_id=tid.strip(), title=title.strip(),
                               status=status.strip(), updated=updated.strip()))
    return out


# --- implementations ----------------------------------------------------------------------

class NullMemory:
    """No durable memory (the default until a vault is configured). A TRUE no-op — every method is
    inert (no I/O, no clock), so the eval path is deterministic and adds nothing to the slice."""

    is_durable = False

    def recall(self, query: str, k: int = 6) -> list[Snippet]:
        return []

    def remember(self, content: str, *, title: str = "", scope: str = "default", tags: str = "") -> None:
        return None

    def append_episode(self, session_id: str, task_id: str, turn: int, record: dict) -> None:
        return None

    def checkpoint_task(self, task: TaskState) -> None:
        return None

    def load_task(self, task_id: str) -> TaskState | None:
        return None

    def list_session_tasks(self, session_id: str) -> list[TaskRef]:
        return []

    def mark_used(self, memory_id: str) -> None:
        return None

    def consolidate(self, session_id: str) -> None:
        return None


class MememMemory:
    """Adapter over memem (lessons) + the on-disk state vault. Construction fails fast if memem
    isn't importable. The vault is memagent-owned (_vault_root), decoupled from memem's state dir."""

    is_durable = True

    def __init__(self) -> None:
        import memem.retrieve  # noqa: F401  — fail fast if memem is absent
        self._vault = _vault_root()

    # --- lessons (unchanged) ---
    def recall(self, query: str, k: int = 6) -> list[Snippet]:
        from memem.retrieve import retrieve
        try:
            hits = retrieve(query, k=k, log_call_type=None, writeback=False)
        except Exception:
            return []
        out: list[Snippet] = []
        for h in hits:
            text = h.get("body") or h.get("title") or ""
            out.append(Snippet(path=h.get("path", ""), text=text, score=float(h.get("score", 0.0))))
        return out

    def remember(self, content: str, *, title: str = "", scope: str = "default", tags: str = "") -> None:
        from memem.operations import memory_save
        try:
            memory_save(content, title=title, scope_id=scope, tags=tags)
        except Exception:
            pass

    # --- episodic cache (lossless, cold) ---
    def _clamp(self, v):
        if isinstance(v, str) and len(v.encode("utf-8")) > _MAX_RECORD_VALUE_BYTES:
            h = _MAX_RECORD_VALUE_BYTES // 2
            return v[:h] + f"\n…[truncated {len(v)} chars]…\n" + v[-h:]
        return v

    def _clamp_record(self, rec: dict) -> dict:
        def acts(lst):
            return [{**a, "args": {k: self._clamp(v) for k, v in a["args"].items()}}
                    if isinstance(a.get("args"), dict) else a for a in (lst or [])]
        rec = dict(rec)
        rec["steps"] = [{**s, "observation": [self._clamp(o) for o in s.get("observation", [])],
                         "action": acts(s.get("action"))} for s in rec.get("steps", [])]
        return rec

    def append_episode(self, session_id: str, task_id: str, turn: int, record: dict) -> None:
        try:
            d = os.path.join(self._vault, "episodic")
            os.makedirs(d, exist_ok=True)
            line = {"v": 1, "session_id": session_id, "task_id": task_id, "turn": turn,
                    "ts": _now_iso(), "record": self._clamp_record(record)}
            with open(os.path.join(d, f"{session_id}.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception:
            pass  # a cache write must never break a session

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
            with open(path, "w", encoding="utf-8") as f:
                f.write(_render_task_md(task, created=created, updated=updated))
            _upsert_session_index(self._vault, task, updated)
        except Exception:
            pass

    def load_task(self, task_id: str) -> TaskState | None:
        try:
            path = os.path.join(self._vault, "tasks", f"{task_id}.md")
            return _parse_task_md(path) if os.path.exists(path) else None
        except Exception:
            return None

    def list_session_tasks(self, session_id: str) -> list[TaskRef]:
        try:
            path = os.path.join(self._vault, "sessions", f"{session_id}.md")
            return _parse_session_index(path) if os.path.exists(path) else []
        except Exception:
            return []

    def consolidate(self, session_id: str) -> None:
        """Session-end sweep: promote durable lessons from the episodic cache into long-term memory
        (the cache→memory loop). Reads the session's JSONL, promotes corrective episodes, remembers
        them. Never raises (a consolidation failure must not break the session)."""
        try:
            from .consolidate import promote_episodes
            path = os.path.join(self._vault, "episodic", f"{session_id}.jsonl")
            if not os.path.exists(path):
                return
            records = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            pass
            for lesson in promote_episodes(records):
                self.remember(lesson["content"], title=lesson["title"], tags=lesson["tags"])
        except Exception:
            pass

    # --- declared now; implemented in step 5 (loud-but-safe so a premature wire is caught) ---
    def mark_used(self, memory_id: str) -> None:
        raise NotImplementedError("mark_used lands in MEMORY-SPEC step 5")


def make_memory(prefer_memem: bool = True):
    """Return MememMemory if memem is importable, else NullMemory (graceful)."""
    if prefer_memem:
        try:
            return MememMemory()
        except Exception:
            pass
    return NullMemory()
