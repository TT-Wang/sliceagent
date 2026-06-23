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
import re
import tempfile


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
from datetime import datetime, timezone

from .code_index import _terms   # reuse the query-term extractor (drops stopwords/short tokens)
from .interfaces import Snippet, TaskRef, TaskState
from .safety import scan_for_threats, redact_text   # persist-guards: block-on-write + redact-on-persist

_MAX_RECORD_VALUE_BYTES = 256 * 1024  # per-value disk safety valve (one pathological output)


def _memory_relevant(text: str, terms: list[str]) -> bool:
    """Relevance gate for the RELEVANT MEMORY tier: keep a recalled lesson only if it shares a
    discriminating term with the goal (whole-word, so 'add' doesn't match 'address'). With no terms
    (un-discriminating query) keep it — we can't gate. This makes the tier relevant-or-nothing
    instead of memem's blind top-k, the same relevance discipline the RELATED CODE map uses."""
    if not terms:
        return True
    blob = (text or "").lower()
    return any(re.search(r"\b" + re.escape(t.lower()) + r"\b", blob) for t in terms)


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


def _skills_dir() -> str:
    """Where consolidation writes promoted-procedure SKILL.md packs — a dir the SkillManager scans
    (default ~/.memagent/skills, so skills are discovered next session). MEMAGENT_SKILLS_DIR overrides."""
    return os.path.expanduser(os.environ.get("MEMAGENT_SKILLS_DIR")
                              or os.path.join("~", ".memagent", "skills"))


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
        "## Goal", task.goal,
        "## Findings", "\n".join(f"- {f}" for f in task.findings),
        # carried slice tiers — JSON-per-bullet so dict items round-trip EXACTLY (no markdown-escape
        # hazard). Without these, resuming a task silently dropped the standing contract / todo / north-
        # star / world model (data loss). Mission is a single verbatim line like Status.
        "## Requirements", "\n".join(f"- {json.dumps(r, ensure_ascii=False)}" for r in task.requirements),
        "## Plan", "\n".join(f"- {json.dumps(p, ensure_ascii=False)}" for p in task.plan),
        "## Mission", task.mission,
        "## World", "\n".join(f"- {json.dumps([k, v], ensure_ascii=False)}" for k, v in task.world.items()),
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
        if isinstance(kv, list) and len(kv) == 2:
            world[kv[0]] = kv[1]
    return TaskState(
        task_id=fm.get("task_id", ""), session_id=fm.get("session_id", ""),
        title=fm.get("title", ""), status=fm.get("status", "active"),
        goal=sec.get("goal", ""),
        findings=_bullets(sec.get("findings", "")),
        requirements=[r for r in _json_bullets("requirements") if isinstance(r, dict)],
        plan=[p for p in _json_bullets("plan") if isinstance(p, dict)],
        mission=sec.get("mission", ""),
        world=world,
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
    _write_atomic(path, "\n".join(lines) + "\n")


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

    def read_episodes(self, session_id: str, *, limit: int | None = None) -> list[dict]:
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

    def consolidate(self, session_id: str) -> None:
        return None


class MememMemory:
    """Adapter over memem (lessons) + the on-disk state vault. Construction fails fast if memem
    isn't importable. The vault is memagent-owned (_vault_root), decoupled from memem's state dir."""

    is_durable = True

    def __init__(self) -> None:
        import memem.retrieve  # noqa: F401  — fail fast if memem is absent
        self._vault = _vault_root()
        self._scope = os.path.basename(os.getcwd()) or "default"   # same-project soft bonus on recall

    # --- lessons ---
    def recall(self, query: str, k: int = 6) -> list[Snippet]:
        """Recall cross-session lessons, then GATE by relevance: drop hits that share no term with
        the goal (memem returns top-k with no floor → cross-domain noise). Pass scope_id so same-
        project lessons get memem's soft bonus, and REINFORCE the surfaced (relevant) ones via
        mark_used — so retrieval feedback tracks what we actually show, not raw top-k. Decay of the
        unreinforced is memem's own job (compute_decay_factor over access_count)."""
        from memem.retrieve import retrieve
        try:
            hits = retrieve(query, k=k, log_call_type=None, writeback=False, scope_id=self._scope)
        except Exception:
            return []
        terms = _terms(query)
        out: list[Snippet] = []
        for h in hits:
            text = h.get("body") or h.get("title") or ""
            if not _memory_relevant(f"{h.get('title', '')} {text}", terms):
                continue                       # relevance gate: relevant-or-nothing, no noise
            self.mark_used(h.get("id", ""))    # reinforce what we surface (feeds memem's decay)
            out.append(Snippet(path=h.get("path", ""), text=text, score=float(h.get("score", 0.0))))
        return out

    def remember(self, content: str, *, title: str = "", scope: str = "default", tags: str = "") -> None:
        # (a) BLOCK on WRITE: a poisoned lesson would re-inject into the slice every turn forever.
        if scan_for_threats(f"{title}\n{content}", scope="strict"):
            return
        # (c) REDACT on PERSIST: never durably store a leaked secret that then re-surfaces.
        content = redact_text(content)
        title = redact_text(title)
        from memem.operations import memory_save
        try:
            memory_save(content, title=title, scope_id=scope, tags=tags)
        except Exception:
            pass

    # --- episodic cache (lossless, cold) ---
    def _clamp(self, v):
        if isinstance(v, str) and len(v.encode("utf-8")) > _MAX_RECORD_VALUE_BYTES:
            h = _MAX_RECORD_VALUE_BYTES // 2
            return redact_text(v[:h] + f"\n…[truncated {len(v)} chars]…\n" + v[-h:])
        if isinstance(v, str):
            return redact_text(v)  # (c) redact every persisted episodic string on its way to the cache
        # #35: recurse into structured (non-string) tool outputs so str leaves inside a dict/list are
        # still byte-bounded + redacted — a huge or secret-bearing nested payload can't slip past.
        if isinstance(v, dict):
            return {k: self._clamp(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [self._clamp(x) for x in v]
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
            ts = _now_iso()
            clamped = self._clamp_record(record)
            line = {"v": 1, "session_id": session_id, "task_id": task_id, "turn": turn,
                    "ts": ts, "record": clamped}
            with open(os.path.join(d, f"{session_id}.jsonl"), "a", encoding="utf-8") as f:
                # #36: default=str — a non-serializable value in a tool output must STRINGIFY, never raise
                # and silently drop the whole turn (the except below would eat it = lost episode + index).
                f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
        except Exception:
            return  # a cache write must never break a session
        self._index_episode(session_id, task_id, turn, ts, clamped)  # additive FTS5 mirror (item 12)

    # --- cross-session FTS5 episode index (item 12; additive, degrades to no-op) ---
    def _episode_index(self):
        """Lazily open the FTS5 episode index (cached). Returns None if FTS5 is
        unavailable — every caller treats None as 'index off' and falls back."""
        idx = getattr(self, "_idx", "unset")
        if idx == "unset":
            try:
                from .search_index import make_episode_index
                idx = make_episode_index()
                if not idx.is_active:
                    idx = None
            except Exception:
                idx = None
            self._idx = idx
        return idx

    def _index_episode(self, session_id, task_id, turn, ts, record) -> None:
        idx = self._episode_index()
        if idx is None:
            return
        try:
            from .search_index import episode_searchable_text
            idx.index_episode(session_id=session_id, task_id=task_id, turn=turn, ts=ts,
                              title=record.get("title", ""), note=record.get("note", ""),
                              text=episode_searchable_text(record))
        except Exception:
            pass

    def search_episodes(self, query: str, *, limit: int = 5,
                        exclude_session: str | None = None,
                        only_session: str | None = None) -> list[dict]:
        """Episode discovery (FTS5). `exclude_session` => cross-session recall; `only_session` =>
        within-session content recall of the long tail (turns past the manifest/index window).
        Returns bounded hit dicts (see search_index.EpisodeIndex.search) or [] when unavailable."""
        idx = self._episode_index()
        if idx is None:
            return []
        try:
            return idx.search(query, limit=limit, exclude_session=exclude_session,
                              only_session=only_session)
        except Exception:
            return []

    def read_episodes(self, session_id: str, *, limit: int | None = None) -> list[dict]:
        """Read the session's episodic cache (the read side of the recall_history tool). Returns the
        raw line dicts in turn order; `limit` keeps only the most recent N. Never raises."""
        try:
            path = os.path.join(self._vault, "episodic", f"{session_id}.jsonl")
            if not os.path.exists(path):
                return []
            out = []
            with open(path, encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        try:
                            out.append(json.loads(ln))
                        except ValueError:
                            continue
            return out[-limit:] if limit else out
        except Exception:
            return []

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
            _write_atomic(path, _render_task_md(task, created=created, updated=updated))
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
        """Session-end sweep: promote durable knowledge from the episodic cache, ROUTED BY TYPE —
        FACTS (corrective lessons) → long-term memory (memem remember); PROCEDURES (repeated smooth
        workflows) → reusable SKILL.md files the SkillManager discovers next session. Both are
        frequency-weighted. Never raises (a consolidation failure must not break the session)."""
        try:
            from .consolidate import promote_episodes, promote_procedures, render_skill
            records = self.read_episodes(session_id)
            if not records:
                return
            for lesson in promote_episodes(records):                 # facts → memem
                self.remember(lesson["content"], title=lesson["title"], scope=self._scope,
                              tags=lesson["tags"])
            skills_dir = _skills_dir()
            for proc in promote_procedures(records):                 # procedures → skill packs
                try:
                    body = render_skill(proc)
                    # (a) BLOCK on WRITE: a poisoned SKILL.md re-injects unscanned every session.
                    if scan_for_threats(body, scope="strict"):
                        continue
                    d = os.path.join(skills_dir, proc["name"])
                    os.makedirs(d, exist_ok=True)
                    with open(os.path.join(d, "SKILL.md"), "w", encoding="utf-8") as f:
                        f.write(redact_text(body))   # (c) redact any secret before it lands on disk
                except Exception:
                    pass
        except Exception:
            pass

    # --- declared now; implemented in step 5 (loud-but-safe so a premature wire is caught) ---
    def mark_used(self, memory_id: str) -> None:
        """Retrieval feedback: reinforce a memory that proved relevant (bumps its access count, which
        feeds memem's decay/ranking). Delegates to memem's primitive — we don't reinvent the decay
        machinery memem already has (obsidian_store.bump_access + decay.compute_decay_factor)."""
        if not memory_id:
            return
        try:
            from memem.obsidian_store import bump_access
            bump_access(memory_id)
        except Exception:
            pass   # feedback is best-effort; must never break a recall


def make_memory(prefer_memem: bool = True):
    """Return MememMemory if memem is importable, else NullMemory (graceful)."""
    if prefer_memem:
        try:
            return MememMemory()
        except Exception:
            pass
    return NullMemory()
