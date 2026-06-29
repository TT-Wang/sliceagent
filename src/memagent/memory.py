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
import threading


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


class MememMemory:
    """Adapter over memem (lessons) + the on-disk state vault. Construction fails fast if memem
    isn't importable. The vault is memagent-owned (_vault_root), decoupled from memem's state dir."""

    is_durable = True

    def __init__(self) -> None:
        import memem.retrieve  # noqa: F401  — fail fast if memem is absent
        self._vault = _vault_root()
        self._scope = os.path.basename(os.getcwd()) or "default"   # same-project soft bonus on recall
        self._idx_lock = threading.Lock()   # serialize the lazy FTS-index open across parallel explorers

    # --- lessons ---
    def recall(self, query: str, k: int = 6, paths: list[str] | None = None) -> list[Snippet]:
        """Recall cross-session lessons, then GATE by relevance: drop hits that share no term with
        the goal (memem returns top-k with no floor → cross-domain noise). Pass scope_id so same-
        project lessons get memem's soft bonus, and `paths` (R1, the files in play at topic-start) so
        memem's paths_context gives lessons tagged with those files a small bonus. REINFORCE the
        surfaced (relevant) ones via mark_used. Decay of the unreinforced is memem's own job."""
        from memem.retrieve import retrieve
        try:
            hits = retrieve(query, k=k, log_call_type=None, writeback=False, scope_id=self._scope,
                            paths_context=paths or None)
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

    def remember(self, content: str, *, title: str = "", scope: str = "default", tags: str = "",
                 paths: list[str] | None = None) -> None:
        # (a) BLOCK on WRITE: a poisoned lesson would re-inject into the slice every turn forever.
        if scan_for_threats(f"{title}\n{content}", scope="strict"):
            return
        # (c) REDACT on PERSIST: never durably store a leaked secret that then re-surfaces.
        content = redact_text(content)
        title = redact_text(title)
        from memem.operations import memory_save
        try:
            memory_save(content, title=title, scope_id=scope, tags=tags, paths=paths or None)  # R1: tag with files
        except Exception:
            pass

    # --- episodic cache (lossless, cold) ---
    def _clamp(self, v):
        if isinstance(v, str) and len(v.encode("utf-8")) > _MAX_RECORD_VALUE_BYTES:
            b = v.encode("utf-8"); h = _MAX_RECORD_VALUE_BYTES // 2   # slice on BYTES (cap is a byte budget);
            # errors="replace" (not "ignore"): a byte cut mid-multibyte-char marks it U+FFFD instead of
            # silently deleting bytes — visible, lossless-ish boundary rather than a quiet corruption.
            head = b[:h].decode("utf-8", "replace"); tail = b[-h:].decode("utf-8", "replace")
            return redact_text(head + f"\n…[truncated {len(v)} chars]…\n" + tail)
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
        # Redact + byte-bound EVERY string leaf of the WHOLE record, not just steps. Earlier this clamped
        # only steps[*].observation/action — so the top-level title/note/markdown/meta (markdown is rendered
        # from the RAW steps + note) reached the durable cache UNREDACTED and could be surfaced by
        # recall_history. _clamp recurses through dict/list, so one call covers the entire record uniformly.
        return self._clamp(rec)

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
        if idx != "unset":
            return idx                       # fast path: already opened (lock-free)
        with self._idx_lock:                 # double-checked: exactly ONE connection is opened + tracked by close()
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

    def close(self) -> None:
        """#33: close the cached FTS5 episode-index connection (WAL checkpoint + release the fd) at
        session end. Idempotent — safe before the index was ever opened ('unset') or after close."""
        idx = getattr(self, "_idx", None)
        if idx is not None and idx != "unset":
            try:
                idx.close()
            except Exception:  # noqa: BLE001
                pass
        self._idx = None

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
            # limit set → keep only the last N parsed dicts via a bounded deque (don't hold the whole
            # session in memory just to slice the tail); limit unset (consolidate) reads all by design.
            from collections import deque
            out = deque(maxlen=limit) if limit is not None else []   # limit=0 ⇒ deque(maxlen=0) keeps ZERO (was: read all)
            with open(path, encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        try:
                            out.append(json.loads(ln))
                        except ValueError:
                            continue
            return list(out)
        except Exception:
            return []

    def episode_manifest(self, session_id: str, k: int) -> tuple[list[dict], int]:
        """(last_k_dicts, total_count) for the PAGED-OUT HISTORY manifest. Reads only the file TAIL and
        parses only ~k records, so the per-turn slice build stays O(k) instead of re-parsing the whole
        session JSONL every turn (which was O(n²) over a long session). Never raises."""
        try:
            path = os.path.join(self._vault, "episodic", f"{session_id}.jsonl")
            if not os.path.exists(path):
                return [], 0
            size = os.path.getsize(path)
            total = 0
            window = min(size, max(65536, (k + 1) * 4096))
            with open(path, "rb") as f:
                for line in f:                 # cheap newline count (no JSON parse) for the '…older' flag
                    if line.strip():
                        total += 1
                f.seek(max(0, size - window))
                tail = f.read()
            rows = tail.splitlines()
            if size > window and rows:
                rows = rows[1:]                # the window may start mid-line → drop the partial leader
            out: list[dict] = []
            for ln in rows:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln.decode("utf-8", "replace")))
                except ValueError:
                    continue
            return out[-k:], total
        except Exception:
            return [], 0

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

    def consolidate(self, session_id: str, *, llm=None, mode: str = "deterministic") -> dict:
        """Session-end sweep: promote durable knowledge from the episodic CACHE (never the slice),
        ROUTED BY TYPE — FACTS (corrective lessons) → memem; PROCEDURES (repeated smooth workflows) →
        reusable SKILL.md the SkillManager discovers next session. `mode="llm"` (with an `llm`) renders
        skill bodies via render_skill_llm (generalized) instead of render_skill (recorded); both fall
        back to deterministic on any failure. RETURNS a stats dict so the caller reports the truth (never a
        blind 'success'); each item is guarded so one bad lesson/skill can't abort the batch. Never raises."""
        stats = {"lessons": 0, "skills": 0, "skills_rejected": 0, "errors": 0}
        try:
            from .consolidate import promote_episodes, promote_procedures, render_skill, render_skill_llm
            records = self.read_episodes(session_id)
            if not records:
                return stats
            for lesson in promote_episodes(records):                 # facts → memem (tagged with files: R1)
                try:
                    self.remember(lesson["content"], title=lesson["title"], scope=self._scope,
                                  tags=lesson["tags"], paths=lesson.get("files"))
                    stats["lessons"] += 1
                except Exception:  # noqa: BLE001 — one bad lesson must not sink the rest
                    stats["errors"] += 1
            skills_dir = _skills_dir()
            for proc in promote_procedures(records):                 # procedures → skill packs (AUTO)
                try:
                    body = (render_skill_llm(proc, llm) if mode == "llm" else render_skill(proc))
                    if write_skill_file(proc["name"], body, skills_dir=skills_dir):   # guarded; None = rejected
                        stats["skills"] += 1
                    else:
                        stats["skills_rejected"] += 1                # validate/threat-scan rejected it
                except Exception:  # noqa: BLE001
                    stats["errors"] += 1
        except Exception:  # noqa: BLE001 — promote_*/read failures never break teardown
            stats["errors"] += 1
        return stats

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
