"""HIPPOCAMPUS — the episodic memory: explicit recall of this session's own past turns. Two
complementary sides live in this one module: the WRITE side (EpisodeSink, buffers one turn's events
and flushes a lossless record via memory.append_episode when the turn closes) and the READ side (the
`history/` virtual files + search_history). Neither ever touches the Slice directly — the cache can
never enter the LLM context except through an explicit read (Markov by construction); this is what
distinguishes HIPPOCAMPUS from PFC (pfc.py, the carried working memory) and from NEOCORTEX
(neocortex.py, the auto-surfaced, distilled lessons vault) — episodic recall is precise, verbatim, and
only happens on request.

WRITE SIDE — episodic cache, the lossless side (MEMORY-SPEC step 1).
An output-only event sink (sibling of LessonMiner): it buffers one turn's events and flushes ONE
record via `memory.append_episode` when the turn closes. It NEVER touches the Slice, so the cache
can never enter the LLM context — Markov by construction. Record shape:
`{steps: [{slice, action:[{name,args,failing}], observation:[...]}], note, meta}` — the SEED slice is
captured once (step 1) plus the turn's accumulated (action, observation) units; lossless for turn recall.

READ SIDE — the episodic cache exposed as READ-ONLY VIRTUAL FILES under `history/` (HistoryFS), which the
model reads/lists/greps with its ordinary file tools. Measured 2026-07-06: the model reaches for
read_file/grep (a pretraining reflex) far more readily than a bespoke recall tool, so surfacing sealed
turns as files converts evicted-fact confabulation 47%->0%. The PAGED-OUT HISTORY manifest in the slice
lists each earlier turn (turn · title · note) WITH the exact read_file("history/turn-N.md") call.
  read_file("history/index.md")     -> the full TIMESTAMPED/TITLED index of this session's turns
  read_file("history/turn-N.md")    -> a specific turn's seal snapshot (actions/observations/note)
  grep <pat> path=history/          -> content search over this session's turn files
  search_history("keywords")        -> FTS5 content search: THIS session + PAST sessions (the one thing
                                       the files can't do — other sessions aren't mounted here)
Reaching back is expected, not a failure: the slice is bounded, so an earlier turn genuinely is not in
front of the model. NON-ACCUMULATION (moat): a read turn is TRANSIENT — it enters context for this loop
only and is never written back into slice state (the slice is rebuilt from the durable stores each turn),
so reads can never rebuild the transcript. history/ + search_history are registered when memory is durable.
"""
from __future__ import annotations

import json
import os
import posixpath
import re
import threading
import time
from collections import deque

from .events import AssistantText, Event, SliceBuilt, ToolResult, TurnEnd, TurnInterrupted
from .pfc import edited_paths_in_code, paths_in_code  # noqa: F401  (paths_in_code kept for back-compat callers)
from .safety import redact_text
from .text_utils import format_ts, now_iso, one_line

# Serializes the count-then-append id assignment in append_subagent_artifact ACROSS THREADS — parallel
# explorers run as threads in ONE process (the scheduler's ThreadPoolExecutor), which is where the id race
# actually happens. FileLock adds cross-PROCESS safety on POSIX, but flock is a no-op on Windows; this
# in-process lock makes the sequential ids collision-proof regardless of flock availability.
_SUBAGENT_APPEND_LOCK = threading.Lock()
# Standing-specialist identity slug (mirrors subagent._VALID_NAME) — storage-side defense in depth:
# roster names become directory names, so anything else must never touch the filesystem.
_ROSTER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,39}$")
# Lessons cap — bounded by CURATION (count), never truncation: newest wins, exact duplicates collapse.
# Deliberately small: the wake seed carries every lesson, so this IS a resident-context budget.
_MAX_LESSONS = 8

_EDIT_TOOL_NAMES = ("edit_file", "append_to_file", "str_replace", "write_file")


def _files_of(event: ToolResult) -> list[str]:
    """CHANGED files for meta['files'] — mirror slice_sink: only SUCCESSFUL edit tools (and the mutated
    paths of a successful execute_code). A read, a dir-scope grep, or a FAILED edit changed nothing, so
    labeling those 'changed/edited' misled recall + mis-classified consolidated lessons (FILE_TOUCHED)."""
    if event.failing:
        return []
    out = []
    args = event.args if isinstance(event.args, dict) else {}   # raw model args may be a non-dict (list/str/number)
    if event.name in _EDIT_TOOL_NAMES:
        p = args.get("path")
        if isinstance(p, str) and p:
            out.append(p)
    if event.name == "execute_code":
        code = args.get("code", "")
        out += edited_paths_in_code(code if isinstance(code, str) else "")   # mutate-only (write/open-w), not reads
    return out


_OBS_KEEP_WHOLE = 2000   # keep an observation WHOLE up to this (covers a ~120-line config / a page of
_OBS_HEAD = 1200         # output), so a value in the MIDDLE survives; beyond it, a generous head+tail.
_OBS_TAIL = 600          # Bounded — the archive is L2 (on disk, not the slice), and recall caps what it serves.


def _obs_excerpt(obs: str) -> str:
    """A BOUNDED excerpt of a tool observation, indented under its trace line, so the actual DATA a turn
    SAW (a value, a grep match, an error) survives into the recallable markdown. The one-line summary
    ('read_file x -> 250 chars, 8 lines') cannot answer 'what was the value?' later — this is the precision
    the cross-slice recall channel needs. Keep the WHOLE observation up to _OBS_KEEP_WHOLE (so a value in
    the MIDDLE of a normal file/output is not lost — the measured gap); only a truly large observation is
    reduced to head+tail. Bounded: the archive is L2 (on disk, not in context until recalled) and
    recall_history caps the total it serves; page-out (#74) already bounds huge tool outputs upstream."""
    o = (obs or "").strip()
    if not o:
        return ""
    body = o if len(o) <= _OBS_KEEP_WHOLE else (o[:_OBS_HEAD] + "\n…⋯…\n" + o[-_OBS_TAIL:])
    return "  " + body.replace("\n", "\n  ")   # indent the excerpt under its "- " trace bullet


def turn_markdown(title: str, steps: list[dict], note: str, meta: dict) -> str:
    """Render a SEALED turn as a clean, self-contained MARKDOWN snapshot — the readable artifact the
    cache holds and the next loop pages back via recall_history (the slice saved into the cache as
    markdown). Distilled, not a raw dump: heading, changed files, outcome, the action→result trace WITH
    a bounded excerpt of each observation (so the data the turn saw is recallable), and the conclusion.
    Built from the buffered turn data alone (no Slice coupling — Markov)."""
    from .tool_summary import summarize_tool_result
    files = meta.get("files") or []
    out = [f"# {title or '(turn)'}"]
    if files:
        out.append(f"**changed files:** {', '.join(files)}")
    if meta.get("stop_reason"):
        out.append(f"**outcome:** {meta['stop_reason']}")
    trace = []
    for st in steps:
        for a, o in zip(st.get("action", []), st.get("observation", [])):
            trace.append("- " + summarize_tool_result(a.get("name", ""), a.get("args", {}), o,
                                                       failing=bool(a.get("failing"))))
            ex = _obs_excerpt(o)        # keep the actual observed DATA, bounded, so recall is USEFUL
            if ex:
                trace.append(ex)
    if trace:
        out.append("\n## what happened\n" + "\n".join(trace))
    if note:
        out.append(f"\n## conclusion\n{note}")
    return "\n".join(out)


class EpisodeSink:
    """Buffers a turn's events; flushes one lossless record on TurnEnd OR TurnInterrupted."""

    def __init__(self, memory, *, session_id: str, task_id_fn, title_fn=lambda: "", outcome_fn=lambda: {}):
        self.memory = memory
        self.session_id = session_id
        self.task_id_fn = task_id_fn   # () -> current task_id (host supplies; Step 3 seam)
        self.title_fn = title_fn       # () -> human title (goal one-liner) for cheap trace-back
        self.outcome_fn = outcome_fn   # () -> {} of task-OUTCOME signals (e.g. requirements_open) for meta
        self._turn = 0
        self._reset()

    def _reset(self) -> None:
        self._steps: list[dict] = []
        self._note = ""
        self._meta = {"failing": False, "files": []}

    def _cur(self) -> dict:
        if not self._steps:
            self._steps.append({"slice": "", "action": [], "observation": []})
        return self._steps[-1]

    def __call__(self, event: Event) -> None:
        if isinstance(event, SliceBuilt):
            # the loop dispatches SliceBuilt for the seed → opens a new step segment
            self._steps.append({"slice": event.rendered, "action": [], "observation": []})
        elif isinstance(event, AssistantText):
            if event.content and event.content.strip():   # content-emitting models' note
                self._note = event.content.strip()
        elif isinstance(event, ToolResult):
            st = self._cur()
            args = event.args if isinstance(event.args, dict) else {}   # coerce: persist a dict so downstream
            st["action"].append({"name": event.name, "args": args, "failing": event.failing})   # (search_index/consolidate) never read a non-dict from the episode
            st["observation"].append(event.output)        # VERBATIM — lossless (not observe()'d)
            note = args.get("note", "")                   # reasoning models' note (empty content)
            if note:
                self._note = note
            if event.failing:
                self._meta["failing"] = True
            self._meta["files"] += _files_of(event)
        elif isinstance(event, TurnEnd):
            self._flush(event.stop_reason, event.usage)   # usage = per-turn TOTAL
        elif isinstance(event, TurnInterrupted):
            self._flush(event.reason, {})                 # abort path: loop returns WITHOUT TurnEnd

    def _flush(self, stop_reason: str, usage: dict) -> None:
        if not self._steps and not self._note and not self._meta["files"]:
            return  # nothing buffered (e.g. the empty TurnEnd right after a TurnInterrupted)
        self._turn += 1
        try:
            try:
                title = self.title_fn() or ""
            except Exception:   # noqa: BLE001 — a title hiccup must not lose the record
                title = ""
            try:
                outcome = self.outcome_fn() or {}   # task-OUTCOME signals (requirements_open) — what
            except Exception:   # noqa: BLE001       # consolidation gates promotion on; never lose a record
                outcome = {}
            meta = {**self._meta, "stop_reason": stop_reason,
                    "ptok": usage.get("prompt_tokens", 0),
                    "ctok": usage.get("completion_tokens", 0),
                    "files": sorted(set(self._meta["files"])),
                    **outcome}
            record = {
                "title": title,            # human breadcrumb for cheap trace-back (topic is task_id)
                "steps": self._steps,      # lossless raw events (full=true / step recall)
                "note": self._note,
                # the SEAL artifact: the turn's slice as a clean MARKDOWN snapshot — what history/turn-N.md
                # serves, so paging a past turn back reads like opening a readable doc.
                "markdown": turn_markdown(title, self._steps, self._note, meta),
                "meta": meta,
            }
            self.memory.append_episode(self.session_id, self.task_id_fn(), self._turn, record)
        finally:
            self._reset()  # reset regardless, so a turn can never bleed into the next


def make_episode_sink(memory, *, session_id: str, task_id_fn, title_fn=lambda: "", outcome_fn=lambda: {}):
    """None for non-durable memory (NullMemory) → host skips it → evals untouched."""
    if not getattr(memory, "is_durable", False):
        return None
    return EpisodeSink(memory, session_id=session_id, task_id_fn=task_id_fn, title_fn=title_fn,
                       outcome_fn=outcome_fn)


_MAX_RECORD_VALUE_BYTES = 256 * 1024  # per-value disk safety valve (one pathological output)


class HippocampusMixin:
    """The durable episodic-cache STORAGE side (lossless turn log on disk). Mixed into MememMemory
    (memory.py) alongside NeocortexMixin (neocortex.py) — `self` at runtime is a concrete MememMemory
    instance, so `self._vault`/`self._idx_lock` (set by MememMemory.__init__) resolve normally via the
    MRO. This is the counterpart EpisodeSink (above) writes through and recall_history's handler
    (below, via `memory.read_episodes`) reads through."""

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
            ts = now_iso()
            clamped = self._clamp_record(record)
            line = {"v": 1, "session_id": session_id, "task_id": task_id, "turn": turn,
                    "ts": ts, "record": clamped}
            from .platform_compat import FileLock
            # FileLock serializes concurrent appenders to this session file (a resumed session reusing its
            # session_id, or a future off-thread writer) so their lines can't interleave into a torn record.
            # Best-effort (real on POSIX, no-op elsewhere); reads already skip an unparsable line either way.
            with open(os.path.join(d, f"{session_id}.jsonl"), "a", encoding="utf-8") as f, FileLock(f):
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
        """Read the session's episodic cache (the read side behind the history/ virtual files and
        search_history). Returns the raw line dicts in turn order; `limit` keeps only the most recent N.
        Never raises."""
        try:
            path = os.path.join(self._vault, "episodic", f"{session_id}.jsonl")
            if not os.path.exists(path):
                return []
            # limit set → keep only the last N parsed dicts via a bounded deque (don't hold the whole
            # session in memory just to slice the tail); limit unset (consolidate) reads all by design.
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

    def append_subagent_artifact(self, session_id: str, artifact: dict) -> str:
        """Archive a subagent's sealed ARTIFACT under the parent SESSION; return its stable handle ('sub-<n>')
        so the parent can recall it via read_file("subagents/<handle>.md"). Mirrors append_episode: append-only
        JSONL at <vault>/subagents/<session>.jsonl, FileLocked so parallel explorers get RACE-SAFE sequential
        ids. Never raises — an archive failure returns '' and the caller falls back to the inline digest."""
        if not session_id:
            return ""
        try:
            d = os.path.join(self._vault, "subagents")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{session_id}.jsonl")
            from .platform_compat import FileLock
            with _SUBAGENT_APPEND_LOCK:                                   # serialize same-process explorer threads
                with open(path, "a+", encoding="utf-8") as f, FileLock(f):   # + cross-process on POSIX
                    f.seek(0)
                    n = sum(1 for ln in f if ln.strip()) + 1             # count-then-append atomic under both locks
                    sid = f"sub-{n}"
                    # _clamp redacts secrets + byte-bounds every string leaf, like append_episode — a child
                    # that quoted a key/token into its report must NOT persist it verbatim on disk.
                    f.write(json.dumps({"v": 1, "id": sid, "ts": now_iso(), "artifact": self._clamp(artifact)},
                                       ensure_ascii=False, default=str) + "\n")   # O_APPEND → writes at EOF
            return sid
        except Exception:
            return ""

    # ── ROSTER — durable standing specialists (v3: hire once, wake many) ─────────────────────────────
    # A NAMED delegation is a standing identity that survives sessions: <vault>/roster/<name>/ holds its
    # profile (identity card), its CAREER (episodes.jsonl of sealed job artifacts), and later its lessons.
    # Identity = archive key, NOT a runtime entity: a dormant specialist costs nothing; waking one is a
    # fresh slice seeded from these files (flat cost regardless of career length).

    @staticmethod
    def _roster_name_ok(name: str) -> bool:
        """Defense-in-depth path guard (spawn already validates): a bad name never touches the filesystem."""
        return bool(name) and bool(_ROSTER_NAME.match(name))

    def _roster_dir(self, name: str) -> str:
        return os.path.join(self._vault, "roster", name)

    def roster_get(self, name: str) -> dict | None:
        """The specialist's profile, or None if never hired. Never raises."""
        if not self._roster_name_ok(name):
            return None
        try:
            with open(os.path.join(self._roster_dir(name), "profile.json"), encoding="utf-8") as f:
                v = json.load(f)
            return v if isinstance(v, dict) else None
        except Exception:
            return None

    def _write_profile_atomic(self, d: str, profile: dict) -> None:
        """Overwrite profile.json via tmp + os.replace — a reader (roster_get, unlocked) or a cross-process
        peer always sees the OLD or NEW file WHOLE, never a torn half-written JSON. The tmp is pid-unique so
        two processes can't clobber each other's staging file."""
        tmp = os.path.join(d, f"profile.json.{os.getpid()}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(d, "profile.json"))

    def roster_hire(self, name: str, kind: str, *, cap: int | None = None) -> dict:
        """ATOMIC get-or-create of a standing identity; returns the AUTHORITATIVE profile (the pre-existing
        one unchanged, or the just-created one) or {} on cap-full/failure. Never raises. The returned dict
        of the caller that PERFORMED the create carries an EPHEMERAL '_created': True marker (never persisted)
        — so exactly ONE caller under a same-name race owns the 'hired' lifecycle announcement (deriving it
        from jobs==0 double-fires, since the race loser also sees jobs==0).

        Race-safety (bug-hunt HIGH): the whole get→cap→create is under _SUBAGENT_APPEND_LOCK so parallel
        spawn threads of the SAME name (the scheduler runs read-only children concurrently) can't both take
        the create path — the loser gets the winner's profile back, so kind-stability is decided once. The
        create uses O_CREAT|O_EXCL (atomic even ACROSS processes); cap is enforced HERE inside the lock."""
        if not self._roster_name_ok(name):
            return {}
        with _SUBAGENT_APPEND_LOCK:
            existing = self.roster_get(name)
            if existing:
                return existing                                  # idempotent — first kind wins (no _created)
            if cap is not None and len(self.roster_list()) >= cap:
                return {}                                        # kernel says no, atomically
            try:
                d = self._roster_dir(name)
                os.makedirs(d, exist_ok=True)
                profile = {"v": 1, "name": name, "kind": kind, "created": now_iso(),
                           "jobs": 0, "last_active": now_iso()}
                ppath = os.path.join(d, "profile.json")
                body = json.dumps(profile, ensure_ascii=False).encode("utf-8")
                try:
                    fd = os.open(ppath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                except FileExistsError:
                    # a cross-process peer claimed the name; its body may not be written yet (the file can be
                    # observably EMPTY for the µs between its O_EXCL and its write). Re-read a few times,
                    # yielding to the peer, so we return THEIR profile rather than {} (which would degrade
                    # this one spawn to a temp). Bounded — a peer that never finishes → temp (a non-issue in
                    # practice; only two processes first-hiring the SAME new name in the same instant race).
                    for _ in range(8):
                        got = self.roster_get(name)
                        if got:
                            return got
                        time.sleep(0.002)
                    return {}
                try:                                             # single write() → the empty window is just
                    os.write(fd, body)                           # open→write (sub-µs), not open→buffered-dump
                finally:
                    os.close(fd)
                return {**profile, "_created": True}             # ephemeral: only the creator owns the announce
            except Exception:
                return {}

    def roster_list(self) -> list[dict]:
        """All standing specialists' profiles, most recently active first. Never raises."""
        try:
            base = os.path.join(self._vault, "roster")
            if not os.path.isdir(base):
                return []
            out = []
            for n in os.listdir(base):
                if self._roster_name_ok(n):
                    p = self.roster_get(n)
                    if p:
                        out.append(p)
            # `or ""` not a default: a present-but-null last_active (hand-edited/legacy record) would make
            # the sort compare None < str → TypeError, crashing roster_list (and everything built on it).
            return sorted(out, key=lambda p: p.get("last_active") or "", reverse=True)
        except Exception:
            return []

    def roster_append_job(self, name: str, artifact: dict) -> str:
        """Append one sealed job to a specialist's CAREER; return its handle ('job-<n>') or ''. NO-OP for
        names never hired (temps don't accumulate careers — hire is deliberate). Race-safe ids + the same
        _clamp redaction as every other archive; bumps the profile's jobs/last_active under the same lock."""
        if not self._roster_name_ok(name) or not self.roster_get(name):
            return ""
        try:
            d = self._roster_dir(name)
            path = os.path.join(d, "episodes.jsonl")
            from .platform_compat import FileLock
            with _SUBAGENT_APPEND_LOCK:
                with open(path, "a+", encoding="utf-8") as f, FileLock(f):
                    f.seek(0)
                    # id = append-count (every non-empty line), the SAME convention as append_episode /
                    # append_subagent_artifact — monotonic + unique. ponytail: an externally-corrupted torn
                    # line would make this drift from roster_read_jobs' parse-count (a missing job-N then
                    # 404s, jobs runs one high); accepted — a torn line is an anomaly, and diverging this one
                    # appender from its siblings to "fix" it costs more than the harmless drift.
                    n = sum(1 for ln in f if ln.strip()) + 1
                    jid = f"job-{n}"
                    f.write(json.dumps({"v": 1, "id": jid, "ts": now_iso(), "artifact": self._clamp(artifact)},
                                       ensure_ascii=False, default=str) + "\n")
                    profile = self.roster_get(name) or {}
                    profile["jobs"] = n
                    profile["last_active"] = now_iso()
                    # W5' lesson curation: append with PROVENANCE (job + date), collapse exact duplicates
                    # (a re-learned lesson refreshes its provenance instead of repeating), cap by count.
                    lesson = self._clamp((artifact.get("lesson") or "").strip())
                    if lesson:
                        lessons = [L for L in (profile.get("lessons") or [])
                                   if isinstance(L, dict)
                                   and L.get("text", "").strip().lower() != lesson.strip().lower()]
                        lessons.append({"text": lesson, "job": jid, "ts": now_iso()})
                        profile["lessons"] = lessons[-_MAX_LESSONS:]
                    self._write_profile_atomic(d, profile)   # tmp+replace: an unlocked reader never sees a torn file
            return jid
        except Exception:
            return ""

    def roster_read_jobs(self, name: str) -> list[dict]:
        """A specialist's career in job order. Torn-line + non-dict tolerant. Never raises."""
        if not self._roster_name_ok(name):
            return []
        try:
            path = os.path.join(self._roster_dir(name), "episodes.jsonl")
            if not os.path.exists(path):
                return []
            out = []
            with open(path, encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        try:
                            v = json.loads(ln)
                        except ValueError:
                            continue
                        if isinstance(v, dict):
                            out.append(v)
            return out
        except Exception:
            return []

    def index_subagent_artifact(self, session_id: str, handle: str, artifact: dict) -> None:
        """W6': additive FTS5 mirror of a sealed subagent artifact, so search_history finds delegated work
        by CONTENT ("what did the auth explorer conclude about refresh?"). NEVER written to the episodic
        JSONL, so the turn timeline (history/, the PAGED-OUT manifest) stays honest: a delegation seal is
        not a turn. Degrades to no-op without FTS5. Redacted like everything persisted.

        CRITICAL: the index is idempotent per (session_id, turn) — index_episode DELETEs the prior row
        with the same key before inserting. So the mirror row's `turn` is the artifact's HANDLE ('sub-3'),
        NOT a constant: a constant (e.g. 0) makes every delegation of a session share one key, and each new
        seal EVICTS all earlier delegated rows (only the last stays searchable). The handle is unique per
        seal AND collision-free with real turns (stored TEXT 'sub-3' != a numeric turn's '3'), so re-indexing
        the same handle correctly replaces only itself."""
        idx = self._episode_index()
        if idx is None or not handle:
            return
        try:
            a = artifact if isinstance(artifact, dict) else {}
            who = a.get("name") or handle
            title = f"[delegated] {who} ({a.get('kind', 'subagent')}): " + one_line(a.get("task", ""), 80)
            body = " ".join(x for x in (a.get("report", ""), " ".join(a.get("findings") or ()),
                                        a.get("lesson", ""), a.get("task", "")) if x)
            idx.index_episode(session_id=session_id, task_id=f"subagent:{handle}", turn=handle, ts=now_iso(),
                              title=self._clamp(title), note=self._clamp(one_line(a.get("lesson") or "", 120)),
                              text=self._clamp(body))
        except Exception:
            pass

    def read_subagent_artifacts(self, session_id: str) -> list[dict]:
        """The parent session's archived subagent artifacts, in spawn order. Torn-line tolerant. Never raises."""
        try:
            path = os.path.join(self._vault, "subagents", f"{session_id}.jsonl")
            if not os.path.exists(path):
                return []
            out = []
            with open(path, encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        try:
                            v = json.loads(ln)
                        except ValueError:
                            continue
                        if isinstance(v, dict):   # a scalar/list line must not reach SubagentFS's .get() → AttributeError
                            out.append(v)
            return out
        except Exception:
            return []


OBS_TAIL = 300         # legacy-record fallback ONLY: per-observation tail when a turn has no stored markdown.
# render_trace returns a fetched turn IN FULL (no read-side content cap) — the bound is the SEAL: the archive
# already excerpts observations at SAVE time (_obs_excerpt), and the turn markdown is a bounded snapshot.


def _short_ts(ts: str) -> str:
    return format_ts(ts)   # "06-16 12:30"


def _tail(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else "…" + s[-n:]


def render_trace(lines: list[dict]) -> str:
    """Page sealed turns back as their clean MARKDOWN snapshot (the seal artifact) — returned IN FULL, no
    read-side size cap (see the constants note: the bound is the seal + transience, not a second read cut;
    a cap here truncated the distilled conclusion at the markdown tail). Falls back to a computed
    action→result trace for older records that predate the stored markdown (per-observation tail only — a
    legacy raw-obs guard; the conclusion/note is kept whole)."""
    from .tool_summary import summarize_tool_result   # fallback path only
    out = []
    for ln in lines:
        rec = ln.get("record", {})
        head = f"\n── turn {ln.get('turn')} · {_short_ts(ln.get('ts',''))} · {rec.get('title') or ''}"
        md = rec.get("markdown")
        if md:                                   # the SEAL artifact — return it directly, in full
            out.append(head + "\n" + md)
        else:                                    # older record without a stored markdown → compute a trace
            block = [head]
            for st in rec.get("steps", []):
                for a, o in zip(st.get("action", []), st.get("observation", [])):
                    summary = summarize_tool_result(a.get("name", ""), a.get("args", {}), o,
                                                    failing=bool(a.get("failing")))
                    block.append(f"  • {summary} → {_tail(o, OBS_TAIL)}")
            if rec.get("note"):
                block.append(f"  ↳ note: {rec['note']}")     # conclusion in full
            out.append("\n".join(block))
    return "\n".join(out).strip() or "(no trace)"


def render_search(mine, cross) -> str:
    """Render a content search. THIS session's matching turns come WITH the read_file call to open the
    turn file — the model searched by content and now has the turn number, so any turn is reachable
    without guessing a number. PAST sessions' FTS5 hits follow as read-only context (their turns aren't
    mounted as files in this session)."""
    out = []
    if mine:
        out.append("# THIS SESSION — content matches (read the full turn with the call shown)")
        for r in mine:
            if str(r.handle).isdigit():
                out.append(f'- turn {r.handle}: {r.preview}  → read_file("history/turn-{r.handle}.md")')
            else:   # W6': a DELEGATED-WORK hit — the seal lives under subagents/, not the turn timeline
                out.append(f'- delegated {r.handle}: {r.preview}  → read_file("subagents/{r.handle}.md")')
    if cross:
        out.append("# CROSS-SESSION RECALL (past sessions — FTS5 over the durable episode index)")
        for r in cross:
            out.append(f"- [{r.handle}] {r.preview}")
    return "\n".join(out) if out else "No content matches found."


# ── history/ — the episodic archive as a read-only VIRTUAL file namespace ─────────────────────────────
# Measured 2026-07-06 (A/B on the gap-detection matrix): the model reaches for read_file/grep — a deeply
# grooved pretraining reflex — but RESISTS the bespoke recall tool, wrongly treating the in-context convo as
# complete. Exposing sealed turns AS files it can read/list/grep converts evicted-fact confabulation 47%→0%
# (recovery 13%→100%). No files hit disk: HistoryFS serves the SAME episodic cache read_episodes reads,
# routed by LocalToolHost whenever a tool path targets `history/`. Read-only (writes rejected upstream); a
# real on-disk file always wins the name (host._history_route), so the virtual view never lies about disk.
HISTORY_MOUNT = "history"
_TURN_FILE = re.compile(r"^turn-(\d+)\.md$")


def _history_leaf(path: str) -> str:
    """The NORMALIZED path tail under the history mount: 'history/turn-2.md' -> 'turn-2.md';
    'history' / 'history/' / 'history//' / 'history/./' -> ''. posixpath.normpath collapses stray '//',
    './' and a leading './' (so read_file("history//turn-1.md") still resolves) — while a nested or
    '..'-escaping path normalizes to something that is NOT a bare index.md/turn-N.md, so it safely misses
    (no traversal; HistoryFS only ever serves exact index.md / turn-N.md names)."""
    p = (path or "").strip().replace("\\", "/")
    if not p:
        return ""
    p = posixpath.normpath(p)          # collapse //, ./, resolve .. — pure string op, no filesystem
    if p == HISTORY_MOUNT:
        return ""
    return p[len(HISTORY_MOUNT) + 1:] if p.startswith(HISTORY_MOUNT + "/") else p


class HistoryFS:
    """Read-only virtual filesystem over THIS session's sealed turns. `index.md` lists every turn (the
    manifest-as-file); each `turn-<N>.md` is that turn's markdown snapshot (the seal artifact, in full).
    Content is served live from the episodic cache (read_episodes) — nothing is written to disk. Duck-typed:
    the LocalToolHost holds one as `_history` and routes read_file/list_files/grep under `history/` here."""

    def __init__(self, memory, session_id: str):
        self._memory = memory
        self._session_id = session_id

    def _lines(self) -> list:
        return self._memory.read_episodes(self._session_id)

    @staticmethod
    def _turn_names(lines) -> list:
        return [f"turn-{ln.get('turn')}.md" for ln in lines if ln.get("turn") is not None]

    def index(self, lines=None) -> str:
        if lines is None:
            lines = self._lines()
        if not lines:
            return "# HISTORY (this session)\n(no earlier turns yet — this is an early turn.)"
        out = ["# HISTORY — your own record of what you did this session "
               "(read a turn: read_file(\"history/turn-<N>.md\"); find by content: search_history(\"keywords\"))"]
        for ln in lines:
            rec = ln.get("record", {})
            title = (rec.get("title") or "(no title)")[:70]
            note = rec.get("note") or ""
            fail = " FAIL" if (rec.get("meta") or {}).get("failing") else ""
            out.append(f"- turn-{ln.get('turn')}.md · {_short_ts(ln.get('ts', ''))} · {title}{fail}"
                       + (f" — {note[:90]}" if note else ""))
        return "\n".join(out)

    def read_file(self, path: str) -> str:
        lines = self._lines()
        leaf = _history_leaf(path)
        if leaf in ("", "index.md"):
            return self.index(lines)
        m = _TURN_FILE.match(leaf)
        if m:
            n = int(m.group(1))
            sel = [ln for ln in lines if ln.get("turn") == n]
            if sel:
                return render_trace(sel)   # the seal snapshot for that turn, in full (same as recall's compact)
            return (f'history/{leaf}: no such turn this session. '
                    f'read_file("history/index.md") for the list of turns.')
        return (f'history/{leaf}: not a history file. Available: index.md, '
                f'{", ".join(self._turn_names(lines)) or "(no turns yet)"}.')

    def listing(self, path: str = HISTORY_MOUNT) -> str:
        names = ["index.md"] + self._turn_names(self._lines())
        return ("\n".join(names)
                + '\n(read index.md for turn titles, or search_history("keywords") to find a turn by content)')

    def _docs_for(self, path, lines) -> list:
        """The (name, text) docs a grep over `path` should scan — SCOPED like ripgrep: the whole namespace
        for the history/ dir, or a single file when `path` targets index.md / a specific turn-N.md."""
        leaf = _history_leaf(path)
        if leaf == "":                     # the history/ dir → the whole namespace
            return [("index.md", self.index(lines))] + [
                (f"turn-{ln.get('turn')}.md", render_trace([ln])) for ln in lines if ln.get("turn") is not None]
        if leaf == "index.md":
            return [("index.md", self.index(lines))]
        m = _TURN_FILE.match(leaf)
        if not m:
            return []                      # a specific non-existent history file → nothing to search
        n = int(m.group(1))
        sel = [ln for ln in lines if ln.get("turn") == n]
        return [(leaf, render_trace(sel))] if sel else []

    def grep(self, pattern: str, *, path: str = HISTORY_MOUNT, output_mode: str = "content",
             context: int = 0, offset: int = 0, limit: int = 50) -> str:
        # ponytail: regex over the rendered turn docs (Python re, not ripgrep — these are virtual). `context`
        # is accepted for arg-parity with the real grep but not implemented (turn docs are seal-bounded).
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"grep: invalid regex ({e})."
        docs = self._docs_for(path, self._lines())   # SCOPE to the requested file, not the whole namespace
        hits, counts = [], {}
        for name, text in docs:
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"history/{name}:{i}:{line}")
                    counts[name] = counts.get(name, 0) + 1
        if not hits:
            return "grep: no matches found."
        if output_mode == "files_with_matches":
            body_lines = [f"history/{n}" for n in counts]
        elif output_mode == "count":
            body_lines = [f"history/{n}:{c}" for n, c in counts.items()]
        else:
            body_lines = hits
        total = len(body_lines)
        window = body_lines[offset:offset + limit]
        if not window:
            return f"grep: no results at offset={offset} ({total} total matches)."
        body = "\n".join(window)
        if offset + limit < total:
            body += f"\n\n[truncated; use offset={offset + limit} to see more]"
        return body


def make_search_history_tool(memory, session_id: str):
    """ToolEntry for search_history: FTS5 content search over PAST sessions AND this session's turns — the
    one thing the greppable history/ files can't do (other sessions aren't mounted as files). Preserves the
    cross-session recall the deleted recall_history(search=…) provided. This-session hits come WITH the
    read_file(\"history/turn-N.md\") call to open them; past-session hits are read-only context. No rein
    needed — each query is a real search returning new info (unlike the old turn-drill loop)."""
    from .pagetable import PageTable
    from .registry import ToolEntry
    # PageTable's episode-xsession backend wraps memory.search_episodes; episode-search-thissession finds
    # this session's turns by content (reachable via read_file once you know the number).
    pages = PageTable(memory=memory, session_id=session_id)

    def _handler(args: dict) -> str:
        q = (args.get("query") or args.get("search") or "")
        q = q.strip() if isinstance(q, str) else ""
        if not q:
            return "search_history: pass a 'query' (FTS5 keywords — AND/OR, \"quoted\", prefix*)."
        mine = pages.lookup(q, kind="episode-search-thissession", k=6)
        cross = pages.lookup(q, kind="episode-xsession", k=6)
        if not mine and not cross:
            return ('No content matches in this or past sessions for that query. Try different keywords, '
                    'or read_file("history/index.md") for this session\'s full turn list.')
        return render_search(mine, cross)

    schema = {"type": "function", "function": {
        "name": "search_history",
        "description": (
            "Find an earlier turn by CONTENT (FTS5 keyword search) across THIS session AND your PAST sessions "
            "— use it when you don't know a turn's number, or to recall how something was solved in an earlier "
            "session. This-session matches come with the read_file(\"history/turn-N.md\") call to open them; "
            "for THIS session you can also read_file(\"history/index.md\") to browse every turn, or grep the "
            "history/ files. Query supports AND/OR, \"quoted phrases\", and prefix*."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "FTS5 keywords to search episode content for."},
        }, "required": ["query"]}}}
    return ToolEntry(name="search_history", schema=schema, handler=_handler, source="builtin")


# ── SUBAGENTS/ virtual namespace — the parent's read-only view of its children's sealed artifacts ──────
# Mirrors HistoryFS (history/), but over read_subagent_artifacts instead of turns. `index.md` is the
# DELEGATED WORK manifest; each `sub-<n>.md` is a child's FULL sealed report — the refinement handle behind
# the bounded digest the spawn tool returned. Served live from the archive; nothing new is written here.
SUBAGENT_MOUNT = "subagents"
_SUB_FILE = re.compile(r"^(sub-\d+)\.md$")
# INSTANCE-NAME alias: subagents/<name>.md resolves to the LATEST artifact sealed under that identity
# (a specialist may do several jobs in one session; sub-N stays the exact per-job handle). The pattern
# mirrors subagent._VALID_NAME; sub-N is matched first so an alias can never shadow a canonical handle.
_NAME_FILE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]{0,39})\.md$")


def _subagent_leaf(path: str) -> str:
    """Normalized tail under the subagents mount ('subagents/sub-2.md' -> 'sub-2.md'; the bare dir -> '')."""
    p = (path or "").strip().replace("\\", "/")
    if not p:
        return ""
    p = posixpath.normpath(p)
    if p == SUBAGENT_MOUNT:
        return ""
    return p[len(SUBAGENT_MOUNT) + 1:] if p.startswith(SUBAGENT_MOUNT + "/") else p


def render_artifact(rec: dict) -> str:
    """A subagent artifact rendered as its full markdown report (what read_file('subagents/sub-N.md') returns).
    Leads with WHO (instance name · kind) and the VERBATIM BRIEF (what they were asked) before the report —
    provenance: the question always travels with the answer, so a narrowly-briefed child is never silently
    cited for broad claims."""
    a = rec.get("artifact") or {}
    who = f"{a['name']} · {a.get('kind', 'subagent')}" if a.get("name") else a.get("kind", "subagent")
    out = [f"# {rec.get('id', 'sub-?')} — {who} · {a.get('status', '?')} · {a.get('steps', '?')} steps"]
    if a.get("coverage"):
        out.append(f"coverage: {a['coverage']}")
    if a.get("change_set"):
        out.append("change-set: " + ", ".join(a["change_set"]))
    if a.get("files"):
        out.append("read: " + ", ".join(a["files"][:20]))
    if a.get("refs"):   # the seal's refinement map back to its INPUTS (what this work was built on)
        out.append("built on: " + ", ".join(a["refs"]))
    brief_task = (a.get("brief") or {}).get("task") or a.get("task", "")
    if brief_task:
        out += ["", "## brief (verbatim task this agent was given)", brief_task]
    findings = a.get("findings") or []
    if findings:
        out += ["", "## findings"] + [f"- {f}" for f in findings]
    if a.get("report"):
        out += ["", "## report", a["report"]]
    if a.get("trace"):   # W6': the action path (locator-grade lines) — how the conclusions were reached
        out += ["", "## trace (actions taken)", a["trace"]]
    return "\n".join(out)


def _artifact_excerpt(rec: dict, n: int = 90) -> str:
    a = rec.get("artifact") or {}
    return one_line(a.get("report") or (a.get("findings") or [""])[0] or a.get("task", ""), n)


class SubagentFS:
    """Read-only virtual FS over THIS session's sealed subagent artifacts (the child→parent seals). Duck-typed
    like HistoryFS: the host holds one and routes read_file/list_files/grep under `subagents/` here."""

    def __init__(self, memory, session_id: str):
        self._memory = memory
        self._session_id = session_id

    def _arts(self) -> list:
        return self._memory.read_subagent_artifacts(self._session_id)

    @staticmethod
    def _names(arts) -> list:
        return [f"{r.get('id')}.md" for r in arts if r.get("id")]

    @staticmethod
    def _by_name(arts, stem: str):
        """The LATEST artifact sealed under instance identity `stem` (a specialist may have several jobs
        this session; the name alias always means the most recent one)."""
        sel = [r for r in arts if (r.get("artifact") or {}).get("name") == stem]
        return sel[-1] if sel else None

    def index(self, arts=None) -> str:
        if arts is None:
            arts = self._arts()
        if not arts:
            return "# DELEGATED WORK (this session)\n(no subagent reports yet.)"
        out = ["# DELEGATED WORK — your subagents' sealed reports this session, the ROSTER "
               "(read one IN FULL: read_file(\"subagents/sub-<N>.md\"); a NAMED agent's latest report "
               "is also at read_file(\"subagents/<name>.md\"))"]
        for r in arts:
            a = r.get("artifact") or {}
            who = f" · {a['name']}" if a.get("name") else ""
            out.append(f"- {r.get('id')}.md{who} · {a.get('kind', 'subagent')} · {a.get('status', '?')} · "
                       f"{a.get('steps', '?')} steps — {_artifact_excerpt(r)}")
        return "\n".join(out)

    def read_file(self, path: str) -> str:
        arts = self._arts()
        leaf = _subagent_leaf(path)
        if leaf in ("", "index.md"):
            return self.index(arts)
        m = _SUB_FILE.match(leaf)
        if m:
            sel = [r for r in arts if r.get("id") == m.group(1)]
            if sel:
                return render_artifact(sel[-1])
            return (f'subagents/{leaf}: no such subagent report this session. '
                    f'read_file("subagents/index.md") for the list.')
        nm = _NAME_FILE.match(leaf)   # after _SUB_FILE: an alias can never shadow a canonical handle
        if nm:
            rec = self._by_name(arts, nm.group(1))
            if rec is not None:
                return render_artifact(rec)
            return (f'subagents/{leaf}: no subagent named {nm.group(1)!r} this session. '
                    f'read_file("subagents/index.md") for the roster.')
        return (f'subagents/{leaf}: not a subagent report. Available: index.md, '
                f'{", ".join(self._names(arts)) or "(none yet)"}.')

    def listing(self, path: str = SUBAGENT_MOUNT) -> str:
        arts = self._arts()
        aliases = sorted({f"{n}.md" for n in ((r.get("artifact") or {}).get("name") for r in arts) if n})
        return "\n".join(["index.md"] + self._names(arts) + aliases) + \
            '\n(read index.md for the delegated-work manifest)'

    def _docs_for(self, path, arts) -> list:
        leaf = _subagent_leaf(path)
        if leaf == "":
            return [("index.md", self.index(arts))] + [(f"{r.get('id')}.md", render_artifact(r)) for r in arts]
        if leaf == "index.md":
            return [("index.md", self.index(arts))]
        m = _SUB_FILE.match(leaf)
        if m:
            sel = [r for r in arts if r.get("id") == m.group(1)]
            return [(leaf, render_artifact(sel[-1]))] if sel else []
        nm = _NAME_FILE.match(leaf)
        if nm:
            rec = self._by_name(arts, nm.group(1))
            return [(leaf, render_artifact(rec))] if rec is not None else []
        return []

    def grep(self, pattern: str, *, path: str = SUBAGENT_MOUNT, output_mode: str = "content",
             context: int = 0, offset: int = 0, limit: int = 50) -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"grep: invalid regex ({e})."
        hits, counts = [], {}
        for name, text in self._docs_for(path, self._arts()):
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"subagents/{name}:{i}:{line}")
                    counts[name] = counts.get(name, 0) + 1
        if not hits:
            return "grep: no matches found."
        if output_mode == "files_with_matches":
            body_lines = [f"subagents/{n}" for n in counts]
        elif output_mode == "count":
            body_lines = [f"subagents/{n}:{c}" for n, c in counts.items()]
        else:
            body_lines = hits
        window = body_lines[offset:offset + limit]
        if not window:
            return f"grep: no results at offset={offset} ({len(body_lines)} total)."
        body = "\n".join(window)
        if offset + limit < len(body_lines):
            body += f"\n\n[truncated; use offset={offset + limit} to see more]"
        return body


# ── ROSTER/ virtual namespace — the standing workforce (parent-readable; a child sees ONLY its own) ────
ROSTER_MOUNT = "roster"
_JOB_FILE = re.compile(r"^(job-\d+)\.md$")


def _roster_leaf(path: str) -> str:
    """Normalized tail under the roster mount ('roster/auth-explorer/job-2.md' -> 'auth-explorer/job-2.md')."""
    p = (path or "").strip().replace("\\", "/")
    if not p:
        return ""
    p = posixpath.normpath(p)
    if p == ROSTER_MOUNT:
        return ""
    return p[len(ROSTER_MOUNT) + 1:] if p.startswith(ROSTER_MOUNT + "/") else p


def render_profile(profile: dict, jobs: list) -> str:
    """A specialist's identity card + career manifest (what read_file('roster/<name>/profile.md') returns)."""
    name = profile.get("name", "?")
    out = [f"# {name} — standing {profile.get('kind', '?')} specialist",
           f"hired: {profile.get('created') or '?'} · jobs: {profile.get('jobs', 0)} · "
           f"last active: {profile.get('last_active') or '?'}"]
    if jobs:
        out += ["", "## career (sealed jobs — read one IN FULL: "
                    f'read_file("roster/{name}/job-<N>.md"))']
        for r in jobs:
            a = r.get("artifact") or {}
            out.append(f"- {r.get('id')}.md · {a.get('status', '?')} · {(r.get('ts') or '')[:10]} — "
                       f"{_artifact_excerpt(r)}")
    return "\n".join(out)


class RosterFS:
    """Read-only virtual FS over the DURABLE standing-specialist archive (<vault>/roster/). Duck-typed like
    SubagentFS; the host routes read_file/list_files/grep under `roster/` here. The parent browses the whole
    workforce; a CHILD is blocked at the SubagentHost guard except for its OWN files (self-memory is not a
    third channel)."""

    def __init__(self, memory):
        self._memory = memory

    def index(self) -> str:
        profiles = self._memory.roster_list()
        if not profiles:
            return ("# ROSTER (standing specialists)\n(none hired yet. Naming a delegation hires one: "
                    "spawn_agent/spawn_explore with name=\"...\" — re-using the name later WAKES the same "
                    "specialist with its career and lessons.)")
        out = ["# ROSTER — your standing specialists (wake one: spawn_agent(agent=<kind>, name=<name>, "
               "task=...); browse one: read_file(\"roster/<name>/profile.md\"))"]
        for p in profiles:
            out.append(f"- {p.get('name')} · {p.get('kind', '?')} · {p.get('jobs', 0)} job(s) · "
                       f"last active {(p.get('last_active') or '?')[:10]}")
        return "\n".join(out)

    def read_file(self, path: str) -> str:
        leaf = _roster_leaf(path)
        if leaf in ("", "index.md"):
            return self.index()
        name, _, rest = leaf.partition("/")
        profile = self._memory.roster_get(name)
        if profile is None:
            return (f'roster/{leaf}: no standing specialist named {name!r}. '
                    f'read_file("roster/index.md") for the roster.')
        if rest in ("", "profile.md"):
            return render_profile(profile, self._memory.roster_read_jobs(name))
        if rest == "lessons.md":
            lessons = profile.get("lessons") or []
            if not lessons:
                return f"roster/{name}/lessons.md: (no lessons recorded yet.)"
            return "\n".join([f"# {name} — lessons from past jobs (advisory priors, may be stale)"] +
                             [f"- {L.get('text', '')}  ({L.get('job', '?')}, {(L.get('ts') or '')[:10]})"
                              for L in lessons if isinstance(L, dict)])
        m = _JOB_FILE.match(rest)
        if m:
            sel = [r for r in self._memory.roster_read_jobs(name) if r.get("id") == m.group(1)]
            if sel:
                return render_artifact(sel[-1])
            return (f'roster/{leaf}: no such job. read_file("roster/{name}/profile.md") for the career.')
        return (f'roster/{leaf}: not a roster file. Available under roster/{name}/: profile.md, '
                f'lessons.md, job-<N>.md.')

    def listing(self, path: str = ROSTER_MOUNT) -> str:
        leaf = _roster_leaf(path)
        if leaf == "":
            names = [p.get("name") for p in self._memory.roster_list()]
            return "\n".join(["index.md"] + [f"{n}/" for n in names if n]) + \
                "\n(read index.md for the standing-specialist roster)"
        name = leaf.partition("/")[0]
        if self._memory.roster_get(name) is None:
            return f"roster/{leaf}: no standing specialist named {name!r}."
        jobs = [f"{r.get('id')}.md" for r in self._memory.roster_read_jobs(name) if r.get("id")]
        return "\n".join(["profile.md", "lessons.md"] + jobs)

    def _docs_for(self, path) -> list:
        leaf = _roster_leaf(path)
        if leaf in ("", "index.md"):
            docs = [("index.md", self.index())]
            if leaf == "":   # whole-mount grep also sweeps every specialist's files
                for p in self._memory.roster_list():
                    n = p.get("name")
                    if n:
                        docs += self._docs_for(f"{ROSTER_MOUNT}/{n}")
            return docs
        name, _, rest = leaf.partition("/")
        profile = self._memory.roster_get(name)
        if profile is None:
            return []
        jobs = self._memory.roster_read_jobs(name)
        if rest == "":
            return ([(f"{name}/profile.md", render_profile(profile, jobs)),
                     (f"{name}/lessons.md", self.read_file(f"roster/{name}/lessons.md"))] +
                    [(f"{name}/{r.get('id')}.md", render_artifact(r)) for r in jobs])
        doc = self.read_file(f"{ROSTER_MOUNT}/{leaf}")
        return [] if doc.startswith(f"roster/{leaf}:") else [(leaf, doc)]

    def grep(self, pattern: str, *, path: str = ROSTER_MOUNT, output_mode: str = "content",
             context: int = 0, offset: int = 0, limit: int = 50) -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"grep: invalid regex ({e})."
        hits, counts = [], {}
        for name, text in self._docs_for(path):
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"roster/{name}:{i}:{line}")
                    counts[name] = counts.get(name, 0) + 1
        if not hits:
            return "grep: no matches found."
        if output_mode == "files_with_matches":
            body_lines = [f"roster/{n}" for n in counts]
        elif output_mode == "count":
            body_lines = [f"roster/{n}:{c}" for n, c in counts.items()]
        else:
            body_lines = hits
        window = body_lines[offset:offset + limit]
        if not window:
            return f"grep: no results at offset={offset} ({len(body_lines)} total)."
        body = "\n".join(window)
        if offset + limit < len(body_lines):
            body += f"\n\n[truncated; use offset={offset + limit} to see more]"
        return body
