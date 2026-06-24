"""Cross-session episode search — a durable SQLite FTS5 index over episodic records.

PORTED SHAPE from /tmp/hermes-agent/tools/session_search_tool.py (discovery / scroll /
read, ZERO LLM — every shape returns actual stored rows). Adapted to memagent's grain:
Hermes indexes per-MESSAGE rows in a live session DB; memagent has no transcript, so we
index per-EPISODE records (one row per turn) appended by episode.py. The index is an
ADDITIVE sidecar over the already-durable JSONL cache — the JSONL stays the source of
truth, this is a queryable mirror. Recall (history.py) is single-session today; this lets
it search ACROSS sessions without ever feeding a growing transcript.

NO-TRANSCRIPT INVARIANT: this never enters the slice. It is a durable store the model
queries on demand (exactly like recall_history), bounded by `limit`/snippet length.

GRACEFUL DEGRADE: if sqlite3 or FTS5 is unavailable, `EpisodeIndex` construction returns
a no-op (index_episode does nothing, search returns []), so the JSONL path is untouched.

PUBLIC SIGNATURES (pinned — other agents code against these verbatim):
    fts5_available() -> bool
    class EpisodeIndex:
        def __init__(self, db_path: str) -> None
        is_active: bool                         # False when FTS5 unavailable / open failed
        def index_episode(self, *, session_id: str, task_id: str, turn: int,
                          ts: str, title: str, note: str, text: str) -> None
        def search(self, query: str, *, limit: int = 5,
                   exclude_session: str | None = None) -> list[dict]
        def close(self) -> None
    episode_searchable_text(record: dict) -> str
    default_index_path() -> str

`search` returns a list of dicts:
    {"session_id","task_id","turn","ts","title","note","snippet","score"}
ordered by FTS5 rank (best first). `snippet` is the FTS5-highlighted excerpt.
"""
from __future__ import annotations

import os
import re

_FTS_TABLE = "episodes"


def _fts_match_query(q: str) -> str:
    """Turn a free-text query into a SAFE FTS5 MATCH expression. Extract word tokens only and quote each
    (so query punctuation/operators - " * : ( ) AND OR NEAR can never trigger a syntax error that silently
    returns nothing), then OR-join them. OR (not AND) is the recall-correct default: a query carries terms
    the target turn won't all contain — meta/ordinal words ("second", "finding"), the user's own framing,
    stray operators — and AND-joining means ONE absent token zeroes the whole result (the 'can't locate my
    second finding' bug: 'second' appeared in no review turn, so the AND failed). With OR, any term surfaces
    the turn and BM25 rank + the relative floor in search() keep it precise. Empty → '' (no search)."""
    toks = re.findall(r"\w+", q or "", flags=re.UNICODE)
    return " OR ".join(f'"{t}"' for t in toks)


def fts5_available() -> bool:
    """True iff this Python's sqlite3 can create an FTS5 virtual table."""
    try:
        import sqlite3
    except Exception:
        return False
    try:
        con = sqlite3.connect(":memory:")
        try:
            con.execute("CREATE VIRTUAL TABLE _probe USING fts5(x)")
            return True
        finally:
            con.close()
    except Exception:
        return False


def default_index_path() -> str:
    """The index lives under the same vault as the episodic JSONL (memory._vault_root).

    Lazy import keeps this module free of a hard memory.py dependency for tests.
    """
    try:
        from .memory import _vault_root
        root = _vault_root()
    except Exception:
        root = os.path.join(os.path.expanduser("~"), ".memagent", "vault")
    return os.path.join(root, "episodic", "index.db")


def episode_searchable_text(record: dict) -> str:
    """Flatten an episode `record` into one searchable blob (title + note + actions +
    observations + files). Deterministic, bounded per-field so one huge observation can't
    dominate the index. Mirrors what history.render_trace surfaces, so a search matches
    what the model would actually read back."""
    parts: list[str] = []
    title = record.get("title") or ""
    if title:
        parts.append(title)
    note = record.get("note") or ""
    if note:
        parts.append(note)
    for st in record.get("steps", []):
        for a in st.get("action", []):
            name = a.get("name") or ""
            args = a.get("args") or {}
            hint = ""
            for k in ("path", "command", "query", "goal"):
                if args.get(k):
                    hint = str(args[k])[:120]
                    break
            parts.append(f"{name} {hint}".strip())
        for o in st.get("observation", []):
            if isinstance(o, str) and o:
                parts.append(o[:500])   # bounded — head of each observation
    meta = record.get("meta", {})
    for f in meta.get("files", []) or []:
        parts.append(str(f))
    return "\n".join(p for p in parts if p)


class EpisodeIndex:
    """SQLite FTS5 mirror of episodic records, queryable across sessions.

    Best-effort throughout: every method swallows its own errors (an index hiccup must
    never break a session — same discipline as memory.append_episode). When FTS5 is
    unavailable or the DB can't open, `is_active` is False and all writes/reads no-op.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.is_active = False
        self._con = None
        if not fts5_available():
            return
        try:
            import sqlite3
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
            # check_same_thread=False so the OPT-IN background-review fork (item 16) can
            # index from its worker thread without a second connection.
            con = sqlite3.connect(db_path, check_same_thread=False)
            con.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE} USING fts5("
                "session_id UNINDEXED, task_id UNINDEXED, turn UNINDEXED, "
                "ts UNINDEXED, title, note, body, "
                "tokenize='porter unicode61')"
            )
            con.commit()
            self._con = con
            self.is_active = True
        except Exception:
            self._con = None
            self.is_active = False

    def index_episode(self, *, session_id: str, task_id: str, turn: int,
                      ts: str, title: str, note: str, text: str) -> None:
        """Insert one episode row. Idempotent per (session_id, turn): a re-index deletes
        the prior row first, so a replayed/duplicate append can't double-count."""
        if not self.is_active or self._con is None:
            return
        try:
            # turn is stored as TEXT (FTS5 UNINDEXED preserves the exact type), so the DELETE
            # must bind the SAME str — int 2 != stored '2' and the row would never be removed.
            self._con.execute(
                f"DELETE FROM {_FTS_TABLE} WHERE session_id = ? AND turn = ?",
                (session_id, str(turn)),
            )
            self._con.execute(
                f"INSERT INTO {_FTS_TABLE} "
                "(session_id, task_id, turn, ts, title, note, body) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, task_id, str(turn), ts, title or "", note or "", text or ""),
            )
            self._con.commit()
        except Exception:
            pass

    def search(self, query: str, *, limit: int = 5,
               exclude_session: str | None = None,
               only_session: str | None = None) -> list[dict]:
        """FTS5 discovery over indexed sessions. Returns bounded hit dicts ordered by rank.
        `exclude_session` drops the current session lineage (cross-session recall). `only_session`
        restricts to ONE session (within-session content recall of the long tail — turns past the
        manifest/index window). Opposite scopings; pass at most one. Never raises."""
        if not self.is_active or self._con is None:
            return []
        match = _fts_match_query(query)
        if not match:
            return []
        try:
            lim = max(1, min(int(limit), 20))
        except (TypeError, ValueError):
            lim = 5
        try:
            rows = self._con.execute(
                f"SELECT session_id, task_id, turn, ts, title, note, "
                f"snippet({_FTS_TABLE}, 6, '«', '»', ' … ', 12) AS snip, "
                f"rank AS score "
                f"FROM {_FTS_TABLE} WHERE {_FTS_TABLE} MATCH ? "
                f"ORDER BY rank LIMIT ?",
                (match, lim + 10),   # over-fetch so exclude_session can't starve the result
            ).fetchall()
        except Exception:
            return []
        out: list[dict] = []
        for r in rows:
            session_id = r[0]
            if exclude_session and session_id == exclude_session:
                continue
            if only_session and session_id != only_session:
                continue
            try:
                turn = int(r[2])
            except (TypeError, ValueError):
                turn = r[2]
            out.append({
                "session_id": session_id,
                "task_id": r[1],
                "turn": turn,
                "ts": r[3],
                "title": r[4],
                "note": r[5],
                "snippet": r[6],
                # #41: FTS5 `rank` is negative (more-negative = better). Negate so callers reading `score`
                # get an intuitive higher-is-better number; result ORDER already follows `rank` directly.
                "score": -float(r[7]) if r[7] is not None else 0.0,
            })
            if len(out) >= lim:
                break
        # RELATIVE FLOOR (counterweight to OR-breadth): keep only hits scoring within 15% of the top hit,
        # always keeping #1. OR-join maximizes recall; this trims the long tail of turns that matched on a
        # single weak/common term, so a precise query still returns a precise set (and a vague one degrades
        # to "the few most relevant", not "30 loosely-related turns"). Degenerate scores (≤0) → keep all.
        if out:
            top = out[0]["score"]
            if top > 0:
                cut = top * 0.15
                out = [out[0]] + [h for h in out[1:] if h["score"] >= cut]
        return out

    def close(self) -> None:
        if self._con is not None:
            try:
                self._con.close()
            except Exception:
                pass
        self._con = None
        self.is_active = False


def make_episode_index(db_path: str | None = None) -> EpisodeIndex:
    """Factory. `db_path` defaults to default_index_path(). Always returns an EpisodeIndex
    (its `is_active` flag tells callers whether FTS5 actually came up)."""
    return EpisodeIndex(db_path or default_index_path())
