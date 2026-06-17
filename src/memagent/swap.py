"""SwapManager — single owner of the working-set PAGE lifecycle (file/dep/skill/ghost/reviewed); every page enters/
leaves the slice THROUGH here. DUCK-TYPED: imports nothing from slice.py (reverse import circular — slice re-imports
the bounds, re-exported there for callers/tests); only prefetch() reaches self.retriever."""
from __future__ import annotations

READ_BUDGET = 4    # recent exploratory reads kept (residue) — SINGLE owner; slice.py re-exports all bounds below
EDIT_CEILING = 8   # max files in the change set
DEP_CEILING = 4    # max read-only DEPENDENCIES of the change set kept co-resident
MAX_GHOSTS = 6     # GHOST INDEX ring — pointers to recently paged-out files/skills
MAX_ACTIVE_SKILLS = 2    # keep only the most-recently-loaded skills active
MAX_SKILL_CHARS = 4000   # a loaded skill body is capped before it enters the slice
MAX_REVIEWED = 8         # bounded ring of history lookbacks done (the recall_history ratchet)
PIN_CEILING = 12         # max files the LLM may deliberately PIN resident (mlock) — the GENEROUS disaster ceiling


class SwapManager:
    """Owns the working set: file load/evict, dep prefetch, skill load/evict, reviewed ratchet, ghosts."""
    def __init__(self, retriever=None):
        self.retriever = retriever

    def load(self, s, path: str, edited: bool = False) -> None:  # add/refresh a file, then evict to bounds
        if not path:
            return
        s.active_files = [p for p in s.active_files if p != path]
        s.active_files.append(path)
        self._ghost_drop(s, "file", path)   # it's back in the working set → no longer a ghost
        if edited:
            s.edited_files.add(path)
        self.evict(s)

    def pin(self, s, path: str) -> None:
        """DELIBERATE growth (mlock): mark a file resident + reclaim-protected so it survives plain-read
        eviction (a multi-file task pins the files it must keep consistent). Bounded by PIN_CEILING — the
        GENEROUS disaster ceiling: past it the kernel FORCE-COMPACTS the least-recent pin (it never refuses
        or errors). The moat holds: growth is TASK-driven and bounded, never history-proportional."""
        if not path:
            return
        if path not in s.pinned:
            s.pinned.append(path)
        del s.pinned[:-PIN_CEILING]   # force-compact the least-recent pins past the disaster ceiling
        self.load(s, path)            # make it resident now (evict keeps it: it's pinned)

    def unpin(self, s, path: str) -> None:
        """Release a pin — the file reverts to ordinary residue (may page out as the working set moves on)."""
        s.pinned = [p for p in s.pinned if p != path]

    def evict(self, s) -> None:
        """Keep the change set (edited, ≤EDIT_CEILING) + PINNED files (deliberate growth, ≤PIN_CEILING) +
        read-only DEPS (≤DEP_CEILING) + most-recent READ_BUDGET reads. Edited/pinned/deps NEVER evict for a
        plain read (re-observation reach must cover them); other reads are residue. No graph → old rule."""
        edited = [p for p in s.active_files if p in s.edited_files][-EDIT_CEILING:]
        edited_set = set(edited)
        pinned_set = {p for p in s.active_files if p in getattr(s, "pinned", ())} - edited_set
        deps = [p for p in s.active_files
                if p in s.protected_deps and p not in edited_set and p not in pinned_set][-DEP_CEILING:]
        deps_set = set(deps)
        reads = [p for p in s.active_files
                 if p not in s.edited_files and p not in deps_set and p not in pinned_set][-READ_BUDGET:]
        keep = edited_set | pinned_set | deps_set | set(reads)
        for p in s.active_files:
            if p not in keep:
                s.edit_anchor.pop(p, None)
                s.edited_files.discard(p)
                self._ghost_add(s, "file", p)   # paged out of OPEN FILES → leave a recovery pointer
        s.active_files = [p for p in s.active_files if p in keep]

    def prefetch(self, s) -> None:
        """Refresh change-set protected DEPS from the code graph BEFORE evict (a dep must never page out then re-pin). No graph → no-op."""
        if hasattr(self.retriever, "deps") and s.edited_files:
            deps: set = set()
            for e in list(s.edited_files)[:EDIT_CEILING]:
                deps.update(self.retriever.deps(e, limit=DEP_CEILING))
            s.protected_deps = deps - s.edited_files
        elif not s.edited_files:
            s.protected_deps = set()

    def load_skill(self, s, name: str, body: str) -> None:  # fold a SKILL into the active tier; evict overflow
        if not name or not body:
            return
        s.active_skills = [sk for sk in s.active_skills if sk["name"] != name]
        s.active_skills.append({"name": name, "body": body[:MAX_SKILL_CHARS]})
        self._ghost_drop(s, "skill", name)   # freshly loaded → not a ghost
        if len(s.active_skills) > MAX_ACTIVE_SKILLS:
            self.evict_skill(s)

    def evict_skill(self, s) -> None:  # drop skills beyond MAX_ACTIVE_SKILLS (oldest first), each → a ghost
        for sk in s.active_skills[:-MAX_ACTIVE_SKILLS]:
            self._ghost_add(s, "skill", sk["name"])   # evicted skill → recovery pointer
        s.active_skills = s.active_skills[-MAX_ACTIVE_SKILLS:]

    def note_review(self, s, mark: str) -> None:  # RATCHET: record a history lookback (bounded) so it isn't re-done
        if mark and mark not in s.reviewed:
            s.reviewed.append(mark)
            del s.reviewed[:-MAX_REVIEWED]

    def _ghost_add(self, s, kind: str, ref: str) -> None:  # paged-out item → bounded recovery POINTER (~0 tokens)
        if not ref:
            return
        s.ghosts = [g for g in s.ghosts if not (g["kind"] == kind and g["ref"] == ref)]
        s.ghosts.append({"kind": kind, "ref": ref})
        s.ghosts = s.ghosts[-MAX_GHOSTS:]

    def _ghost_drop(self, s, kind: str, ref: str) -> None:  # back IN the slice → no longer a ghost
        s.ghosts = [g for g in s.ghosts if not (g["kind"] == kind and g["ref"] == ref)]


_DEFAULT_SWAP = SwapManager()   # retriever-free ops; touch_file/add_skill in slice.py delegate here
