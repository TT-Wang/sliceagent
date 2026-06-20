"""SwapManager — single owner of the working-set PAGE lifecycle (file/dep/skill/ghost/reviewed); every page enters/
leaves the slice THROUGH here. The memory plane of a DEMAND-PAGED SNAPSHOT MACHINE: the slice is a CACHE, not a log,
so eviction is always safe (a re-fault re-reads from the durable store). DUCK-TYPED: imports nothing from slice.py
(reverse import circular — slice re-imports the bounds, re-exported there for callers/tests); only prefetch() reaches
self.retriever. SELF-TUNING (automatic, no model): a re-read of a file still in the recency ring (a REFAULT) proves
the budget was momentarily too tight, so the kernel grants ITSELF a brief reclaim-protection (Linux mm/workingset
refault detection, scaled down) AND widens its OWN read budget one notch (s.read_budget, bounded by s.read_ceiling)
— so the working set grows to TASK need, not to a fixed ceiling. hit/miss/refault/evict are counted (s.io) so the
moat is MEASURED, not asserted."""
from __future__ import annotations

READ_BUDGET = 4    # FLOOR for the exploratory-read residue — the lean DEFAULT, NOT a hard cap. The kernel GROWS the
                   # live budget (s.read_budget) on refault thrash up to READ_BUDGET_MAX, bidirectionally with the
                   # overflow tighten ladder. "Bounded" = no PASSIVE/history-proportional growth (Markov current-state),
                   # never a fixed size ceiling. SINGLE owner; slice.py re-exports all bounds below.
READ_BUDGET_MAX = 16  # per-slice DISASTER CEILING for refault-driven growth. A single COHERENT task rarely needs more
                      # resident reads than this; genuine BREADTH ("review the repo") is delegated to the subagent SWARM
                      # (each child a fresh lean slice), NOT served by inflating one slice toward the context window
                      # (that invites context-rot / lost-in-the-middle — see auto-memory: kernel-architecture).
# bound ≠ size: the change set + dependency closure are RELEVANCE sets, kept resident IN FULL (evict no
# longer truncates them — see evict/prefetch). These two constants are now PHYSICAL fan-in backstops
# (guard against a pathological 100s-of-callers symbol), generous and well above any coherent change —
# NOT relevance caps. swap_tools.py reads them for the occupancy display; the real overflow handler is
# the physical-context tighten ladder, not these.
EDIT_CEILING = 32  # physical backstop on the change set (a coherent change is far smaller); not a relevance cap
DEP_CEILING = 64   # per-symbol caller fan-out backstop — keep ALL direct callers that break on the change
                   # (caller-first ranked in code_index.deps); generous physical guard, not a relevance cap.
MAX_GHOSTS = 6     # GHOST INDEX ring — pointers to recently paged-out files/skills (also the refault recency window)
MAX_ACTIVE_SKILLS = 2    # keep only the most-recently-loaded skills active
MAX_SKILL_CHARS = 4000   # a loaded skill body is capped before it enters the slice
MAX_REVIEWED = 8         # bounded ring of history lookbacks done (the recall_history ratchet)
PIN_CEILING = 12         # max files the LLM may deliberately PIN resident (mlock) — the GENEROUS disaster ceiling
HOT_TTL = 3              # steps a REFAULT-promoted file stays kernel-protected (self-tuning; not the model)
HOT_CEILING = 4  # bound the kernel-granted soft-pin set — never an accumulating tier (decoupled from
                 # DEP_CEILING so raising the dep ceiling doesn't widen the refault soft-pin set)


class SwapManager:
    """Owns the working set: file load/evict, dep prefetch, skill load/evict, reviewed ratchet, ghosts."""
    def __init__(self, retriever=None):
        self.retriever = retriever

    def load(self, s, path: str, edited: bool = False) -> None:  # add/refresh a file, then evict to bounds
        if not path:
            return
        io = getattr(s, "io", None)
        if path in s.active_files:                       # already resident — a cache HIT
            if io is not None:
                io["hit"] += 1
        elif self._is_ghost(s, "file", path):           # re-read of a recently-evicted page — a REFAULT
            if io is not None:
                io["refault"] += 1
            self._promote(s, path)                       # budget was too tight → kernel grants a brief soft-pin
            self._grow(s)                                # …and widens its OWN read budget one notch (bidirectional ladder)
        elif io is not None:                             # first sight — a cold MISS
            io["miss"] += 1
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
        """Keep the change set (edited) + PINNED + HOT + the WHOLE dependency closure (relevance, not a
        count) + most-recent read_budget exploratory reads; page the rest OUT — non-lossy, since every
        evicted file leaves a GHOST recovery pointer and re-reads on demand (a REFAULT, which also widens
        the budget). Edited/pinned/hot/deps never evict for a plain read (re-observation reach must cover
        them). NOTE: whether OPEN FILES should evict at all within a loop is an OPEN design decision (see
        the bound-is-relevance discussion); this is the re-faultable middle ground pending that call."""
        edited_set = {p for p in s.active_files if p in s.edited_files}
        pinned_set = {p for p in s.active_files if p in getattr(s, "pinned", ())} - edited_set
        hot_set = {p for p in s.active_files if p in getattr(s, "hot", {})} - edited_set - pinned_set
        protect = edited_set | pinned_set | hot_set
        deps_set = {p for p in s.active_files if p in s.protected_deps and p not in protect}
        read_budget = getattr(s, "read_budget", READ_BUDGET)   # LIVE adaptive budget (grows on refault); floor = READ_BUDGET
        reads = [p for p in s.active_files
                 if p not in s.edited_files and p not in deps_set and p not in protect][-read_budget:]
        keep = protect | deps_set | set(reads)
        io = getattr(s, "io", None)
        for p in s.active_files:
            if p not in keep:
                s.edit_anchor.pop(p, None)
                s.edited_files.discard(p)
                if io is not None:
                    io["evict"] += 1
                self._ghost_add(s, "file", p)   # paged out of OPEN FILES → leave a recovery pointer
        s.active_files = [p for p in s.active_files if p in keep]

    def prefetch(self, s) -> None:
        """Refresh change-set protected DEPS from the code graph BEFORE evict (a dep must never page out then re-pin).
        Also AGE the kernel-granted soft-pins one step (runs once per build = once per step) so refault protection is
        temporary, never an accumulating tier. No graph → deps no-op; the hot decay still runs."""
        if getattr(s, "hot", None):
            s.hot = {p: t - 1 for p, t in s.hot.items() if t - 1 > 0}
        r = self.retriever
        # Snapshot pre-edit def-names of files READ but not yet edited, so we can later see what an edit
        # REMOVED/MOVED (pre-edit defs - current defs). Cheap: reads the already-cached code graph.
        if hasattr(r, "def_names") and hasattr(s, "pre_defs"):
            for p in s.active_files:
                if p not in s.edited_files and p not in s.pre_defs:
                    s.pre_defs[p] = r.def_names(p)
        if hasattr(r, "deps") and s.edited_files:
            edited = list(s.edited_files)   # the WHOLE change set (relevance, not a count) — materialize ONCE
            deps: set = set()
            for e in edited:
                deps.update(r.deps(e, limit=DEP_CEILING))   # all direct callers that break on the change
            s.protected_deps = deps - s.edited_files
            # CHANGE-SET CLOSURE (symbol-aware): names the edits removed/moved, then the dependents whose
            # CURRENT tokens still reference one — precise dangling call-sites. SILENT on feature-adds
            # (nothing removed) so it never inflates non-refactor tasks; render_closure further filters to
            # the UNOPENED ones (self-extinguishing once the model opens the site to fix/confirm).
            if hasattr(r, "def_names") and hasattr(r, "ref_tokens") and hasattr(s, "stale_deps"):
                removed: set = set()
                for e in edited:
                    removed |= (s.pre_defs.get(e, set()) - r.def_names(e))
                s.stale_deps = ({d for d in s.protected_deps if r.ref_tokens(d) & removed}
                                if removed else set())
        elif not s.edited_files:
            s.protected_deps = set()
            if hasattr(s, "stale_deps"):
                s.stale_deps = set()

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

    def _is_ghost(self, s, kind: str, ref: str) -> bool:  # is this ref currently in the (recency-bounded) ghost ring?
        return any(g["kind"] == kind and g["ref"] == ref for g in s.ghosts)

    def _promote(self, s, path: str) -> None:
        """Kernel grants ITSELF a brief reclaim-protection for a thrashing (refaulted) page — refault-driven,
        NO model involvement (the validated automatic-beats-active-asker path). Bounded by HOT_CEILING; decays
        in prefetch (HOT_TTL steps), so it can never become an accumulating tier."""
        hot = getattr(s, "hot", None)
        if hot is None:
            return
        hot[path] = HOT_TTL
        while len(hot) > HOT_CEILING:        # drop the oldest soft-pin (insertion-ordered) past the ceiling
            hot.pop(next(iter(hot)))

    def _grow(self, s) -> None:
        """A REFAULT proves the resident read budget was momentarily too tight for THIS task (thrash).
        The kernel widens its OWN live budget one notch — the BIDIRECTIONAL counterpart to the overflow
        tighten ladder (which shrinks it). Bounded by the per-slice disaster ceiling (s.read_ceiling);
        kernel-driven from MEASURED thrash (s.io refault), NO model involvement. Monotone within a
        session (a task that needed N reads keeps needing ~N); never history-proportional, so the moat
        holds — growth is TASK/refault-driven and window-bounded, and the kernel can always say no at
        the ceiling. Same signal + same place as _promote: extend self-tuning from a per-file soft-pin
        to the budget itself (Linux mm/workingset, scaled up)."""
        cur = getattr(s, "read_budget", READ_BUDGET)
        ceiling = getattr(s, "read_ceiling", READ_BUDGET_MAX)
        if cur < ceiling:
            s.read_budget = cur + 1

    def _ghost_add(self, s, kind: str, ref: str) -> None:  # paged-out item → bounded recovery POINTER (~0 tokens)
        if not ref:
            return
        s.ghosts = [g for g in s.ghosts if not (g["kind"] == kind and g["ref"] == ref)]
        s.ghosts.append({"kind": kind, "ref": ref})
        s.ghosts = s.ghosts[-MAX_GHOSTS:]

    def _ghost_drop(self, s, kind: str, ref: str) -> None:  # back IN the slice → no longer a ghost
        s.ghosts = [g for g in s.ghosts if not (g["kind"] == kind and g["ref"] == ref)]


_DEFAULT_SWAP = SwapManager()   # retriever-free ops; touch_file/add_skill in slice.py delegate here
