import os

# Correct, full implementation of the task scheduler after all ten turns.
# VALIDATION ONLY -- never shown to the benchmarked agents.
#
# Two invariants make the turn-7 and turn-9 regressions vanish:
#   * remove_task purges the removed name from EVERY other task's dependency
#     set (not just its own entry), so no dangling edge is ever left behind
#     for topo_order()/waves()/ready() to trip on.
#   * run() computes the failed/skipped frontier TRANSITIVELY: a task is
#     skipped iff ANY task it depends on (directly or through a chain) has
#     status 'failed' or 'skipped'. Because run walks in topo order, each
#     task's dependencies already have a final status when it is reached.

REFERENCE = '''\
"""A tiny deterministic task scheduler over a dependency graph.

Design notes:
  * ``_deps`` maps task name -> set of DIRECT dependency names.
  * "task depends_on dep" => ``dep`` runs before ``task``.
  * Deterministic throughout: alphabetical tie-breaks, no randomness/clock.
"""


class Scheduler:
    def __init__(self):
        self._deps = {}

    # ----- graph construction --------------------------------------------
    def add_task(self, name):
        if name not in self._deps:
            self._deps[name] = set()

    def add_dependency(self, task, depends_on):
        if task == depends_on:
            raise ValueError("a task cannot depend on itself")
        self.add_task(task)
        self.add_task(depends_on)
        # Would adding task->depends_on create a cycle? That happens iff
        # depends_on can already reach task through existing edges.
        if self._reaches(depends_on, task):
            raise ValueError("adding this dependency would create a cycle")
        self._deps[task].add(depends_on)

    def _reaches(self, start, target):
        """True if ``start`` can reach ``target`` following dependency edges."""
        seen = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur == target:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self._deps.get(cur, ()))
        return False

    # ----- graph reads ----------------------------------------------------
    def tasks(self):
        return sorted(self._deps.keys())

    def dependencies(self, name):
        return sorted(self._deps[name])

    # ----- removal --------------------------------------------------------
    def remove_task(self, name):
        if name not in self._deps:
            raise KeyError(name)
        del self._deps[name]
        for deps in self._deps.values():
            deps.discard(name)

    # ----- cycle detection ------------------------------------------------
    def has_cycle(self):
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in self._deps}

        def visit(n):
            color[n] = GRAY
            for m in self._deps.get(n, ()):
                if color.get(m, WHITE) == GRAY:
                    return True
                if color.get(m, WHITE) == WHITE and visit(m):
                    return True
            color[n] = BLACK
            return False

        for n in self._deps:
            if color[n] == WHITE and visit(n):
                return True
        return False

    # ----- ordering -------------------------------------------------------
    def topo_order(self):
        if self.has_cycle():
            raise ValueError("graph contains a cycle")
        # Kahn's algorithm with an alphabetically-sorted frontier.
        indeg = {n: len(self._deps[n]) for n in self._deps}
        # dependents[d] = tasks that directly depend on d
        dependents = {n: [] for n in self._deps}
        for task, deps in self._deps.items():
            for d in deps:
                dependents[d].append(task)
        frontier = sorted(n for n in self._deps if indeg[n] == 0)
        order = []
        while frontier:
            n = frontier.pop(0)
            order.append(n)
            newly_free = []
            for m in dependents[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    newly_free.append(m)
            for m in newly_free:
                # insert keeping frontier sorted
                lo, hi = 0, len(frontier)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if frontier[mid] < m:
                        lo = mid + 1
                    else:
                        hi = mid
                frontier.insert(lo, m)
        return order

    def waves(self):
        if self.has_cycle():
            raise ValueError("graph contains a cycle")
        placed = set()
        levels = []
        remaining = set(self._deps.keys())
        while remaining:
            level = sorted(
                n for n in remaining if self._deps[n] <= placed
            )
            # (level is non-empty because the graph is acyclic)
            levels.append(level)
            placed.update(level)
            remaining -= set(level)
        return levels

    # ----- ready ----------------------------------------------------------
    def ready(self, done):
        done = set(done)
        return sorted(
            n for n in self._deps
            if n not in done and self._deps[n] <= done
        )

    # ----- running --------------------------------------------------------
    def run(self, runner):
        status = {}
        for name in self.topo_order():
            # A task is skipped if any direct dependency ended failed/skipped.
            # In topo order those dependencies already have final status.
            blocked = any(
                status.get(d) in ("failed", "skipped")
                for d in self._deps[name]
            )
            if blocked:
                status[name] = "skipped"
                continue
            try:
                runner(name)
                status[name] = "ok"
            except Exception:
                status[name] = "failed"
        return status

    # ----- serialization --------------------------------------------------
    def to_dict(self):
        return {
            "tasks": sorted(self._deps.keys()),
            "deps": {n: sorted(self._deps[n]) for n in self._deps},
        }

    @classmethod
    def from_dict(cls, d):
        s = cls()
        for name in d.get("tasks", []):
            s.add_task(name)
        for name, deps in d.get("deps", {}).items():
            s.add_task(name)
            for dep in deps:
                s.add_task(dep)
                s._deps[name].add(dep)
        return s
'''


def apply(workdir):
    store = os.path.join(workdir, "taskdag", "scheduler.py")
    with open(store, "w") as f:
        f.write(REFERENCE)
