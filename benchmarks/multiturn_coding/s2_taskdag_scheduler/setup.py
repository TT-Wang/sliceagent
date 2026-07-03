import os

# Seed project: a tiny deterministic task scheduler over a dependency graph.
# This is the starting repo BEFORE turn 1. It only knows how to BUILD the graph
# (register tasks and edges and read them back). It has no ordering, cycle
# detection, running, removal, or serialization -- those arrive across the 10
# dependent user turns. The adjacency representation (name -> set of direct
# dependency names) is documented in the module docstring and every later turn
# builds on it.

SEED_SCHEDULER = '''\
"""A tiny deterministic task scheduler over a dependency graph.

Design notes (READ THIS before changing behavior):
  * The graph is kept as a single dict ``_deps`` mapping
        task name -> set(names of that task's DIRECT dependencies).
    Registering a task with no dependencies stores an empty set.
  * "task depends_on dep" means ``dep`` must run BEFORE ``task``; it is
    recorded by adding ``dep`` to ``_deps[task]``.
  * Everything is deterministic: there is no randomness, wall-clock, or
    network use anywhere, and any ordering exposed to callers is sorted.

Public API (do not rename without updating callers):
  add_task(name), add_dependency(task, depends_on),
  tasks(), dependencies(name)
"""


class Scheduler:
    def __init__(self):
        # name -> set of direct dependency names
        self._deps = {}

    # ----- graph construction --------------------------------------------
    def add_task(self, name):
        """Register a task. Idempotent: re-adding an existing task is a no-op
        and never clears its recorded dependencies."""
        if name not in self._deps:
            self._deps[name] = set()

    def add_dependency(self, task, depends_on):
        """Record that ``task`` depends on ``depends_on`` (so ``depends_on``
        must run first). Both tasks are auto-registered if new."""
        self.add_task(task)
        self.add_task(depends_on)
        self._deps[task].add(depends_on)

    # ----- graph reads ----------------------------------------------------
    def tasks(self):
        """Sorted list of all registered task names."""
        return sorted(self._deps.keys())

    def dependencies(self, name):
        """Sorted list of ``name``'s DIRECT dependencies."""
        return sorted(self._deps[name])
'''

SEED_README = '''\
# taskdag

A tiny deterministic task scheduler over a dependency graph, used as a teaching
toy.

Current capabilities:
  * register tasks and dependency edges (`add_task` / `add_dependency`)
  * read the graph back (`tasks` / `dependencies`)

See `taskdag/scheduler.py` for the design notes. Keep the adjacency design
(`name -> set of direct dependency names`); new features should build on
`_deps` rather than replacing it.
'''

SEED_TEST = '''\
"""Smoke tests that ship with the seed. These exercise only the seed feature
set (graph construction and reads). Keep them green."""
from taskdag.scheduler import Scheduler


def test_add_and_list():
    s = Scheduler()
    s.add_task("a")
    s.add_task("b")
    s.add_dependency("b", "a")
    assert s.tasks() == ["a", "b"]
    assert s.dependencies("b") == ["a"]


def test_add_dependency_autoregisters():
    s = Scheduler()
    s.add_dependency("y", "x")
    assert s.tasks() == ["x", "y"]
    assert s.dependencies("y") == ["x"]
    assert s.dependencies("x") == []


def test_add_task_idempotent():
    s = Scheduler()
    s.add_dependency("b", "a")
    s.add_task("b")  # must not wipe b's dependency on a
    assert s.dependencies("b") == ["a"]


if __name__ == "__main__":
    test_add_and_list()
    test_add_dependency_autoregisters()
    test_add_task_idempotent()
    print("seed tests ok")
'''


def setup(workdir):
    pkg = os.path.join(workdir, "taskdag")
    os.makedirs(pkg, exist_ok=True)
    with open(os.path.join(pkg, "__init__.py"), "w") as f:
        f.write("from .scheduler import Scheduler\n")
    with open(os.path.join(pkg, "scheduler.py"), "w") as f:
        f.write(SEED_SCHEDULER)
    with open(os.path.join(workdir, "README.md"), "w") as f:
        f.write(SEED_README)
    tests = os.path.join(workdir, "tests")
    os.makedirs(tests, exist_ok=True)
    with open(os.path.join(tests, "test_seed.py"), "w") as f:
        f.write(SEED_TEST)
