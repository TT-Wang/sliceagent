"""Reconstruction-quality telemetry — measures whether the slice KEEPS what the model needs, so the
slice's central bet becomes a NUMBER instead of an anecdote.

A pure OBSERVER sink: it consumes the loop's events and accumulates counters. It emits no events and
mutates no slice — completely off the moat. Deterministic, bounded, zero LLM. Wire it into a
dispatcher alongside slice_sink (the eval harness does this) and read `.summary()` after the run.

Signals:
  - re_reads : a read_file on a path ALREADY read within the last `window` steps. Within a turn the prior
    read is still in the accumulated transcript, so a re-read signals the model isn't using its resident
    context; across turns it's the reconstruction-MISS signal (the seed/seal didn't carry what it needed).
  - recalls  : accesses to the paged-out episodic cache — a read_file/list_files/grep under history/, or a
    search_history call. Recovery from the cold cache (the model knew it forgot). (Was recall_history, now
    deleted: the same recovery happens through the history/ virtual files.)
  - reads    : total successful read_file calls on REAL files (the denominator for a re-read RATE; a
    history/ read is a recall, counted above, not a source re-read).
"""
from __future__ import annotations

from .events import Event, StepEnd, ToolResult

RE_READ_WINDOW = 6          # a path re-read within this many steps counts as a likely reconstruction miss
_MAX_TRACKED = 256          # bound the per-path last-seen map (never a transcript)


def _is_history_path(path) -> bool:
    """True if a tool path targets the virtual history/ namespace (recovery from the paged-out cache)."""
    if not isinstance(path, str):
        return False
    p = path.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    p = p.rstrip("/")
    return p == "history" or p.startswith("history/")


class Telemetry:
    """Callable event sink that counts reconstruction-cost signals. Read `.summary()` after a run."""

    def __init__(self, window: int = RE_READ_WINDOW):
        self.window = window
        self.step = 0
        self.reads = 0
        self.re_reads = 0
        self.recalls = 0
        self._last_read: dict[str, int] = {}   # path -> step it was last read (bounded)

    def __call__(self, e: Event) -> None:
        if isinstance(e, StepEnd):
            self.step += 1
        elif isinstance(e, ToolResult) and not e.failing:
            path = (e.args or {}).get("path")
            hist = _is_history_path(path)
            if e.name in ("read_file", "list_files", "grep") and hist:
                self.recalls += 1               # recovery from the paged-out cache via the history/ files
            elif e.name == "search_history":
                self.recalls += 1               # cross-session content recall (FTS5)
            elif e.name == "read_file":         # a REAL source read (history reads counted above, not here)
                self.reads += 1
                if path:
                    last = self._last_read.get(path)
                    if last is not None and (self.step - last) <= self.window:
                        self.re_reads += 1          # read again soon after — slice didn't carry it
                    self._last_read[path] = self.step
                    if len(self._last_read) > _MAX_TRACKED:   # bound: drop the oldest entry
                        del self._last_read[min(self._last_read, key=self._last_read.get)]

    def summary(self) -> dict:
        rate = round(self.re_reads / self.reads, 3) if self.reads else 0.0
        return {"reads": self.reads, "re_reads": self.re_reads,
                "re_read_rate": rate, "recalls": self.recalls}


def make_telemetry_sink() -> Telemetry:
    """A Telemetry instance IS the sink (it's callable) AND carries the counters to read afterward."""
    return Telemetry()
