"""Cost + reliability metrics — the moat-MEASURING observer (borrowed from Kimi's usage/telemetry layer,
re-expressed for the slice thesis).

The project's whole bet is that per-turn cost stays FLAT as the conversation grows (the slice rebuilds a
bounded seed each turn) while a transcript agent's climbs linearly. That bet is only credible if it's a
NUMBER. This sink makes it one: the headline signal is `per_turn_fresh` — the FRESH (non-cache-read) input
tokens per turn — which should stay flat for memagent and climb for a log-based agent.

Pure OBSERVER, like its sibling `Telemetry`: consumes the loop's events, accumulates counters, emits nothing,
mutates no slice — completely off the moat. It reads the TYPED usage breakdown the llm adapter now produces
(`input_other`/`input_cache_read`/`input_cache_creation`/`output`, from llm._usage_dict). Per-step usage is
accumulated from StepEnd; TurnEnd snapshots the per-turn fresh-input total and resets — so no double-counting
with TurnEnd's cumulative `total`. Wire it into a dispatcher alongside slice_sink/telemetry and read
`.summary()` afterward; `record_error(kind)` folds in the llm error buckets from errors.classify().
"""
from __future__ import annotations

from .events import (ApiRetry, Event, SliceTightened, StepEnd, ToolResult, TurnEnd,
                     TurnInterrupted)


class CostMetrics:
    """Callable event sink accumulating cost + reliability metrics. Read `.summary()` after a run."""

    def __init__(self) -> None:
        self.turns = 0
        self.steps = 0
        self.input_other = 0          # FRESH (non-cache-read) input tokens — the real cost driver
        self.input_cache_read = 0     # input served from the provider prompt cache (~0.1x price)
        self.input_cache_creation = 0
        self.output = 0
        self.per_turn_fresh: list[int] = []   # input_other per TurnEnd — THE moat curve (flat vs climbing)
        self.tool_calls = 0
        self.tool_failures = 0
        self.retries = 0
        self.overflows = 0
        self.errors: dict[str, int] = {}      # classify() kind -> count
        self._turn_fresh = 0                  # accumulator for the in-progress turn

    def __call__(self, e: Event) -> None:
        if isinstance(e, StepEnd):
            self.steps += 1
            self._add(e.usage)
        elif isinstance(e, (TurnEnd, TurnInterrupted)):
            # #56: snapshot + reset on BOTH clean and PARKED turn-ends. Without TurnInterrupted, a parked
            # turn's fresh tokens were dropped from the moat curve AND its accumulator bled into the next
            # turn (double-count); turns/per_turn_fresh undercounted on every interruption.
            self.turns += 1
            self.per_turn_fresh.append(self._turn_fresh)
            self._turn_fresh = 0
            if isinstance(e, TurnInterrupted):
                self.errors[f"park:{e.reason}"] = self.errors.get(f"park:{e.reason}", 0) + 1
        elif isinstance(e, ToolResult):
            self.tool_calls += 1
            if e.failing:
                self.tool_failures += 1
        elif isinstance(e, ApiRetry):
            self.retries += 1
        elif isinstance(e, SliceTightened):
            self.overflows += 1

    def _add(self, usage: dict | None) -> None:
        if not usage:
            return
        fresh = usage.get("input_other", 0) or 0
        self.input_other += fresh
        self._turn_fresh += fresh
        self.input_cache_read += usage.get("input_cache_read", 0) or 0
        self.input_cache_creation += usage.get("input_cache_creation", 0) or 0
        # output: prefer the typed key, fall back to the legacy one (older usage dicts)
        self.output += usage.get("output", usage.get("completion_tokens", 0)) or 0

    def record_error(self, kind: str) -> None:
        """Fold an llm error bucket (errors.classify()['kind']) into the failure histogram. Called by the
        host's retry/closeout path; the loop itself stays observer-only."""
        if kind:
            self.errors[kind] = self.errors.get(kind, 0) + 1

    def summary(self) -> dict:
        input_total = self.input_other + self.input_cache_read + self.input_cache_creation
        hit = round(self.input_cache_read / input_total, 3) if input_total else 0.0
        ptf = self.per_turn_fresh
        return {
            "turns": self.turns,
            "steps": self.steps,
            "input_other": self.input_other,
            "input_cache_read": self.input_cache_read,
            "input_cache_creation": self.input_cache_creation,
            "output": self.output,
            "cache_hit_rate": hit,                                   # cache-read / total input
            "per_turn_fresh": list(ptf),                            # the moat curve
            "avg_turn_fresh": round(sum(ptf) / len(ptf), 1) if ptf else 0.0,
            "peak_turn_fresh": max(ptf) if ptf else 0,
            "tool_calls": self.tool_calls,
            "tool_failures": self.tool_failures,
            "retries": self.retries,
            "overflows": self.overflows,
            "errors": dict(self.errors),
        }


def make_metrics_sink() -> CostMetrics:
    """A CostMetrics instance IS the sink (callable) AND carries the counters to read afterward."""
    return CostMetrics()
