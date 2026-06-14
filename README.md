# memagent

A **memory-native coding agent**. Its core bet is a different memory model from every mainstream agent:

> **Don't accumulate the transcript — reconstruct a small, deterministic working state every turn.**

Mainstream agents (and the strong open-source ones — OpenClaw, Hermes) accumulate a growing message history and **LLM-summarize it when it nears the context window** ("transcript + compaction"). memagent never accumulates: each turn it rebuilds a bounded **Active Memory Slice** from ground truth — the live files, the last error (verbatim), a counted action tally, recent actions, and retrieved context — and sends only that.

## Why

- **Bounded by construction** — per-turn context stays flat regardless of session length (no grow-to-window sawtooth).
- **Faithful** — context is re-read from ground truth, not a lossy summary of the conversation.
- **Auditable** — you can print the exact, small input the model saw each turn and know *why* it decided.
- **Cheap at scale** — validated: on long/iterative tasks the slice cut tokens up to ~60–80% and wall-clock ~70% vs a transcript loop, with identical test pass rates.

This is the opposite of the field's default ("bigger windows + summarize"): **remember less, reconstruct precisely.**

## Status

Early. The **core idea is validated** in a ~250-line JS prototype (see [`prototype/`](prototype/)) through controlled experiments vs a classic transcript loop. The production build is Python (aligns with [memem](https://github.com/TT-Wang/memem)).

## Architecture (build / borrow / plug)

The discipline: **own the thin differentiated core, borrow the thick commodity periphery.**

- **Build (the moat):** the slice loop, the typed memory tiers + per-tier compaction, the reconstruction. Plus thin glue: permission gate, verification orchestration, subagents, resume.
- **Plug:** [memem](https://github.com/TT-Wang/memem) as the retrieval + cross-session memory engine (behind a `Retriever` interface).
- **Borrow:** LLM SDKs, tree-sitter (repo map), ripgrep (search), a container sandbox, MCP (tool breadth), a TUI lib, SWE-bench (evals). Patterns from Aider / OpenHands / Hermes.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full component map, the four core interfaces, and the phased plan.

## The differentiator, in one line

> **Deterministic reconstruction from ground truth** — vs the incumbents' **accumulate-then-LLM-summarize**.
