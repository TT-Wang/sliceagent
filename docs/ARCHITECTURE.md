# memagent — architecture

## Principle

Hourglass / thin-waist: a **thin differentiated core** (the slice loop + typed tiers) with **commodity layers above and below** that are standard, verified, and swappable. Spend attention almost entirely on the core; borrow everything else.

## The core loop (the moat)

Every turn:

1. **Reconstruct** the Active Memory Slice from deterministic tiers + retrieval (no accumulated history).
2. **One** model call with `[system+task, slice]`.
3. **Run** the tool calls; fold results into the tiers.
4. **Append** the full record to a durable on-disk log (the event store — never sent to the model).
5. Repeat.

Memory model = **event-sourcing**: the durable log is the event store; the slice is a materialized view rebuilt each turn. The bet is the slice is a **sufficient statistic** for the next decision, so full history is redundant.

## The memory tiers (typed, per-tier compaction)

| Tier | Source | Policy |
|------|--------|--------|
| Task | the goal | stable (in system prompt → cacheable) |
| Current error | last failing tool output | verbatim, deterministic, auto-cleared on a clean run |
| Action tally | tool calls | **counted**, render only repeated/failing (anti-loop) |
| Recent actions | last K steps | sliding window |
| Working set (artifacts) | files being edited | **deterministic, re-read fresh** — never a retrieval guess |
| Discovery set | retrieval (memem) | top-k relevant, recall-biased, agent-correctable |

Different memory types get different compaction: facts → dedup, attempts → count, artifacts → fresh, error → verbatim. (Uniform summarization, the incumbent approach, loses the wrong things.)

## Coping with imperfect retrieval

Retrieval is not 100%. The loop — not the retriever — is the recovery mechanism:
- Keep the **working set deterministic** (retrieval error confined to discovery, not to files being edited).
- Give the agent **exact tools** (grep, symbol search) as the precision backstop to fuzzy embeddings.
- **Errors → re-queries** (the error tier feeds the next search).
- **Verify against oracles** (tests/typecheck/lint) before "done" — independent of retrieval accuracy.
- Net: retrieval accuracy sets *cost* (search turns), oracles guarantee *correctness*.

## The four core interfaces (the core depends on these, not implementations)

- `LLMClient` — provider-agnostic completion + tool-calling (borrow: official SDKs / LiteLLM).
- `Retriever` — per-turn retrieval + cross-session memory (plug: memem).
- `ToolHost` — tool execution behind a sandbox; local / container / MCP (borrow: container, OpenHands runtime, MCP).
- `Oracle` — verification: run tests / typecheck / lint (borrow: the project's own runners).

## Component map

| Component | Build / Borrow / Plug | Source |
|---|---|---|
| Slice loop, tiers, renderer | BUILD | core |
| Permission gate, verification orchestration, subagents, resume | BUILD (thin) | core |
| LLM client | BORROW | official SDKs / LiteLLM |
| Retrieval + cross-session memory | PLUG | memem |
| Repo map / symbol index | BORROW | tree-sitter (Aider's design) |
| Search (grep/glob) | BORROW | ripgrep |
| Edit/patch application | BORROW | diff lib (Aider's search/replace format) |
| Tool execution + sandbox | BORROW | container / OpenHands runtime |
| Tool breadth (git/web/db/…) | BORROW | MCP client + servers |
| Code-as-action (script→RPC, collapse pipelines into one turn) | BORROW | Hermes `execute_code` pattern |
| Streaming / TUI / diff-approval | BORROW | Textual·Rich / Ink (Cline UX) |
| Eval harness | BORROW | SWE-bench(-lite) |

## Phased plan

- **P1 — safe + capable on a real repo (MVP):** core behind the 4 interfaces · container sandbox + permission gate · ripgrep + tree-sitter repo map · two-tier artifacts (deterministic working-set + memem discovery-set) · tests-as-oracle · errors→re-query.
- **P2 — good to use:** streaming + diff-approval · resume · MCP client · config.
- **P3 — smart at scale:** memem cross-session memory · subagents · SWE-bench eval loop.

## Positioning vs incumbents

OpenClaw and Hermes are both **transcript + LLM-summarization-compaction** (verified by source inspection): accumulate, then summarize near the window. memagent is **deterministic bounded reconstruction** — genuinely distinct. Edge is largest at scale / long-horizon / cost-sensitive / auditability-critical work. Borrow their periphery (sandbox backends, TUI, code-as-action); do not adopt their loop.
