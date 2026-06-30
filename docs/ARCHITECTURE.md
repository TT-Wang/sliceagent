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
| Streaming / TUI / diff-approval | BORROW | Rich + prompt_toolkit / Ink (Cline UX) |
| Eval harness | BORROW | SWE-bench(-lite) |

## Phased plan

- **P1 — safe + capable on a real repo (MVP):** core behind the 4 interfaces · container sandbox + permission gate · ripgrep + tree-sitter repo map · two-tier artifacts (deterministic working-set + memem discovery-set) · tests-as-oracle · errors→re-query.
- **P2 — good to use:** streaming + diff-approval · resume · MCP client · config.
- **P3 — smart at scale:** memem cross-session memory · subagents · SWE-bench eval loop.

## Positioning vs incumbents

OpenHands and Hermes are both **transcript + LLM-summarization-compaction** (Hermes verified by source inspection): accumulate, then summarize near the window. memagent is **deterministic bounded reconstruction** — genuinely distinct. Edge is largest at scale / long-horizon / cost-sensitive / auditability-critical work. Borrow their periphery (sandbox backends, TUI, code-as-action); do not adopt their loop.

## Refined thesis: *relevant* bounded context

Validation on real repos (SWE-bench) sharpened the thesis. Bounded context is the moat **only if it is *relevant* bounded context** — "remember less" wins only if you remember the *right* less, dynamically, around what the agent is doing **now**. Every context bug we hit was one disease: *the slice omitted what the agent needed this turn, and (no transcript) the omission was unrecoverable* — it couldn't scroll back to what it saw last turn.

**Failure that taught it:** a 12.7k-char `config.py`; OPEN FILES used a static head+tail truncation (first 1KB + last 0.5KB), so the method to edit (mid-file) was invisible. The agent flailed ~30 steps re-reading and guessing an edit it couldn't see. Toy evals never exposed this — their files fit entirely in the slice. *Lesson: evals must stress the design's load-bearing axis (file/repo size), or "all green" is a false signal.*

**Two universal rules (apply to every tier):**
1. **Compact by relevance to the current focus, not by position/recency/static rule.** Head+tail is position-blind and will, by construction, drop exactly what matters.
2. **Capture every ephemeral signal back into a tier.** No transcript ⇒ tool results, reads, operations, and the *current focus* vanish unless explicitly folded into state.

**Refined by studying Kimi** (transcript+compaction, mostly recency-based — *no* active-file/focus object). It dodges the large-file flail not via smart retention but via **tool contracts**: `Read` returns a *faithful contiguous window* (never head+tail; the model pages to the region via grep+offset); `Edit` matches against **disk** and says *"re-Read if stale"* on mismatch (decoupling *can-I-edit* from *is-it-in-context*); bounding clears stale tool-result *payloads* while keeping a *reference*. So:

- **Faithful over lossy for action targets** — never head+tail a file the agent might edit; lossy caps are only for incidental output (logs/searches).
- **Relevance-*default* region + model navigation** — the reconstructor windows around the focus by default, *and* the model can re-aim (`read_region`/offset). Don't bet everything on the heuristic.
- **Decouple action from context** — edit against disk; a mismatch drives a *targeted re-read* next turn (deterministic 1-step recovery, not a flail).
- **Evict to a reference, not to nothing** — keep a pointer (`path:lines, re-read to view`) so recovery is one step away.

**memagent's edge over Kimi:** relevance-driven retention (keep the active edit region at full fidelity, reference-only the rest) is strictly better than Kimi's recency-only clearing. Our bug was being *recency/position-dumb* in the one place the design is meant to be smart.

*In one line:* **relevant bounded context = faithful, focus-targeted views of what you're acting on + cheap references to the rest + actions that recover via re-read, not retention.**
