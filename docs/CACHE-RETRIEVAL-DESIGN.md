# Cache & Retrieval Evolution — `slice → cache → memem`

Design proposals for the cross-turn memory channel, sparked by a source-level study of **MiMo Code**
(Xiaomi's opencode-based agent, `github.com/XiaomiMiMo/MiMo-Code`) and grounded in our own measured gaps.

Scope: the within-session `previous slice → cache → current slice` channel and its cross-session tail.
Four functions, ordered by value × moat-fit.

**Status (2026-06-24):** **F1 + F2 SHIPPED** (`consolidate_checkpoint` in `slice.py`; rebuild-from-checkpoint
overflow breadcrumb in `loop.py`; wired in `cli.py`) — offline suite 101 green (+`test_checkpoint.py`, +1
overflow test), adversarial-review clean. **F1 steady-state render was deliberately NOT added** — it would
duplicate the existing findings/world/requirements/change-set tiers (the moat's lean-slice rule); the
checkpoint earns its place only where those tiers are gone, i.e. the F2 overflow rebuild. **F3 + F4 DEFERRED**
(both memem-coupled; F4's clean form is native to F3's FTS5 store and an approximation against memem's
uncertain scores — not worth the risk in the frozen cross-session domain).

---

## 0. Context — where we are

memagent's memory is three layers:

- **L1 slice** — rebuilt **once per turn**, bounded by **relevance** (the moat: cache-not-log). `slice.py`,
  `regions.py`.
- **L2 episodic cache** — a **per-turn LOG**: each sealed turn is archived as a markdown snapshot
  (`episode.turn_markdown`) and paged back on demand via `recall_history` (`history.py`), advertised by the
  PAGED-OUT HISTORY manifest (`render_cache_manifest`). Backed today by `MememMemory.read_episodes`.
- **L3 memem** — cross-session lessons (Obsidian vault + embeddings + FTS5).

Recent measurement (`evals/probe_recall_battery.py`, DeepSeek + gpt-5.5, 11 cross-slice scenarios) showed
the channel is **structurally sound** for ordinary data — but surfaced three things this doc addresses:

1. **No curated "where are we" snapshot.** The always-on cross-turn summary is the lossy 300-char
   conversation one-line (`CONVO_MSG_CHARS`) + scattered `findings` / `world` / `requirements`. The dense
   state only exists as a turn LOG you must `recall` the right slice of.
2. **Overflow sheds raw messages** (drop-oldest exchange / micro-compaction) rather than rebuilding from a
   distilled state.
3. **L2 requires memem** — under `NullMemory` the manifest is empty and the whole channel is dead. A
   *within-session* concern shouldn't depend on the *cross-session* store.

### What MiMo Code does (the reference)

- Typed markdown files (`MEMORY.md` · `checkpoint.md` · `notes.md` · `tasks/<id>/progress.md`) under
  `scope` (global/projects/sessions) + **one SQLite FTS5 index** (whole-file rows, `scope`/`type` filter
  columns, BM25, OR-joined tokens, **relative score floor 0.15×top**, always-keep-#1). FTS5 over vectors
  **for reviewability** (read/edit/delete the `.md`).
- A **background checkpoint-writer subagent** fires **early** at context-budget thresholds (20/40/60/80% for
  small windows) — concurrent, non-blocking — and writes a **structured `checkpoint.md`** (11 token-budgeted
  sections, deduped against prior titles, per-task progress reconciled).
- On overflow it **rebuilds from the checkpoint** + project memory + task progress + recent tail
  (per-section token budgets), as a synthetic boundary message.

The deep contrast: **we have a lossless turn LOG + on-demand recall; MiMo has a curated always-on
SNAPSHOT.** They are complementary. We are missing the snapshot.

### Non-goals (memagent wins to preserve)

- **Relevance-bounded, rebuilt-per-turn slice** (the moat). MiMo injects budgeted-but-not-relevance-gated
  memory; do **not** regress to "inject everything."
- **Lossless per-turn recall.** The checkpoint is lossy; it *complements* recall, never replaces it.
- Borrow MiMo's **ideas**, not its apparatus (Effect fibers, 11 sections, fork-modes, spillover).

---

## Function 1 — Curated CHECKPOINT tier (consolidation layer)  ★ highest value

**What.** A single dense, structured, deduped **session-state snapshot** carried across turns and rendered
as a prominent slice tier — the "where are we" the model can read directly, without choosing a turn to
recall.

**Why.** Closes gap #1 (the goal-2 conversation-ring residual). Today the dense state is fragmented across
`findings`/`world`/`requirements`/conversation; nothing fuses them into one legible snapshot for resume or
long sessions.

**Shape (task-agnostic, bounded sections).** Adapt MiMo's 11 sections down to memagent's tiers:

```
# CHECKPOINT (the live state of this task — distilled, carried across turns)
## intent        — the standing goal / current sub-goal           (~1 line; from goal + requirements)
## decisions     — choices made + why                             (from findings tagged decision/ruled-out)
## state         — change-set + key world facts                   (from edited_files + world)
## findings      — established facts still load-bearing           (from the findings tier, deduped)
## open          — blockers / next step                           (from open_report + convergence)
```

**Two build modes:**

- **Deterministic (ship first, $0):** `consolidate(slice) -> Checkpoint` assembles the snapshot from the
  EXISTING durable tiers (`findings` + `finding_source`, `world`, `requirements`, `edited_files`,
  `open_report`). No extra LLM call — it's a *re-projection* of state we already carry. This alone gives a
  legible "where are we."
- **LLM-distilled (upgrade):** a cheap consolidation call (the "checkpoint-writer" idea) that compresses +
  dedupes the sections. Gate behind a config flag; only worth it for long sessions.

**Moat-fit.** It is a **carry-forward distillation at the seal boundary** — same class as `findings`/`world`
(which `seal()` already preserves), not a within-loop cut. Section budgets keep it bounded
(`bound-is-relevance-not-size`: the bound is the seal). It **complements** lossless recall (checkpoint =
dense always-on; recall = precise on-demand).

**Implementation sketch.**
- `slice.py`: a `Checkpoint` projection + `consolidate()` invoked in/after `seal()` (deterministic mode), or
  a hook for the LLM mode. Store on `Slice` as a durable field (carried, like `world`).
- `regions.py`: `render_checkpoint(...)` + a `REGION_ORDER` entry. Placement: high-salience but volatile →
  the recency tail (beside WORKSPACE STATE / ACTIVE FOCUS), or a dedicated near-top slot if measurement
  shows resume needs primacy.
- `episode.py`: persist the checkpoint as `checkpoint.md` in the cache (so it survives process restart and
  feeds Function 2's rebuild).

**Verification.** Extend `evals/probe_recall_battery.py` with "where are we / what did we decide / what's
next" asks answered from the checkpoint **without** a `recall_history` call; a multi-turn resume probe
(restart mid-task → the checkpoint re-orients the agent in one turn).

**Risks.** Staleness (must refresh each seal); duplication with `findings`/`world` (make those the *source*,
render the checkpoint as a view, not a parallel store); LLM-mode cost (gate it).

---

## Function 2 — Rebuild-from-checkpoint on overflow + proactive consolidation

**What.** (a) On context overflow, **rebuild from the checkpoint + recent tail** instead of only
dropping the oldest exchange / clearing tool bodies. (b) Fire the consolidation **proactively at pressure
thresholds**, not just at the turn boundary, so a fresh checkpoint already exists when overflow hits.

**Why.** Closes gap #2. Today the overflow handler (`loop.py`, the `ContextOverflow` path: micro-compaction
→ drop-oldest-whole-exchange) sheds *raw* messages; rebuilding from a *distilled* checkpoint preserves the
session's intent/decisions/knowledge. MiMo's early-fire avoids the "checkpoint too late" race (overflow
before the snapshot is fresh).

**Design.**
- **Pressure gauge** in `run_turn`: track `prompt_tokens / window` and cross thresholds (e.g. 50/70/85% —
  mirror MiMo's `pressureLevel` 0/1/2/3). On a newly-crossed threshold, refresh the checkpoint (deterministic
  is free; LLM-mode only at the higher thresholds). This is at the **boundary of necessity**, not a routine
  within-loop cut — moat-compatible.
- **Overflow rebuild:** extend the `ContextOverflow` handler — when micro-compaction (#77) is exhausted,
  instead of dropping whole exchanges, **insert the checkpoint as the distilled state** and keep the recent
  tail (the 10–20K "anchor" MiMo preserves). Keep tool_call↔reply pairings valid (the existing micro-compact
  invariant).

**Moat-fit.** Overflow is already a *forced-cut* path (the moat permits degrade there). Rebuilding from the
checkpoint is strictly better than dropping — less is lost. The pressure-fire stays out of the happy path
(only when the window is genuinely filling).

**Implementation sketch.** `loop.py` `run_turn`: a `_pressure(messages)` helper + a threshold set; the
`ContextOverflow` block gains a "rebuild from checkpoint" branch above the drop-oldest fallback;
`episode.py` exposes the latest `checkpoint.md`.

**Verification.** A long-turn overflow probe: force overflow, then ask for the intent/decisions — they
survive via the checkpoint (vs. lost under today's drop-oldest). Confirm the offline suite's
`test_loop_overflow.py` invariants (message validity, no infinite loop, breadcrumb-once) still hold.

**Risks.** Consolidation cost at thresholds (gate LLM-mode); rebuild-boundary correctness (reuse the
breadcrumb + valid-pairing machinery from #77).

---

## Function 3 — Lean files + FTS5 for L2 (reviewable, memem-decoupled)

**What.** Make the **within-session episodic cache** a plain-markdown-files + **SQLite FTS5** store,
**independent of memem**. L2 = files+FTS5 (within-session); L3 = memem stays for cross-session lessons.

**Why.** Closes gap #3 (the flagged memem dependency). MiMo proves a within-session memory works as
files+FTS5: **reviewable** (read/edit/delete the `.md`), greppable, portable, and **needs no cross-session
store**. Today `NullMemory` ⇒ dead channel.

**Design (mirrors MiMo's store).**
- On-disk: `.memagent/cache/<session>/{checkpoint.md, notes.md, turns/<n>.md, tasks/<id>/progress.md}`
  (the page-out blobs from #74 already live under `.memagent/`).
- Index: one SQLite FTS5 table — `path, scope (session|project|global), type (turn|checkpoint|notes|
  progress|lesson), body, fingerprint(size+mtime)`; BM25; lazy reconcile-on-search via fingerprints;
  triggers keep the virtual table synced.
- A `FtsCache` backend implementing the existing `read_episodes` / `search_episodes` / `append_episode`
  contract, so `history.py` + the manifest are unchanged. `make_memory()` wires `FtsCache` for **L2 even
  when memem (L3) is absent** → the within-session channel works standalone.

**Moat-fit.** L2 is the durable cache (not the slice); swapping its backend changes nothing about the
slice/recall contracts. Net simplification: the within-session channel stops depending on the heavier
cross-session stack.

**Implementation sketch.** `memory.py`: new `FtsCacheMemory` (SQLite via stdlib `sqlite3`); `make_memory`
composition: `FtsCache` (L2) ⊕ optional `MememMemory` (L3 lessons). `episode.py` writes `.md`; the FTS
index reconciles them.

**Verification.** The recall battery passes with **lessons disabled but FtsCache enabled** (within-session
recall works without memem). Reviewability check: the `.md` files are human-readable + editable, and an edit
is picked up on next search (fingerprint reconcile).

**Risks.** Adds a SQLite-FTS5 store we own (vs. delegating to memem); reconciliation correctness. **Gated on
the memem freeze** — design now, build when memem is in scope.

---

## Function 4 — Retrieval tuning (scope/type filter · BM25 relative floor · always-keep-#1)

**What.** Tune the retrieval ranking: OR-joined tokens (recall) + BM25 + a **relative** score floor (keep
hits ≥ `floorRatio × topScore`, default ~0.15) + **always keep the #1 hit** + `scope`/`type` filters.

**Why.** memagent's memem recall currently gates by **term-overlap** (drop hits sharing no goal term —
`_memory_relevant`). A **relative** floor adapts to corpus size (small corpora have low IDF where an absolute
gate fails); always-keep-#1 avoids empty results on a thin match; scope/type filters sharpen
(this-session-episodes vs cross-session-lessons vs project-memory).

**Design.** Apply at the ranking seam used by `recall_history(search=…)` and memem `recall`:
`pagetable.py` (`_episodes_search_thissession`, `_episodes` x-session) + `memory.py` `recall`. With Function
3's FTS5 this is native (`bm25()` + a floor pass); against memem today it's an approximation over its scores.

**Moat-fit.** Pure retrieval-quality tuning — no slice/loop change.

**Verification.** A recall-precision eval: for a known fact, the right turn/lesson ranks #1 and survives the
floor; cross-domain noise is dropped; a tiny corpus still returns its single best hit.

**Risks.** Minor; a tuning knob (`floorRatio`) to expose + measure.

---

## Sequencing

1. **F1 (checkpoint) + F2 (rebuild + proactive)** — the high-value, moat-coherent pair. Ship **F1
   deterministic** first (free re-projection of existing tiers), then **F2 rebuild**, then the
   pressure-threshold fire, then optionally the LLM-distilled checkpoint. Measure on the recall battery +
   a long-turn overflow probe at each step.
2. **F4 (retrieval tuning)** — cheap, independent; do alongside F1/F2.
3. **F3 (files+FTS5 L2)** — the larger structural change; **deferred until the memem freeze lifts**, but
   de-risked by MiMo's working example. Design captured here so it's ready.

## One-line summary

Keep the moat (relevance-bounded slice + lossless recall) and **add the curated checkpoint MiMo has and we
lack** — a dense always-on snapshot that fixes cross-turn intent, powers a smarter overflow rebuild, and
(eventually) rides a lean, reviewable, memem-independent files+FTS5 cache.

> Sources: [XiaomiMiMo/MiMo-Code](https://github.com/XiaomiMiMo/MiMo-Code) (source read: `packages/opencode/src/{memory,session,plugin,actor}`);
> measured gaps: `evals/probe_recall_battery.py`, `evals/memory_recall_test.py`.
