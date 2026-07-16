# SliceAgent — End-Game Context Architecture (v3.1)

> Status: implemented kernel and migration path · corrected 2026-07-17

## Thesis

SliceAgent is not a transcript compressor and not a host-side intent classifier. It is a context operating
system around a semantic model:

> The user and model own meaning. The host owns identity, provenance, persistence, dependency selection,
> fidelity, execution, and verification boundaries.

“Bounded” means session length cannot force transcript accumulation. It does **not** mean every active slice is
small. A hard request may retain a large dependency closure when that is the information required to reason
correctly. Elasticity changes representation only after relevance has been decided.

## One durable event history, one derived work model

The application event ledger sits above workspace-local journals. It records immutable, idempotent facts:

- one `user_utterance` for one logical request;
- accepted `work_delta` effects;
- each `context_transition` between workspace epochs;
- optional sealed `child_artifact` effects (child computation itself returns directly in its tool outcome);
- the one terminal `response_delivered` event.

Persistence redacts secrets without changing source positions. Active Work binds to those canonical persisted
bytes; the live CURRENT REQUEST remains verbatim at recency. A normal restart faults cited source IDs from their
archived application-session ledgers, so it validates the same source hash instead of treating a new process or
the redacted event as different text.

The ledger is history. `WorkGraph` is its immutable semantic projection, not a second history and not a prose
summary. Each item has:

- a stable item, request-root, and logical-turn identity;
- exact digest-bound source ranges;
- lifecycle state;
- monotonic dependency edges;
- workspace-epoch-scoped resource references;
- typed evidence and delivered-output references.

Workspace-local semantic-transition journals remain the commit point for tool effects; the application ledger is
appended only after that commit, so it cannot claim a rolled-back delta. Conversely, startup idempotently backfills
a missing `response_delivered` projection from an already-sealed terminal response artifact after a crash gap.

The graph rejects stale revisions, dependency and supersession cycles, erased provenance, unknown roots, and
illegal transitions. It is optional working state for user-relevant commitments that must survive turns—not a
scheduler and not a prerequisite for tools. `update_work` never mirrors child launch, settlement, or one-turn
synthesis. The model cannot forge `delivered` or `verified`; only the host can attach the real sealed response
artifact. Verification truth remains in typed observations and receipts.

## Logical turns and runtime segments

A user turn is logical, not synonymous with one model loop or one workspace:

```text
user utterance / request root
  ├─ segment 0 · workspace epoch 0
  ├─ context transition
  ├─ segment 1 · workspace epoch 1
  └─ final response delivery
```

Workspace switching keeps the interface, provider connection, task, request root, and exact request alive. The
target segment starts automatically—there is no synthetic `go`, no duplicate user row, no rerouting, and no
source-segment “switching” answer presented as the final response. Multiple distinct workspace edges are allowed
within a bounded transition budget; repeating an already-traversed directed edge is rejected as a loop.

The transition protocol is crash-visible (`prepared → activated → continuing`). A new process reuses the same
application session/ledger namespace and restores the source-linked graph. In-process continuation is automatic;
post-crash recovery reconstructs and reports the interruption but deliberately does not launch an unrequested
model call during startup. The next explicit user admission receives a fresh collision-proof logical identity and
retires the recovered transport ticket.

## The compiler: dependencies first, elasticity second

Each provider seed is compiled in this order:

1. Start from the exact current request and any unresolved user-relevant Active Work commitments.
2. Add request-root ownership and transitive dependency edges.
3. Expand exact user sources and typed evidence/resource/output references.
4. Deduplicate by semantic identity and workspace epoch.
5. Fetch only selected live resources.
6. Choose physical fidelity under the provider window.
7. Place one paired prior user/assistant adjacency and the exact current request at the tail.

This order is the core correction. The legacy design fetched and rendered roughly thirty global regions, then
asked an elasticity controller what to trim. The new design does not eagerly fetch repo maps, code discovery,
memory, history, roster, worktree, or open files unless the unresolved dependency closure names that class of
resource. A file reference from workspace epoch 0 is never rendered as live world state in epoch 1. The checkpoint
restores the epoch that authored the active workspace, so a process restart cannot reset stale resources to
“current.”

Exact active instructions are lossless. If they cannot fit, the system fails visibly rather than silently
summarizing away a user obligation. Large recoverable observations can use locators or fresh re-observation.

## Interaction continuity without a transcript

The live CURRENT REQUEST appears once. Production does not carry the last three rounds verbatim.

One exact paired prior user/assistant exchange is retained as an adjacency object so `yes`, `go`, `that`,
corrections, and option selections have the same local antecedent they would have in a transcript agent. It is
labeled as historical adjacency, never as the current directive or world evidence. A normal restart reconstructs
that one pair from the latest immutable turn artifact; under context pressure the pair degrades to its artifact
locator instead of truncating either side. Older exchanges remain immutable artifacts reached explicitly. Exact
artifact handles are resolvable across a logical turn's workspace stores, while discovery/listing remains local so
unrelated archives are not injected into context.

A constant-size projection of the latest sealed execution receipt is also always present. This removes the old
failure where receipts appeared only if a lexical classifier guessed that the user was asking about execution.

## Four proof families

The compiler and prompt keep these non-interchangeable:

| Proof family | Establishes | Does not establish |
|---|---|---|
| Fresh observation / OPEN FILES | Current visible world state | Omitted bytes or future behavior |
| Canonical execution receipt | Requested, started, rejected, settled, failed, or indeterminate execution | That the desired end-state holds |
| Sealed response artifact | The text delivered to the user | That the answer was correct or acted upon |
| Child outcome / optional artifact | What a particular child reported, with qualifiers | Parent-certified workspace truth |

Notes and retrieved memories are leads. A receipt cannot prove a file is correct; a response cannot prove a
command ran; a child summary cannot silently become a direct observation.

## Delegation

Delegation is an ordinary tool path:

```text
spawn_agent call
  → child runs in an isolated slice
  → ChildOutcome(full normalized report, status, evidence metadata, optional locators)
  → parent receives that tool result and continues once

             ├─ TUI observes lifecycle events
             └─ artifact store persists opportunistically
```

The scheduler already returns parallel calls in provider-call order. In normal execution there is no host fan-in
state machine, synthetic user message, report reopening, transcript reset, or WorkGraph transition for child lifecycle.
The first ordinary tool-free parent response is the answer; the host does not reject it with phrase classifiers or
force a second “response-only” model pass.

Children receive a self-contained objective and scoped sources, not the parent transcript. Their private trajectory
never enters the parent. The complete canonical, redacted report does. This preserves the recursive slice thesis:
context grows with reports relevant to the live task, not with child reasoning history. A failed child may return an
explicitly partial report; an evidence-free explorer returns no accepted testimony.

`ChildOutcome` is the computational truth. It carries operational status, report completion/hash/size, stop cause,
evidence account, source coverage, usage, and optional report/evidence locators. Its report body appears exactly once,
in the tool result. A small `child_outcome` effect drives receipts and the TUI without duplicating prose. A
`child_artifact` effect exists only when persistence commits. Artifact, index, roster, or memory-mirror failure after
a safe report exists adds a persistence warning; it cannot turn completed computation into an indeterminate child.
Actual unresolved provider or writable execution remains indeterminate.

Active Work records only real user commitments that need cross-turn continuity. It is never required before
delegation and never advances merely because a child started or settled. This separation removes the stale-revision
race while keeping launch/queued/running/failed/ready telemetry mechanically answerable from receipts.

The parent synthesizes every returned report, preserves failed/partial coverage gaps, and independently verifies
load-bearing claims against live source. An optional artifact is a later refinement/recovery handle, not the delivery
channel and not a correctness certificate.

If the process dies after a child ToolResult is journaled but before the parent model call, recovery keeps those
bytes in the immutable interrupted-turn artifact and advertises that exact locator for one resumed seed. This is a
crash-only repair seam, not a second fan-in path; the pointer clears after the next successful seal.

## Simplified model surface

The production model sees `update_work` only as an optional cross-turn commitment tool. It is not needed for an
ordinary turn, tool call, delegation, or response. Older requirements, plan, world-scratchpad, and generic per-tool
`note` channels remain compatibility surfaces while migrations run; they do not gate publication.

The stable prompt is an operating kernel rather than an autobiography. It states request authority, Active Work,
dependency-first context, proof-family rules, autonomy, verification, workspace continuation, and concise user
communication. Tool mechanics stay in live schemas.

## Permanent cognitive address space

The model-facing floor plan is now the read-only `@sliceagent/` ContextFS, available through ordinary
read/list/grep tools in every primary workspace. It mounts canonical evidence events and seals, the Hippocampal
history index, PFC Active Work, typed Neocortical knowledge, and the roster without exposing private physical
state paths. Its manifest reports unavailable/degraded/unknown providers explicitly; absence never becomes a
negative fact.

Self-inspection treats that floor plan as the only canonical runtime description: start at
`@sliceagent/index.md`, read only the relevant region, and use `@sliceagent/memory/status.md` for the bounded
general summary. Raw compatibility inventory is isolated at `@sliceagent/memory/diagnostics.md` and read only on
an explicit diagnostic request. Compatibility counts are not layer sizes or a knowledge backlog, and
consolidation is derivation rather than migration. Source files and private stores are inspected
only for an explicit implementation-debugging request, never to reconstruct a competing brain model.

The primary workspace is a relative-path focus and PROJECT-knowledge scope, not a prison. A `ReachSet` adds only
grounded narrow roots for exact user targets, while ContextFS remains a separate read-only capability. A live
workspace transition atomically replaces workspace-owned resources and PROJECT scope while retaining the same
model connection, logical request, and application event ledger. Delayed background review and consolidation
carry the stable project identity captured when their source work ran, so a later switch cannot relabel memory.

The memory floor plan preserves the brain model:

- L0 / Hippocampus indexes immutable application events and canonical seals;
- L1 / PFC is Active Work, rebuilt from source-linked state;
- L2 / Neocortex is typed USER, PROJECT, and CRAFT knowledge;
- SliceAgent owns one typed L2 model. When its structured protocol is present, Memem is the primary semantic
  retrieval backend over stable canonical record IDs; native FTS/lexical search is whole-query failover only,
  never co-ranked. SQLite remains record/provenance/lifecycle authority until Memem can transactionally
  round-trip the full typed envelope. Memem is never a layer owner or L0/L1 durability switch.

The roster and skills are adjacent capabilities, not additional memory layers. Physical co-location in a legacy
vault does not change their architectural role.

See `MEMORY-LAYERS-DESIGN.md` for scope, lifecycle, physical storage, and failure semantics.

## Non-negotiable invariants

- One logical request has one user event and one request root.
- The current request is exact and appears once per seed.
- No host paraphrase can outrank its source.
- Scheduler lifecycle cannot mutate or gate user-commitment state.
- Model deltas cannot manufacture delivery or verification.
- A workspace transition is transport, never task completion.
- Old-epoch resources are locators, never current-world observations.
- Receipt, response, child, and observation evidence never substitute for one another.
- Context growth follows active dependency growth, not elapsed turns.
- Missing context means unselected or unknown, never false.
- Every L1 source event has a canonical `@sliceagent/evidence/events/<id>.md` locator.
- Native memory failure cannot disable exact history or Active Work, and cannot render as a false empty set.
- Workspace identity and PROJECT knowledge scope are stable across Git worktrees and isolated across projects.

## Migration boundary

`IntentState`, legacy discourse admission, plan/world fields, and old region renderers remain as tolerant
checkpoint adapters while parity tests run. In the Active Work path they no longer select production context or
appear as competing model tools. They can be deleted after old checkpoint support has a versioned migration and
the behavioral corpus passes on both the reasoner and weakest supported model tier.

The end state is one unified loop:

```text
immutable events → request + optional Active Work → dependency closure → selective fetch → elastic seed
       ↑                                                                   ↓
 sealed evidence/output ← optional observer projections ← ordinary model/tool trajectory
```

That is SliceAgent’s thesis in operational form: retain exactly what the live work depends on, preserve exact
authority and provenance, let the slice expand when the problem demands it, and never confuse history volume with
context relevance.
