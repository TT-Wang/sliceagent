# SliceAgent — End-Game Context Architecture (v3)

> Status: implemented kernel and migration path · 2026-07-12

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
- sealed `child_artifact` effects;
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
illegal transitions. The model can maintain child work through the small `update_work` API; it cannot forge
`delivered` or `verified`. `ready` means a child contribution is prepared for the final response. Only the host
can turn it into `delivered` by attaching the real sealed response artifact. `verified` remains a compatible
host-owned state for embedding hosts; production does not manufacture it from generic tool success, and keeps
verification truth in typed observations/receipts until an explicit work binding exists.

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

1. Start from every unresolved Active Work item, including unresolved children beneath a progress response.
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
| Child artifact | What a particular child observed/reported, with qualifiers | Parent-certified workspace truth |

Notes and retrieved memories are leads. A receipt cannot prove a file is correct; a response cannot prove a
command ran; a child summary cannot silently become a direct observation.

## Delegation

`spawn_agent` carries a `work_item_id`; it is required on the production Active-Work-bound parent and remains
optional only for compatibility hosts without that state seam. The ID must already name a nonterminal child of the
current request. That immutable binding travels through the child brief, sealed artifact, typed effect, application
ledger, parent receipt, and checkpoint. A missing, nonexistent, cross-request, request-root, or terminal binding is
rejected before launch. Child testimony is attached to that work item only after the artifact seals, avoiding
mid-turn graph-revision races.

For staged breadth, the full currently promised coverage frontier is created in Active Work before the first
delegation wave launches. A wave is only a concurrency window: later partitions remain `open` rather than living
in progress prose. On a clean model stop, the host gives current-root `open`/`in_progress` children one bounded
reconciliation pass; `ready` work may be delivered and `waiting_user` may yield. The same unchanged frontier is
never retried indefinitely. This is typed lifecycle arithmetic, not a semantic quality or permission gate.

Children receive their self-contained objective and scoped sources, not the parent transcript. A parent still
synthesizes and independently verifies load-bearing claims.

Delegation fan-in has three orthogonal host-derived dimensions. **Operational status** says whether a child ran and
sealed. **Explorer evidence** says whether the bounded child artifact retained content observations rather than only
navigation, and records omitted/truncated views without pretending to judge claim correctness. **Parent use** says
whether the bounded digest was delivered and whether the exact sealed report was opened completely. These dimensions
must never collapse into a single “agent succeeded” bit. Each child also declares `digest_ok` or `report_required` in
its brief. The latter gets one non-blocking completion advisory when its report remains unread; after that the model
may either consume it or finish with an explicit partial-scope limitation. Ordinary delegation is never held behind a
semantic quality gate.

The volatile `DELEGATION FAN-IN` region is the bounded live projection of those facts. On seal, the same relations are
folded into Active Work evidence references so a restart can reconstruct them without persisting a rival mutable
manifest. Exact report reads emit typed artifact identity and complete/partial read coverage. This keeps the parent
slice bounded while making “requested, launched, sealed, evidence retained, digest delivered, report consumed”
separately answerable.

## Simplified model surface

When Active Work is bound, the production model sees one semantic state tool: `update_work`. The older
requirements, plan, world-scratchpad, and generic per-tool `note` channels remain readable/executable only for
legacy checkpoints and embedding hosts; they are hidden from the production schema and excluded from the new
compiler. This prevents multiple state APIs from competing to describe the same task.

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
- A root cannot disappear while a required child remains unresolved.
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
immutable events → Active Work frontier → dependency closure → selective fetch → elastic seed
       ↑                                                        ↓
 sealed evidence/output ← typed effects and host verification ← model/tool trajectory
```

That is SliceAgent’s thesis in operational form: retain exactly what the live work depends on, preserve exact
authority and provenance, let the slice expand when the problem demands it, and never confuse history volume with
context relevance.
