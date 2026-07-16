# SliceAgent — Cognitive Memory and ContextFS Architecture

> Status: implemented core, compatibility surfaces retained · 2026-07-12  
> Companion: `ENDGAME-CONTEXT-DESIGN.md`

## Decision

SliceAgent keeps the existing brain vocabulary, but gives every part one owner and one model-facing address:

| Brain region | Runtime layer | Question answered | Authority |
|---|---|---|---|
| Sensory Cortex | live observations | What is true now? | Fresh observation for current-world facts |
| Hippocampus | L0 evidence/history | What exactly happened or was said? | Immutable persisted evidence for the past |
| PFC | L1 Active Work | What are we doing and what remains open? | Derived control state; never new user authority |
| Neocortex | L2 knowledge | What durable lead may help again? | Provenance-linked lead; never current-world proof |

Memem is not a brain layer. SliceAgent owns one typed L2 semantic model; the SQLite knowledge repository is the
sole authority for record identity, provenance, lifecycle, freshness, revision, and sensitivity. When the
structured Memem protocol is installed, Memem is the primary live semantic retrieval backend for that model.
Native lexical/FTS search is failover and migration support, never a co-ranked second authority. Removing or
breaking Memem must not disable evidence, history, Active Work, typed knowledge, or recovery.
The roster and skills are adjacent capabilities: the roster tracks standing workers, while skills package
executable know-how. Their persistence does not make either one a fourth memory layer.

The model never needs physical paths to understand this architecture. It reads the permanent read-only virtual
namespace `@sliceagent/` through its ordinary file tools.

## The invariant

```text
world ──observe──> Sensory Cortex
                      │
                      ▼
immutable events + sealed artifacts (L0 / Hippocampus)
                      │ derive / replay
                      ▼
             Active Work (L1 / PFC)
                      │ consolidate with source refs
                      ▼
       typed knowledge (L2 / Neocortex)
                      │ stable-id semantic projection
                      ▼
     Memem primary retrieval / native failover
```

Every durable claim may cite downward. No lower-authority summary may overwrite its source:

- L1 cites L0 source-event IDs and evidence handles.
- L2 cites L0 evidence with a digest and observation metadata.
- Memem returns stable external IDs that must resolve to SliceAgent-owned records. Orphaned, out-of-scope,
  inactive, or secret hits are rejected; old unprovenanced vault notes never form an automatic recall tail.
  Memem cannot create authority by ranking something highly.
- The current exact request outranks L1 and L2. Fresh observation outranks stored project knowledge for factual
  claims about the live world.

## ContextFS: one stable floor plan

`@sliceagent/` is always routed before physical workspace resolution and is always read-only. Its root manifest
reports live availability and degradation rather than guessing from object presence.

```text
@sliceagent/
├── index.md
├── evidence/
│   ├── index.md
│   ├── events/       application user/work/transition/delivery events
│   ├── turns/        immutable canonical turn seals
│   ├── children/     immutable canonical child seals
│   └── receipts/     execution lifecycle receipts embedded in turn seals
├── history/
│   ├── index.md      Hippocampal index over canonical turn seals
│   ├── turn-N.md     current application-session aliases
│   ├── sessions/     sessions retained in the current workspace store + exact artifact IDs
│   └── search.md
├── work/
│   ├── active.md
│   ├── plan.md
│   ├── dependencies.md
│   └── receipts.md
├── memory/
│   ├── index.md
│   ├── status.md       bounded general self-inspection summary
│   ├── diagnostics.md  raw host-counted inventory, only on explicit request
│   ├── user/index.md
│   ├── project/index.md
│   ├── craft/index.md
│   └── records/<knowledge-id>.md
└── roster/
    └── index.md
```

The root stays readable when a provider is absent or failing. Missing means unavailable or unknown, never empty
or false. A native-knowledge query failure therefore renders degraded status; it does not become “0 memories.”

For self-inspection, this virtual surface is canonical. The model reads `@sliceagent/index.md`, then only the
region needed by the request. `@sliceagent/memory/status.md` is the bounded general-answer surface: current-scope
typed counts, global compatibility-layout state, project-scoped consolidation state, and component-local health.
Raw compatibility aggregates live separately at `@sliceagent/memory/diagnostics.md` and are read only for an
explicit inventory/backend question. Those aggregates have heterogeneous units and may overlap: they are neither
layer sizes nor a typed-knowledge backlog. Consolidation selectively derives provenance-linked L2 records while
its source remains L0; it is not migration. The status page explicitly separates the three memory layers from
indexes, retrieval backends, roster, subagents, and skills.
Implementation modules and raw private stores are debugging inputs only when the user explicitly asks to debug
the implementation; they are not an alternate self-description.
For a general “check your memory system” or “what can you see?” request, the root plus this status page is the
complete answer. Other region indexes are content drill-down surfaces, not a traversal checklist.

Compatibility aliases (`artifacts/`, `history/`, `subagents/`, `roster/`) remain during migration. Model-facing
indexes prefer `@sliceagent/`, which cannot be shadowed by a same-named repository directory.

Subagents do not inherit the parent ContextFS. Their schemas omit the ContextFS capability marker and runtime
guards reject parent-private paths. A child receives a self-contained brief plus explicit sealed grants only.

## L0 — evidence and Hippocampus

L0 has two canonical record families:

1. The application event ledger: one exact persisted user event per logical request, plus work deltas, workspace
   transitions, child-artifact effects, and response-delivery facts.
2. Immutable artifact seals: turn records, tool lifecycle receipts, response bytes, and child reports.

The Hippocampus is an index/read discipline over those records, not another truth store. Canonical history reads
turn artifacts directly and therefore remains available even if the legacy episodic JSONL mirror fails. Exact
cited event IDs can fault from archived application ledgers without listing unrelated sessions.

Persisted text is secret-redacted at the durability boundary. User-event redaction preserves length so Active
Work ranges and digests still validate against the canonical persisted bytes.

The legacy episodic JSONL/FTS path remains for compatibility and consolidation input. It is a mirror written only
after the canonical core seal commits; it is not the response or recovery commit point.

## L1 — Active Work / PFC

Active Work is a rebuildable `WorkGraph`, not prose autobiography. It contains request roots, child work,
dependencies, resource/evidence/output references, and lifecycle state.

`@sliceagent/work/active.md` renders:

- the open dependency closure;
- exact prior user spans resolved from the event ledger;
- the current logical request verbatim in the standalone mounted document;
- direct source locators such as
  `@sliceagent/evidence/events/<user-event-id>.md`;
- explicit degraded/unavailable source state rather than reconstructed language.

Workspace switching creates a new runtime segment while retaining the same logical request and graph. The model
connection and interface remain alive. Switching changes the primary workspace and PROJECT knowledge scope; it
does not reset USER or CRAFT knowledge.

## L2 — typed native knowledge / Neocortex

The native SQLite repository is always available in the base installation. Each immutable `KnowledgeRecord`
contains:

- a path-safe stable ID;
- kind: preference, fact, lesson, or procedure;
- independent USER, PROJECT, and CRAFT scope keys;
- content and applicability;
- one or more digest-bound `KnowledgeSourceRef` values for active records;
- authority and proof-family labels;
- freshness, sensitivity, lifecycle status, metadata, and supersession links.

Lifecycle is explicit:

```text
candidate ──provenance review──> active ──> superseded / retracted / expired ──> tombstoned
legacy_unprovenanced ───────────> candidate or active only after provenance is attached
```

An active record cannot be written without a source. Direct store writes cannot forge supersession topology;
the atomic supersede operation owns that transition. Feedback records exposure, opening, citation, application,
validation, correction, contradiction, or retraction separately from record meaning.

### Scope

Stable project identity is private host state:

- Git worktrees share the identity of their Git common directory.
- Non-Git workspaces receive a registry UUID.
- Concurrent first registration is cross-process locked.
- Repository files cannot choose or impersonate the private project key.

Every native query hard-filters all non-null scope axes before ranking. Two repositories with the same basename
cannot share PROJECT memory. Cross-project native recall is off; a future explicit UI may request it as a
separate operation.

### Admission into the slice

Knowledge is pushed through one bounded compilation seam:

- up to two active USER preference records are standing collaboration leads;
- PROJECT records enter only after hard scope filtering and lexical relevance to the exact request;
- CRAFT records enter only when task wording matches them;
- stale records and dependency-revision-drifted PROJECT observations remain explicit-pull only;
- PROJECT diagnostic/bug records are issue-shaped: only `open`, `unresolved`, or `current` may auto-push;
  resolved/superseded reports remain searchable evidence instead of repeatedly narrating an old bug as live;
- every entry is labelled by axis and points to its canonical record;
- a non-empty admitted knowledge block survives Active Work dependency selection, then competes normally in
  physical elasticity and can degrade to `@sliceagent/memory/index.md`.

This is not transcript accumulation. Relevance is decided first; elasticity decides full text versus locator
afterward. There is no arbitrary wall-clock decay for USER preferences or CRAFT procedures; retirement is an
explicit freshness/lifecycle/revision decision. The current request and fresh observation remain higher authority.

## Memem

Memem is installed with the optional `memory` package extra. Importability alone is insufficient: SliceAgent
activates it only when the structured external-index protocol is present and reports health after an observed
rebuild/search/write. Production always constructs `LocalMemory`; the backend split is:

- `KnowledgeRepository` (SQLite): sole typed record truth and host diagnostics;
- `MememKnowledgeIndex`: primary semantic retrieval when healthy;
- `NativeKnowledgeIndex`: whole-query availability fallback only, never merged into successful Memem results.

The protocol uses `memory_index_upsert(external_id, value, primary_index, cues, scope_id, ...)`,
`memory_index_remove(external_id)`, and hard-scoped `retrieve(..., scope_mode="hard")`. It preserves the full
high-fidelity value in Memem's human-readable Markdown, but FTS, BM25, embeddings, contradiction detection, and
graph linking see only one primary abstraction plus bounded cue anchors. Stable external identity bypasses
Memem's fuzzy dedup/merge because SliceAgent has already decided record identity. USER, PROJECT, and CRAFT use
separate hard partitions; every returned ID is resolved through the canonical repository and its predicates are
checked again before PFC admission. A successful empty Memem result stays empty. Native full-body search is used
only if the semantic operation fails, avoiding a quiet return to historical-report noise.

### Memora mechanism borrowed, and limits

This representation is informed by Microsoft Memora at commit
[`dec3f8f2`](https://github.com/microsoft/Memora/tree/dec3f8f2444eace7004fc084abe1be9f3d88270e)
(2026-06-16), specifically its `MemoryEntry.value` / one-to-one `index` / `cue_indices` split and its documented
rule that values are not directly indexed. SliceAgent borrows only that *harmonic representation* mechanism:
full typed value, one primary abstraction, several distinct retrieval cues. It does not copy Memora's Chroma
storage, prompted multi-step retrieval, or RL/GRPO policy; those would add machinery without strengthening the
L0→L1→L2 authority chain. SliceAgent cues are deterministic PFC/consolidator metadata, tags, applicability, and
path anchors; an LLM cue generator is not on the write path.

Memem Markdown is not yet the sole canonical record store. That retirement requires transactional APIs that can
round-trip the complete typed envelope (source refs/digests, all three scope axes, lifecycle/supersession,
freshness/resource revision, sensitivity, feedback, and runtime metadata), atomic compare/update, full scans,
and crash-recovery tests. Calling it canonical before those APIs exist would create two authorities or lose the
contracts this refactor was built to protect.

## Physical storage

Physical placement is an implementation detail and is intentionally outside the model's workspace boundary.
By default the private state root is `~/.sliceagent` (or `SLICEAGENT_CACHE_DIR`):

```text
~/.sliceagent/
├── core/<workspace-key>/...       canonical artifacts, checkpoints, journals
├── event-ledger/<session>.jsonl   application events
├── knowledge/knowledge.db         typed L2 + native FTS/lexical index
├── registry/projects.json         private stable project identities
├── vault/episodic/...             legacy L0 compatibility mirror
├── vault/tasks|sessions/...       legacy L1 compatibility projections
└── vault/roster|subagents/...     standing-agent compatibility stores
```

When enabled, Memem keeps its rebuildable indexes under `MEMEM_DIR` (default `~/.memem`) and the human-readable
projection under `MEMEM_OBSIDIAN_VAULT` (default `~/obsidian-brain/memem`). Those paths remain independently
configurable and do not become another SliceAgent layer or authority.

Private state is 0700/0600 where the platform supports POSIX modes. It is not placed in every repository because
that would create accidental commits, worktree duplication, permission problems, repository prompt injection,
and ambiguity when a project has several worktrees or is moved. The model still gets the ergonomic benefit of
“memory inside every workspace” through the permanent virtual mount.

An explicitly shared, reviewed project-memory file can be added later as an import/export source. It must not
silently become the private canonical store.

## Reach and workspace boundaries

The primary workspace is a focus and relative-path base, not the whole capability boundary. `ReachSet` contains:

- the primary workspace;
- exact grounded focus roots under the user's home;
- the separate read-only internal `@sliceagent/` namespace.

An explicit absolute home target can admit only its narrow containing directory. HOME itself, filesystem root,
and conventional credential directories are never auto-admitted. `change_workspace` changes project identity,
configuration, indexes, and primary scope without reconnecting the model.

## Failure semantics

- ContextFS provider failure: canonical path stays present and reports degraded/unavailable.
- Native L2 failure: no false-empty count; exact history and Active Work continue.
- Memem absent/failing: the whole query uses native L2 search; L0 and L1 continue.
- Legacy episode mirror failure: canonical artifact history continues.
- Legacy episode rows carry the stable project ID; background review and shutdown consolidation use the
  project binding captured by their originating workspace, never a later foreground scope.
- Missing cited event: Active Work keeps the event ID and direct locator, marks exact source unavailable, and
  never reconstructs plausible wording.
- Workspace switch: the same logical request continues; PROJECT memory scope binds before the target runtime is
  exposed, and a failed binding aborts publication instead of serving target tools against the prior project.

## Migration boundary

The broad `Memory` interface, `MememMemory` class name, episodic JSONL/FTS, task/session markdown, and shadowable
archive aliases remain compatibility surfaces. The host reports per-channel write attempts/failures and a
non-destructive retirement gate. Delete compatibility only after all five explicit gates pass:

1. canonical L0 equivalence;
2. canonical L1 equivalence;
3. canonical L2 equivalence;
4. legacy read-fallback coverage;
5. compatibility-write health.

The default is `blocked/unproven`; a failure overrides a stored pass, and `ready` never deletes anything. This
prevents silent dual authority while allowing a versioned migration/equivalence evaluator to publish proof.

## Acceptance gates

- `where are your memories?` returns L0 history, L1 work, L2 knowledge and the `@sliceagent/` locator without a
  raw-path filesystem safari.
- `check your memory system` reads bounded canonical status, reports exactly three layers, preserves every
  count's unit/scope, and never converts compatibility telemetry into an L2 migration or consolidation backlog.
- A workspace switch preserves the logical request and model connection while changing PROJECT scope.
- Current request beats USER knowledge; fresh sensory evidence beats PROJECT knowledge.
- A prior exact statement resolves from L0, not from L2 summary.
- Memem absence leaves native history/work/knowledge available; Memem success is never co-ranked with native.
- A Memem hit without a resolvable canonical ID, exact scope, active lifecycle, or allowed sensitivity is absent.
- Full historical-report bodies are pull-only and cannot re-enter ranking through FTS, embeddings, or graph links.
- Compatibility retirement remains blocked until the five reported proof gates pass.
- A native query failure renders degraded, not zero.
- Same-basename repositories and concurrent first startup cannot share or split identity incorrectly.
- ContextFS is unreadable to isolated children unless an exact sealed artifact is granted through the child
  capability channel.
