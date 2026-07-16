# Subagent System — Direct Outcome Architecture

> Status: runtime contract for 0.3.0 · 2026-07-17

## Thesis

A child is an isolated sliceagent computation, not a workflow node and not a second memory system.

> The child receives a brief, runs privately, and returns one complete normalized outcome. The parent receives that
> outcome through the ordinary tool-result channel. UI and persistence observe the return; neither delivers it.

“Bounded” means child trajectory and session history do not accumulate in the parent. It does not mean every report
has a tiny fixed size. The report may expand to the task's legitimate complexity within the configured completion
budget; the private child transcript never crosses the boundary.

## The whole path

```text
parent model calls spawn_agent
  │
  ├─ scheduler admits the call under the shared child/provider limits
  │
  └─ child runs in a fresh slice
       ├─ tools and model trajectory remain private
       ├─ canonical report is normalized and redacted once
       └─ ChildOutcome returns as the spawn_agent tool result
              │
              ├─ parent model sees the full report and synthesizes normally
              ├─ TUI observes lifecycle metadata
              └─ artifact/memory stores persist opportunistically
```

There is no second fan-in protocol. In particular, the host does not:

- replace the ordinary tool transcript with a synthetic user message;
- reopen reports from an artifact store before the parent can continue;
- create or advance WorkGraph children for scheduler lifecycle;
- reject the parent's tool-free answer with a phrase classifier;
- force a response-only repair pass.

The scheduler already preserves provider-call order even when children finish out of order. That ordered list of tool
results is the fan-in.

## Public call contract

Core mode exposes one read-only child kind:

```text
spawn_agent(agent="explorer", task="…", scope=[…], exclusions=[…], report_shape="…")
```

Advanced mode may also expose named specialists, writable kinds, and exact artifact grants. The live schema is
authoritative. `work_item_id` is not a spawn field: delegation never requires bookkeeping in Active Work.

Good delegation units are independent, source-weight-bounded questions. For a broad review, map the repository and
give each explorer a coherent path/question set rather than one directory name or the entire repository. Submit
independent children in one logical batch when practical. Provider capacity may queue them; the scheduler owns those
physical waves. If the parent announces another semantic wave, it must actually launch it before claiming coverage.

Stay single-threaded for tightly coupled implementation. Parallel readers compose; parallel writers can make mutually
inconsistent decisions unless isolated behind an explicit merge design.

## Brief down

`SubagentBrief` is the child input boundary. It contains:

- the self-contained objective;
- exact applicable user constraints and their source identities;
- scope and exclusions;
- requested report shape;
- optional immutable input-report grants;
- drift policy.

It does not contain the parent transcript, parent fan-out mechanics, progress prose, or a WorkGraph lifecycle ID. A
child sees the files and exact granted artifacts it needs, not an autobiography of how the parent reached the task.

## Outcome up

`ChildOutcome` is immutable and contains:

| Field | Meaning |
|---|---|
| `status` | `succeeded`, `failed`, `cancelled`, or `indeterminate` |
| `report` | Complete canonical report bytes accepted for the parent; may be explicitly partial |
| `report_completion` | `complete`, `partial`, `absent`, or `unknown` |
| `stop_reason`, `stop_cause` | Mechanical child/provider termination facts |
| `partial`, `error` | Honest incomplete-work boundary |
| `kind`, `name`, `launch_ordinal` | Stable identity and ordered fan-in metadata |
| `evidence_status`, `evidence_account` | What workspace evidence the child actually retained |
| `source_coverage_status` | Whether a synthesiser consumed/cited its granted reports |
| `usage` | Child model usage folded into the parent turn once |
| `report_locator`, `evidence_locator`, `artifact_id` | Optional refinement/recovery handles |
| `persistence_warnings` | Fail-soft observer failures after computation exists |

The rendered tool result has an explicit report envelope:

```text
[child 2 · explorer · succeeded · evidence content retained]

BEGIN CHILD REPORT
<complete normalized report>
END CHILD REPORT
Archive: artifacts/<id>.md                 # only when available
Evidence: artifacts/<id>/evidence/index.md # only when available
```

The report body occurs exactly once. A small `child_outcome` effect carries only metadata for the TUI, receipts, and
journal. A `child_artifact` effect is emitted only if canonical artifact persistence commits.

## Evidence boundary

Children are testimony sources, not parent-certified truth.

- A successful content observation can support an explorer report.
- Navigation-only calls such as listing or globbing do not support source claims.
- An evidence-free explorer returns no accepted report, even if it produced persuasive prose.
- A source-paged or truncated view remains explicitly partial.
- A failed child may return usable partial testimony, clearly labeled failed/partial.
- The parent preserves qualifiers and verifies load-bearing conclusions against live source.

The optional evidence archive retains exact child-visible observations and hashes. It is a refinement surface; its
existence does not prove the report is correct.

## Persistence is an observer

Once a safe normalized report exists, computation is complete independently of storage.

| Failure | Child disposition |
|---|---|
| Report normalization/redaction cannot produce a safe envelope | Determinate failure; no report accepted |
| Provider or writable execution may still be physically active | `indeterminate` |
| Parent cancellation closes a read-only child before publication | `cancelled` |
| Artifact store, parent-ref handoff, legacy mirror, index, or roster fails after report exists | Preserve computed status and report; add persistence warning |

The parent turn journal records the full tool outcome. The artifact store adds later recall and recovery; it is not the
only current-turn copy and cannot relabel success as indeterminate.

If the process dies after that journal append but before the parent's synthesis call, unprepared-turn recovery
materializes the journal as one immutable interrupted-turn artifact. On the next startup, the existing latest-receipt
context advertises that exact artifact for one resumed turn only. The parent reads the journaled tool-result text and
continues normally; recovery does not rebuild a fan-in packet, copy reports into every seed, or mutate Active Work.
The transient pointer disappears after the next successful turn seal.

## Child execution profile

Explorers normally use two internal stages:

1. A short fast navigation phase gathers typed workspace observations.
2. One full-reasoning, tool-free synthesis writes the report from retained evidence.

The reserved synthesis runs only after evidence exists. Cancellation, unresolved provider state, and evidence-free
navigation never mint a wrapper recovery call. Streaming is assembled off the main thread so read timeouts measure
inter-chunk stalls rather than total reasoning duration. A monotonic whole-request ceiling and a bounded close grace
handle genuine stalls. The model runner is the sole retry owner.

These stages are child-internal. The parent still sees one `spawn_agent` call and one `ChildOutcome`.

## Scheduling and budgets

- Read-only children may overlap; writable children remain serialized barriers.
- A process-wide provider lease limits physically live requests, including requests whose local watchdog fired but
  whose transport has not closed.
- Queued children are distinct from started children in receipts and the TUI.
- Child budget reservations subtract from one running remainder across waves; serialized children cannot each receive
  the full parent balance.
- Results are returned in provider-call order, never completion order.
- An indeterminate lifecycle child stops admission of the unstarted tail; a determinate failed child does not erase
  independent sibling work.

## Parent trajectory and delivery

A tool-bearing assistant response may contain brief UI progress prose. That prose is not a delivered answer and is
stored as empty semantic assistant content alongside its native tool calls. Provider-required hidden reasoning remains
available for protocol replay. After the ordered child tool results, the parent takes one normal model step. It may use
more tools to verify claims, or it may give a tool-free synthesis. The first tool-free response is delivered.

This prevents the failure where “I will do two waves” or “the report is above” became pinned conversation state and
later fooled the model into believing it had already answered.

## UI contract

The live matrix is an observer over typed lifecycle events:

```text
agents 6 · 2 working · 3 ready · 1 failed
  id  agent       state      current             ops  time
  1   explorer    working    model responding     12  01:08
  2   explorer    ready      report ready         18  00:54
```

“Ready” means the report computation succeeded. It does not mean an artifact was sealed. Optional artifact locators
are displayed separately. `ops` is the child's settled tool-operation count, not tokens or model passes. Long report
bodies remain model-visible but are not dumped into the live matrix.

## Active Work separation

Active Work tracks user commitments that need cross-turn continuity: what the user is waiting for, unresolved
dependencies, and real delivery state. It does not track six explorer processes. Consequently:

- `spawn_agent` does not require an Active Work item;
- child settlement does not mutate graph revision or status;
- seal and crash recovery do not replay child lifecycle into the graph;
- open graph items do not block a tool-free answer.

Receipts and `child_outcome` effects answer operational questions such as requested/started/failed/ready. Active Work
answers the different question: what user-relevant work remains.

## Advanced grants and standing specialists

Exact persisted artifacts can still be granted to a later child or attached to a standing specialist's career. A
mutable name is resolved once to an immutable job handle. Children cannot re-grant sibling reports. If persistence
failed, the current parent still receives the report, but that report cannot be granted or added to a later career;
the persistence warning states the limitation.

## Compatibility boundary

0.2 checkpoints may contain child-artifact WorkGraph evidence and fan-in manifests. Their readers remain tolerant so
old state is inspectable. New turns do not create those bindings or inject a `DELEGATION FAN-IN` context region.
`fan_in.py` and `attach_child_artifacts` are legacy read/migration helpers, not live execution machinery.

## Non-negotiable invariants

- One child call produces one ordered tool result.
- In normal execution, the full accepted report reaches the parent without an artifact read; a process-crash
  recovery advertises the exact interrupted-turn locator instead of inventing a second live delivery path.
- The report body appears exactly once in model context.
- UI and persistence never decide whether computation is delivered.
- Tool-bearing prose is presentation-only; only a final response enters conversation continuity.
- Child lifecycle never mutates user-commitment state.
- Evidence-free explorer prose never becomes testimony.
- Storage failure cannot relabel completed computation.
- Actual unresolved physical effects remain indeterminate.
- Parent synthesis preserves failed/partial coverage and verifies material claims proportionally.

## Release gates

The core regression must demonstrate:

1. Two or more children may finish in reverse order but reach the next parent call in invocation order.
2. A report marker beyond the old 800-character excerpt boundary is present in parent context.
3. No synthetic fan-in user message or artifact loader is called.
4. Tool-bearing progress prose is absent from the semantic trajectory.
5. Artifact-store failure still returns a successful full report plus a warning and no `child_artifact` effect.
6. Delegation without Active Work succeeds and leaves WorkGraph bytes/revision unchanged.
7. The first normal tool-free response is the sole delivered final answer.
8. The TUI moves every child from queued/running to ready/failed without freezing on archival.
9. A crash after a full child ToolResult is journaled exposes a readable immutable report locator in the real resumed
   provider seed, then retires that recovery pointer after the next successful seal.

That is the entire architecture: brief down, outcome up, ordinary continuation.
