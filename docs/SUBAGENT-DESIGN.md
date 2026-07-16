# Subagent System — Design Spec

> Status: runtime-aligned design (v2; current behavior checked 2026-07-14). Consolidates the
> map-reduce-recall model, the coherence-limit analysis,
> the four subsystem deep-dives (delegation · inheritance · reading · hippocampus), and the typed
> seal contract. Extends the existing `run_subagent` (src/sliceagent/subagent.py); does **not** add
> a workflow engine.
>
> **"Bounded" here means history-bounded, not fixed-size** (see [CORE-DESIGN.md](../CORE-DESIGN.md) §1–2):
> the orchestrator's per-turn cost is decoupled from session length, career length, and the fleet's raw
> work-volume — because children seal to digests, so the parent absorbs bounded summaries, not raw
> transcripts. It is *not* independent of how many child digests are actively relevant this turn, nor of
> the current task's complexity. The seal removes the dead weight of history; it does not shrink the
> legitimate cost of the work in front of the parent right now.

## Current runtime (2026-07-14)

- Every child model call drains and assembles SSE off the main thread, even without a UI renderer. The parent
  receives only low-rate typed activity (`awaiting_model`, `model_active`, `reasoning`, `writing`, tool/retry
  state), never the child's private token stream.
- Provider admission is physical, not merely logical: one process-wide lease per provider/account caps live
  sockets (four by default), survives a watchdog returning `indeterminate`, and is released only after the
  stream or SDK worker actually exits. Capacity wait and execution share one monotonic deadline; cancellation
  before async admission makes a later socket start impossible.
- The model runner is the single retry owner; provider SDK retries are disabled. Exhausted attempts,
  truncation, or cancellation do not trigger a wrapper-level “report recovery” request. Any usable report,
  observations, trace, usage, and WorkGraph binding instead seal as an explicitly partial artifact.
- Explorers default to two planned stages: at most six fast, tool-using evidence-navigation model steps, then
  one full-reasoning, tool-free synthesis. The handoff is allowed after a clean navigation exit or after that
  explicit navigation ceiling **only when the host captured typed workspace observations**; a ceiling handoff
  carries an authoritative “coverage is incomplete—report gaps” note. The navigator disables `run_turn`'s
  generic max-step closeout, so the reserved synthesis is the only extra model call. Cancellation, uncertain
  provider/tool state, truncation, token-budget failure, and evidence-free navigation never enter synthesis.
  `AGENT_EXPLORER_NAV_STEPS` changes the navigation ceiling (clamped below the child maximum), while
  `AGENT_EXPLORER_REASONING=fast|full` keeps a single-stage profile when explicitly selected.
- Lifecycle admission is bounded to four concurrent children per wave. The first pair launches immediately;
  later children enter on a short ramp, and requested/queued/active/terminal states remain distinct in the
  subagent matrix. Read-only and writable children both obey the delegation ceiling; writable children remain
  serialized barriers and time out as `indeterminate` when workspace effects cannot be disproved.
- Canonical child publication and its downward parent reference commit under the launch turn's seal lock.
  Therefore cancellation or parent sealing can win before publication, but cannot split an accepted child
  artifact from the parent reference that makes it reachable.
- Under context pressure, only old read-result bodies in the next provider-request copy are reduced to
  locator/hash/head-tail views; the canonical in-turn trajectory is not rewritten. The independently bounded
  artifact capsule preserves locators, hashes, retained bytes, and explicit truncation rather than pretending
  omitted bytes were reviewed.

## 0. TL;DR

Turn today's prose-returning `run_subagent` into **bounded-cost map-reduce over agents**. A parent
delegates a fan-out of independent tasks; each child is itself a sliceagent that **seals a structured,
archived artifact** (not a transcript); an optional **synthesiser** (also a subagent) reduces N artifacts
to a digest; the parent absorbs **digest + a visible manifest** and **recalls** any child's full detail
on demand — which pages back out at the next seal. Per-turn resident cost is **bounded by the digests in
play this turn, not by the fleet's raw work-volume or the session's length** — because children seal to
digests, the parent never carries their raw transcripts. That is the moat, expressed recursively.

The whole system is **two bounded seals**: the *brief* (parent → child) and the *artifact* (child →
parent). Both are refinable (carry a handle back to the finer level) and honest (report what they
compressed). Everything below is a consequence of that.

## 1. Goal & why it matters

The defining failure of multi-agent systems is context blow-up: a child's raw trajectory floods the
parent, so orchestration gets expensive and incoherent fast. sliceagent's **seal-and-recall** is exactly
the mechanism that solves it — so subagent orchestration is not a side feature, it is the *killer
application* of the cache-not-log thesis: bounded cost **per level of delegation**, a property a
transcript orchestrator structurally cannot match at scale.

## 2. Scope — what fan-out is actually for (the coherence limit)

Fan-out suitability is determined by **coupling**, not task category. Two failure modes:

1. **Detail loss at the seal** — recoverable here (see §7.3: refinable seals + manifest).
2. **Conflicting implicit decisions during parallel work** — *irrecoverable* at the seal, because the
   incoherence happened during execution. No summariser can reconcile two children that assumed
   different things.

Therefore:

| Pattern | Fan out? | Why |
|---|---|---|
| Investigation / search (map over independent sources/files) | ✅ | no shared decisions |
| Verification / judgment (N independent checks of one claim) | ✅ | independent judgment, reduce by vote |
| Independent-section generation | ⚠️ | only if sections truly don't share assumptions |
| Coherent implementation (coupled change across files) | ❌ | single-thread, or worktree-isolate + merge |

**Division of labor:** subagents *scout and check*; the parent *synthesises and writes* on the single
thread. The design encodes this: parallel children are **read-only** (§7.2) — read = investigate
(composable), write = implement (coherent, must stay single-threaded). The read-only constraint **is**
the coherence boundary.

## 3. Non-goals (hold this line)

- **No workflow engine** — the small execution scheduler provides ordering, bounded admission, cancellation,
  and launch ramping; it is not a lane manager, DAG runtime, DSL, or event-bus middleware (that is Raven's
  Spine and would dilute the lean-kernel moat).
- **No new agent type** — a subagent *is* a sliceagent; a synthesiser *is* a subagent. Build "agent"
  once; it nests.
- **No third channel** — children couple only through the two seals. No shared mutable `Slice`, no
  back-chat.
- **No unbounded nesting / fan-out** — depth, pool size, and N stay bounded; the kernel can always say no.

## 4. Core model (math → design constraints)

- **Fixed point:** `agent = Slice + run_turn(may spawn agents)`. Self-similar → one mechanism at every
  level.
- **Graded seal:** sealing is coarse-graining (`trajectory → artifact → digest`), recall is refinement.
  **Constraint:** a seal MUST ship its refinement handle, or the parent hallucinates the detail it can't
  see (the #114/#116 theorem).
- **Computation is a map-reduce DAG** — bounded because every edge carries a sealed artifact, not a
  transcript.
- **Coherence is an atlas cocycle** — each child slice is a local chart on the shared workspace/memory
  manifold; charts must agree on overlaps (a child and parent reading the same file see the same bytes).
  This is self-critique #4 (SwapManager/seed view) restated as a law: **do not fork the file view.**

## 5. The two seals (unifying frame)

- **DOWN — the brief** (parent → child): "what you need to know." The parent compresses its context into
  a self-contained task; refinement handle = files/artifacts the child may read.
- **UP — the artifact** (child → parent): "what I found." The child seals its trajectory into a bounded,
  honest report; refinement handle = the child's own archived turns.

Bounded both ways, refinable both ways, honest both ways. There is nothing else.

## 6. Data contracts (typed sketch)

```python
from dataclasses import dataclass, field

@dataclass
class SubagentBrief:            # DOWN seal: parent -> child
    kind: str                   # which agents/*.md spec runs (explorer/reviewer/synthesiser/...)
    task: str                   # the SELF-CONTAINED goal (child starts from a fresh Slice)
    context: str = ""           # minimal facts the child needs — it has none of the parent's slice
    scope: str = ""             # boundary: which files/dirs are in / out of scope
    returns: str = ""           # EXACTLY what to report back — steers what detail survives the seal
    budget_steps: int = 20
    budget_reads: int = 0       # 0 => kind default (EXPLORER_READ_BUDGET, etc.)
    refs: list[str] = field(default_factory=list)   # handles the child MAY recall to refine the brief

@dataclass
class SubagentArtifact:         # UP seal: child -> parent. Stored as an episode (role="subagent").
    id: str                     # stable handle: "sub-<session>-<n>"
    parent: str                 # parent session/turn that spawned it
    kind: str
    task: str
    status: str                 # ok | error | max_steps | aborted
    steps: int
    digest: str                 # bounded conclusion — the parent reads THIS first
    findings: list[str]         # structured distilled findings (the sealed FINDINGS tier)
    change_set: list[str] = field(default_factory=list)   # edited files (empty for read-only children)
    # --- HONEST fields: defeat "decisive results hide missed detail" ---
    coverage: str = ""          # what WAS and was NOT covered ("reviewed 3 of 5 files; skipped auth")
    uncertainty: str = ""       # assumptions made / what the child is unsure of
    conflicts: list[str] = field(default_factory=list)    # cross-source disagreements (synthesiser fills)
    # --- refinement handle (up): where the parent recalls full detail behind the digest ---
    refs: list[str] = field(default_factory=list)         # ["subagents/sub-<n>/turn-2.md", ...]
```

The **synthesiser** is a subagent of `kind="synthesiser"` whose output is itself a `SubagentArtifact`:
its `digest` is the merged conclusion, its `findings`/`conflicts` surface cross-child agreement and
*disagreement*, and its `refs` point at the N child artifacts it reduced.

## 7. Mechanics

Flow: `delegate → run → seal → archive → (reduce) → absorb → recall`.

For the canonical local path, the immutable child write and its launch-turn reference are one
linearizable publication. Child publication and parent sealing share the turn-store lock: cancellation
may win before publication and write nothing, but once publication starts it completes both facts before
returning. If sealing wins first, the retired launch rejects the child before its write. The filesystem
crash-recovery path still closes the unavoidable process-death gap by rediscovering children through their
sealed parent lineage.

### 7.1 Delegation (parent → child) — prompt + mechanism

Tools: `spawn(brief) -> artifact_id` and `spawn_parallel(briefs) -> [artifact_id]` (bounded read-only
pool). The parent's system-prompt guidance teaches three decisions:

1. **Whether** — *"Delegate independent investigation/verification you can run in parallel. Do NOT
   delegate coupled work whose sub-results must agree — hold that on your own thread."*
2. **How to decompose** — into subtasks whose results *compose* (minimal shared assumptions).
3. **What to hand off** — *"The child starts with a fresh slice and no memory of this conversation. Give
   it a self-contained task, the minimal context it needs, an explicit scope, and exactly what to report
   back."* The `returns` field is the lever that controls detail loss **from the source**: ask for the
   specifics you need and the child seals *those*, not a vague summary.

Risk: an under-specified brief → the child works on a wrong understanding. Mitigated on the child side
(§7.2): the child reports what it was missing rather than assuming.

### 7.2 Child inheritance & coupling (the fixed point in practice)

The child **is** a sliceagent (`run_subagent`: fresh `Slice`, same `run_turn`).

- **Shared, read-only (atlas overlap):** the workspace + code-index / SwapManager *view* (same bytes as
  the parent — the cocycle law), the thread-safe LLM view (`_profile_llm`), the memory backend (child can
  recall cross-session lessons), the read-only tool subset, and the repo-map system prefix (cache
  prefix-share across children).
- **Fresh (isolation = bounded cost):** the `Slice` — its own working set, findings, budget, and goal.
  The child does **not** see the parent's trajectory.
- **Coupling = the two seals only.** No shared `Slice`, no side-channel.
- **Recursion:** the child may spawn its own children (bounded depth) — self-similar down.
- **Subagent-aware self-model** (extends #115): *"You are a subagent completing a delegated task. You
  don't have the parent's context — work from your brief; if a decision needs context you weren't given,
  say so in your report rather than inventing it. Your job ends when you seal a report."*

Risk: parallel read-only children see a **snapshot at fan-out time**; concurrent parent edits are
invisible. Fine for investigation, and precisely why writes stay serial.

### 7.3 Parent reads findings without losing detail

Three tiers + a discipline:

- **Digest** (resident) — the synthesiser's bounded conclusion; read first.
- **Manifest** (resident, bounded — the visible-cache-manifest) — one line per child: *what it covered,
  what it found, its coverage/uncertainty, → recall handle.* Rendered as a `DELEGATED WORK` region
  (reuse the PAGED-OUT HISTORY tier in regions.py). This advertises that detail exists.
- **Artifacts** (paged out, recallable) — full child findings, addressable, with provenance: every digest
  claim traces to `read_file("subagents/sub-N.md")`.

Two non-negotiables:

1. **Honest seals beat decisive results.** `coverage`, `uncertainty`, and `conflicts` are first-class;
   the synthesiser *surfaces* "sub-1 and sub-4 disagree on the auth assumption" and "sub-2 covered 3 of 5
   files" rather than laundering them into a clean merge. This is what *triggers* the parent to drill.
2. **Grounding discipline** (parent prompt, the #114/#116 lesson): *"The digest summarises sealed work.
   When a decision depends on a child's specific — a value, a signature, an edge case — do NOT trust the
   digest; recall that child's artifact and ground on it. The manifest says which child holds what."*

Advantage: each drill-down is a **recall** — pages in, used, pages back out next seal. The parent can
drill into ten artifacts across ten turns at resident cost bounded by what's active each turn, not by the
number of artifacts accumulated over the session.

Risk (highest-stakes dependency in the system): the parent only recovers detail if it *recognises* it
needs to. A confidently-wrong digest whose manifest doesn't flag the gap → no drill → the industry's
detail-loss failure returns. **The honest fields are load-bearing; do not cut this corner.**

### 7.4 Hippocampus management

Principle: **a subagent artifact is just an episode from a nested agent** — reuse the machinery.

- **Storage:** `append_episode` with `role="subagent"` + `parent` link + stable id, carrying the
  `SubagentArtifact`. Same JSONL, same `FileLock`, same FTS5 index (so `search_history("auth flow")`
  finds a child's finding).
- **Addressable via HistoryFS:** `subagents/sub-3.md` is served exactly like `history/turn-N.md` from
  `read_episodes`. Recall "just works" — no new read path.
- **The archive is a TREE mirroring the spawn DAG.** A child has its own episode history *and* its
  artifact is an episode in the parent's history, so HistoryFS exposes nested depth:
  - `subagents/sub-3.md` → the artifact (the child's seal)
  - `subagents/sub-3/turn-2.md` → the child's *internal* turn (drill deeper)
  
  The scale-space made navigable: zoom digest → artifact → the child's own reasoning, each level paged
  out and recallable on demand.
- **Consolidation (neocortex):** treat a child's findings like any turn's findings — mine durable
  lessons into memem, archive the rest ("consolidate by salience"). Hierarchy: parent findings ⊇
  absorbed digest ⊇ archived artifacts ⊇ children's internal trajectories.
- **Retention/GC:** none new — artifacts age out like episodes; FTS5 keeps them searchable cross-session.

Risk: deep trees mean more round-trips to drill; bounded depth + a digest at every level keeps it sane
(one level is the common case).

## 8. Invariants / guardrails

1. **Moat:** parent absorbs only `digest + manifest` — never a child's raw trajectory. Resident cost
   independent of N and child steps.
2. **Refinement handle:** every seal carries its recall pointer. No lossy seal without a way back.
3. **Cocycle:** child and parent share ONE file view (don't fork SwapManager/seed).
4. **Read-only ⇒ parallel-safe;** write-capable children are serial until worktree isolation (Phase 5).
5. **Serialized absorb** (the `Slice` isn't thread-safe); bounded pool size, bounded N, bounded depth.

## 9. Reuse map (what's new is small)

| Piece | Reuse | New |
|---|---|---|
| Child execution | `run_subagent` / `run_turn` | brief plumbing |
| Artifact storage | hippocampus episode archive + `FileLock` + FTS5 | `role="subagent"` record + id/parent |
| Addressable recall | `HistoryFS` (`history/*`) | `subagents/*` mount (+ nested `sub-N/turn-M`) |
| Manifest render | PAGED-OUT HISTORY tier (regions.py) | a `DELEGATED WORK` region |
| Parallel run | thread-safe `_profile_llm` view | bounded pool + `spawn_parallel` tool |
| Reduce | a subagent (`spec`) | a `synthesiser` agents/*.md + prompt |
| Consolidation | neocortex mine-lessons | (unchanged) |

Genuinely new code: type the seal, archive it, a manifest region, a bounded parallel spawn, one
synthesiser spec. Everything else is existing primitives pointed one level up.

## 10. Build order (de-risked — highest dependency first)

- **P0 — Seal contract with honest fields.** `SubagentBrief` + `SubagentArtifact` (incl.
  `coverage`/`uncertainty`/`conflicts`/`refs`). Everything hangs off this.
- **P1 — Hippo storage + HistoryFS `subagents/` mount.** Artifacts addressable → recall works.
- **P2 — Manifest tier + grounding discipline.** The anti-confabulation piece (load-bearing).
- **P3 — Structured brief + coupling guidance + subagent self-model.** The prompt layer.
- **P4 — Parallel fan-out + synthesiser kind.** Mechanism, once contract + read-path are proven.
- **P5 — later/bigger:** isolated-write children (worktree, merge-back). *Prereq: the Slice
  decomposition (self-critique #1) for safe parallel absorb.*

## 11. Eval gates (the moat, at the subagent level)

- **Bounded-peak:** parent resident bytes after absorbing N artifacts ≤ `digest + manifest` bound,
  independent of N and of each child's step count. (The headline assertion.)
- **Drill-down:** parent recalls a specific child's full detail; it pages back out next seal.
- **Manifest-visibility (anti-confabulation):** a gap_detection-style probe — parent asked about a
  child's specific *recalls* rather than inventing it; and a child that skipped coverage says so.
- **Conflict surfacing:** two children given contradictory premises → the synthesiser's artifact reports
  the conflict rather than a false merge.
- **Parallel correctness:** N read-only children → N valid archived artifacts, no races; absorb serial.
- **No-regression:** existing subagent tests + full suite green.

## 12. Open decisions

1. **Artifact storage** — reuse the episode JSONL with `role="subagent"` (recommended; HistoryFS then
   surfaces both `history/` and `subagents/`) vs a separate store.
2. **Synthesiser trigger** — N ≥ 3 default, env-configurable?
3. **Parallelism surface** — explicit `spawn_parallel` tool the model calls (recommended) vs host
   auto-parallelising sequential spawns within a turn.
4. **Slice decomposition sequencing** — serialized absorb makes P4 correct *without* it; do #1 before P5
   (write-capable parallel), not before P4. Agree?

## 13. One sentence

**Build one recursive agent; make its seal a graded projection that always ships a visible, honest
manifest; spawn it in parallel because it's read-only; reduce fan-outs with a synthesiser that is itself
just another agent — and keep coherent writes on the single thread.**

---

# v3 — Roster Edition (standing specialists)

> Status: design (v3, 2026-07-09). Supersedes v2's build order from P3 onward. Adds the **hire/wake
> lifecycle**: subagents stop being ephemeral calls and become **named, durable specialists** with a
> dedicated prompt, lessons learned, and memories of previous work — all as functions of the sealed
> archive. Grounded by the 2026-07-09 prior-art red-team (5 refuters): every *feature* here exists
> somewhere (Letta persistent agents, CC agent-teams resume, CrewAI roles, Reflexion/ExpeL lessons);
> the unoccupied territory is the **conjunction as substrate invariant** — dormant at zero cost, wake
> at cost bounded independent of career length (sized to identity + last-K, not the whole career),
> career = typed sealed artifacts at stable addresses,
> lessons = bounded curated tier with per-lesson provenance, identity = archive key with no runtime residue.

## v3.1 Core reframe: hire once, wake many

The parent's delegation primitive changes from "create a worker" to "**get me the worker who does this**":

- roster hit  → **WAKE**: rehydrate the specialist from its archive (fresh slice; cost bounded by identity + last-K, not career length)
- miss + name → **HIRE**: mint the identity (one-time; roster-cap-gated — kernel can say no)
- no name     → **TEMP**: today's anonymous `sub-N` (default; zero ceremony, session-scoped)

One tool (`spawn_agent` grows `name`), backward compatible. No auto-router/classifier: the ROSTER manifest
(visible, bounded, with the copy-paste wake call) is the routing table and the parent is the manager —
task-agnostic kernel, model-owned judgment (the 38%→100% visible-manifest lesson).

## v3.2 A specialist = the three-layer brain, per identity (fixed point deepened)

| layer | main agent | specialist |
|---|---|---|
| PFC | bounded slice per turn | fresh slice at WAKE — bounded regardless of career length (sized to identity + last-K) |
| hippocampus | episodes + history/ | its CAREER: sealed artifacts of past jobs (+ traces, v3.7) |
| neocortex | memem consolidation | its LESSONS: curated distillate of its own jobs |

Storage (durable, beside session archives):

```
<vault>/roster/<name>/
  profile.json     # identity card: kind, created, jobs, last_active  (rendered as profile.md)
  lessons.json     # curated lessons [{text, job, ts}] — bounded by CURATION (K), never truncation
  episodes.jsonl   # its sealed artifacts (career), same _clamp redaction as session archives
```

## v3.3 Wake-seed contract (bounded by construction)

identity (name + IMMUTABLE kind prompt + career stats) · lessons (≤K, advisory priors with provenance)
· career manifest (last K jobs, one line + handle each) · the new brief (verbatim) · self-model:
"You are <name>, resumed from your sealed reports. Your memories are only what those reports say — if
the answer needs more, say so. The workspace may have moved since; re-read files, don't trust your
reports' quoted code." (#114 one level down + the cocycle law.)

Full career detail stays paged out; the specialist may read ITS OWN past jobs (own-namespace carve-out
in the isolation guard). Self-memory ≠ third channel; sibling memory still needs an explicit grant (v3.5).

## v3.4 Identity + brief provenance (W1)

The artifact gains `name` (instance identity) and `brief` (the VERBATIM prompt given — task + grants).
Provenance rule extended into delegation: whoever reads a report can see what its author was ASKED —
a child briefed narrowly cannot be silently cited for broad claims. index.md becomes the roster-style
manifest: `name · kind · brief-gist · status · handle`. Named leaf alias: `subagents/<name>.md`.

## v3.5 Capability grants (W2) — the governed handle channel

`grants: [exact handles]` on spawn; the child guard becomes default-deny + grant-list. Rules that keep
"children couple only through seals" true: EXACT handles only (never a dir or index.md), spawn-time
existence validation + list cap (kernel says no), NO transitive propagation (grandchildren default empty).
A grant is a pointer to a seal.

## v3.6 Synthesiser (W3) = a child granted all N handles

No special machinery: a `synthesiser` AgentSpec (read-only tools) + grants = the N sibling handles.
Prompt: page one artifact at a time, cite every merged claim to its handle, surface CONFLICTS and
coverage gaps rather than laundering them. Seal populates `refs` = the cited handles, so the synthesis
itself ships its refinement map (invariant: every seal ships its handle). Bounded lossless reduce:
peak O(K) for any N; dropped detail stays one read_file away.

## v3.7 Lessons tier (W5') — reuse the standing-requirements pattern

Seal-time reflection: artifact gains optional `lesson` ("what would you tell the next you?", one line).
Curation on write into lessons.json: add / supersede / drop, hard cap K, per-lesson provenance
(job handle + date). Seed injection frames them as PRIORS, not rules. Kill-switch eval: lesson-efficacy
A/B (specialist-with-lessons vs fresh temp on the same task class) — if no measured gain, the tier dies
(reasoning-trace-tier precedent: 0/4 → reverted).

## v3.8 Risks (gates, not footnotes)

1. **Lesson poisoning** — wrong self-assessed lesson biases every wake → provenance + advisory framing +
   supersede/drop + the A/B kill-switch.
2. **Confab amplification** — persona + "memories" is the maximal trap (weak models invent their past
   WITH a self-model) → the abstention line is load-bearing; eval probes with leading questions (usersim).
3. **Roster sprawl** — temps by default; hire is deliberate; cap + retire; staleness visible (last_active).
4. **Specialist tunnel vision** — lessons speed the K-th similar job, may blind the novel one → parent
   always has the fresh-temp option; measure, don't assume.

## v3.9 Build order + eval gates

W1 identity+provenance (S) → W2 grants (S) → W3 synthesiser (S) → W4' durable roster + hire/wake +
own-namespace carve-out (M) → W5' lessons (M) → W6' trace archiving + FTS5 dual-write role='subagent' (M).

Gates: wake-cost FLAT vs career length (1 vs 20 jobs) · lesson-efficacy A/B (kill-switch) · persona
confab probe (leading questions → abstain + cite handles) · roster visibility (does the parent actually
wake staff when a specialist exists? measured).

## v3.10 One sentence

The hippocampus stops storing just what happened and starts storing **who your agent works with** — a
team whose salaries are zero, whose memories are seals, and whose résumés are grep-able.
