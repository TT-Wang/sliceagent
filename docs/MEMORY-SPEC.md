# Memory Architecture Spec — the State Vault

> **Status:** **Steps 1–4 (core) BUILT + validated** (2026-06-16; plan hardened by an 8-agent workflow
> that caught 18 issues incl. 2 blockers).
> - **Step 3** topic-switching: `session.py` (`Session` + `new_topic`/`switch_topic` model tools) +
>   `OTHER OPEN THREADS` render tier + Slice-or-Session-aware `make_build_slice`/`slice_sink`/`mining`.
> - **Topic router** (`route_topic` + `Session.continue_topic`): a cheap host-side call routes each
>   user message → continue / new / resume *before* mutating any topic, so no junk topics; `continue`
>   keeps the topic's context. Real-model demo: new→continue→new→resume routed correctly, calc.py
>   accreted add/sub/mul across continue+resume, 2 topics (no junk).
> - **Step 4 (core)** consolidation: `consolidate.py` `promote_episodes` (cache→memory: corrective
>   episodes promoted, deduped, secrets excluded, declarative) + `MememMemory.consolidate` (reads the
>   session JSONL, remembers lessons) wired at session end in `cli.py`.
>
> Remaining: step-4 refinements (route procedures→skills, frequency/retrieval weighting) + **step 5**
> (retrieval-feedback: `mark_used` + decay). Target: the interactive/session use case.
> New code: `episode.py` (sink), `taskstate.py` (mappers), `interfaces.py` (`TaskState`/`TaskRef` +
> frozen `Memory` surface + `is_durable`), `memory.py` (vault I/O), `cli.py` (wiring); tests
> `tests/test_episode.py` (9) + `tests/test_task_state_roundtrip.py` (7). Eval suite 9/9 = moat untouched.
>
> **One sentence:** memem becomes the durable, structured **state vault** (episodic → task →
> session → long-term); the active memory slice is the bounded, reconstructed **view** of the
> current task's state. The transcript is replaced not by storing history, but by storing
> *reconstructable state* — and by separating lossless **recording** from bounded **presentation**.

Provenance: extends the slice thesis (Markov, bounded-relevant context) and memem (cross-session
lessons). Borrows the load discipline from Claude Code's memory model — a concise always-loaded
**index** (`MEMORY.md`) + **on-demand** topic files — and the consolidation pattern (review
sessions, promote by statistical signal) from Anthropic's scheduled-consolidation approach.

---

## 1. Problem & principle

The slice assumes **one active task**. `Slice.reset(goal)` wipes everything; not resetting pollutes
the next topic. So topic-switching and "resume this later" are unsolved: reset() *loses* the task,
no-reset *contaminates* the next. A transcript "solves" this only by unbounded accumulation that
also mushes topics together (and degrades the model on long sessions).

**Principle (codify this):**
> **Cache everything** (lossless episodic) · **present bounded** (the slice the LLM sees) ·
> **promote selectively** (only reusable / repeated / corrective / actually-retrieved facts) ·
> **route by type** (facts→memory, procedures→skills, transient→age-out) ·
> **load via index + on-demand fetch**, never the whole store.

This is the bounded-context thesis applied to *memory across tasks/time*: a transcript grows with
`topics × turns`; this grows with `live topics` (each bounded; cold ones distilled away).

---

## 2. Architecture — three layers

| Layer | What | Loaded into LLM context? | Lifetime / bound |
|---|---|---|---|
| **Episodic cache** | Lossless log of every turn (rendered slice + action + observation + metadata). NO distillation. | **Never** (cold storage; recovery + consolidation source) | Per session; archived/retained, then pruned |
| **Working slice** | The bounded, reconstructed state the model acts on (the moat — unchanged). | **Yes, every turn** | Per turn (recomputed) |
| **Long-term vault** | Consolidated, distilled: task-state records (resumable) + memory/lesson records (facts). | **Index always; details on-demand** | Durable, cross-session |

### Scopes (orthogonal to layers)
- **Global** (cross-topic, never parked): user preferences, repo-level facts, cross-session lessons (memem).
- **Session**: the set of task-states for this session (the topics index).
- **Task-local** (parked on switch, restored on resume): working set, error, findings, recent, `edited_files`, `since_edit`.

---

## 3. Data structures (concrete)

### 3.1 Episodic record (lossless cache) — AS BUILT
Append-only JSONL, **one line per turn**, at `<vault>/episodic/<session_id>.jsonl`. Cheap, lossless,
never fed to the LLM.
```json
{"v":1, "session_id":"...", "task_id":"...", "turn":7, "ts":"2026-06-16T...",
 "record": {
   "steps":[{"slice":"<rendered slice the LLM saw THIS step>",
             "action":[{"name":"str_replace","args":{...},"failing":false}],
             "observation":["<verbatim tool output>"]}],
   "note":"root cause: ...",
   "meta":{"stop_reason":"end_turn","ptok":12000,"ctok":1500,"failing":false,"files":["a.py"]}}}
```
**Implementation note:** the record uses a single canonical `steps[]` array (per-step
`{slice, action, observation}`) — so a multi-step turn pairs each rendered slice with *its own*
actions (a flat per-turn `action`/`observation` would mis-pair across steps). The earlier sketch's
redundant flat arrays were dropped: it removes duplication and lets the disk size-clamp cover every
value. Tokens (`ptok`/`ctok`) and `stop_reason` come from `TurnEnd.usage` (the per-turn total), and
the buffer also flushes on `TurnInterrupted` (the abort/`max_steps` path emits no `TurnEnd`).

### 3.2 Task-state record (resumable; the reconstruction source)
One markdown file per task (memem-native), `<vault>/tasks/<task_id>.md`. This is the **distilled**
state — NOT per-turn renders.
```markdown
---
type: task-state
session_id: ...
task_id: ...
title: "Fix Content-Length on empty GET"
status: active | parked | done | abandoned
created: ...   updated: ...
links: [other task_ids]
tags: [python, requests]
---
## Goal
<the task statement>
## Findings (decision log — bounded, deduped)
- Root cause: ...
- Ruled out: ...
## Working set (refs only — files are re-read live as ground truth)
- requests/models.py  (anchor: "def prepare_body")
- requests/sessions.py
## Status
last_error: "" | "<verbatim>"   since_edit: 2
## Resolution
<what fixed it / outcome>  (filled at done)
```

### 3.3 Session index
`<vault>/sessions/<session_id>.md` — the source for the `OTHER OPEN THREADS` tier.
```markdown
---
type: session
session_id: ...   started: ...
---
## Tasks
- task_id · "title" · parked · updated 12:31
- task_id · "title" · active · updated 12:40
```

### 3.4 Long-term memory record (consolidated facts/lessons — memem, refined)
```markdown
---
type: memory          # memory | lesson
scope: <project>      tags: [python]
retrieved_count: 0    last_retrieved: null    # retrieval-feedback fields (NEW)
created: ...
---
<declarative FACT, not an imperative — "str_replace no-ops unless its snippet is unique" ✓>
```
Plus a concise, always-loaded `MEMORY.md` **index** (Claude-Code-style; bounded).

---

## 4. `Memory` interface extension

The moat never imports memem; everything stays behind the `Memory` protocol (interfaces.py).
`NullMemory` implements all of these as no-ops (eval determinism).

```
# --- long-term (exists) ---
recall(query, k=6) -> [Snippet]            # index/top-k, always-loaded tier
remember(content, *, title, scope, tags)   # write a fact/lesson
mark_used(memory_id)                        # NEW: retrieval feedback (reinforce)

# --- episodic cache (NEW) ---
append_episode(session_id, task_id, turn, record)   # lossless; never recalled to LLM

# --- task state / resume (NEW) ---
checkpoint_task(task_state)                 # write/update a task-state record
load_task(task_id) -> TaskState | None      # for resume → reconstruct slice
list_session_tasks(session_id) -> [TaskRef] # the topics index (id, title, status)

# --- consolidation (NEW) ---
consolidate(session_id)                     # cache -> long-term, per §7 policy
```

---

## 5. Session & topic lifecycle (host-side orchestration; not in the loop core)

A `Session` holds `{topic_id -> Slice}` with one active. The loop and slice are unchanged; this is
a layer above them.

- **Tools the model sees** (default = continue): `new_topic(goal)`, `switch_topic(topic_id)`.
  The active slice always renders a bounded `# OTHER OPEN THREADS` tier (id · title · status) from
  `list_session_tasks`, so the model knows what it can resume.
- **Continue** (no tool call): keep the active slice.
- **New topic**: `checkpoint_task(active)` → park; create a fresh `Slice` (global tiers persist:
  memem recall + prefs); set active.
- **Resume**: `checkpoint_task(active)` → `load_task(id)` → repopulate a `Slice`'s task-local tiers
  → set active. Files are re-read live, so resume is *curated state + fresh ground truth*.
- **Routing safety:** bias to **continue**; a switch/new is an explicit, **recoverable** action
  (wrong guess → the topics index lets model or user correct next turn — same principle as
  "failed str_replace → re-read").

Per-turn the episodic cache is appended (lossless) regardless of topic.

---

## 6. Reconstruction (slice ← vault)

`make_build_slice` is unchanged for the active task. The only new thing is **rehydration on
resume**: `load_task(id)` repopulates the `Slice` fields —
`goal, findings, active_files, edit_anchor, last_error, edited_files, since_edit` — then the normal
reconstruction runs (re-reads OPEN FILES from disk, retrieves RELATED CODE, recalls MEMORY). No
file contents are stored in the vault; only refs — ground truth always comes from disk.

---

## 7. Consolidation (cache → long-term) — the heart

### Triggers (never per-turn — that's transcript bloat)
- **Task end** — a validated episode (`end_turn` after an error was hit and cleared). *Already
  mined today by `LessonMiner`; generalize it.*
- **Topic park** — checkpoint task-state (always) + consider promotions.
- **Session end / idle** — a sweep over the session's episodic cache ("dreaming").

### Promote a cache signal to long-term only if it is:
1. **Reusable** beyond this task (useful in a *future* conversation) — not transient task state.
2. **Repeated / clustered** — happened ≥2× or a recurring correction (frequency = signal).
3. **Corrective** — an error→resolution episode (high signal; what we mine today).
4. **Retrieved-and-used** — reinforce memories that were recalled and helped; **decay/prune those
   never retrieved** after N sessions (`retrieved_count` / `last_retrieved`).

### Route by type (do NOT dump everything into memory)
- declarative **fact** → memory record (memem).
- reusable **procedure/workflow** → a **skill** (`.memagent/skills/`).
- **path/file-specific** rule → a scoped rule (future; cf. Claude Code `.claude/rules/`).
- **one-off task detail** → stays in the episodic cache and **ages out** — not promoted.

### Exclusions (hard)
- secrets / API keys / PII (we already secret-scrub sandbox env; extend to consolidation input).
- generic platitudes ("write good code").
- duplicates of existing memories (dedupe before write).

### Phrasing
Declarative facts, never imperatives ("Project uses pytest" ✓ / "Always run pytest" ✗) — already
enforced in `LessonMiner`'s distill prompt.

---

## 8. Load discipline (bounded — the key to not becoming a transcript)
- **Always loaded into the slice:** a *concise index* — `recall(task, k)` (RELEVANT MEMORY tier) +
  `OTHER OPEN THREADS` (session task index). Both bounded.
- **On demand:** full task-state (on resume), a detailed memory (on retrieval), a skill body (when
  loaded). Fetched, not pre-loaded.
- **Cold / never in context:** the episodic cache. Read only for recovery, audit, or consolidation.

---

## 9. Security & retention
- Episodic cache is **lossless** → it may contain sensitive tool output. Local-only, gitignored,
  with a **retention policy** (prune after N days / on session close once consolidated).
- **Threat-scan** injected skills / recalled memories for prompt-injection before they enter the
  slice (Hermes pattern) — matters once skills/memory come from untrusted or shared sources.
- Exclude secrets from anything promoted to durable memory.

---

## 10. Principles (codified)
1. Cache everything · present bounded · promote selectively · route by type · index + on-demand.
2. Markov: the slice is the sufficient, bounded **view** of the current task; the vault is the
   durable **state**.
3. Bounded growth: long-term store grows with *distinct reusable facts* and *live topics*, not with
   turns. Retrieval feedback keeps it high-signal.
4. Task-agnostic & LLM-agnostic: no language/tool/model specifics in the core; specifics at the edges.
5. The moat never imports memem — all of this lives behind the `Memory` interface; `NullMemory`
   keeps evals deterministic.
6. Ground truth is disk: store *refs*, re-read files live; resume = curated state + fresh truth.

---

## 11. Build path (MVP first; validate each)
1. **Episodic cache** — `append_episode` (JSONL). Lossless logging; enables recovery/audit and is
   the consolidation source. *Lowest risk, immediate value.*
2. **Task-state round-trip** — serialize `Slice` ↔ task-state record; `checkpoint_task` / `load_task`.
   **Validate:** checkpoint a task, reset, `load_task`, reconstruct slice → same working state.
3. **Session + topics** — `Session` manager, `OTHER OPEN THREADS` tier, `new_topic`/`switch_topic`.
   **Validate:** scripted A→B→A — B stays clean, A restores.
4. **Consolidation** — task-end trigger; generalize `LessonMiner` to the promotion criteria + type
   routing. **Validate:** a corrective episode becomes a fact; a one-off does not.
5. **Retrieval feedback** — `mark_used`, decay unused. **Validate:** used memories rank up; dead ones prune.

---

## 12. Open questions
- **Resume fidelity:** distilled task-state is lossy — is findings + working-set refs enough to
  resume a complex half-done task, or do we need a richer decision log? (Fallback: the episodic
  cache has everything for a deep reconstruction if needed.)
- **Switch-detection accuracy:** model-routed via tools, biased to continue, recoverable — measure
  false-switch / false-continue rates on real multi-topic sessions.
- **Consolidation cadence:** task-end vs session-end vs idle "dreaming" — and how much compute it costs.
- **Episodic retention:** how long to keep lossless logs before pruning once consolidated.
</content>
