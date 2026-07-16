# Changelog

All notable changes to sliceagent. Format follows [Keep a Changelog](https://keepachangelog.com/);
this project aims for [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Intent Fidelity v2.** One shared, ephemeral turn contract now separates user-authorized action spans
  from quoted/reported prior output, grounds historical references in sealed responses, resolves numbered
  items and launch-ordered subagents, and helps the model carry terse confirmations onto the immediately
  pending proposal or explicitly selected numbered option. This intent metadata is advisory context, not a
  second host-side permission system.
- **Safe update path.** `sliceagent update` now updates a positively identified canonical `uv tool`
  install before any API-key, workspace, plugin, or MCP startup. Editable/direct installs are preserved,
  alternative package managers receive exact guidance, and `/update` points active sessions to the safe
  process boundary.
- **Seamless workspace handoff.** `/cwd <path>` and the model-facing `change_workspace` control tool now
  stage and validate the target runtime, durably seal the current turn, and atomically replace only
  workspace-owned resources. The terminal application, model client, token/cost counters, and connection stay
  alive; target preparation failures roll back to the untouched current workspace.
- **Canonical execution receipts.** Every tool lifecycle now records requested, rejected-before-start,
  physically started, settled, and effect-applied states under one invocation identity. Sealed receipts flow
  into turns, recovered crashes, child-artifact references, terminal completion, and receipt-grounded recall;
  large-task aggregates remain exact without replaying thousands of operation rows into the prompt.
- **Receipt-aware mind-model evaluation.** The paired old-autobiography/operating-contract harness now proves
  the substituted system-prompt diff on one Git revision and workspace, retains full replies, rejects
  screen-derived ground truth, and scores lifecycle claims, abstentions, corrections, and required coverage
  directly against sealed receipts.

### Changed
- **One typed memory model, with Memem as retrieval rather than authority.** L0 remains canonical sealed
  evidence, L1 remains rebuildable Active Work, and native SQLite owns provenance-linked USER/PROJECT/CRAFT
  L2 records. Memem 2.10's structured external-index protocol is the primary semantic ranker when available:
  stable IDs, hard scope partitions, one primary abstraction, and bounded cue anchors are always resolved back
  through SliceAgent's lifecycle, revision, sensitivity, and provenance predicates. Resolved or revision-drifted
  project diagnostics stay explicit-pull instead of being re-injected as current bugs; a whole-query native
  fallback preserves availability without co-ranking historical full-body noise.
- **Autonomous admission kernel.** General permission modes, host-side turn-authority denials,
  reconciliation locks, and loop-policy refusals are retired. Ordinary requested work now flows directly;
  the host retains only a narrow, high-confidence safeguard for commands that could catastrophically damage
  the machine (such as formatting a device or recursively deleting `/` or the user's home).
- **Calmer, informative Rich TUI.** The footer no longer pins task text; it restores total-token and
  dollar-savings meters. The live reasoning/progress row also shows only current activity (or the active plan
  step), never the task's first prompt. Assistant replies have more breathing room, and the composer drops its
  redundant title. Combined greetings such as “hi how are you” stay on the cheap chitchat path instead of
  becoming a durable task title.
- **Natural workspace intent.** “Go/open/work in the hunter workspace” now navigates without requiring
  implementation-shaped wording. Safe fallback path probes remain observational, and an assistant's exact
  workspace-path clarification carries naturally into a terse confirmation such as “yes.”
- **Elastic observation flow.** Pipelines and fallback branches composed entirely of read-only commands
  (for example `find … | sort` and `ls … 2>/dev/null || ls …`) remain ordinary inspection. A failed operation
  does not poison unrelated reads, and terminal messages report the actual failed action without internal
  policy jargon.
- **Capability-shaped intent grounding.** Effects are understood by governing action, concrete target, and
  command family rather than by tool name or stray keywords. Coordinated directives retain each capability;
  quoted filenames, exact commands, dotfiles, file sets, requirements, VCS/package operations, and adjacent
  “go” or “continue” confirmations remain grounded without turning intent recognition into authorization.
- **Typed evidence selection.** One `EvidenceQuery` now owns execution-recall family, predicate, and scope.
  Receipt projections distinguish exact aggregates from failure details and latest-turn from task-wide recall,
  while the slice remains elastic for active complexity and history-bounded with respect to transcript age.
- **Plugin execution-admission APIs are intentionally retired.** Plugins continue to register tools, skills,
  and MCP servers, but the old `register_hook` / `intent_effect` permission surfaces and hook-bearing
  `load_plugins` tuple are removed with the policy-gate machinery. This is a deliberate pre-1.0 API break;
  ordinary advisory behavior belongs in tool results or skills, not hidden host-side action gates.

### Fixed
- **Repository-review placeholders receive one response-only correction.** Skills may declare a typed terminal
  deliverable contract; the review contract persists in PFC across recovery and workspace handoff, distinguishes
  consumed child evidence from user-visible synthesis, and gives an obvious private-output placeholder one
  unpublished response nudge. It does not enforce headings, grade report quality, or block a later terminal answer;
  ordinary tasks and workspace transport remain outside this narrow reminder.
- **Child reports and parent seals now publish as one logical fact.** Canonical child artifact writes and
  launch-turn references are linearized against parent sealing. Cancellation may stop a child before
  publication, but can no longer arrive between the immutable write and required reference to create an
  accepted orphan; crash recovery retains the existing parent-lineage repair path.
- **Broad subagent reviews no longer collapse into timeout cascades.** Repository audits use ignore-aware,
  source-weighted scopes and staged 2–3-child waves instead of one exhaustive child per directory. Every child
  drains and assembles SSE off the main thread even without a UI sink, so an ordinary long reasoning response
  is not mistaken for 60 seconds of transport silence. The completion-budget-aware monotonic watchdog remains
  configurable with `LLM_HARD_TIMEOUT_SEC`; provider SDK retries stay disabled and the model runner is the one
  visible retry owner. A process-wide physical-call lease now bounds live requests per provider account,
  including sockets whose logical watchdog already returned, while pre-admission cancellation cannot start a
  late request and capacity wait shares the same absolute deadline as execution.
- **Canonical installs can actually self-update.** The updater now recognizes the standard
  `python = "3.12"` field emitted by current `uv tool` receipts while continuing to reject custom
  requirements, resolver settings, and non-public package sources.
- **Private state modes do not leak into shared skills.** Personal vault/config/default-skill files remain
  owner-only, while an explicitly selected project/shared skill directory creates collaborator-readable
  `SKILL.md` files and preserves an existing file mode across atomic rewrites.
- **Workspace switches preserve the live collaboration state.** One application session identity now spans
  in-process workspace handoffs; conversation, standing user intent, proposals, and every open topic survive
  A→B→A, while file residency, observed facts, revisions, receipt snapshots, and old artifact handles are
  restored only from the selected workspace. Handoff publication stages fallible derived objects first, and
  teardown/logging failures can no longer falsify a switch that already committed.
- **Durable evidence has an unambiguous order and explicit gaps.** A crash-durable monotonic turn order is
  shared by journals, artifacts, and checkpoints (including recovery and idempotent retries); legacy
  same-second ties expand as partial evidence instead of task-ID guessing. Corrupt immutable records now
  propagate bounded gap metadata into execution/quality coverage and the artifact index rather than silently
  producing exact counts over survivors.
- **Intent continuity no longer drifts across attribution or redaction.** A user who adopts wording from an
  assistant recommendation retains a live imperative; quoted/fenced proposal examples cannot authorize a
  later “yes”; nested numbered details cannot steal top-level ordinals; and persistence-only secret masks
  preserve source offsets so exact user clauses still point to the same words after recovery.
- **User-reported failures close only on real proof.** Stale tool errors clear only after the exact failed call
  succeeds. Test/build reports clear only after an explicit file-tool edit followed by a status-propagating
  shell command whose test/lint/type/build family matches the report; sequential families accumulate only
  until the next edit. Help, list, collect, dry-run, skip, or exit-zero modes do not count, and source text
  inside `execute_code` is never treated as a mutation or verification receipt.
  Neutral lifecycle cancellations remain visible as “not run” without becoming blockers, evidence, warning
  completions, or policy errors; catastrophic safety refusals remain explicit and adverse.
- **Child budgets remain hard batch ceilings.** Delegated token reservations now carry across scheduler waves,
  so interleaved effects or serialized writable children cannot each receive the parent's full remaining cap;
  children rejected before physical start no longer dilute a valid sibling's reservation.
- **Subagent fan-out now degrades gracefully under provider pressure.** A wave admits at most four child model
  loops, launches the first pair immediately, and ramps later starts instead of creating a synchronized provider
  burst. Explorers default to at most six fast evidence-navigation steps followed by one full, tool-free
  synthesis. Clean navigation or planned navigation-budget exhaustion may hand off only typed workspace
  evidence; the latter carries an explicit incomplete-coverage note. The fast navigator suppresses the generic
  max-step closeout, avoiding a redundant billed call, while cancellation, uncertainty, fatal provider/tool
  stops, token exhaustion, truncation, and evidence-free runs never synthesize. The live matrix is driven by typed
  queued/starting/model-active/reasoning/writing/tool/
  retry/terminal phases rather than detail-string inference. Exhausted attempts, truncated output, cancellation,
  or a final-synthesis failure start no wrapper-level recovery call: usable report text, observations, trace,
  usage, and WorkGraph binding seal as an explicitly partial artifact, while cancellation still wakes retry
  backoff and prevents late publication into a replacement task.
- **Hung read deadlines stay live.** Timeout-enabled pure reads use bounded daemon workers and a short grace
  period; unresolved readers become typed indeterminate outcomes, later barriers are cancelled, Ctrl-C remains
  responsive, and a permanently stuck reader no longer freezes the turn or process exit.
- **Read scheduling keeps truthful lifecycle boundaries.** Lifecycle children no longer disable adjacent read
  deadlines; exhausted daemon capacity settles as a typed not-started cancellation; late indeterminate results
  still close later effects; and a start journal crossing a deadline cannot enter its handler or route a late
  lifecycle edge into a newer turn/workspace.
- **Registry outcomes stay typed at every edge.** Preflight-cancelled calls bypass execution-only effect
  factories, availability is admitted exactly once before the durable start boundary, direct validation
  failures never construct execution effects, and registry replacement cannot mix an admitted handler with a
  stale effect factory. Invalid explicit statuses become indeterminate instead of successful, and byte or
  unrenderable extension results remain inside the canonical outcome path.
- **Streaming deadlines cannot leak output into a later turn.** On platforms without a synchronous SIGALRM
  lease, both Chat Completions and Responses interactive paths use the blocking daemon watchdog rather than
  abandoning an SSE worker that could continue emitting deltas after timeout.
- **Plugin/MCP extension startup is transactional and bounded.** Plugin packages support relative imports and
  fully roll back registry, skill, MCP, and module state on failure. MCP timeouts cancel discarded workers,
  malformed configuration/descriptors are contained before spawn, and remote tool names cannot control
  page-out paths or labels.
- **The catastrophic floor survives extension failure and common shell spelling.** A crashing/no-op earlier
  preflight hook cannot skip the later machine-safety check, and high-confidence POSIX escaped executable names
  or quoted raw-device assignments are recognized without treating escaped literal home/glob operands as wipes.
- **Varying re-inspection loops get an advisory recovery cue.** Eight distinct observation calls returning the
  same meaningful result add one model-only nudge to synthesize, act, or ask. It never rejects a tool or emits a
  user-facing policy error.

## [0.2.0] — 2026-07-10

A typed re-architecture of the core around a single canonical design (elastic slice, execution kernel,
crash-safe persistence), plus a rebuilt Rich TUI. No change to the thesis — same history-bounded,
task-elastic slice — but the invariants are now enforced by typed subsystems rather than convention.

### Added
- **Typed core kernel.** The active slice is split into typed semantic regions (intent, task, evidence,
  working set, continuity, turn runtime), each owning its own seal/reset lifecycle, replacing the flat
  field-classification table. A provenance-tagged **intent ledger** captures verbatim user directives with
  their source handles, deterministically promotes explicit constraint language (`must`/`never`/`only`),
  and records supersession/satisfaction as explicit transitions.
- **Execution kernel.** Structured tool outcomes (succeeded/failed/cancelled/**indeterminate**), ordered
  read/write waves (pure reads parallel, effects as barriers), reconciliation gates that block dependent
  actions after an unprovable outcome, and one capacity-preflighted model-call path.
- **Crash-safe persistence.** A pending-journal → immutable-artifact → checkpoint compare-and-swap commit
  order with idempotent replay, a full error taxonomy, and an OS-backed workspace lease, so an interrupted
  turn resumes without re-running side effects. Dependency-scoped workspace revisions stale only the claims
  whose inputs actually changed.
- **Rebuilt TUI.** A single `TurnProgress` event reducer folds loop events into an immutable snapshot;
  the renderer projects that snapshot (semantic tool buckets, bounded plan/tally, width-responsive
  diagnostics, one live-status owner) instead of interpreting events independently.

### Changed
- Elastic residency is centralized: a pressure controller chooses graded fidelity globally
  (full → excerpt → digest → locator) so no region can consume the window unnoticed.
- Per-turn cost framing corrected throughout the docs to **history-bounded, task-elastic** — bounded with
  respect to session length, not a fixed size regardless of task breadth.

### Fixed
- **Intent ledger and progress signals no longer lost on resume.** Checkpoint state is deep-frozen on
  write; the resume and crash-recovery paths now thaw it fully at the boundary (and the deserializers
  accept any `Mapping`), so binding user directives and their provenance survive a restart instead of
  being silently dropped.
- **A completed first-turn objective no longer stays pinned as a mandatory, un-pageable block** — a clean
  turn with no unresolved state marks it pageable background; unresolved state or explicit continuation
  keeps it active.
- Security/reliability hardening from external review: WAL/recovery redacts tool-call arguments; process
  kill reaps the whole spawn-captured process group; `code_review` and the read-only git policy refuse
  external-diff/textconv and config injection; subagent children charge their tokens to the parent budget,
  stay isolated from the parent's private state, and write durable state 0600/0700.

## [0.1.18] — 2026-07-09

### Added
- **The standing roster is now visible** to the agent — a bounded STANDING SPECIALISTS manifest in the slice
  advertises your named specialists (and the `read_file("roster/index.md")` / wake calls), so a fresh
  session discovers them instead of never finding the feature. The roster read is bounded-work (rank by a
  cheap stat, parse only the top-K), so it stays flat per turn as the roster grows.
- **A calibrated `reviewer` agent kind** — a read-only, parallel reviewer whose prompt enforces a severity
  rubric and four anti-cry-wolf disciplines (read the adjacent comment, trace tainted data to its real
  consumer, single-user-local threat model, refute your own finding). `spawn_agent(agent="reviewer", …)`.

### Changed
- **One delegation tool.** `spawn_explore` / `spawn_subagent` are collapsed into `spawn_agent(agent=<kind>,
  name?, grants?)` (measured parallel-fan-out parity). The delegation mental model is rewritten around two
  orthogonal dials — the KIND (`agent=`) and the IDENTITY (`name=` → omit for a one-shot temp, pass to HIRE
  a standing specialist you can WAKE later). Old tool names still work.
- **No roster hire cap.** A dormant specialist is just files on disk; the bound is on the surfaced view, not
  the stored count.

### Fixed
- **Streaming hard-deadline** is re-raised (not downgraded to a partial truncation) so a wall-clock stall is
  actually retried.
- **Auto-approve safety:** a broad `AGENT_AUTO_APPROVE` glob no longer silently approves `git push`, package
  publishes, or `rmdir` — they fall through to a confirmation.
- Delegation guidance is spliced into the system prompt again (its gate keyed on a removed tool); a path
  auto-grant no longer trips on version-shaped tokens; ripgrep subprocesses decode as UTF-8; a topic-park
  failure no longer masquerades as "no such topic"; an unknown scheduler access type serializes safely; the
  virtual roster FS resolves `profile.json` as well as `profile.md`.

## [0.1.17] — 2026-07-09

### Added
- **Standing specialist roster — hire once, wake many.** A named delegation
  (`spawn_agent`/`spawn_explore` with `name="…"`) now HIRES a durable specialist; re-using the name WAKES
  it, rehydrated from its own sealed archive at flat cost regardless of career length (identity is an
  archive key, not a running process). A woken specialist is seeded with a bounded identity block: its
  recent career manifest, its lessons, and an abstention self-model ("your memories are only your sealed
  reports — say so beyond them"). The hippocampus now records who your agent works with, what they were
  told, and what they found.
- **Instance identity + verbatim-brief provenance.** Every sealed artifact carries the instance `name` and
  the exact brief it was given, so a report's reader always sees the question alongside the answer;
  `subagents/index.md` is the roster and `subagents/<name>.md` aliases a specialist's latest job.
- **Capability grants.** A parent wires one child's sealed report to another by granting an exact handle in
  the brief — default-deny, one-hop (no re-grant), spawn-time existence-checked — so children still couple
  only through seals.
- **Synthesiser agent + lessons + searchable delegated work.** A read-only `synthesiser` kind reduces N
  granted sibling reports into one cited synthesis; a seal-time `LESSON:` reflection is curated into the
  specialist's profile (deduped, capped, provenance-tagged); and delegated work is dual-written to the FTS5
  index so `search_history` finds it by content without touching the turn timeline.

### Fixed
- **Race-safe hire + durable-store hardening.** Concurrent same-name hires are atomic (in-process lock +
  `O_EXCL` create + cap enforced under the lock), profile writes are atomic (tmp + `os.replace`), and the
  roster tolerates corrupt/legacy records (null date fields no longer crash the wake seed or the roster
  listing) — surfaced and fixed across three adversarial bug-hunt rounds.

## [0.1.16] — 2026-07-08

### Added
- **Subagent structured artifacts.** A delegated child now seals a typed report into a `subagents/`
  archive and hands the parent a bounded digest plus a `read_file("subagents/sub-N.md")` recall handle —
  the cache-not-log moat applied to delegation. The parent absorbs O(1)-sized digests while the child's
  full detail stays paged out on disk yet fully recallable (manifest at `subagents/index.md`, per-child
  reports, grep). Ids are assigned race-safely under concurrent fan-out, secrets are redacted on persist,
  and a child is isolated from the parent's `history/`, siblings' `subagents/`, and `search_history`.

### Fixed
- **FileLock flush-before-release.** The advisory lock now flushes buffered writes before releasing the
  lock, so a concurrent appender counting lines can't race a half-written file (surfaced by parallel
  subagent fan-out).

## [0.1.15] — 2026-07-08

### Added
- **Anthropic prompt-cache breakpoint.** On a Claude/Anthropic endpoint the stable system prefix is now
  marked with a `cache_control` breakpoint, so Anthropic serves the whole prefix from cache on later
  same-prefix turns (it was previously a no-op stub). Gated to Claude endpoints — the default DeepSeek /
  OpenAI path is byte-for-byte unchanged.

### Changed
- **Slice field lifecycle is now explicit and enforced.** Every `Slice` field is classified in one table
  (carry / reset / custom at the turn boundary), and the suite fails if a field is added without a
  lifecycle decision — or if `seal()`/`reset()` mishandle it. Closes a class of silent bugs where state
  could leak across tasks or accumulate across turns.

### Fixed
- **Episode-writer concurrency.** An advisory file lock (real on POSIX, graceful no-op elsewhere) now
  serializes concurrent appenders to the same session log so records can't interleave into a torn line.

## [0.1.14] — 2026-07-08

### Changed
- **RECENT CONVERSATION tier now keeps the last few completed turns verbatim** (was an
  800-char head-cut applied every turn). Fixes cross-turn reference resolution: a
  recommendation or conclusion stated at a reply's *tail* now survives, so a follow-up like
  "go with your recommendation" resolves against it instead of falling to recall and grabbing
  an older, keyword-matching turn. The bound is the turn *count*, not a byte cap — per-turn
  peak stays flat across session length (older turns still page out to `history/`).

### Added
- **Value-provenance clarify cue:** the agent treats a concrete value it did not observe
  (a number / id / port / path) as an unstated requirement — it asks or leaves an obvious
  placeholder instead of inventing a plausible default. Scoped to *unsourced* values, so it
  never second-guesses a value it legitimately has. (Measured: absent-value confabulation
  0.83 → 0.25, with no over-abstention on values the agent does have.)

### Fixed
- **Clean exit on Ctrl-C during shutdown:** a Ctrl-C that lands in the session-end memory
  consolidation (a slow subprocess LLM call) no longer dumps a traceback — the whole
  shutdown sequence is guarded so an interrupt just quits.

## [0.1.12] — 2026-07-03

**Native Windows support — one command, no WSL:**
`irm https://raw.githubusercontent.com/TT-Wang/sliceagent/main/install.ps1 | iex`
(installs pinned uv + sliceagent with its own Python 3.12, and SHA256-verified Git Bash + ripgrep when missing; user-scoped, no admin).

### Added
- `platform_compat` seam: shell commands run under **Git Bash** on Windows (the model's bash-syntax
  tool calls work unchanged), win32 process groups + `taskkill` tree-kill, drive-letter/MSYS path
  extraction for shell-path grants.
- Forward-slash path contract in all model-facing output on Windows (`rg --path-separator /`,
  normalized rel paths in the code index / repo map / listings / `@file` completion) and a
  win32-only system-prompt note about Git Bash path quoting.
- CI `windows-latest` cell (full suite green on Windows) + a Windows-footgun lint
  (`scripts/check_windows_footguns.py`) that keeps the Unix-ism bug class out permanently.

### Changed
- memem floor → **2.9.7** (Windows-importable memory backend; also carries 2.9.6's
  legacy-frontmatter auto-repair).

### Known Windows limitations
- Interactive PTY sessions (`terminal_open`) are not yet available natively (clean refusal; use
  `run_command` / `proc_start`). Planned: pywinpty bridge.

**Linux/macOS: zero behavior change.** Every Windows branch is platform-gated; POSIX call paths are
byte-identical, pinned by identity tests, and verified by an adversarial POSIX-regression review.

## [0.1.11] — 2026-07-03

OSS-polish pass (triaged from two external reviews; the confirmed quick wins).

### Fixed
- `_as_text` (tool-result coercion): a tool returning `None` now becomes `""` (not the literal string
  `"None"`) and `bytes` are decoded (not rendered as a `b'…'` repr) before entering the slice.
- Rebrand leftovers: `sliceagent.toml.example`, the example plugin, and `.gitignore` still referenced
  the old `memagent` / `.memagent/` name and paths — updated to `sliceagent` / `.sliceagent/`.
- `.gitignore` now ignores `.sliceagent/` (a session writes paged-output blobs into a project-local
  `.sliceagent/blobs/`, which was showing up as untracked repo cruft); `.dockerignore` excludes it too.
- The generated config reference (`docs/CONFIGURATION.md`, an internal/untracked doc) is regenerated
  from `envspec` so `AGENT_MODEL` shows no default (it is required); a drift-guard test keeps it honest
  and skips cleanly where the untracked doc is absent (CI / installed package).
- `install.sh`: removed a dead `REPO` (git-URL) variable — the installer tracks the PyPI release, one
  canonical path; `ROADMAP.md` updated to match (it still described the old `git+…` install).

### Docs
- README: the benchmark section now notes the model is swappable via `AGENT_MODEL` and points at the
  reproducible `evals/` harnesses; added a **pre-1.0 stability** statement (SemVer, 0.x may change,
  breaking changes in the changelog, numbers are directional).

## [0.1.10] — 2026-07-03

### Fixed
- Startup banner: the block wordmark clipped its right edge — the final "t" of "sliceagent" — on
  terminals narrower than ~91 columns, because the frame's indent + padding + border overhead pushed
  the 79-column art past the window. The layout now sheds that chrome as the window narrows, so the
  full name renders from ~85 columns up (wide windows keep the roomy framing).

## [0.1.9] — 2026-07-03

Bug-hunt round 2 (deep-core lenses, full 3-vote adversarial verify). Five confirmed fixes.

### Security
- The durable debug log's `_scrub_args` now redacts secrets in NESTED dict/list tool arguments
  (e.g. an MCP call's `{config:{api_key:…}}` or `{headers:{Authorization:…}}`) — top-level-only
  redaction leaked them to `~/.sliceagent/logs/**/durable-log.jsonl` in plaintext.
- `AGENT_PROXY=none` (documented "force a DIRECT connection") now truly forces direct: the httpx
  client is built with `trust_env=False`, so an ambient `HTTPS_PROXY` can no longer silently route
  your API traffic through a proxy you told sliceagent to bypass (both first build and `/model` hops).

### Fixed
- `str_replace`/`edit_file` no longer flip a whole file to CRLF when it's mostly-LF but contains a
  single embedded `\r\n` (a byte literal, an HTTP fixture, a merge artifact). CRLF is now detected by
  DOMINANCE, not mere presence — uniformly-CRLF Windows files are still preserved.
- Session-end consolidation no longer crashes (and silently discards ALL of a session's promoted
  lessons/skills) when the episodic log contains one malformed record.
- `/undo` and `/cwd` (and `/plugins`) no longer corrupt output or crash Rich with a MarkupError when a
  path contains brackets — e.g. a Next.js `app/[id]/page.tsx` route or `~/proj/[slug]`.

## [0.1.8] — 2026-07-02

Launch-day bug-hunt round: a security fix plus config-robustness and /model correctness.

### Security
- `run_command` in the default *teenager* mode no longer auto-runs exec/write commands disguised as
  "read-only": `env <program>`, `sort -o FILE`, `date -s`, `tree -o FILE`, `uniq IN OUT`, and
  `git grep --open-files-in-pager` now take the confirm path (arbitrary code-exec / file-overwrite
  confirm-bypass). Read-only siblings like `du -s` / `grep -o` still auto-run.

### Fixed
- Boot no longer cross-wires a prefs-pinned provider's key with the DEFAULT provider's endpoint (an
  env `gpt-5.5` @ OpenAI pin on a DeepSeek-default config used to 401 every call on relaunch).
- A prefs `provider`/model pin whose `[providers.<id>]` table was removed from config.toml is now
  dropped at boot instead of forcing a model onto the wrong endpoint.
- The wizard no longer silently ERASES other providers' API keys when config.toml is unparseable —
  the old file is moved to a non-clobbering `.bak` first (keys recoverable).
- `sliceagent config` / `config --use` / the init wizard no longer crash on a non-UTF-8 config.toml
  or a scalar value under `[providers]` (they degrade).
- `llm.switch()` (a `/model` provider hop) closes the replaced HTTP client — no more fd leak per hop.
- The `/model` reasoning menu offers `high` (not a misleading `max`) for OpenRouter models, and offers
  the levels the *target* provider supports when the pick rebinds the endpoint.
- The wizard's typed provider fallback rejects an unknown choice instead of silently configuring
  OpenRouter.

## [0.1.7] — 2026-07-02

### Fixed
- `/model`: the current env-configured model row is now labeled `current (env)` instead of the
  provider family name — it could masquerade as a configured provider (e.g. an env `gpt-5.5` on a
  DeepSeek endpoint showed as "deepseek").

## [0.1.6] — 2026-07-02

The clear config journey: one wizard, two doors, an honest /model.

### Added
- **`/config`** — manage LLM providers *inside* sliceagent: the same wizard as first-run onboarding
  (provider → model → key → live test), then the config hot-reloads and the new provider shows up
  in `/model` immediately.
- **`/model` switches providers for real** — the menu lists ONLY configured providers' models
  (saved model + suggestions, labeled by provider), and picking one switches **model + endpoint +
  key together** (the old menu changed the model string but never the endpoint). The last-picked
  provider is remembered across sessions.

### Changed
- Typed `/model <name>` stays same-endpoint (documented as such); the mismatch warning now points
  at `/config` + the `/model` menu instead of `config --use`.

## [0.1.5] — 2026-07-02

### Changed
- Wizard step order is now **provider → model → key** ("choose what you want, then prove you can") —
  the key is the last thing typed, so the live test follows it immediately.

## [0.1.4] — 2026-07-02

### Fixed
- **Wizard menus render as a proper vertical list.** The single-line selector wrapped with six long
  provider labels, and its clear-one-line redraw stacked copies of itself down the screen. New
  `_menu_select`: one option per row, width-clamped labels (wrap impossible), in-place cursor-up
  redraws, explicit `\r\n` in raw mode. PTY-tested, including the anti-stacking invariant.

## [0.1.3] — 2026-07-02

The five-door provider lineup + a wizard that feels like one.

### Added
- **Providers**: OpenRouter (hundreds of models, one key — now the first door), OpenAI,
  **Anthropic/Claude** (new — via Anthropic's OpenAI-compatible endpoint), DeepSeek, Moonshot/Kimi,
  plus custom endpoints. All five ride the single adapter — zero new dependencies.
- **OpenRouter quirks**: reasoning intent maps to OpenRouter's unified `reasoning` object (works
  WITH tools — the raw `reasoning_effort` param never could); tool-calling requests pin routing to
  hosts that honor every param (`require_parameters`) so nothing silently degrades; per-call
  `usage.cost` is parsed into the cost meter.
- `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` / `OPENROUTER_API_KEY` accepted as key fallbacks.

### Changed
- **Wizard UX**: API key shows `******` as you type (no more invisible field); provider and model
  are arrow-key menus (↑/↓ + Enter, Esc cancels) with per-provider model suggestions +
  "type another model id…". Scripted/CI runs keep the plain typed flow.

## [0.1.2] — 2026-07-02

Nothing may hang the user — three fixes from a live first-run reproduction.

### Fixed
- **Repo-map walk is hard-bounded** (`max_dirs` budget): the first slice build can no longer hang for
  minutes when the workspace root is huge (a home directory mistaken for a project, a giant monorepo).
  Output caps existed; now the walk itself is bounded — worst case ~1s, maps what it saw.
- **The OS-account home never counts as a project root**, independent of `$HOME`: a stray
  `package.json` in the real home no longer turns the entire home directory into a "project"
  (the prior guard compared against `$HOME`, which containers/sandboxes/sudo can override).
- **Ctrl-C during the slice build cancels the turn cleanly** (`· cancelled`) instead of crashing the
  REPL with a traceback — the build phase ran before the turn's interrupt handling.

## [0.1.1] — 2026-07-02

First-run onboarding, hardened by a live stranger walkthrough.

### Added
- **Auto-onboarding** — a bare interactive `sliceagent` with nothing configured now drops straight
  into the guided setup wizard (provider → key → model → live test), then continues into the session.
  No more "go run `sliceagent init`" bounce. Piped/CI runs keep the print-and-exit gate.
- **Installer handles everything** — `install.sh` now also installs ripgrep (brew when available,
  else an isolated ~2 MB static binary; no sudo) and pins `uv tool install --python 3.12`, so the
  curl path has zero prerequisites even when the default Python is 3.9/3.10 (conda base, Ubuntu 22.04).

### Changed
- README: the curl installer is the primary install path; PyPI/pip is the "you manage the Python"
  alternative. (Project renamed memagent → sliceagent the same day, before 0.1.1: PyPI's
  name-similarity check blocks "memagent".)

## [0.1.0] — 2026-07-02

First public release.

### Added
- **`sliceagent init`** — guided first-run setup (provider, API key, model); tests the key and writes
  `~/.sliceagent/config.toml` (0600). Config-persisted keys mean the next run needs no env vars.
- **`sliceagent config` / `config --list` / `config --path`** — discover every setting, its default, and
  current value. New central env-var registry (`envspec.py`) is the single source of truth.
- **`sliceagent help` / `version`** subcommands.
- **Startup config validation** — a typo'd enum (e.g. `AGENT_POLICY`) now warns instead of silently
  defaulting.
- **Always-pinned live UI** (`AGENT_TUI=live`) — a bordered input box stays at the bottom while output
  streams above it, in the normal buffer (native copy/paste).
- **Lexical topic router** (default) — routing no longer costs an LLM round-trip per follow-up;
  `AGENT_ROUTER=llm` restores the classifier. Measured identical to the LLM router on continue/resume.
- **Per-tool wall-clock timeout** (`AGENT_TOOL_TIMEOUT`, opt-in) — a stuck tool no longer hangs the turn.
- Docs: `QUICKSTART.md`, `CONTRIBUTING.md`.
- **One-command install** — `install.sh` (bootstraps `uv` → isolated tool install), a `Dockerfile`, and a
  README `## Install` (uv / pipx / docker). MIT `LICENSE` declared in metadata; `NOTICE` records
  third-party attributions; `SECURITY.md` documents the threat model + a disclosure path.
- **Three permission modes** — `baby-sitter` / `teenager` (default) / `let-it-go`, all sharing a
  catastrophic-command floor; `/mode` + `/model` two-tier selector menus; **Esc = undo**.
- **MCP spawn-security screen** (`mcp_security.py`) — refuses a shell-interpreter MCP entry that does network
  egress or writes OS-persistence surfaces, before it is spawned.
- **`read_file` window** — `offset`/`limit` + a default view cap + a `<system>` footer (a large file no
  longer floods the slice); a `glob` file-finder; `grep` `output_mode`/`--type`/context; `str_replace`
  `replace_all`; model pricing single-sourced in `model_catalog`; `with_retry` honors `Retry-After`.
- **"$ saved" cost meter** — shows dollars saved vs a full-transcript agent, re-priced live on `/model`.
- **CI** (`.github/workflows/ci.yml`, ubuntu+macOS × py3.11/3.12: install + lint + tests),
  `scripts/run_tests.sh`, a `ruff` config + `[dev]` extra, single-sourced version, and contribution
  scaffolding (`CODE_OF_CONDUCT.md`, issue/PR templates).

### Removed
- The full-screen Textual UI (`AGENT_TUI=textual`) and its `textual` dependency. The inline `rich` REPL is
  the proven default (native copy/paste/scrollback on any terminal); `AGENT_TUI=live` remains for the
  always-pinned composer, `off` for plain stdout.

### Changed
- Default UI is the inline `rich` REPL (native copy/paste on any terminal). The composer is a bordered,
  bottom-pinned box.
- Permission confirms are arrow-key selectable (Yes / No / Always) instead of typed.
- The user's message is echoed the instant Enter is pressed — before any routing/LLM work (no input lag).
- `AGENT_PROXY=off` now forces a direct connection (previously misread as a proxy URL); the network route
  is shown on startup (`net=…`).
- Read/list tool cards show the action header only (no content dump); commands and failures show output.

### Reliability
- Hook callbacks are guarded: advisory hooks (budget/oracle/plugin) degrade gracefully on error; the
  permission hook fails **closed** (denies) on error.
- Streaming assembly survives a malformed chunk (skipped) and a mid-stream drop (salvages the partial, or
  re-rolls via retry when nothing was assembled).
- Session teardown is guarded and bounded — a stuck MCP server / index write can't freeze exit.

### Core
- Initial Python core: the slice/cache-not-log loop, typed memory tiers, reconstruction seam,
  event-sink host. Tools, skills, subagents, MCP, plugins, sandbox, permission policy,
  session/topic resume, durable cross-session memory (memem).
