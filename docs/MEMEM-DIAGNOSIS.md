# memem (L3) Layer — Diagnosis & Action Plan

Scope: memagent's cross-session memory layer — (1) distill→memem, (2) retrieve→slice, (3) the missing transcript→skill. Compared against EverOS (engine), EverMe/evermem-claude-code (plugin), and Hermes `/learn`. Generated 2026-06-25 from a 5-reader workflow + a supplemental EverMe study; every claim verified against source.

## 1. Diagnosis

### Distill (memories → memem) — VERDICT: **HEALTHY** (one real stub, one real security gap, rest are honest design limits)

**Evidence (verified against source):**
- Two-write design is sound: a LIVE per-turn miner (`mining.py:84-122` `LessonMiner._on_turn_end`) plus a session-end batch sweep (`memory.py:463-490` `consolidate`). Both go through the same `Memory.remember()` seam → memem stays behind the interface (moat: llm-agnostic ✓).
- The corrective gate is genuinely 4-fold, not 3 — the input under-described it. `mining.py:90-101`: `stop_reason=="end_turn"` **and** `self._errors` present **and** `not s.last_error` (turn ended clean) **and** `s.edited_files` (a real fix was made) **and** key not in `_saved`. The `edited_files` requirement is the load-bearing gate that prevents mining "Lesson: <user's query>" from no-op turns. This is correct and well-reasoned.
- `is_self_inflicted()` (`mining.py:35-41`) filters host-guardrail errors so confinement/permission hits teach nothing — task-agnostic substring match. Good.
- Title is the **pitfall signature, never the goal** (`mining.py:121`, `pitfall_signature` strips the `Error:` prefix at `mining.py:50-54`) — recall matches on the engineering failure, not user phrasing. This is a deliberate, correct fix.
- `make_miner()` returns `None` for `NullMemory` (`mining.py:174-179`) → graceful degrade ✓.
- `promote_episodes`/`promote_procedures` (`consolidate.py:62-98`, `114-150`) are **pure** (no I/O, no LLM), dedupe by signature/shape, frequency-weight via `Counter`, cap procedures at 3. Clean and testable.
- Provenance is stamped on auto skills (`consolidate.py` render via `skill_provenance.frontmatter_line(AUTO)`) → curator can prune only auto skills ✓.

**Gaps/bugs (verified):**

1. **REAL — secret can leak into the LLM distill prompt** (`mining.py:138-161`). `_distill()` sends `task` + `pitfall` to `self.llm.complete()`. Threat-scan/redact happens only in `remember()` (`memory.py:304-309`), i.e. **after** distill. An API key in an error message reaches the LLM in plaintext. The deterministic consolidate path scans `_is_secret` *before* building (`consolidate.py:74`), but the LLM mining path does not. **This is the one fix-now security item in distill.**

2. **REAL but a known stub — `render_skill()` is deterministic-only** (`consolidate.py:153-168`). The module docstring and the function docstring both say "the LLM-distill upgrade (generalizing the steps) slots in here" but `render_skill(proc)` takes only a dict, no `llm` param, no mode flag. Skills are *recorded* procedures, not *generalized* ones. Verified: no LLM call anywhere in the function.

3. **MINOR — misleading wording.** `render_skill` emits "Auto-distilled from N run(s)" (`consolidate.py` body) while the docstring elsewhere correctly calls it "a RECORDED procedure." Use "Observed from N run(s)" until LLM generalization actually lands.

4. **DESIGN LIMIT (not a bug) — `MAX_PROCEDURES=3` silent cap** (`consolidate.py:24,148`). Top-3-by-freq promoted, rest dropped silently. Correct spam guard; add a debug log only.

5. **DESIGN LIMIT — in-session dedup only.** `_saved` (`mining.py:72`) is session-local; cross-session dedup falls to memem's `memory_save` token_set_ratio (`operations.py:19-68`: ≥0.92 reject, 0.70–0.92 Haiku-merge). This is the correct division of labor (BORROW-periphery: memem owns dedup), but it is a *silent* reliance.

6. **CORRECTION to the input:** the input claims `is_self_inflicted` could misclassify and *lose* data with no log. True there's no log, but this is intentional filtering of host errors, low severity — not data loss of real lessons.

### Retrieve (memem → slice) — VERDICT: **HEALTHY**

**Evidence (verified):**
- Single read seam: `slice.py:891-893` calls `pages.lookup(goal, kind="memory-lessons", k=6)` once per build, cached by goal (`recall_cache[goal]`). One memem call per topic, not per turn ✓.
- `recall()` (`memory.py:283-302`) order is correct: `retrieve(query, k=k, writeback=False, scope_id=self._scope)` → relevance-gate via `_memory_relevant` → `mark_used()` **only on surfaced hits** → return `Snippet[]`. Reinforcement tracks what the agent actually sees, not raw top-k. This is the right feedback discipline.
- `_memory_relevant()` (`memory.py:45-53`) is a whole-word discriminating-term gate (`\bterm\b`, lowercased), tolerant when no terms extracted. Task-agnostic, portable. Turns memem's blind top-k into relevant-or-nothing.
- `render_memory()` (`slice.py:655-661`) wraps every lesson in `wrap_untrusted(kind="memory")` each turn (re-applied, not persisted); suppresses empty tier. Untrusted fencing is correct.
- Tier is STABLE, reconstructed once per build from live recall — bound-is-relevance, no within-loop size cut ✓.

**Gaps/bugs (verified):**

1. **MINOR (correct-by-design) — empty-terms keeps all hits** (`memory.py:50-51`). When the query yields no discriminating terms, the gate returns `True` for everything, so memem's k=6 is the only filter. Low practical risk (memem already disambiguates), but it does relax "relevant-or-nothing" on un-discriminating goals.

2. **UNDER-USE, not bug — no score floor.** `_memory_relevant` is a term gate with no confidence floor; relies entirely on memem's RRF ranking + k=6. Acceptable (memem owns scoring).

3. **The input's "scope param mismatch BUG" is NOT a bug — confirmed.** `memory.py:313` passes `scope_id=scope` correctly; the adapter renames `scope`→`scope_id` at the call site. Working as intended.

### memem backend fit — adapter uses memem correctly; several capabilities under-used

The adapter is faithful: it talks only to `retrieve`, `memory_save`, `bump_access` and lets memem own embeddings, BM25, FTS5, RRF fusion, 3-phase decay, and dedup (BORROW-periphery ✓). Under-used memem capabilities (all confirmed against the input's retrieve.py line refs):

- **`paths_context`** (retrieve.py:712-716) — never populated. The slice knows `s.active_files`; passing them would give a 1.05× bonus to memories tagged with the files being edited. Cheap, moat-safe.
- **`writeback=True`** (retrieve.py:950-960) — adapter passes `writeback=False` and manually loops `mark_used()`. Not wrong, but the manual loop reinforces *every kept hit*, which is actually *better* than blind writeback (reinforces only surfaced). Keep the manual loop; this is a deliberate, correct divergence, not a defect.
- **`should_demote` / explicit layer management** (decay.py:123-141) — never called. memagent trusts memem's ranking decay and never demotes. Fine for now (no curator yet).
- **Cross-vault episode pooling** (search_index.py) — memagent searches only its own vault. By design (vault decoupled); not a gap to fix.

**Capability comparison table:**

| Capability | memagent today | EverOS | Gap |
|---|---|---|---|
| Distill trigger | Per-turn LIVE miner + session-end deterministic sweep | Sync pipeline + async OME strategies on event bus | memagent has no **user-initiated** distill (no `/learn`) |
| Fact extraction | Deterministic corrective-pitfall, optional 1-shot LLM | LLM atomic-fact extraction always on | memagent LLM path is optional + has a secret-leak gap |
| Procedure→skill | `promote_procedures`→`render_skill` (RECORDED, no LLM) | `extract_agent_skill` with cosine clustering + LLM | No **generalization**; no clustering of similar procedures |
| Skill ranking | freq-weight + `skill_usage` sidecar (`bump_use` on load, skills.py:153) | cosine top-K, MAX_SKILLS_IN_PROMPT=10 | memagent caps via MAX_ACTIVE_SKILLS; no usage→memem decay |
| Consolidation merge | dedup by sig/shape, no cluster merge | EpisodeReflector LLM merge, deprecation links | No cluster-aware merge / deprecation chain |
| Retrieval ranking | memem 3-way RRF + term gate (delegated) | 4-layer hierarchical, MaxSim child rerank, LR eviction | No L2 MaxSim fact-rerank (delegated to memem; acceptable) |
| Live re-index | episodic JSONL + FTS5 mirror on write | cascade daemon watches markdown, sha256 re-embed | memem owns vault re-index; memagent episodic is append-only (fine) |

## 2. What to borrow from EverOS — moat-safe only

1. **Cluster-before-promote for procedures.** EverOS `extract_agent_skill` clusters episodes by `task_intent` cosine before writing a skill. **Change:** in `promote_procedures` (`consolidate.py:114-150`), before the action-shape dedup, optionally group candidates whose `goal` embeddings (via memem's embedder — borrow, don't reinvent) are close, and promote the cluster centroid. Keep it pure by accepting an injected similarity fn so the function stays LLM-agnostic and testable. Effort M. Moat-safe: still deterministic structure, memem owns embeddings.

2. **Deprecation links instead of silent overwrite.** EverOS marks superseded memories `deprecated_by` rather than deleting. **Change:** when consolidate re-promotes a fact whose sig already exists, pass a `related`/`supersedes` field through `remember()`→`memory_save` so memem's link signal (retrieve.py link bonus) reflects the chain. Effort S. Moat-safe (just frontmatter).

3. **Make consolidation strategies configurable (hot-enable).** EverOS gates reflection behind `ome.toml`. memagent's `consolidate()` always runs both promoters. **Change:** add `cfg` flags `promote_facts` / `promote_procedures` / cap overrides (mirrors the existing `cfg.mine = deterministic|llm|off`). Effort S. Moat-safe.

4. **DO NOT borrow** EverOS's always-on LLM reflection or its 4-layer hierarchical retriever. That would duplicate what memem already owns (RRF, decay) and violate BORROW-periphery. memagent's delegation to memem is the correct posture.

## 3. The missing function — transcript → reusable SKILL (Hermes `/learn`)

**Reconciliation (the key question):** memagent's `consolidate.py` "routes procedures → skills" is **real and working** (`memory.py:476-488` writes `~/.memagent/skills/{name}/SKILL.md`), but it is **automatic, deterministic, session-end, and RECORDED-only**. There is genuinely **no foreground user-initiated path** — confirmed by `grep` (no `learn` tool in `cli.py`/`tools.py`) and by `skill_provenance.py:5` which states verbatim: *"memagent has no foreground tool that writes skills today — consolidate.render_skill is the ONLY writer."* So Hermes `/learn` is a **genuine missing function**, not a duplicate.

**Decision: EXTEND, don't replace.** Keep `consolidate` as the background auto-path. Add a foreground `/learn` that reuses the *same* storage (`skills_dir` + `render_skill` schema + the SKILLS tier) but differs on three axes: trigger (user), distillation (LLM generalization, the very upgrade `render_skill` was stubbed for), and provenance (`USER`, never auto-pruned — the `set_authoring_origin` ContextVar at `skill_provenance.py` already exists for exactly this).

**Concrete design:**

- **Trigger:** a foreground `learn` tool (model-callable) + a `/learn` CLI command, following Hermes (`learn_prompt.py`): open-ended, sources = current session transcript / a path / pasted notes. Default source when empty = "the workflow we just went through" → read from the **episodic cache** (`read_episodes(session_id)`), NOT the slice (preserve the no-transcript invariant — the episodic cache is the lossless replay store, `memory.py:4`).
- **Distill prompt:** borrow Hermes' embedded `_AUTHORING_STANDARDS` (house-style frontmatter + When-to-Use / Process / Pitfalls / Verification sections). This is finally the implementation of the `render_skill` LLM upgrade — add an `llm`-bearing variant `render_skill_llm(proc, llm)` as a **separate pure-ish function** (keep deterministic `render_skill` intact), so the moat seam is unchanged. Prompt instruction: "generalize the recorded steps into reusable prose; phrase as declarative process, not session-specific narration."
- **Skill schema:** identical to `render_skill` output (name/description frontmatter + body) so the existing `SkillManager.discover()` (`skills.py:95-126`) parses it unchanged. Set `provenance: user` via `set_authoring_origin(USER)`.
- **Storage:** existing `~/.memagent/skills/{name}/SKILL.md`. Recall/reuse: existing SKILLS tier (`render_skills`, `MAX_ACTIVE_SKILLS`, loaded on `skill()` call into ACTIVE SKILL tier). No new tier needed.
- **Dedup/quality gate:** borrow Hermes' frontmatter-first validation + name-collision check + atomic-write-then-security-scan-with-rollback (`skill_manager_tool.py:291-327`, `579-585`). memagent already has `scan_for_threats(scope="strict")` on skill write (`memory.py:481`) — wire it into the `/learn` path identically.
- **Moat-fit:** ✓ task-agnostic (no skill-type heuristics), ✓ llm-agnostic (LLM only inside the new render fn, behind the same `Memory`/llm seam, deterministic `render_skill` remains the fallback when no llm), ✓ no-transcript (reads episodic cache, not slice), ✓ BORROW (Hermes authoring standards + memem storage, nothing reinvented).
- **Secret handling:** run `scan_for_threats` on the source material **before** sending to the distill LLM (same fix as distill-bug #1) — `/learn` reads a transcript that may contain secrets.

## 4. Prioritized ACTION PLAN

### FIX-NOW (real bugs in §1)

**F1 — Threat-scan before LLM distill (security).**
- What: in `mining.py:_distill`, call `scan_for_threats`/skip if `_is_secret(pitfall or task)` **before** `self.llm.complete()`. Mirror `consolidate.py:74`.
- Files: `mining.py:138-161` (+ import from `safety`).
- Effort: S. Risk: low (only blocks tainted episodes from the LLM; deterministic path still mines).
- Verify: unit test — episode with `API_KEY=sk-...` in pitfall, `mode="llm"`, assert `llm.complete` not called and either deterministic fallback or no-mine.

**F2 — Fix misleading "Auto-distilled" wording.**
- What: change `render_skill` provenance line to "Observed from N run(s)" until F-BUILD-B lands.
- Files: `consolidate.py` (the `prov = ...` line).
- Effort: S. Risk: none. Verify: existing render_skill test asserts new string.

**F3 — Log the silent procedure cap.**
- What: when `len(cand) > cap` in `promote_procedures`, emit a debug log of how many were dropped.
- Files: `consolidate.py:148-150`. Effort: S. Risk: none. Verify: test with 5 distinct smooth workflows asserts cap=3 + log emitted.

### BUILD (the `/learn` function — §3)

**B1 — `render_skill_llm(proc, llm)` (the stubbed upgrade).**
- What: separate pure function that LLM-generalizes recorded steps into Hermes-standard sections; falls back to deterministic `render_skill` on llm failure/absence.
- Files: new fn in `consolidate.py`; embed Hermes `_AUTHORING_STANDARDS`.
- Effort: M. Risk: med (LLM output quality) — gate behind validation + provenance. Verify: golden-output test with a stub llm; assert frontmatter parses via `SkillManager._load`.

**B2 — `learn` foreground tool + `/learn` CLI command.**
- What: read default source from `read_episodes(session_id)`; **scan_for_threats first**; call `render_skill_llm`; write via the existing skills-dir path with `set_authoring_origin(USER)`; frontmatter+collision validation; atomic-write-then-scan-then-rollback.
- Files: `tools.py` (register tool), `cli.py` (command), reuse `memory.py` write path, `skill_provenance.py`.
- Effort: M. Risk: med. Verify: integration test — run a 4-step session, call `/learn`, assert a USER-provenance SKILL.md exists and `SkillManager.discover()` catalogs it; assert a transcript with a secret is redacted/blocked before the LLM.

### BORROW (EverOS adoptions — §2)

**R1 — `paths_context` into recall.** Pass `s.active_files` to `retrieve(paths_context=...)` in `memory.py:recall`. Effort S. Risk low. Verify: assert retrieve called with non-empty paths_context when slice has active files; recall_cache T2 test still green.

**R2 — Configurable consolidation flags.** Add `cfg.promote_facts/promote_procedures/caps`. Files `config.py`, `consolidate.py`, `memory.py:consolidate`. Effort S. Risk low. Verify: flag off → no facts written.

**R3 — Cluster-before-promote (optional similarity fn).** Inject an embedder into `promote_procedures` to group near-duplicate goals before shape-dedup. Files `consolidate.py`, wiring in `memory.py`. Effort M. Risk med (keep pure via injection). Verify: two semantically-equal-but-different-shape workflows collapse to one skill.

**R4 — Deprecation/supersedes link on re-promoted facts.** Thread a `supersedes` field through `remember`→`memory_save`. Effort S. Risk low. Verify: re-promoting a known sig sets the link frontmatter.

---

**Honest bottom line:** Distill and Retrieve are genuinely **healthy** — the corrective gate, pitfall-titling, pure promoters, term-gate, untrusted fencing, and memem delegation are all correct and moat-compliant. There are exactly **two real defects worth fixing now** (F1 the secret-leak-into-LLM, F2 the wording), one **honest stub** (`render_skill` never got its LLM upgrade), and one **genuinely missing feature** (foreground `/learn`). Everything else is correct-by-design delegation to memem or deliberate divergence — do not invent work there.

Files of record: `/Users/tongtao/Desktop/memagent/src/memagent/mining.py`, `/Users/tongtao/Desktop/memagent/src/memagent/consolidate.py`, `/Users/tongtao/Desktop/memagent/src/memagent/memory.py`, `/Users/tongtao/Desktop/memagent/src/memagent/slice.py`, `/Users/tongtao/Desktop/memagent/src/memagent/skills.py`, `/Users/tongtao/Desktop/memagent/src/memagent/skill_provenance.py`, `/Users/tongtao/Desktop/memagent/src/memagent/skill_usage.py`.

---

## Addendum — EverMe / evermem-claude-code (the product/plugin layer)

Both EverMind Claude-Code plugins are **thin cloud clients**: rule-based hooks ship raw turns to the EverMem/EverOS *backend*, which does the distillation.
- **Distill:** Stop/SessionEnd hooks parse the transcript (rule-based, NO client LLM) and POST raw turn-pairs to the cloud (`evermem` → `/api/v1/memories`; `EverMe` → `/mem/agent-memory` with `flush:true`). Backend extracts episodes/profiles.
- **Retrieve:** UserPromptSubmit hook auto-injects (push) + MCP tools (`everme_search`/`everme_context`) for pull + a **queryless SessionStart context snapshot** (server-rendered profile). EverMe renders **multi-section results** (episodic / profile / skills / cases).
- **Transcript→skill:** NEITHER plugin does it client-side — delegated to EverOS "Reflection", not exposed. **Confirms Hermes `/learn` is the template for #3.**

Net vs memagent: memagent's local, LLM-distilled **named taxonomy** (facts/decisions/procedures) in transparent markdown is *richer* than "ship raw, let the cloud guess."

**Borrowable from EverMe (moat-safe, additive to the EverOS borrows):**
- **B-EM1 — multi-section RELEVANT MEMORY rendering:** split the recalled tier by kind (lessons vs decisions vs skills) so the model distinguishes "what you learned" from "what you decided." memagent flattens to one list (`render_memory`). Effort S.
- **B-EM2 — queryless topic-start memory snapshot:** on a new topic, surface a no-query "recent + most-reinforced" snapshot, not only keyword recall. Effort S–M.
- **NOT borrowable:** the cloud backend / cross-device sync — memagent is local + single-machine by design.

---

## IMPLEMENTED (2026-06-25) — the FIX-THE-PATH + plan execution

The layering invariant is now enforced: **L1 slice ─seal→ L2 cache ─distill→ L3 memem ─recall→ L1 slice**, with distill sourcing *only* the cache and recall feeding *only* the slice.

**FIX-THE-PATH (the layering violation).** Removed the per-turn `LessonMiner` that read the live slice. `mining.py` is now helpers-only (`is_self_inflicted`, `pitfall_signature`); all distillation is cache-only via `memory.consolidate → consolidate.promote_episodes / promote_procedures` (read `read_episodes`, never the slice). `promote_episodes` absorbed the miner's self-inflicted filtering — it now collects all failing observations and titles the lesson by the **last non-self-inflicted** one, so a self-inflicted-only episode mines nothing while a real error after a self-inflicted one still mines.

**FIX-NOW.**
- **F1 (security):** `render_skill_llm` scans the recorded material (`_is_secret` + `scan_for_threats` strict) **before** the LLM call and falls back to the deterministic render on any secret/threat/no-llm/failure — no secret ever reaches a distill LLM. (The cache is already redacted on archive; `write_skill_file` scans strict + redacts + writes atomically.)
- **F2 (wording):** prov line is now "Observed from N successful run(s) this session." (was "Auto-distilled").
- **F3 (cap):** `promote_procedures` logs the dropped-at-cap count instead of dropping silently.

**BUILD (#3 — transcript → reusable skill).**
- **B1 — `render_skill_llm`:** the stubbed generalization is real — deterministic frontmatter + LLM-generalized body, scan-first, degrades to recorded steps without an LLM.
- **B2 — `/learn` + `write_skill`:** Hermes pattern — `build_learn_prompt` instructs the live agent to read THIS session from the **cache** (`recall_history`) and author a SKILL via the new foreground `write_skill` tool. The tool owns provenance (`provenance: user` — never auto-pruned) and the guarded write (validate frontmatter + scan strict + redact + atomic), so a model cannot forge AUTO provenance or smuggle an unscanned skill onto disk. `/learn` is routed in the CLI as a real turn. **Live-validated end-to-end (DeepSeek): the agent read the cache and wrote a `provenance: user` skill.**

**BORROW.**
- **R1 — file-context (`paths`):** lessons are tagged with their files on write (`remember → memory_save(paths=...)`); recall passes the files-in-play at topic-start to `retrieve(paths_context=...)`. Snapshotted at first recall and cached by goal, so it preserves the one-memem-call-per-topic property (per-turn `active_files` would otherwise bust the cache).
- **R3 — cluster-before-promote:** `promote_procedures` collapses near-duplicate-GOAL workflows via a lexical Jaccard test (`_near_dup_goal`) — no embedder/memem dependency, task-agnostic — keeping the higher-frequency one.
- **R2 — config:** `AGENT_MINE` gates consolidation end-to-end (off → skip; deterministic → recorded; llm → generalized).
- **R4 / B-EM2 (queryless snapshot) — DEFERRED:** both require **memem API changes** the current interface doesn't expose (`memory_save` has no `supersedes`; `retrieve` has no queryless mode), and the user's standing rule is to keep memem behind its interface. R4's intent is partially served by memem's own merge-band dedup (0.70–0.92 → merge). B-EM1 (multi-section render) deferred as low-value: memem-recalled items are predominantly one kind (corrective facts); skills already live in a separate tier.

**Verification:** offline suite **103/103 green** (added B1/B2/R1/R3 tests in `test_consolidate.py`; migrated the removed-miner behavior; updated stale fakes for the new optional params). Live `/learn` probe (`evals/probe_learn.py`) **PASS**.

---

## Adversarial review (2026-06-25) — verdict + fixes

A 4-lens adversarial review (invariant · security · correctness · integration) + synthesis ran over the changes.

**Confirmed safe:** the cache-only-distill invariant holds (no layer-skip); F1 sends no secret/injection to the LLM; `write_skill` cannot forge AUTO provenance, escape the skills dir, or persist an unscanned body; `promote_episodes` self-inflicted selection is correct; R1 does not bust the per-topic recall cache; R3's Jaccard does not over-collapse distinct intents; no dangling `LessonMiner` references.

**One real defect found + fixed (root-cause).** `pagetable._episodes_search_thissession` overloaded the `exclude_session` field as `only_session` — it worked only because every caller happened to pass the current session id. It did **not** leak in practice (the only caller, `history.py`, set it; the slice's `PageTable` — which left it `None` — never invokes that kind), so the reviewer's "leaks via slice.py" severity was over-stated. But it was a foot-gun one wiring change away from a cross-session leak. Fixed at the invariant level: `PageTable` now holds **one** `session_id` (cross-session reads EXCLUDE it; within-session reads filter ONLY to it), the within-session content search **fails closed** when no session is set (`not self.session_id → []`, never `only_session=None`), and both construction sites pass it. Regression test: `thissession_search_fails_closed_without_a_session`.

**Defense-in-depth (the two low F1 notes), also closed.** `render_skill_llm` now `redact_text`s and `scan_for_threats`-checks the LLM's **return** body (not just the input material) before emitting, so a secret/injection the model might produce never lives in the skill string even transiently — independent of the downstream `write_skill_file` sanitize.

**Final:** suite **103/103 green** (incl. the new fail-closed regression); live `/learn` PASS. No open findings.
