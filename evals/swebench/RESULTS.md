# Head-to-head: slice vs transcript (the thesis test)

The core bet is that **Markov reconstruction (the slice) beats transcript accumulation on
efficiency at equal capability** — not that it raises pass rate. To test it we ran memagent
(slice) against **mini-swe-agent** (the SWE-agent team's canonical *transcript* baseline) on
the same SWE-bench Lite instances, same model (`gpt-5.5`), same local clones (no deps either
— a fair, isolated transcript-vs-slice comparison).

## Result (3 `pallets/flask` instances, large files)

| instance | memagent (slice) | mini-swe-agent (transcript) | token saving |
|---|---|---|---|
| flask-4992 | 8 steps / 47k tok | 19 steps / 221k tok | 4.7× |
| flask-4045 | 3 steps / 34k tok | 14 steps / 193k tok | 5.7× |
| flask-5063 | 15 steps / 203k tok | 30 steps / 424k tok | 2.1× |
| **total** | **26 steps / 284k tok** | **63 steps / 838k tok** | **2.95×** |

- **Equal capability:** both resolve **0/3** (these instances are underdetermined — e.g.
  flask-4992's issue proposes a `mode` param; the hidden test wants `text`; both agents
  faithfully implemented `mode`). So the comparison is purely about cost at the same outcome.
- **~2.4× fewer steps, ~2.95× fewer tokens** for the slice, at the same result.
- **The mechanism is the moat.** mini's per-step *prompt* tokens climb monotonically as the
  transcript accumulates — e.g. flask-4992: `1.4k → 2.3k → 5.7k → 10k → 15k → 19.6k`; the
  slice stays ~flat (~6k) by reconstructing bounded, relevant state each turn. The advantage
  **widens with task length**: on the longest task (flask-5063, 30 steps) the transcript
  saturated to ~424k tokens.

## How to reproduce
- memagent: `evals/swebench/run_agent.py` (slice loop on a local clone) → predictions → official scorer.
- transcript baseline: `mini-extra swebench-single` (Docker) or `scratch/mini_run.py` (LocalEnvironment,
  same clone), capturing per-call token usage. Both scored with `swebench.harness.run_evaluation`.

## Takeaway
At equal capability the slice is **~3× cheaper** and its per-turn cost is **flat** where the
transcript's **grows unbounded** — the predicted advantage, and it compounds on longer tasks.
Pass-rate is model/instance-bound (and these instances are underdetermined); the *efficiency*
delta is the thesis, and it holds.

---

# Trustworthy pass-rate: SWE-bench Verified (official pinned images)

The Lite-on-local-images runs above gave noisy 0/N (underspecified instances + local dep drift,
e.g. werkzeug deprecation cascades). To get a *trustworthy* resolution number we ran on **SWE-bench
Verified** (human-validated for solvable issues + fair tests) and scored with the **official pinned
images** (`--namespace swebench`), which fixes the dep-drift contamination.

memagent (slice), gpt-5.5, 3 Verified instances:

| instance | result | steps | tokens |
|---|---|---|---|
| **pallets__flask-5014** | **✅ RESOLVED** | 7 | 41k |
| pytest-dev__pytest-10356 | ✖ unresolved (converged, wrong fix) | 13 | 212k |
| pytest-dev__pytest-10051 | ✖ unresolved (hit max_steps; large repo) | 40 | 175k |

**1/3 resolved**, 0 errors, 0 empty patches — memagent's first *trustworthy* benchmark resolution,
and the win was efficient (7 steps / 41k tokens). n=3 is tiny, but this is a real, fair data point:
on the credible benchmark with the correct env, the slice agent **does** resolve real issues. Next:
larger Verified N for a leaderboard-comparable rate, and the slice-vs-transcript head-to-head on the
same Verified set.

---

# Slice redesign for reasoning-model latency (the re-derivation tax)

**Problem.** On a *reasoning* model (deepseek-v4), the slice's strength became a latency cost.
Wall-time per step tracks **completion/reasoning tokens generated**, not prompt size (prefill is
parallel/fast — confirmed by `scratch/timing_trace.py`: slice rebuild + ripgrep + tools ≈ 1% of
wall, the LLM call ≈ 99%, and the expensive steps were the ones emitting 1500–4400 reasoning tokens).
Because the slice carries **no transcript**, a reasoning model **re-derives** the situation from
scratch each turn — the dropped tokens *were* the prior reasoning a transcript agent reuses. Net:
the slice saves total/prompt tokens but can pay a per-step re-derivation tax in reasoning tokens
(and thus wall-clock). This is the genuine tension between Markov (state, not history) and reasoning
models (which amortize thinking across a transcript).

**Fixes (all preserve Markov-boundedness — bounded, reconstructed tiers, no transcript):**

1. **FINDINGS tier (anti-re-derivation).** The model records a distilled one-line *conclusion* per
   turn (root cause, confirmed fix, ruled-out hypothesis, "task done"); these accumulate into a
   bounded, deduped `WHAT YOU'VE ESTABLISHED` tier it reuses instead of re-deriving. Captured for
   **free** via an optional `note` arg injected into every tool schema (`tools.with_note`) — reasoning
   models emit empty message *content* while tool-calling, so a tool ARG, not message text, is the only
   reliable capture point. *Controlled A/B (identical mid-task slice ± findings, deepseek-v4-flash):*
   **45% lower median reasoning tokens** (69 vs 126) and it **eliminates catastrophic re-derivation
   bursts** (a 3276-ctok step without findings → ≤136 with).
2. **Tail-preserving observations (`observe()`).** Tool output was truncated head-only (`out[:200]`),
   hiding the **verdict** (`1 passed`, `FAILED`, the exception) that lives at the END of test/trace
   output. The agent literally couldn't see its test passed and re-ran forever. Now keep a little head
   + the whole tail.
3. **WORKING DIRECTORY context.** The slice never told the model its cwd, so it thrashed on
   `cd`/absolute paths hunting for files. Now states the root and "commands already run here; use
   relative paths, no cd."
4. **Runner-broken anti-loop escalation.** When a verification command repeatedly fails to *start*
   (missing runner: `command not found`, `no module named pytest`, exit 127) — distinct from a code
   assertion failing — the action-tally tier tells the agent to finish rather than keep retrying.

**End-to-end effect (synthetic bug-fix with test-runner verification, deepseek-v4-flash, n=4):**

| build | converges | steps | reasoning tok |
|---|---|---|---|
| before (head-only obs, no cwd) | 0/3 | 12 (max) | 2000–3000 |
| + working-dir context | 2/3 | 5–12 | 600–2000 |
| + tail-preserving observations | **4/4** | **5–6** | **~640** |

The changes target **waste** (re-derivation, path-thrashing, blind test re-runs), not the
irreducible fix-design reasoning — so on a genuinely hard, exploration-dominated instance with no
such pathology (pylint-4551) the result is parity-within-noise, as expected; the wins show up
wherever the pathology exists. Token/step efficiency is preserved (the `note` schema adds a modest
per-step prompt cost; reasoning tokens — the latency/cost driver on these models — drop where
re-derivation occurred).

---

# Principle-driven design audit (Markov-sufficiency · task-agnostic · extract-don't-truncate)

After fixing the design principles ("size tracks current-work complexity not history"; "general,
never task-type-specific"; "keep what the next decision needs, never truncate it"), we audited the
slice and fixed the violations:

1. **Working set: change-set-protected, not fixed-K.** The `K=4` cap was a per-slice *size* cap
   (north-star violation) with *recency* eviction (Markov violation — a 5th touch could evict a file
   the current multi-file edit still needs). Now the **change set** (every edited file, up to 8) is
   protected from eviction; only the 4 most-recent *reads* are kept (residue). A 6-file change keeps
   all 6; exploration stays tight.
2. **Discovery tracks current focus.** The retrieval query now folds in the latest finding (the
   agent's current conclusion), so on a large repo RELATED CODE surfaces what the *next* decision
   needs — not the static task terms (targets large-repo non-convergence).
3. **Generalized off Python/pytest.** Pytest-specific runner markers → POSIX-general
   `command-unavailable` (exit 127/126) + a general "repeated-failing → won't change; fix or finish";
   prompt language "test runner" → "verification command".
4. **Bug fixed** (exposed by the audit on a real run): `list_files`' directory path was being tracked
   as a working-set "file" — now excluded.

**Validation:** eval suite **9/9** (core 3/3 + stress 6/6; `repo_fix` 11→8 steps, `multi_file` 4→3
— some *improved*, none regressed); `js_fix` PASS confirms task-agnosticism; a new **non-pytest
missing-command** test fires the *general* anti-loop and converges in 5 steps; on the real
`requests-1142` instance the slice is **cleaner and leaner** (8→4 files, 0 directory noise, avg
prompt 13.8k→13.0k, reasoning 2445→2178/step, wall 303→239s).

---

# Convergence fix: over-verification (state-driven, general)

**Problem.** The agent makes the correct edit, then spends turns *re-running/re-reading checks it
already passed* — and on hard instances where it can't get a clean green from the *hidden* test it
never feels "done" → max_steps (`requests-1142`; `calc_eval`). The slice didn't represent
verification *progress*, and the stop gate ("tests pass") was often unsatisfiable.

**Fix (general · Markov · no platitude).** A state-driven `# CONVERGENCE CHECK` tier: `since_edit`
counts tool calls since the last successful edit; the tier fires **only** when a change set exists,
there's **no current error**, and `since_edit ≥ 2` (escalating to "STOP NOW" at ≥4), telling the
agent to write the final summary unless it has a *specific* new edit. Task-agnostic (counts
edits vs non-edits — no tool/language names), a pure function of state, and *suppressed whenever
something is broken* (a failing check keeps the error set), so it never cuts off active fixing.

**Result — fixes the loop AND improves perf (the over-verification was pure waste):**

| | before | after |
|---|---|---|
| calc_eval | 13 steps / 44.8k tok | **4 / 10.3k** |
| strutils_build | 8 / 20.5k | **3 / 7.1k** |
| wide_fix | 14 / 44.0k | **9 / 27.0k** |
| core suite total | 75.1k tok | **27.4k (−64%)** |
| stress suite total | 110.0k tok | **94.3k (−14%)** |
| `requests-1142` | max_steps (spun) | **converges (`end_turn`)** |

Eval suite stays **9/9** — accuracy preserved (a premature stop would fail the independent
verifiers), and multi-file cases (`multi_file`, `wide_fix`, `feature_add`) still pass (no
premature-stop regression). The nudge is soft (the model may continue for a real edit), so it
shrinks wasted steps/tokens/time without sacrificing correctness.

---

# Processing-time: non-reasoning fast mode (5 new Verified, deepseek)

memagent was slower wall-clock than mini despite far fewer tokens/steps, because on a *reasoning*
model wall-time is bound by **completion/reasoning tokens generated sequentially**, and the slice
provokes deep per-step re-derivation (1,000–2,800 reasoning tok/step → 9–27s/call). Investigated
four configs on the same 5 Verified instances; **all resolve identically (2/5: xarray-3151,
xarray-3677)** — capability is model-bound, not config-bound:

| config | steps | tokens | wall | resolve |
|---|---|---|---|---|
| deepseek-v4-flash, thinking ON (baseline) | 56 | 905k | 758s | 2/5 |
| thinking OFF, no nudge | 50 | 590k | 307s | **1/5** (over-engineers diffs) |
| thinking OFF + minimality nudge | 50 | 644k | 563s | 2/5 |
| **deepseek-chat (non-reasoning) + max_tokens** | 41–57 | 542–773k | 529–593s | 2/5 |

**Robust finding:** going non-reasoning cuts **per-call LLM latency ~10×** (9–27s → 1–2s; probe:
667 vs 23 completion tok) — the reasoning-burst bottleneck removed, with resolve preserved. The
slice's reconstructed *state* substitutes for per-step re-derivation, so a non-reasoning model is
"enough."

**Honest caveat — total wall-clock is high-variance:** the *same* config ran xarray-3151 in 4
steps/84s one run and 13 steps/299s the next, so 5-instance totals (758→529→593s) sit within the
noise — the *per-call* speedup is the reliable claim, not a precise "Nx total." Two structural
reasons the total improves less than 10×: per-instance step-count variance, and once the LLM is
fast, **clone/setup becomes the wall-clock floor** (network-bound, not LLM).

**Shipped (all LLM-agnostic, in `llm.py`):** `AGENT_REASONING=full|fast` (neutral intent mapped
per-provider: deepseek→`thinking:disabled`, o-series/gpt-5→`reasoning_effort`, else no-op);
`max_tokens` guard (default 8192, configurable) — fixes truncation→retry stalls (requests-2931
309s→143s, clean); minimality nudge (keeps fixes surgical — recovered the resolve pure thinking-off
lost). Reverted a "never rewrite a file" nudge — it caused str_replace step-bloat (net-harmful).

**Recommendation:** run fast mode = non-reasoning model (`deepseek-chat`) or `AGENT_REASONING=fast`
+ the max_tokens guard. Removes the reasoning-burst cost (the actual complaint) while keeping the
token/step advantage and 2/5 resolve. A trustworthy wall-clock delta needs multiple runs per config.
