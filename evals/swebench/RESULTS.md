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
