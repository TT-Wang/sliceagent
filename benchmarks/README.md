# sliceagent benchmarks — multi-turn coding scenarios

The three self-designed **long-horizon multi-turn** coding tasks from the [README benchmark §3](../README.md#benchmark). They exist to test the one thing single-turn benchmarks can't: whether a bounded-slice agent keeps **per-turn context flat** (and stays as capable) as a coding session grows, vs a transcript agent whose context accumulates.

Each task is revealed as a **fixed sequence of dependent, pre-written turns** (`prompts.json`), one at a time — deterministic and identical for every agent, so it's a fair head-to-head with no simulated-human variance. The full spec is *not knowable upfront*: later turns build on (and regress) earlier work, so **no agent can one-shot a task**, a transcript genuinely grows, and a slice does not.

## The scenarios (`multiturn_coding/`)

| scenario | turns | what it stresses |
|---|--:|---|
| `s1_longhorizon_debug` | 6 | a `tinykv` store built up over 6 dependent turns (nested transactions, tombstones, a regression fix) — history matters most here |
| `s2_taskdag_scheduler` | 10 | a dependency-DAG task scheduler grown over 10 dependent turns (topo order, cycles, waves, failure-skip runs, two regression fixes) |
| `s3_intervalset_algebra` | 10 | half-open interval algebra grown over 10 dependent turns (canonical merge, remove/split, union/intersection/difference, two regression fixes) |

Each folder has: `meta.json` (turns, step cap, notes), `prompts.json` (the ordered turns), `setup.py` (writes the starting repo into a fresh workdir), `verify.py` (the independent pass/fail check), and `reference_fix.py` (a worked solution, for reference). Everything is stdlib-only and deterministic.

## Run it

Needs `pip install "sliceagent[tui]"` and an LLM configured (`sliceagent init`, or export `LLM_API_KEY` +
`AGENT_MODEL`). The runner uses the same env-over-config provider resolution as the app, preserves one stable
task objective across turns, and seals turn-local runtime state after every prompt.

```bash
python benchmarks/run.py                          # all three
python benchmarks/run.py --scenario s1_longhorizon_debug
AGENT_REASONING=high python benchmarks/run.py     # match the published run
```

It drives sliceagent over each scenario's turns, scores the final repo with `verify.py`, and prints pass + per-call **peak input**, tokens (input/cached/output), wall, and steps — per turn and total.

## Published results (sliceagent vs OpenAI Codex, both `gpt-5.5` at `high`)

| metric | sliceagent | Codex | % of Codex |
|---|--:|--:|--:|
| solved | 3 / 3 | 3 / 3 | parity |
| peak input (median) | **16k** | 2.08M | **0.8%** |
| peak input (mean) | **17k** | 2.06M | **0.8%** |
| total tokens | **2.28M** | 26.7M | **9%** |
| cost (cache-aware) | **$1.30** | $9.43 | **14%** |
| wall (total) | **1,069s** | 1,761s | **61%** |

On `s3` (10 turns), Codex's transcript reached a **2.44M-token** single-request peak while sliceagent held **16k** — a 149× gap that widens with session length. See [`README.md`](../README.md#benchmark) for the full picture including the public [ColBench](https://huggingface.co/datasets/facebook/collaborative_agent_bench) run.

## Subagent fan-out (README §4) — `subagent_fanout.py`

A separate head-to-head where a ColBench-style human-sim tells **both** agents to fan out (one explorer subagent per module) then asks parent-only follow-ups over a 6-turn session. Both agents delegate (Codex has its own `spawn_agent`); the question is what the **orchestrator** pays to run a fleet. Each agent's own subagent tokens are counted — Codex's child threads are recovered from its `~/.codex/sessions` rollout files — for a true total-vs-total. Needs the **Codex CLI installed + logged in** in addition to a configured LLM.

```bash
ARCH_MODE=humansim AGENT_REASONING=high CODEX_EFFORT=high \
  ARCH_RUNS=3 python benchmarks/subagent_fanout.py   # 3 runs + aggregate
```

Result (N=3): sliceagent's **orchestrator peak stays ~17k while Codex's climbs to ~360k** (it re-carries the whole transcript each turn), for **~21× smaller orchestrator peak and ~3.7× lower true total** on a delegation-heavy session. See [`README.md` §4](../README.md#benchmark) for the table and caveats.
