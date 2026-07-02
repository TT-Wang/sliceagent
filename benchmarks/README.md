# sliceagent benchmarks — multi-turn coding scenarios

The three self-designed **long-horizon multi-turn** coding tasks from the [README benchmark §3](../README.md#benchmark). They exist to test the one thing single-turn benchmarks can't: whether a bounded-slice agent keeps **per-turn context flat** (and stays as capable) as a coding session grows, vs a transcript agent whose context accumulates.

Each task is revealed as a **fixed sequence of pre-written turns** (`prompts.json`) — deterministic and identical for every agent, so it's a fair head-to-head with no simulated-human variance. Later turns depend on earlier work, so a transcript grows and a slice does not.

## The scenarios (`multiturn_coding/`)

| scenario | turns | what it stresses |
|---|--:|---|
| `s1_longhorizon_debug` | 6 | a `tinykv` store built up over 6 dependent turns (nested transactions, tombstones, a regression fix) — history matters most here |
| `s2_largefile_bug` | 1 | one planted bug deep inside a ~2,700-line file (a copy of CPython 3.13's `argparse`) — large-file navigation |
| `s3_multifile_refactor` | 1 | rename + rethread a required parameter through a 16-module `eventbus` package — multi-file consistency |

Each folder has: `meta.json` (turns, step cap, notes), `prompts.json` (the ordered turns), `setup.py` (writes the starting repo into a fresh workdir), `verify.py` (the independent pass/fail check), and `reference_fix.py` (a worked solution, for reference).

## Run it

Needs `pip install "sliceagent[tui]"` and an LLM configured (`sliceagent init`, or export `LLM_API_KEY` + `AGENT_MODEL`).

```bash
python benchmarks/run.py                          # all three
python benchmarks/run.py --scenario s1_longhorizon_debug
AGENT_REASONING=high python benchmarks/run.py     # match the published run
```

It drives sliceagent over each scenario's turns, scores the final repo with `verify.py`, and prints pass + per-call **peak input**, tokens (input/cached/output), wall, and steps — per turn and total.

> **s2 note:** its bug is planted in a copy of CPython **3.13**'s `argparse`, so `setup.py` must run on a 3.13 interpreter (the agent and verifier are version-independent). Run that one under Python 3.13.

## Published results (sliceagent vs OpenAI Codex, both `gpt-5.5` at `high`)

| metric | sliceagent | Codex | % of Codex |
|---|--:|--:|--:|
| solved | 3 / 3 | 3 / 3 | parity |
| peak input (median) | **20k** | 172k | **12%** |
| peak input (mean) | **21k** | 665k | **3%** |
| total tokens | **1.0M** | 5.5M | **18%** |
| cost (cache-aware) | **$0.60** | $2.12 | **28%** |
| wall (total) | **534s** | 659s | **81%** |

On `s1` (6 turns), Codex's transcript reached a **1.65M-token** single-request peak while sliceagent held **15k** — a 112× gap that widens with session length. See [`README.md`](../README.md#benchmark) for the full picture including the public [ColBench](https://huggingface.co/datasets/facebook/collaborative_agent_bench) run.
