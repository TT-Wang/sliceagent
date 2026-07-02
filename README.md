# sliceagent

[![CI](https://github.com/TT-Wang/sliceagent/actions/workflows/ci.yml/badge.svg)](https://github.com/TT-Wang/sliceagent/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/sliceagent.svg)](https://pypi.org/project/sliceagent/) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

> **A coding agent that reconstructs a small, exact working context every turn — instead of accumulating a chat transcript and summarizing it when it overflows.**

That one change is the whole product. Because context is rebuilt from ground truth each turn rather than piled up:

- **Cheap at scale** — per-turn input stays flat no matter how long the session runs; no grow-to-window sawtooth, so tokens (and cost) don't balloon on long tasks.
- **Less wall-clock** — a small, stable prompt each turn means less to send and less to reason over; long, iterative work finishes faster.
- **No context rot, no compaction** — every turn re-reads the live files and the last error verbatim, so nothing drifts into a lossy summary and there is no history to compact.

The field's default is *bigger windows + summarize*. sliceagent does the opposite: **remember less, reconstruct precisely.**

*Pre-1.0: on `0.x`, CLI flags, config keys, and APIs may change between releases; breaking changes are noted in the [CHANGELOG](CHANGELOG.md).*

**Contents:** [How it works](#how-it-works) · [Benchmark](#benchmark) · [Install & quickstart](#install--quickstart) · [Usage](#usage) · [License](#license) · [Acknowledgements](#acknowledgements) · [Contact](#contact)

## How it works

sliceagent's memory is organized like a brain: fast, lossy **perception** of the live world; a small **working memory** for the current task; a **hippocampus** that records what just happened; and a **neocortex** that distills durable lessons. Every turn *reconstructs* a bounded working set from these — it never replays a growing transcript.

| Region | Role |
|---|---|
| **Sensory cortex** — live perception | Re-derives the world each turn: git state, project facts, repo map. Never stored or recalled. |
| **Prefrontal cortex** — working memory | The carried **Slice**: bounded, provenance-tagged state (findings, plan, change-set), sealed at each turn boundary. |
| **Hippocampus** — episodic memory | Losslessly records each turn; pages a specific past turn back in on demand. |
| **Neocortex** — long-term memory | Distills successful episodes into durable cross-session lessons, auto-surfaced when relevant. |

```text
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│        PFC        │ │  Sensory Cortex   │ │    Hippocampus    │ │     Neocortex     │
│  working memory   │ │  live perception  │ │  episodic memory  │ │  durable lessons  │
└─────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘
          │                     │                     │                     │
          └─────────────────────┴──────────┬──────────┴─────────────────────┘
                                           ▼
      ┌────────────────────────────────────────────────────────────────────────┐
      │            GLOBAL WORKSPACE  —  this turn's reconstructed seed           │
      │        (carried slice + live views + relevant lessons + prompt)         │
      └────────────────────────────────────────────────────────────────────────┘
                                           ▼
                         ┌───────────────────────────────────┐
                         │             LLM turn              │
                         │ tool calls accumulate within-turn │
                         └───────────────────────────────────┘
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │   PFC updated · turn sealed to memory   │
                      └─────────────────────────────────────────┘

  ↻  next turn: only the PFC slice carries forward —
     everything else re-derives live from disk.
```

Each turn faults in exactly what the turn references — the carried slice, live views, and any relevant lessons — and hands the model that bounded **Seed**. The model acts; observations fold back into working memory; at the turn boundary the episode is sealed into the hippocampus; on success the neocortex distills a durable lesson. Net effect: **per-turn context stays flat no matter how long the session runs.**

## Benchmark

Two questions decide whether reconstructing context every turn actually works: does it stay as **capable** as a transcript agent, and does it keep **per-turn cost flat** as a session grows? All three benchmarks are head-to-head vs **OpenAI Codex** on the same model (`gpt-5.5`) at matched reasoning.

### 1. In-turn capability — Terminal-Bench 2.0 (public)

A TB2.0 task is a single turn, so it's a clean test of raw within-turn ability. On the 32 tasks both agents completed cleanly:

| | pass rate | wins |
|---|--:|:--:|
| **sliceagent** | **18 / 32** | 4 |
| OpenAI Codex | 18 / 32 | 4 |

**Dead even** — reconstruct-every-turn matches a state-of-the-art agent on in-turn tasks, no capability tax.

### 2. Multi-turn — ColBench (public: Meta SWEET-RL)

Collaborative coding over multiple rounds with a simulated human — the memory model *does* matter here. 20 backend tasks, both `gpt-5.5` at `high`:

| | solved | peak input | total tokens | cost |
|---|--:|--:|--:|--:|
| **sliceagent** | **20 / 20** | **5.2k** | **308k** | **$0.44** |
| OpenAI Codex | 20 / 20 | 13.4k | 769k | $0.55 |

Same capability — and sliceagent's per-turn context is **2.6× smaller**, at **2.5× fewer tokens** and **1.3× cheaper**.

### 3. Long-horizon multi-turn — self-designed coding scenarios

Iterative coding sessions (a 6-turn debugging build, a large-file bug, a multi-file refactor) — where a transcript really piles up. Both `gpt-5.5` at `high`:

| | solved | peak input (median) | total tokens | cost | wall |
|---|--:|--:|--:|--:|--:|
| **sliceagent** | **3 / 3** | **20k** | **1.0M** | **$0.60** | **534s** |
| OpenAI Codex | 3 / 3 | 172k | 5.5M | $2.12 | 659s |

Same capability — **8–32× smaller per-turn context, 5.5× fewer tokens, 3.6× cheaper, 1.2× faster.** On the long-horizon task, Codex's transcript reached a **1.65M-token** single-request peak while sliceagent held **15k** — a 112× gap that **widens the longer the session runs.**

> The pattern across all three: **capability holds, and the cost gap grows with session length** — exactly the flat-per-turn-cost thesis. Numbers are small-N, single-trial, same-model, from our harness (reproducible under [`evals/`](evals/)); "solved" is solution correctness, scored identically for both agents.

## Install & quickstart

One command — Linux, macOS, WSL2:

```bash
curl -fsSL https://raw.githubusercontent.com/TT-Wang/sliceagent/main/install.sh | sh
```

**The installer handles everything**: `uv`, its own Python 3.12, ripgrep, and sliceagent — in an isolated tool env, no sudo, no prerequisites, no conflict with any Python you already have (conda base at 3.10? Rosetta-Intel conda on an M-series Mac? Doesn't matter). Then just:

```bash
sliceagent          # first run drops you straight into guided setup, then start chatting
```

Setup happens once, **in-process**: pick a provider, paste your API key (shown as `******`, live-tested), and it writes `~/.sliceagent/config.toml` (0600) so every later run just starts. Re-configure anytime with `/config` in-session or `sliceagent init`. There is **no default model** — sliceagent never picks one for you.

<details>
<summary><b>Alternative: install from PyPI yourself</b> (you manage the Python — needs ≥ 3.11)</summary>

```bash
uv tool install --python 3.12 "sliceagent[tui]"     # uv — fetches Python itself
pipx install "sliceagent[tui]"                      # pipx
pip install "sliceagent[tui]"                       # plain pip (use a venv)
```

If `pip` refuses with `Requires-Python >=3.11`: `conda create -n sliceagent python=3.12 -y && conda activate sliceagent`, then pip install. Prefer env vars over the wizard? Export **both** `LLM_API_KEY` and `AGENT_MODEL` (plus `LLM_BASE_URL` for non-OpenAI endpoints). `ripgrep` is recommended (code search degrades gracefully without it).
</details>

Footprint is light (no torch). `pip install -e .` works for a clone. Homebrew / Docker arrive in v0.2. → Full walkthrough in **[QUICKSTART.md](QUICKSTART.md)**.

## Usage

Run `sliceagent` in your project and type what you want in plain language. It rebuilds its working context, investigates, edits (auto-applied or confirmed, per your mode), and can run your tests to verify. A turn looks like:

```text
❯ why does retry_with_backoff drop the last attempt? fix it

  🔍 grep "retry_with_backoff"   📖 read errors.py:40-72   ✎ edit errors.py
  ┌─ assistant ─────────────────────────────────────────────┐
  │ The loop exits on `attempt == max` before the final      │
  │ sleep+retry, so the last attempt never runs. Changed the  │
  │ bound to `attempt <= max` and added a regression test.    │
  └──────────────────────────────────────────────────────────┘
  ✓ done · 4 steps · 6.1k tokens
```

Attach a file or path to your message with `@`: `@src/errors.py explain the backoff`.

**In-session commands** (type `/help` for the full list):

| Command | What it does |
|---|---|
| `/config` · `/model` · `/reasoning` | add/switch providers · switch model / reasoning effort (persists) |
| `/mode` | permission mode: **baby-sitter** (confirm each edit + command) · **teenager** (default; confirm risky ones) · **let-it-go** (auto-run all but catastrophic) |
| `/undo` | revert the last edit(s) |
| `/cwd <path>` | change the workspace root mid-session |
| `/cost` | tokens and estimated $ spent this session |
| `/skills` · `/tools` · `/mcp` · `/plugins` · `/agents` | list what's available to the agent |
| `/threads` · `/resume` | switch between, or resume, parked topics |
| `/learn <note>` | save a durable lesson yourself |
| `/plan` | draft a plan before it starts editing |
| `Ctrl-C` · `exit` | interrupt the turn · quit |

It can edit code (workspace-confined, reversible with `/undo`), run shell commands and interactive processes through a sandbox (`local` by default, `docker` for full isolation), search the tree and the web, delegate decomposable work to subagents (each on its own bounded slice), and remember lessons across sessions. Three permission modes gate it, all with a hard floor on catastrophic commands; secrets are scrubbed from anything it runs or logs.

**Configuration.** `sliceagent config --list` prints every setting. Set them persistently in `~/.sliceagent/config.toml` (written by `init`), or override any one via an environment variable:

| Setting | Default | Purpose |
|---|---|---|
| `AGENT_MODEL` | *(required)* | the model id to run |
| `AGENT_POLICY` | `teenager` | permission mode |
| `AGENT_SANDBOX` | `local` | `local` or `docker` (isolated) |
| `AGENT_MAX_STEPS` | `60` | per-turn step ceiling |
| `SLICEAGENT_VAULT` | `~/.sliceagent/vault` | where episodic memory + task state persist (cross-session memory is on by default) |
| `AGENT_VERIFY_CMD` | *(unset)* | test command used as the verification oracle |

## License

**MIT** — see [LICENSE](LICENSE). Third-party components and their licenses are listed in [NOTICE](NOTICE). Security policy + threat model: **[SECURITY.md](SECURITY.md)**.

## Acknowledgements

sliceagent's design was informed by two excellent open-source agents: **[Hermes](https://github.com/NousResearch/hermes)** (MIT) and **[Kimi Code](https://github.com/MoonshotAI/kimi-code)**. A few peripheral utilities are ported from Hermes (see [NOTICE](NOTICE)); most of the rest are patterns we studied and reimplemented on our own terms. Cross-session memory is powered by [memem](https://github.com/TT-Wang/memem). With thanks to their authors.

## Contact

Questions, feedback, or ideas — open an [issue](https://github.com/TT-Wang/sliceagent/issues) or reach out: **[tongtao.wang@gmail.com](mailto:tongtao.wang@gmail.com)**. (Security reports: please follow [SECURITY.md](SECURITY.md) instead.)
