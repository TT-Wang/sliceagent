# sliceagent

[![CI](https://github.com/TT-Wang/sliceagent/actions/workflows/ci.yml/badge.svg)](https://github.com/TT-Wang/sliceagent/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/sliceagent.svg)](https://pypi.org/project/sliceagent/) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

> **A coding agent that reconstructs a history-bounded, task-elastic working context every turn — instead of accumulating a chat transcript and summarizing it when it overflows.**

That one change is the whole product. Because context is reconstructed from active semantic state and live ground truth each turn rather than piled up:

- **History-bounded cost** — when active task state stays stable, per-turn input does not grow merely because the session is older; there is no transcript-driven grow-to-window sawtooth.
- **Task-elastic focus** — simple tasks stay lean, while real user constraints, coupled files, and unresolved evidence can expand the working slice when needed.
- **Recoverable live state** — every turn re-observes relevant workspace state and faithfully carries current failures; retired detail pages out behind stable handles instead of forcing routine transcript compaction.

The field's default is *bigger windows + summarize*. sliceagent does the opposite: **carry what remains active; archive and recover the rest.**

*Pre-1.0: on `0.x`, CLI flags, config keys, and APIs may change between releases; breaking changes are noted in the [CHANGELOG](CHANGELOG.md).*

**Contents:** [How it works](#how-it-works) · [Benchmark](#benchmark) · [Install & quickstart](#install--quickstart) · [Usage](#usage) · [License](#license) · [Acknowledgements](#acknowledgements) · [Contact](#contact)

## How it works

<p align="center">
  <img src="assets/sliceagent-core-loop.gif" width="840"
       alt="The core loop: a transcript agent re-sends its entire growing history every turn (208k to 1.66M tokens over 6 turns), while sliceagent rebuilds a history-bounded, task-elastic seed from the carried slice, live files, and lessons, then seals each turn to disk, and the hippocampus pages past turns back into future seeds on demand — peak input stayed ~12-15k in the s1 benchmark, 112x smaller by turn 6.">
</p>

sliceagent's memory is organized like a brain: fast, lossy **perception** of the live world; an elastic **working memory** for the current task; a **hippocampus** backed by always-on local artifacts; and a typed native **neocortex** for provenance-linked USER, PROJECT, and CRAFT knowledge. Every turn *reconstructs* a history-bounded working set from these — it never replays a growing transcript. With Memem's structured-index protocol (2.10+) installed, Memem is the primary semantic retrieval backend for typed L2; it is not another brain layer or a second record authority.

| Region | Role |
|---|---|
| **Sensory cortex** — live perception | Re-derives only live resources named by the active dependency closure; unrelated repo maps, history, and memory are not eagerly injected. |
| **Prefrontal cortex** — working memory | Source-linked **Active Work** for genuine unresolved, cross-turn user commitments; it is not a shadow scheduler for tools or subagents. |
| **Hippocampus** — episodic memory | Seals every turn into the always-on local artifact store; optional child-report artifacts add re-readable locators without gating direct report delivery. |
| **Neocortex** — long-term memory | Stores scoped, provenance-linked USER, PROJECT, and CRAFT records in one typed model; Memem provides primary semantic retrieval when available, with native search as failover. |

The implementation contracts are documented in [End-game context design](docs/ENDGAME-CONTEXT-DESIGN.md) and
[Memory layers design](docs/MEMORY-LAYERS-DESIGN.md).

The exact current request is admitted once into an application event ledger and one Active Work root. Context
is selected from that graph's unresolved dependency closure before physical elasticity is applied. After
execution, the sealed turn carries a canonical receipt distinguishing requested, rejected, started, settled,
and applied work. A constant-size receipt projection remains visible without constructing an autobiography
from conversational residue.

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
                      │  PFC updated · turn sealed to artifact  │
                      └─────────────────────────────────────────┘

  ↻  next turn: the active slice remains resident;
     live views re-derive, archived detail returns by handle.
```

Each turn faults in what the active task references — the carried slice, live views, selected artifacts, and applicable typed USER, PROJECT, or CRAFT knowledge — and hands the model an elastic **Seed**. The model acts; observations fold back into working memory; at the turn boundary the episode is sealed into an immutable local artifact and the next checkpoint is published. Qualifying evidence may then consolidate into a native typed lesson; optional semantic retrieval may index it but is not required. Net effect: **for stable active task state, context does not grow merely with session age; it can still expand with genuine task complexity.**

## Benchmark

On public benchmarks, sliceagent matches Codex's solve rate while using 2.5× fewer tokens and 1.3× less cost on ColBench, and up to 149× smaller peak input on long sessions.

Two questions decide whether reconstructing context every turn actually works: does it stay as **capable** as a transcript agent, and does it keep **per-turn cost history-bounded as the session grows** — sized to the current task, not the accumulated history? All four benchmarks are head-to-head vs **OpenAI Codex** on the same model (`gpt-5.5`) — the fourth adds a third question: what does it cost to *orchestrate a subagent fleet*.

### 1. In-turn capability — Terminal-Bench 2.0 (public)

A TB2.0 task is a single turn, so it's a clean test of raw within-turn ability. On the 32 tasks both agents completed cleanly:

| metric | sliceagent | OpenAI Codex |
|---|--:|--:|
| **pass rate** | **18 / 32 (56%)** | 18 / 32 (56%) |
| wins (exclusive) | 4 | 4 |
| median steps / task | **10** | 27 |

**Dead even, 4 wins each** — reconstruct-every-turn matches a state-of-the-art agent on in-turn tasks with no capability tax.

### 2. Multi-turn — ColBench (public: Meta SWEET-RL)

Collaborative coding over multiple rounds with a simulated human — the memory model *does* matter here. 20 backend tasks, both `gpt-5.5` at `high`:

| metric | sliceagent | OpenAI Codex | % of Codex |
|---|--:|--:|:--:|
| **solved** | **20 / 20** | 20 / 20 | parity |
| peak input · median | **5,191** | 13,415 | **39%** |
| peak input · mean | **5,188** | 13,424 | **39%** |
| input tokens · total | **284k** | 760k | 37% |
| ↳ served from cache | 50% | 57% | — |
| output tokens · total | 24.3k | 8.8k | 276% |
| **total tokens** | **308k** | 769k | **40%** |
| rounds · median | 3 | 3 | — |
| wall · median/turn | 26s | 26s | 100% |
| **cost** (cache-aware $) | **$0.44** | $0.55 | **80%** |

Same capability — **2.6× smaller per-turn context, 2.5× fewer tokens, 1.3× cheaper**, at parity wall.

### 3. Long-horizon multi-turn — self-designed coding scenarios

Iterative coding sessions where a transcript really piles up. Each scenario is a **fixed sequence of 6–10 dependent, pre-written turns revealed one at a time** — later turns build on (and regress) earlier work, so **no agent can one-shot them** and byte-identical turns go to both agents. Both `gpt-5.5` at `high`. Reproduce with [`benchmarks/run.py`](benchmarks/) (the scenarios are in [`benchmarks/multiturn_coding/`](benchmarks/multiturn_coding)):

| metric | sliceagent | OpenAI Codex | % of Codex |
|---|--:|--:|:--:|
| **solved** | **3 / 3** | 3 / 3 | parity |
| peak input · median | **16,330** | 2,075,987 | **0.8%** |
| peak input · mean | **16,734** | 2,056,732 | **0.8%** |
| input tokens · total | **2.21M** | 26.4M | 8% |
| ↳ served from cache | 84% | 91% | — |
| output tokens · total | **63.4k** | 355.4k | 18% |
| **total tokens** | **2.28M** | 26.7M | **9%** |
| wall · total | **1,069s** | 1,761s | **61%** |
| **cost** (cache-aware $) | **$1.30** | $9.43 | **14%** |

Per task—the transcript agent's peak input scales with the session while the slice remained within a narrow band in these runs:

| scenario | agent | solved | peak input | total tokens | wall |
|---|---|:--:|--:|--:|--:|
| **s1** long-horizon debug (6 turns) | sliceagent | ✓ | **14,769** | 499k | 257s |
| | Codex | ✓ | 1,655,714 | 5.17M | 465s |
| **s2** dependency-DAG scheduler (10 turns) | sliceagent | ✓ | **19,104** | 924k | 461s |
| | Codex | ✓ | 2,075,987 | 10.0M | 656s |
| **s3** interval-set algebra (10 turns) | sliceagent | ✓ | **16,330** | 854k | 351s |
| | Codex | ✓ | 2,438,496 | 11.5M | 641s |

Same capability — **109–149× smaller peak context, 11.7× fewer tokens, 7.3× cheaper, 1.6× faster.** On `s3`, Codex's transcript reached a **2.44M-token** single-request peak while sliceagent held **16k** — a 149× gap that **widens the longer the session runs.**

### 4. Subagent fan-out — hosting a delegation fleet ([`benchmarks/subagent_fanout.py`](benchmarks/subagent_fanout.py))

A ColBench-style human-sim (a staff engineer) **explicitly tells both agents to fan out** — one explorer subagent per module across a 6-module service — then asks four parent-only follow-ups: a 6-turn session, 2 fan-out turns + 4 follow-ups. Codex `exec` ships its **own** `spawn_agent` primitive, so **both agents genuinely delegate.** The question isn't who *can* delegate; it's what the **orchestrator** pays to run a fleet. Both `gpt-5.5` at `high`; each agent's own subagent tokens are counted — Codex's child threads recovered from its session rollouts — for a true total-vs-total. **N = 3 runs, mean [min–max]:**

| metric | sliceagent | OpenAI Codex | % of Codex |
|---|--:|--:|:--:|
| subagent spawns | 14 | 11.7 | both fan out |
| **orchestrator peak** (largest single request) | **17,362** [15.7k–20.3k] | 362,595 [332k–387k] | **4.8%** |
| delegated · own children | 341,515 | 568,027 | 60% |
| **true total** (orchestrator + children) | **610,612** [536k–656k] | 2,235,243 [2.11M–2.32M] | **27%** |

Per turn (mean of 3 runs) — the orchestrator's context is what caps how large a fleet you can keep running:

| turn | sliceagent orchestrator | OpenAI Codex orchestrator |
|---|--:|--:|
| 1 · fan-out | 37,191 | 119,397 |
| 2 · fan-out | 36,817 | 258,580 |
| 3 · follow-up | 39,447 | 282,281 |
| 4 · follow-up | 82,232 | 305,855 |
| 5 · follow-up | 41,447 | 338,507 |
| 6 · follow-up | 31,963 | 362,595 |

In these runs, sliceagent's orchestrator peak stayed roughly flat because it carried child results rather than child trajectories; its largest single request averaged **~17k** across the six-turn workload. A transcript orchestrator **re-carries the whole session every turn**, so the same delegation-heavy session reached a **~21× larger orchestrator peak and ~3.7× more total tokens** once both agents' children were counted. Parent context does not grow with child trajectory length, though it may still grow when more delegated results are genuinely relevant. Delegation is table stakes; the architectural advantage is direct child outcomes, optional re-readable artifacts, and a history-bounded parent.

*N = 3 runs, single model, one opponent, needs the Codex CLI installed. A value-recall sub-check varied wildly run-to-run (sliceagent 1–3 / 3, Codex 0–2 / 3) — it turns on a behavioral re-read choice, so it is within noise and **not** part of the claim. The defensible result is the orchestrator-context and total-token gap, which is structural and held across all three runs (orchestrator 8.8–14.3×, total 3.2–4.3×).*

<details>
<summary><b>How the cost numbers are calculated</b> (exact token counts × published rates)</summary>

Cache-aware, at `gpt-5.5` list rates — **$1.25 / 1M** fresh input, **$0.125 / 1M** cached input (a 10× discount on the prompt prefix the provider serves from its cache), **$10 / 1M** output:

```
cost = fresh_in × $1.25/M  +  cached_in × $0.125/M  +  output × $10/M
       where  fresh_in = total input − cached input
```

Applied to the summed token counts from the runs above:

**ColBench** (N = 20, all tasks summed)

| line item | sliceagent | OpenAI Codex |
|---|--:|--:|
| fresh input | 140,403 × $1.25/M = $0.176 | 330,719 × $1.25/M = $0.413 |
| cached input | 143,125 × $0.125/M = $0.018 | 429,719 × $0.125/M = $0.054 |
| output | 24,265 × $10/M = $0.243 | 8,786 × $10/M = $0.088 |
| **total** | **$0.436** | **$0.555** |

**Self-designed long-horizon** (N = 3, summed)

| line item | sliceagent | OpenAI Codex |
|---|--:|--:|
| fresh input | 347,238 × $1.25/M = $0.434 | 2,293,552 × $1.25/M = $2.867 |
| cached input | 1,866,752 × $0.125/M = $0.233 | 24,075,264 × $0.125/M = $3.009 |
| output | 63,372 × $10/M = $0.634 | 355,365 × $10/M = $3.554 |
| **total** | **$1.301** | **$9.430** |

One honest wrinkle worth naming: Codex's append-only transcript actually earns a *higher* cache-hit rate (57% vs 50% on ColBench, 91% vs 84% here) — a long stable prefix caches well. It still costs more, because its raw input volume is an order of magnitude larger; a cheaper per-token rate can't outrun many more tokens. That's the whole point of the slice: fewer tokens to bill in the first place.

*Line items are rounded to the nearest $0.001; each total is the exact sum of unrounded per-token costs, and matches the cost row in the tables above.*

</details>

> The pattern across all four is evidence for the **history-bounded-cost thesis in these scenarios**: capability held while per-turn cost tracked the current task rather than accumulated history. The same pattern extended to subagent orchestration. "Solved" is solution correctness, scored identically for both agents. ColBench is [public](https://huggingface.co/datasets/facebook/collaborative_agent_bench); the long-horizon scenarios are reproducible under [`benchmarks/`](benchmarks/).
>
> **These are early, small-scale results** — modest task counts (N = 32 / 20 / 3 tasks; §4 is one task × 3 runs), single trial per task, one model, one opponent. Treat them as a directional signal, not a settled claim. We're actively expanding to larger and more varied test sets, more trials, and more baselines, and will update these numbers as that work lands.

## Install & quickstart

One command — Linux, macOS, WSL2:

```bash
curl -fsSL https://raw.githubusercontent.com/TT-Wang/sliceagent/main/install.sh | sh
```

**Windows** — one command in PowerShell, fully native (no WSL, no admin):

```powershell
irm https://raw.githubusercontent.com/TT-Wang/sliceagent/main/install.ps1 | iex
```

(Installs `uv` + sliceagent, and Git Bash + ripgrep if you don't have them — the agent's shell commands run under Git Bash, same as other coding agents. Persistent process and interactive PTY tools are optional and disabled by default everywhere; `AGENT_ADVANCED_TOOLS=1` enables them where supported, but `terminal_open` is not available on native Windows yet. Prefer WSL2? The Linux one-liner above works there as-is.)

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

Native evidence, Active Work, history, and typed knowledge are included in the base install. The
`sliceagent[tui,memory]` extra adds optional Memem; when its structured-index protocol is available (Memem 2.10+),
SliceAgent uses it as primary L2 retrieval and falls back to native search on a whole-query failure. Older Memem
versions are reported as unavailable for this protocol rather than used as an unscoped recall tail. Typed record
truth and `@sliceagent/` durability remain available without it.

If `pip` refuses with `Requires-Python >=3.11`: `conda create -n sliceagent python=3.12 -y && conda activate sliceagent`, then pip install. Prefer env vars over the wizard? Export **both** `LLM_API_KEY` and `AGENT_MODEL` (plus `LLM_BASE_URL` for non-OpenAI endpoints). `ripgrep` is recommended (code search degrades gracefully without it).
</details>

### Updating

For installs created by the one-line installer:

```bash
sliceagent update
```

The command updates only when it can positively identify SliceAgent's isolated `uv` tool environment;
it never replaces an editable checkout or guesses at a manager-owned environment. On Windows, exit
SliceAgent and follow the external process guidance it prints. If your installed version predates
`sliceagent update`, re-run the one-line installer; the installer is deliberately safe to re-run.

Self-managed installs stay self-managed: `uv tool upgrade sliceagent`, `pipx upgrade sliceagent`, or
`python -m pip install --upgrade "sliceagent[tui]"` (or `sliceagent[tui,memory]` when using Memem). Source checkouts should pull first, then run
`uv sync --all-extras`.

Footprint is light (no torch), and `pip install -e .` works for a clone. Distribution is currently through
the one-line installers and PyPI; no Homebrew formula or prebuilt SliceAgent container image is advertised.
Docker is available separately as the optional POSIX/WSL2 command-sandbox backend. → Full walkthrough in
**[QUICKSTART.md](QUICKSTART.md)**.

## Usage

Run `sliceagent` in your project and type what you want in plain language. It rebuilds its working context,
investigates, edits, and can run your tests to verify. Ordinary requested work proceeds without permission
prompts; only a narrow high-confidence catastrophic-command safeguard can refuse execution. A turn looks like:

```text
❯ why does retry_with_backoff drop the last attempt? fix it

  │ 1 search · 2 read  retry.py, tests/test_retry.py
  │ write retry.py
  │ plan 2/3 · add a regression test
  ◌ 2/3 add a regression test · Running pytest -q · 00:12
  │ run pytest -q
  │   38 passed
  │ agent note · The focused regression now passes.
  ─ assistant ───────────────────────────────────────────────

    The loop exits on `attempt == max` before the final
    sleep+retry, so the last attempt never runs. Changed the
    bound to `attempt <= max` and added a regression test.

  ───────────────────────────────────────────────────────────
  ✓ turn saved · plan 3/3 · 2 passes · 4 read · 1 edit · 1 cmd · 00:18
```

Attach a file or path to your message with `@`: `@src/errors.py explain the backoff`.

**In-session commands** (type `/help` for the full list):

| Command | What it does |
|---|---|
| `/config` · `/model` | add/switch providers · switch model / reasoning effort (persists) |
| `Esc` | revert the last edit(s) |
| `/cwd [path]` | show the workspace; with a path, atomically switch it without restarting the UI or model client |
| `/cost` | tokens and estimated $ spent this session |
| `/skills` · `/tools` · `/mcp` · `/plugins` · `/agents` | list what's available to the agent |
| `/threads` · `/resume` | switch between, or resume, parked topics |
| `/learn <note>` | save a durable lesson yourself |
| `/plan` | show the current task plan |
| `/update` | show the safe process-boundary update command |
| `Ctrl-C` · `exit` | interrupt the turn · quit |

Public `/` palette: `/config` · `/model` · `/cwd` · `/learn` · `/plan` · `/cost` · `/update` · `/threads` · `/resume` · `/plugins` · `/mcp` · `/skills` · `/tools` · `/agents` · `/help` · `/exit`.
The typed compatibility aliases `/reasoning`, `/undo`, and `/switch` remain accepted, but stay out of the
palette because `/model`, Esc, and `/resume` are their clearer public spellings.

File mentions accept exact workspace paths such as `@app/jobs/[id]/page.tsx`; quote paths containing spaces,
for example `@"docs/my guide.md"`.

Prefix an unrelated request with `New task:` to start it with fresh task state while parking the current task
for `/resume`. Ambiguous follow-ups deliberately continue the active task so context is never discarded on a
guess.

It can edit code in the primary workspace and grounded focus roots (reversible with `/undo`), run regular shell commands through a sandbox (`local` by default, `docker` for container isolation on POSIX/WSL2), search the tree and the web, and delegate decomposable research to a fresh one-shot read-only explorer (each child gets its own history-bounded, task-elastic slice). Native Windows uses the local backend; run SliceAgent inside WSL2 if you want the Docker backend. That narrow surface is the demo default. Set `AGENT_ADVANCED_AGENTS=1` to expose writable and named specialist delegation (and nested delegation when `AGENT_SUBAGENT_DEPTH` is raised above its default of `1`), or `AGENT_ADVANCED_TOOLS=1` to expose persistent process and interactive terminal tools. The flags are independent. Ordinary work runs directly from the user's request; the host retains only a narrow safeguard against high-confidence catastrophic shell commands. Secrets are scrubbed from anything persisted or logged.

Every clean or interrupted agent task turn is sealed into the always-on local artifact/checkpoint path.
Subagent reports return directly to the parent in launch order; when optional artifact persistence succeeds,
they also gain durable re-readable locators. The model reads exact evidence, project history, Active Work, and
typed knowledge through the permanent read-only `@sliceagent/` namespace in every workspace.
`@sliceagent/memory/status.md` is the bounded general summary; raw host-counted inventory lives separately at
`@sliceagent/memory/diagnostics.md` for explicit diagnostic requests. Compatibility counts are not layer sizes
or an L2 backlog. Memem is primary semantic retrieval when enabled; task recovery and typed record truth do
not depend on it.

Automatic knowledge push is lifecycle- and revision-aware: stale or dependency-drifted PROJECT observations and
resolved diagnostic reports remain explicitly searchable, but do not reappear as if they described the current
workspace. USER preferences and reusable CRAFT procedures do not decay merely because time passed.

If a timeout or disconnect leaves an operation's side effects uncertain, SliceAgent records that uncertainty
durably and shows it to the model as grounding on later turns. It can re-observe relevant live state before
making claims, but the receipt does not block ordinary work, task switching, undo, or workspace navigation.
Ambiguous recovery journals still stop startup before plugins or MCP processes run because that boundary
protects the integrity of the durable local store rather than interpreting user intent.

**Configuration.** `sliceagent config --list` prints every setting. Set them persistently in `~/.sliceagent/config.toml` (written by `init`), or override any one via an environment variable:

| Setting | Default | Purpose |
|---|---|---|
| `AGENT_MODEL` | *(required)* | the model id to run |
| `AGENT_SANDBOX` | `local` | `local`, or `docker` on POSIX/WSL2 (native Windows: use `local` or run under WSL2) |
| `AGENT_MAX_STEPS` | `60` | per-turn step ceiling |
| `AGENT_CONTEXT_WINDOW` | *(catalog or unset)* | explicit provider window for strict per-call preflight; unknown models otherwise use compatibility mode |
| `AGENT_ADVANCED_AGENTS` | *(off)* | enable writable and named specialists; unlock the nested surface subject to the depth ceiling |
| `AGENT_SUBAGENT_DEPTH` | `1` | delegation depth ceiling; raise it to permit nested advanced agents |
| `AGENT_DELEGATION_TIMEOUT` | `900` | hard ceiling in seconds for one child-agent wave; raise for unusually slow providers |
| `AGENT_EXPLORER_REASONING` | `staged` | fast evidence navigation followed by one full, tool-free final synthesis (`fast`/`full` are single-stage overrides) |
| `AGENT_EXPLORER_NAV_STEPS` | `6` | fast-navigation model-step ceiling for staged explorers; one separate synthesis step stays reserved |
| `LLM_HARD_TIMEOUT_SEC` | *(completion-budget derived)* | absolute per-call watchdog; provider-agnostic default allows the configured completion cap at a conservative generation rate (minimum 180s) |
| `LLM_STREAM_CLOSE_GRACE_SEC` | `2` | bounded wait to prove a cancelled/timed-out SSE request physically closed before any retry |
| `LLM_PROVIDER_MAX_INFLIGHT` | `4` | process-wide physical request cap per provider account; indeterminate calls hold their slot until the transport closes |
| `AGENT_ADVANCED_TOOLS` | *(off)* | enable persistent process and interactive terminal tools |
| `SLICEAGENT_CACHE_DIR` | `~/.sliceagent` | always-on local checkpoints, immutable artifacts, and recovery journals |
| `SLICEAGENT_VAULT` | `~/.sliceagent/vault` | legacy episodic/task/roster compatibility records (not canonical typed L2) |
| `AGENT_VERIFY_CMD` | *(unset)* | test command used as the verification oracle |

DeepSeek official-API configurations should move from the retiring `deepseek-chat` / `deepseek-reasoner`
aliases to `deepseek-v4-flash` or `deepseek-v4-pro`. SliceAgent keeps the old names temporarily compatible,
but new provider setup and model suggestions use the V4 names.

## License

**MIT** — see [LICENSE](LICENSE). Third-party components and their licenses are listed in [NOTICE](NOTICE). Security policy + threat model: **[SECURITY.md](SECURITY.md)**.

## Acknowledgements

sliceagent's design was informed by two excellent open-source agents: **[Hermes Agent](https://github.com/NousResearch/hermes-agent)** (MIT) and **[Kimi Code](https://github.com/MoonshotAI/kimi-code)**. A few peripheral utilities are ported from Hermes (see [NOTICE](NOTICE)); most of the rest are patterns we studied and reimplemented on our own terms. [memem](https://github.com/TT-Wang/memem) provides primary semantic retrieval for SliceAgent-owned typed L2 knowledge when the memory extra is installed. Its value/primary-index/cue representation was also informed by Microsoft [Memora](https://github.com/microsoft/Memora) (MIT). Neither is another brain layer, and local artifacts and recovery do not depend on them. With thanks to their authors.

## Contact

Questions, feedback, or ideas — open an [issue](https://github.com/TT-Wang/sliceagent/issues) or reach out: **[tongtao.wang@gmail.com](mailto:tongtao.wang@gmail.com)**. (Security reports: please follow [SECURITY.md](SECURITY.md) instead.)
