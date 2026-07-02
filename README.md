# sliceagent

[![CI](https://github.com/TT-Wang/sliceagent/actions/workflows/ci.yml/badge.svg)](https://github.com/TT-Wang/sliceagent/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml) [![status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)

A **coding agent with a new context-engineering framework**, built for long-horizon work. Its core bet is a different memory model from every mainstream agent:

> **Don't accumulate the transcript — reconstruct a small, deterministic working state every turn.**

Mainstream agents accumulate a growing message history and **LLM-summarize it when it nears the context window** ("transcript + compaction"). sliceagent never accumulates: each turn it rebuilds a bounded **Active Memory Slice** from ground truth — the live files, the last error (verbatim), a counted action tally, recent actions, and retrieved context — and sends only that.

**Contents:** [Why](#why) · [What it can do](#what-it-can-do) · [How it works](#how-it-works--the-brain-model) · [Install](#install) · [Quickstart](#quickstart) · [Usage](#usage) · [Benchmarks](#benchmarks) · [Under the hood](#under-the-hood) · [License](#license)

## Why

- **Bounded by construction** — per-turn context stays flat regardless of session length (no grow-to-window sawtooth).
- **Faithful** — context is re-read from ground truth, not a lossy summary of the conversation.
- **Auditable** — you can print the exact, small input the model saw each turn and know *why* it decided.
- **Cheap at scale** — validated: on long/iterative tasks the slice cut tokens up to ~60–80% and wall-clock ~70% vs a transcript loop, with identical test pass rates.

This is the opposite of the field's default ("bigger windows + summarize"): **remember less, reconstruct precisely.**

## What it can do

sliceagent is an interactive terminal coding agent. Point it at a repo, describe the task in plain language, and it investigates, edits, and verifies.

- **Edit code** — create, modify, and refactor files. Edits are workspace-confined and reversible with `/undo`.
- **Run commands** — execute shell commands, launch background processes, and drive interactive terminals (REPLs, servers, `ssh`) through a sandbox — `local` by default, `docker` for full isolation.
- **Investigate** — grep and search the tree, read line-numbered context, and trace a bug from its live error. Deterministic; no embeddings, no index to stale.
- **Search the web** — fetch a page or run a keyless search when a task needs current information.
- **Delegate** — fan out large, decomposable work to subagents, each on its own bounded slice, returning a summary instead of a transcript.
- **Extend** — add tools via **MCP** servers, prompt-packs via **skills** (`SKILL.md`), or full **plugins** — all through one registry.
- **Remember across sessions** — durable lessons are distilled and auto-surfaced when relevant (via [memem](https://github.com/TT-Wang/memem)); park a topic and `/resume` it later.
- **Stay in control** — three permission modes with a hard floor on catastrophic commands; secrets are scrubbed from anything it runs or logs.

## How it works — the brain model

sliceagent's memory is organized like a brain: fast, lossy **perception** of the live world; a small **working memory** for the current task; a **hippocampus** that records what just happened; and a **neocortex** that distills durable lessons. Every turn *reconstructs* a bounded working set from these — it never replays a growing transcript.

| Region | Module | Role |
|---|---|---|
| **Sensory cortex** — live perception | `sensory_cortex.py` | Re-derives the world each turn: git state, project facts, repo map. Never stored or recalled. |
| **Prefrontal cortex** — working memory | `pfc.py` | The carried **Slice**: bounded, provenance-tagged state (findings, plan, change-set), sealed at each turn boundary. |
| **Hippocampus** — episodic memory | `hippocampus.py` | Losslessly records each turn; `recall_history` pages a specific past turn back in on demand. |
| **Neocortex** — long-term memory | `neocortex.py` | Distills successful episodes into durable cross-session lessons, auto-surfaced when relevant. |

```text
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│        PFC        │ │  Sensory Cortex   │ │    Hippocampus    │ │     Neocortex     │
│      pfc.py       │ │ sensory_cortex.py │ │  hippocampus.py   │ │   neocortex.py    │
│  working memory   │ │  live perception  │ │  episodic memory  │ │  durable lessons  │
└─────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘
          │                     │                     │                     │
          └─────────────────────┴──────────┬──────────┴─────────────────────┘
                                           │
                                           ▼
      ┌────────────────────────────────────────────────────────────────────────┐
      │                 GLOBAL WORKSPACE  —  this turn's seed                  │
      │                 seed.py  make_build_slice() / build()                  │
      │           + prompt.py  (SYSTEM_PROMPT, stable cache prefix)            │
      └────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
                         ┌───────────────────────────────────┐
                         │             LLM turn              │
                         │ tool calls accumulate within-turn │
                         └───────────────────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │               PFC updated               │
                      │ pfc.py  slice_sink() folds events back  │
                      └─────────────────────────────────────────┘

  ↻  next turn: the PFC slice carries forward —
     everything else re-derives live from disk.
```

Each turn, `seed.py` faults in exactly what the turn references — the carried PFC slice, live sensory-cortex views, and any relevant neocortex lessons — and hands the model that bounded **Seed**. The model acts; observations fold back into working memory; at the turn boundary the episode is sealed into the hippocampus; on success, the neocortex consolidates it into a durable lesson. Net effect: **per-turn context stays flat no matter how long the session runs.**

## Status

Early, but the **core bet is validated** — see the measured head-to-head benchmarks below. The production build is Python and aligns with [memem](https://github.com/TT-Wang/memem).

## Install

Straight from PyPI (any one of):

```bash
uv tool install --python 3.12 "sliceagent[tui]"     # uv (recommended — fetches Python itself)
pipx install "sliceagent[tui]"                      # pipx (needs Python ≥ 3.11 on PATH)
pip install "sliceagent[tui]"                       # plain pip (needs Python ≥ 3.11)
```

Or the one-command bootstrap (installs `uv` if needed, then sliceagent in an isolated tool env — **no prerequisites**, works even when your default Python is 3.9/3.10):

```bash
curl -fsSL https://raw.githubusercontent.com/TT-Wang/sliceagent/main/install.sh | sh
```

Footprint is light (no torch). `pip install -e .` works for a clone too. `ripgrep` is recommended (code search degrades gracefully without it). Homebrew / Docker arrive in v0.2.

## Quickstart

```bash
sliceagent init            # guided setup: provider, API key, model → ~/.sliceagent/config.toml (tests your key)
sliceagent                 # start the agent
```

`init` writes the config so the next run needs no env vars. Prefer env vars? Export **both** `LLM_API_KEY` and `AGENT_MODEL` (plus `LLM_BASE_URL` for non-OpenAI endpoints) and skip `init` — there is no default model; sliceagent never picks one for you. Discover every setting with `sliceagent config --list`.

→ Full walkthrough in **[QUICKSTART.md](QUICKSTART.md)** · **[CONTRIBUTING.md](CONTRIBUTING.md)** · **[CHANGELOG.md](CHANGELOG.md)**

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
| `/model` · `/reasoning` | switch model / reasoning effort (persists) |
| `/mode` | permission mode: **baby-sitter** (confirm each edit + command) · **teenager** (default; confirm risky ones) · **let-it-go** (auto-run all but catastrophic) |
| `/undo` | revert the last edit(s) |
| `/cwd <path>` | change the workspace root mid-session |
| `/cost` | tokens and estimated $ spent this session |
| `/skills` · `/tools` · `/mcp` · `/plugins` · `/agents` | list what's available to the agent |
| `/threads` · `/resume` | switch between, or resume, parked topics |
| `/learn <note>` | save a durable lesson yourself |
| `/plan` | draft a plan before it starts editing |
| `Ctrl-C` · `exit` | interrupt the turn · quit |

**Configuration.** `sliceagent config --list` prints every setting. Set them persistently in `~/.sliceagent/config.toml` (written by `init`), or override any one via an environment variable:

| Setting | Default | Purpose |
|---|---|---|
| `AGENT_MODEL` | *(required)* | the model id to run |
| `AGENT_POLICY` | `teenager` | permission mode |
| `AGENT_SANDBOX` | `local` | `local` or `docker` (isolated) |
| `AGENT_MAX_STEPS` | `60` | per-turn step ceiling |
| `SLICEAGENT_VAULT` | `~/.sliceagent/vault` | where episodic memory + task state persist (cross-session memory is on by default) |
| `AGENT_VERIFY_CMD` | *(unset)* | test command used as the verification oracle |

## Benchmarks

The bet — *flat per-turn cost from reconstruction, at capability parity* — is measured, not asserted. All runs use `gpt-5.5`.

**The moat: per-turn input stays flat while a transcript grows.** Head-to-head vs Kimi Code (a strong transcript-based agent) on hard multi-turn tasks:

| Scenario | sliceagent peak input | Kimi Code peak input | ratio |
|---|--:|--:|:-:|
| long-horizon debug | **7.5k** | 64.5k | **8.6×** |
| large-file bug | **7.7k** | 37.0k | 4.8× |
| multi-file refactor | **5.9k** | 28.2k | 4.8× |

Across a broader 22-scenario set: median peak input **10k (sliceagent) vs 23k (Kimi Code)** — and sliceagent's per-turn input barely moves (2.6k → 7.5k over 50 steps) while the transcript climbs 16k → 64k.

**Capability is at parity on these samples.** 22/22 vs 21/22 passed on the parity set; on 3 SWE-bench Verified instances sliceagent resolved 1/3 (scored by the official harness); TerminalBench-core standalone accuracy 0.625 (N=16).

**Same work, far fewer tokens.** On SWE-bench Lite vs a transcript agent, same instances: **26 steps / 284k tokens vs 63 steps / 838k** — ~2.4× fewer steps, ~3× fewer tokens (both resolved 0/3 — underdetermined instances, equal capability).

> Numbers are small-N and honestly reported: the consistent, reproducible signal is the **flat per-turn cost**, not a capability leap. The win shows up in **multi-turn real use** (where a transcript grows), not single-turn SWE-bench (which structurally can't show it).

## Under the hood

The core is `openai`-free (only `llm.py`/`cli.py` import the SDK), so the whole loop is testable offline with a fake LLM. Layout under `src/sliceagent/`:

- **moat:** `pfc.py` (the `Slice` dataclass, typed tiers, `slice_sink`) + `seed.py` (the reconstruction seam `make_build_slice`) + `prompt.py` (`SYSTEM_PROMPT`), `loop.py` (`run_turn`/`run_step` — stateless core over contracts).
- **contracts:** `interfaces.py` (`LLMClient`/`ToolHost`/`Retriever`/`Oracle`), `events.py` (the loop's only output path), `hooks.py` (policy seam: `OracleHook`/`PermissionHook`/`BudgetHook`).
- **engineering:** `access.py` + `scheduler.py` (resource-conflict model → safe parallel tools), `errors.py` (error classification + retry/backoff), `sandbox.py` (execution backend), `policy.py` (permission chain).
- **default impls:** `tools.py` (`LocalToolHost`), `llm.py` (`OpenAILLM`), `code_index.py` (`RipgrepCodeIndex`) + `retriever.py` (`NullRetriever`), `oracle.py`, `cli.py` (event-sink host).

The loop dispatches events; the host composes sinks (slice-updater, durable log, terminal). Ships a local `ToolHost` (workspace-confined file ops + sandboxed shell) and a ripgrep-backed `CodeIndex` (falls back to `NullRetriever` when `rg` isn't on PATH).

**Safety (P1.5).** Two independent layers:
- *Safe execution* (`tools.py` + `sandbox.py`): file ops are confined to the workspace root — path traversal out of it is rejected — and shell runs through a `Sandbox` backend. `BaseSandbox` owns output capping; backends implement `_exec()`: **`LocalSandbox`** (subprocess, cwd-confined, timeout, **secret-env scrubbing** so model-run commands can't read your API keys) and **`DockerSandbox`** (container — workspace bind-mounted same-path, network off by default, only configured env enters). Pick via `AGENT_SANDBOX`/`[sandbox]`. Code-as-action stays backend-portable via `sandbox.python_cmd`.
- *Authorization* (`policy.py`): an ordered `PolicyChain` behind the `PermissionHook`. Three modes via `AGENT_POLICY`, **all of which block catastrophic commands** (`rm -rf /`, `sudo`, `curl … | sh`, writes to `/etc`, key/cred reads, force-push): **`teenager`** (default — auto-applies file edits, asks before shell commands), **`baby-sitter`** (asks before every edit *and* command; "always" memorizes for the session), **`let-it-go`** (runs everything except the catastrophic floor). A non-interactive/headless run auto-proceeds on a confirm-mode (still catastrophic-gated); legacy `guard`/`ask`/`readonly`/`allow` still resolve. Hooks can also mutate via `prepare_messages` (inject context before the LLM call) and `transform_tool_result` (rewrite/redact output before it enters the slice).

**Subagents (`spawn_subagent`).** The slice thesis applied recursively: for large, decomposable work the model delegates a self-contained sub-task to a child agent. The child runs its OWN loop with a fresh slice in the SAME workspace, then returns **only a compact summary** — the parent's slice never sees the child's transcript, so parent context stays bounded no matter how much the child did. It's a ToolHost wrapper (`subagent.py`), so the loop is unchanged (one tool call → a summary string); depth-capped (`AGENT_SUBAGENT_DEPTH`, default 1) against runaway recursion, and the child runs under the same permission policy. Verified live: the model delegated two modules to two children that produced correct code, with the parent slice holding only the two `spawn_subagent` summaries.

**Code-as-action (`execute_code`).** Beyond one-call-per-tool, the model can write a single Python script that performs many file/shell actions and prints one short result — collapsing N tool round-trips into one turn (the strongest context reducer). The script runs **in the LocalSandbox** (cwd-confined, secret-scrubbed, timed-out) with a no-import helper API (`read_file`/`write_file`/`append_file`/`str_replace`/`list_files`/`run`); the workspace is on `sys.path` so freshly-written modules import cleanly. Only stdout returns. Files it reads/edits via the helpers are folded back into the OPEN FILES working set (paths parsed from the script), so code-as-action coheres with the slice instead of bypassing it — the agent doesn't re-read what a script already touched. It carries the same trust level as `run_command` (arbitrary execution) and is gated by the same policy (`readonly` blocks it). RPC-back-to-parent for parent-only tools (memem/MCP) is the documented upgrade.

**Extensions (MCP · skills · plugins).** sliceagent extends through one tool registry that every source feeds:
- *MCP* (`mcp_client.py`): declare servers in `[mcp_servers.*]`; their tools appear as `mcp__server__tool` (official MCP SDK, stdio).
- *Skills* (`skills.py`): `SKILL.md` prompt-packs (see above) discovered from `.sliceagent/skills`.
- *Plugins* (`plugins.py`): a directory with `plugin.toml` + an `__init__.py` exposing `register(ctx)`. Through `ctx` a plugin contributes tools/skills/MCP-servers/hooks into the **existing** seams — no privileged surface; plugin tools run through the same sandbox + policy + scheduler. Discovered from `.sliceagent/plugins` (+ `[plugins].dirs`). See [`examples/plugins/hello`](examples/plugins/hello).

**Code-discovery tier (CodeIndex).** `code_index.py` fills the RELATED CODE tier from a real repo: each turn it ripgreps the working tree for the identifiers in the task **plus the current error** (which usually names the missing symbol), ranks files by how many distinct query terms they hit, and returns line-numbered context windows — deterministic, no embeddings, no network. `repo_map()` gives a compact file→definitions skeleton for orientation (not folded into every turn, to keep context bounded). tree-sitter is the precision upgrade for definition extraction (drop-in at `_defs_in()`); v1 uses ripgrep + regex.

**Memory tier (memem) — a closed read/write loop.** `memory.py` plugs [memem](https://github.com/TT-Wang/memem) in as the cross-session `Memory` (the RELEVANT MEMORY tier). It's behind the `Memory` interface; memem indexes a curated lesson vault, *not* source code (code discovery is the separate `CodeIndex` above).
- *Read:* each task recalls relevant lessons via memem's hybrid retrieval into the slice.
- *Write (`neocortex.py`):* after a task **succeeds**, consolidation distills a durable lesson from what happened and `remember()`s it — so a future similar task recalls it. This is what makes sliceagent memory-*native*. It's an event sink, signal-dense by construction: it mines **only a validated episode** (a successful turn in which an error was hit and then cleared — no error / no success / no lesson), dedups within a session, and prints `💡 learned: …`. `AGENT_MINE=deterministic` (default — cheap, no extra LLM call) | `llm` (one-shot distillation for a crisper lesson) | `off`.

Configure via **`sliceagent.toml`** (persistent; see [`sliceagent.toml.example`](sliceagent.toml.example)) or env vars (one-off overrides). Precedence: env > project `sliceagent.toml` > user `~/.sliceagent/config.toml` > default. Keys: `AGENT_POLICY` (`baby-sitter`/`teenager`/`let-it-go`), `AGENT_MINE`, `AGENT_SUBAGENT_DEPTH`, `AGENT_MODEL`, `SLICEAGENT_VAULT` (memory location), `AGENT_VERIFY_CMD` (tests as the Oracle), `AGENT_MAX_TOKENS`, `SHOW_SLICE=1`; plus `[skills]`, `[mcp_servers]`, `[plugins]` sections.

## Architecture (build / plug / integrate)

The discipline: **own the thin differentiated core, keep the thick commodity periphery on well-known building blocks.**

- **Build (the moat):** the slice loop, the typed memory tiers + per-tier compaction, the reconstruction. Plus thin glue: permission gate, verification orchestration, subagents, resume.
- **Plug:** [memem](https://github.com/TT-Wang/memem) as the retrieval + cross-session memory engine (behind a `Retriever` interface).
- **Integrate:** LLM SDKs, tree-sitter (repo map), ripgrep (search), a container sandbox, MCP (tool breadth), a TUI lib, SWE-bench (evals).

## The differentiator, in one line

> **Deterministic reconstruction from ground truth** — vs the incumbents' **accumulate-then-LLM-summarize**.

## License

**MIT** — see [LICENSE](LICENSE). Third-party components and their licenses are listed in [NOTICE](NOTICE).

Security policy + threat model: **[SECURITY.md](SECURITY.md)**.

## Acknowledgments

sliceagent's design was informed by two excellent open-source agents: **[Hermes](https://github.com/NousResearch/hermes)** (MIT) and **[Kimi Code](https://github.com/MoonshotAI/kimi-code)**. A few peripheral utilities are ported from Hermes (see [NOTICE](NOTICE)); most of the rest are patterns we studied and reimplemented on our own terms. With thanks to their authors.
