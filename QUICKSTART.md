# Quickstart

Get from zero to your first completed task in ~5 minutes. For the architecture behind the CLI, see the
canonical [Core Design](CORE-DESIGN.md): **history-bounded, task-elastic, and recoverable by construction**.

## 1. Install

One command — Linux, macOS, WSL2 (installs `uv`, its own Python, ripgrep, and sliceagent in an isolated tool env):

```bash
curl -fsSL https://raw.githubusercontent.com/TT-Wang/sliceagent/main/install.sh | sh
```

On **Windows** (native, no WSL needed):

```powershell
irm https://raw.githubusercontent.com/TT-Wang/sliceagent/main/install.ps1 | iex
```

Or install from PyPI yourself (you manage the Python — needs ≥ 3.11):

```bash
uv tool install --python 3.12 "sliceagent[tui]"   # or: pipx install "sliceagent[tui]" / pip install "sliceagent[tui]"
```

The base install includes native typed knowledge and canonical history. Use `sliceagent[tui,memory]` only when
you also want the optional Memem semantic index (its structured protocol requires Memem 2.10+).

From source (development):

```bash
git clone https://github.com/TT-Wang/sliceagent && cd sliceagent
uv sync --all-extras         # or: pip install -e ".[tui]"
```

The interactive UI (rich + prompt_toolkit) ships in the `tui` extra — the curl installer and the PyPI
commands above include it; everything still works without it (plain stdout).

Update a one-line-installer release later with `sliceagent update`. For versions older than that command,
re-run the one-line installer; it upgrades the isolated tool in place.

If you installed with uv/pipx/pip or from source, keep using that manager (`uv tool upgrade sliceagent`,
`pipx upgrade sliceagent`, `python -m pip install --upgrade "sliceagent[tui]"`, or pull +
`uv sync --all-extras`).

## 2. Set up your provider

```bash
sliceagent
```

Nothing to memorize — the **first run walks you through setup automatically**: pick a provider
(OpenRouter — hundreds of models with one key — or OpenAI, Anthropic/Claude, DeepSeek, Moonshot/Kimi, or a custom OpenAI-compatible endpoint), paste your API key (hidden),
choose a model. It **tests the key with one request**, writes `~/.sliceagent/config.toml` (mode `0600` —
it holds your key), and drops you straight into the session. Every later `sliceagent` just starts.
(Re-configure anytime — add or switch providers — with `sliceagent init`.)

Prefer environment variables? Skip `init` and export them instead:

```bash
export LLM_API_KEY="sk-…"
export LLM_BASE_URL="https://api.moonshot.cn/v1"   # omit for OpenAI
export AGENT_MODEL="kimi-k2.7-code"
```

Env vars always override the config file. See every knob with `sliceagent config --list`.

DeepSeek official-API users should migrate saved `deepseek-chat` / `deepseek-reasoner` model names to
`deepseek-v4-flash` or `deepseek-v4-pro` via `/model` or `sliceagent init`. The old names remain temporary
wire-compatible aliases during DeepSeek's retirement window, but new setup no longer offers them.

## 3. Run it

```bash
sliceagent
```

You get an inline prompt with a pinned input box. Type a task — e.g. *"add a `--json` flag to the CLI and a
test for it"* — and watch it work. The conversation stays in your normal terminal scrollback, so
**select + copy/paste and scroll work natively** on any terminal (including macOS Terminal.app).

The demo defaults to the narrow core: the parent can edit and run regular commands, while delegation creates
a fresh, one-shot, read-only explorer. Advanced surfaces are explicit opt-ins:

```bash
export AGENT_ADVANCED_AGENTS=1   # writable and named specialists; nesting uses AGENT_SUBAGENT_DEPTH
export AGENT_ADVANCED_TOOLS=1    # persistent processes and interactive terminal/PTY tools
```

The flags are independent; leave either unset to keep that surface out of the model's tool set. Delegation
depth defaults to `1`; raise `AGENT_SUBAGENT_DEPTH` only if you intentionally want advanced agents to spawn
children. Local checkpoints and immutable turn/subagent artifacts are always on and can be read through
`artifacts/`. Semantic cross-session lessons are an optional derived layer; recovery does not depend on them.
An uncertain timeout pauses only the current turn so later calls in that batch cannot overtake an operation
whose outcome is unknown. The receipt remains advisory evidence on later turns: the agent should re-observe
relevant state before relying on it, but ordinary work, topic changes, and workspace switches remain available.

Useful keys & commands:

- **Enter** sends · **Ctrl-J** inserts a newline · **Ctrl-C** aborts the current turn · **Ctrl-D** quits.
- `/help` lists slash commands · `/plan` shows the agent's plan · `/threads` lists topics · `/exit` quits.
- Start an unrelated task explicitly with `New task: ...`; the current task is parked for `/resume`.

UI modes (via `AGENT_TUI`): `rich` (default, inline) · `live` (always-pinned box, streams above it) ·
`off` (plain stdout, good for pipes/CI).

Sandbox note: `AGENT_SANDBOX=docker` is supported on POSIX and inside WSL2. On native Windows, use the
default `local` backend or launch SliceAgent inside WSL2; native-Windows Docker is rejected at startup so a
Windows host path cannot be mistaken for a valid Linux-container workspace path.

## 4. Reading the output

The status bar shows the active workspace and model together with cumulative session token use and money
saved. The **fresh** input number is the non-cached portion accumulated in this session. For a per-turn curve,
start with `AGENT_METRICS=1` and use `/cost`; with active task state held stable, that per-turn input does not
grow merely because the session is older, though it can expand when the task gains genuine constraints, files,
or evidence.

## Troubleshooting

Common snags: no API key (run `sliceagent init`), `rg` (ripgrep) not installed, or an MCP server that fails to
start. For a model whose context window is not in the catalog, set `AGENT_CONTEXT_WINDOW` to its documented
token limit to enable strict per-call capacity rejection; leaving it unset uses the explicit compatibility
mode. `sliceagent config --list` shows every setting and its current value.
