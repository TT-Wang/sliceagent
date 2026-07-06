# Quickstart

Get from zero to your first completed task in ~5 minutes.

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

From source (development):

```bash
git clone https://github.com/TT-Wang/sliceagent && cd sliceagent
uv sync --all-extras         # or: pip install -e ".[tui]"
```

The interactive UI (rich + prompt_toolkit) ships in the `tui` extra — the curl installer and the PyPI
commands above include it; everything still works without it (plain stdout).

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

## 3. Run it

```bash
sliceagent
```

You get an inline prompt with a pinned input box. Type a task — e.g. *"add a `--json` flag to the CLI and a
test for it"* — and watch it work. The conversation stays in your normal terminal scrollback, so
**select + copy/paste and scroll work natively** on any terminal (including macOS Terminal.app).

Useful keys & commands:

- **Enter** sends · **Ctrl-J** inserts a newline · **Ctrl-C** aborts the current turn · **Ctrl-D** quits.
- `/help` lists slash commands · `/plan` shows the agent's plan · `/threads` lists topics · `/exit` quits.

UI modes (via `AGENT_TUI`): `rich` (default, inline) · `live` (always-pinned box, streams above it) ·
`off` (plain stdout, good for pipes/CI).

## 4. Reading the output

The status bar reads `model · net · policy · Σ tokens · fresh`. The **fresh** number is the one to watch:
it's the per-turn non-cached input cost, and sliceagent's whole design keeps it flat as a session grows.

## Troubleshooting

Common snags: no API key (run `sliceagent init`), `rg` (ripgrep) not installed, or an MCP server that fails to
start. `sliceagent config --list` shows every setting and its current value.
