# Quickstart

Get from zero to your first completed task in ~5 minutes.

## 1. Install

```bash
git clone https://github.com/TT-Wang/memagent && cd memagent
uv sync                      # or: pip install -e .
```

memagent needs Python ≥ 3.11. The interactive UI (rich + prompt_toolkit) ships in the `tui` extra and is
installed by default; everything still works without it (plain stdout).

## 2. Set up your provider

```bash
memagent init
```

`init` walks you through it: pick a provider (Moonshot/Kimi, OpenAI, DeepSeek, or a custom OpenAI-compatible
endpoint), paste your API key (hidden), choose a model. It then **tests the key with one request** and writes
`~/.memagent/config.toml` (mode `0600` — it holds your key). The next `memagent` just works.

Prefer environment variables? Skip `init` and export them instead:

```bash
export LLM_API_KEY="sk-…"
export LLM_BASE_URL="https://api.moonshot.cn/v1"   # omit for OpenAI
export AGENT_MODEL="kimi-k2.7-code"
```

Env vars always override the config file. See every knob with `memagent config --list`.

## 3. Run it

```bash
memagent
```

You get an inline prompt with a pinned input box. Type a task — e.g. *"add a `--json` flag to the CLI and a
test for it"* — and watch it work. The conversation stays in your normal terminal scrollback, so
**select + copy/paste and scroll work natively** on any terminal (including macOS Terminal.app).

Useful keys & commands:

- **Enter** sends · **Ctrl-J** inserts a newline · **Ctrl-C** aborts the current turn · **Ctrl-D** quits.
- `/help` lists slash commands · `/plan` shows the agent's plan · `/threads` lists topics · `/exit` quits.

UI modes (via `AGENT_TUI`): `rich` (default, inline) · `live` (always-pinned box, streams above it) ·
`textual` (full-screen) · `off` (plain stdout, good for pipes/CI).

## 4. Reading the output

See **[docs/OUTPUT.md](docs/OUTPUT.md)** — what the banner, tool cards (`┊ ✓ …`), the plan checklist, the
boxed reply, and the status bar (`model · net · policy · Σ tokens · fresh`) mean. The **fresh** number is the
one to watch: it's the per-turn non-cached input cost, and memagent's whole design keeps it flat as a session
grows.

## Troubleshooting

Stuck? See **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** (no API key, `rg` missing, proxy/region,
MCP server failures, copy/paste per UI mode).
