# Changelog

All notable changes to memagent. Format follows [Keep a Changelog](https://keepachangelog.com/);
this project aims for [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **`memagent init`** — guided first-run setup (provider, API key, model); tests the key and writes
  `~/.memagent/config.toml` (0600). Config-persisted keys mean the next run needs no env vars.
- **`memagent config` / `config --list` / `config --path`** — discover every setting, its default, and
  current value. New central env-var registry (`envspec.py`) is the single source of truth.
- **`memagent help` / `version`** subcommands.
- **Startup config validation** — a typo'd enum (e.g. `AGENT_POLICY`) now warns instead of silently
  defaulting.
- **Always-pinned live UI** (`AGENT_TUI=live`) — a bordered input box stays at the bottom while output
  streams above it, in the normal buffer (native copy/paste).
- **Lexical topic router** (default) — routing no longer costs an LLM round-trip per follow-up;
  `AGENT_ROUTER=llm` restores the classifier. Measured identical to the LLM router on continue/resume.
- **Per-tool wall-clock timeout** (`AGENT_TOOL_TIMEOUT`, opt-in) — a stuck tool no longer hangs the turn.
- Docs: `QUICKSTART.md`, `docs/OUTPUT.md`, `docs/TROUBLESHOOTING.md`, `CONTRIBUTING.md`.

### Changed
- Default UI is the inline `rich` REPL (native copy/paste on any terminal); full-screen Textual is opt-in
  (`AGENT_TUI=textual`). The composer is a bordered, bottom-pinned box.
- The user's message is echoed the instant Enter is pressed — before any routing/LLM work (no input lag).
- `AGENT_PROXY=off` now forces a direct connection (previously misread as a proxy URL); the network route
  is shown on startup (`net=…`).
- Read/list tool cards show the action header only (no content dump); commands and failures show output.

### Reliability
- Hook callbacks are guarded: advisory hooks (budget/oracle/plugin) degrade gracefully on error; the
  permission hook fails **closed** (denies) on error.
- Streaming assembly survives a malformed chunk (skipped) and a mid-stream drop (salvages the partial, or
  re-rolls via retry when nothing was assembled).
- Session teardown is guarded and bounded — a stuck MCP server / index write can't freeze exit.

## [0.1.0]
- Initial Python core: the slice/cache-not-log loop (`slice.py`, `loop.py`), typed memory tiers,
  reconstruction seam, event-sink host. Tools, skills, subagents, MCP, plugins, sandbox, permission policy,
  session/topic resume, durable memory (memem). Core idea validated in the JS `prototype/`.
