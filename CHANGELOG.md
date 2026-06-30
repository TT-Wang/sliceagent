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
- **One-command install** — `install.sh` (bootstraps `uv` → isolated tool install), a `Dockerfile`, and a
  README `## Install` (uv / pipx / docker). MIT `LICENSE` declared in metadata; `NOTICE` credits the Hermes
  (MIT) ports + Kimi-Code patterns; `SECURITY.md` documents the threat model + a disclosure path.
- **Three permission modes** — `baby-sitter` / `teenager` (default) / `let-it-go`, all sharing a
  catastrophic-command floor; `/mode` + `/model` two-tier selector menus; **Esc = undo**.
- **MCP spawn-security screen** (`mcp_security.py`) — refuses a shell-interpreter MCP entry that does network
  egress or writes OS-persistence surfaces, before it is spawned.
- **`read_file` window** — `offset`/`limit` + a default view cap + a `<system>` footer (a large file no
  longer floods the slice); a `glob` file-finder; `grep` `output_mode`/`--type`/context; `str_replace`
  `replace_all`; model pricing single-sourced in `model_catalog`; `with_retry` honors `Retry-After`.
- **"$ saved" cost meter** — shows dollars saved vs a full-transcript agent, re-priced live on `/model`.
- **CI** (`.github/workflows/ci.yml`, ubuntu+macOS × py3.11/3.12: install + lint + tests),
  `scripts/run_tests.sh`, a `ruff` config + `[dev]` extra, single-sourced version, and contribution
  scaffolding (`CODE_OF_CONDUCT.md`, issue/PR templates).

### Removed
- The full-screen Textual UI (`AGENT_TUI=textual`) and its `textual` dependency. The inline `rich` REPL is
  the proven default (native copy/paste/scrollback on any terminal); `AGENT_TUI=live` remains for the
  always-pinned composer, `off` for plain stdout.

### Changed
- Default UI is the inline `rich` REPL (native copy/paste on any terminal). The composer is a bordered,
  bottom-pinned box.
- Permission confirms are arrow-key selectable (Yes / No / Always) instead of typed.
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
