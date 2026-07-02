# Changelog

All notable changes to sliceagent. Format follows [Keep a Changelog](https://keepachangelog.com/);
this project aims for [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.6] — 2026-07-02

The clear config journey: one wizard, two doors, an honest /model.

### Added
- **`/config`** — manage LLM providers *inside* sliceagent: the same wizard as first-run onboarding
  (provider → model → key → live test), then the config hot-reloads and the new provider shows up
  in `/model` immediately.
- **`/model` switches providers for real** — the menu lists ONLY configured providers' models
  (saved model + suggestions, labeled by provider), and picking one switches **model + endpoint +
  key together** (the old menu changed the model string but never the endpoint). The last-picked
  provider is remembered across sessions.

### Changed
- Typed `/model <name>` stays same-endpoint (documented as such); the mismatch warning now points
  at `/config` + the `/model` menu instead of `config --use`.

## [0.1.5] — 2026-07-02

### Changed
- Wizard step order is now **provider → model → key** ("choose what you want, then prove you can") —
  the key is the last thing typed, so the live test follows it immediately.

## [0.1.4] — 2026-07-02

### Fixed
- **Wizard menus render as a proper vertical list.** The single-line selector wrapped with six long
  provider labels, and its clear-one-line redraw stacked copies of itself down the screen. New
  `_menu_select`: one option per row, width-clamped labels (wrap impossible), in-place cursor-up
  redraws, explicit `\r\n` in raw mode. PTY-tested, including the anti-stacking invariant.

## [0.1.3] — 2026-07-02

The five-door provider lineup + a wizard that feels like one.

### Added
- **Providers**: OpenRouter (hundreds of models, one key — now the first door), OpenAI,
  **Anthropic/Claude** (new — via Anthropic's OpenAI-compatible endpoint), DeepSeek, Moonshot/Kimi,
  plus custom endpoints. All five ride the single adapter — zero new dependencies.
- **OpenRouter quirks**: reasoning intent maps to OpenRouter's unified `reasoning` object (works
  WITH tools — the raw `reasoning_effort` param never could); tool-calling requests pin routing to
  hosts that honor every param (`require_parameters`) so nothing silently degrades; per-call
  `usage.cost` is parsed into the cost meter.
- `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` / `OPENROUTER_API_KEY` accepted as key fallbacks.

### Changed
- **Wizard UX**: API key shows `******` as you type (no more invisible field); provider and model
  are arrow-key menus (↑/↓ + Enter, Esc cancels) with per-provider model suggestions +
  "type another model id…". Scripted/CI runs keep the plain typed flow.

## [0.1.2] — 2026-07-02

Nothing may hang the user — three fixes from a live first-run reproduction.

### Fixed
- **Repo-map walk is hard-bounded** (`max_dirs` budget): the first slice build can no longer hang for
  minutes when the workspace root is huge (a home directory mistaken for a project, a giant monorepo).
  Output caps existed; now the walk itself is bounded — worst case ~1s, maps what it saw.
- **The OS-account home never counts as a project root**, independent of `$HOME`: a stray
  `package.json` in the real home no longer turns the entire home directory into a "project"
  (the prior guard compared against `$HOME`, which containers/sandboxes/sudo can override).
- **Ctrl-C during the slice build cancels the turn cleanly** (`· cancelled`) instead of crashing the
  REPL with a traceback — the build phase ran before the turn's interrupt handling.

## [0.1.1] — 2026-07-02

First-run onboarding, hardened by a live stranger walkthrough.

### Added
- **Auto-onboarding** — a bare interactive `sliceagent` with nothing configured now drops straight
  into the guided setup wizard (provider → key → model → live test), then continues into the session.
  No more "go run `sliceagent init`" bounce. Piped/CI runs keep the print-and-exit gate.
- **Installer handles everything** — `install.sh` now also installs ripgrep (brew when available,
  else an isolated ~2 MB static binary; no sudo) and pins `uv tool install --python 3.12`, so the
  curl path has zero prerequisites even when the default Python is 3.9/3.10 (conda base, Ubuntu 22.04).

### Changed
- README: the curl installer is the primary install path; PyPI/pip is the "you manage the Python"
  alternative. (Project renamed memagent → sliceagent the same day, before 0.1.1: PyPI's
  name-similarity check blocks "memagent".)

## [0.1.0] — 2026-07-02

First public release.

### Added
- **`sliceagent init`** — guided first-run setup (provider, API key, model); tests the key and writes
  `~/.sliceagent/config.toml` (0600). Config-persisted keys mean the next run needs no env vars.
- **`sliceagent config` / `config --list` / `config --path`** — discover every setting, its default, and
  current value. New central env-var registry (`envspec.py`) is the single source of truth.
- **`sliceagent help` / `version`** subcommands.
- **Startup config validation** — a typo'd enum (e.g. `AGENT_POLICY`) now warns instead of silently
  defaulting.
- **Always-pinned live UI** (`AGENT_TUI=live`) — a bordered input box stays at the bottom while output
  streams above it, in the normal buffer (native copy/paste).
- **Lexical topic router** (default) — routing no longer costs an LLM round-trip per follow-up;
  `AGENT_ROUTER=llm` restores the classifier. Measured identical to the LLM router on continue/resume.
- **Per-tool wall-clock timeout** (`AGENT_TOOL_TIMEOUT`, opt-in) — a stuck tool no longer hangs the turn.
- Docs: `QUICKSTART.md`, `CONTRIBUTING.md`.
- **One-command install** — `install.sh` (bootstraps `uv` → isolated tool install), a `Dockerfile`, and a
  README `## Install` (uv / pipx / docker). MIT `LICENSE` declared in metadata; `NOTICE` records
  third-party attributions; `SECURITY.md` documents the threat model + a disclosure path.
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

### Core
- Initial Python core: the slice/cache-not-log loop, typed memory tiers, reconstruction seam,
  event-sink host. Tools, skills, subagents, MCP, plugins, sandbox, permission policy,
  session/topic resume, durable cross-session memory (memem).
