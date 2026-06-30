# Configuration reference

_Auto-generated from `src/memagent/envspec.py` — do not edit by hand (`python scripts/gen_config_reference.py`)._

memagent reads **44** environment variables across **6** groups; every value is validated at startup (a misspelled enum warns instead of silently defaulting). Run `memagent config --list` to see the resolved value of each on your machine. Secrets (🔒) are read from the environment / config and never printed.

## agent

| variable | default | description |
|---|---|---|
| `AGENT_ALLOW_PLUGINS` | — | Set truthy to load project/user plugins. |
| `AGENT_AUTO_APPROVE` | — | Comma-separated globs of pre-approved safe commands (skip prompt). |
| `AGENT_BACKGROUND_REVIEW` | — | Set truthy to run an off-thread reviewer that consolidates lessons after each turn. |
| `AGENT_COMPLETION_TOKENS` | `8192` | Per-REQUEST completion cap (max output tokens); distinct from the AGENT_MAX_TOKENS turn budget. |
| `AGENT_EXPLORER_REASONING` | `fast` | Reasoning effort for read-only explorer children. |
| `AGENT_MAX_STEPS` | `60` | Per-turn step ceiling (runaway backstop); raise for deep analysis. |
| `AGENT_MAX_TOKENS` | — | Per-session token budget (parks the turn when exhausted). |
| `AGENT_MINE` | `deterministic` | Lesson-mining mode for end-of-session consolidation. |
| `AGENT_MODEL` | `gpt-5.5` | LLM model id to drive the agent. |
| `AGENT_MODEL_FALLBACK` | — | Larger-context model to switch to ONCE if the context overflows even after compaction (secondary net; the bounded slice is the primary). |
| `AGENT_POLICY` | `teenager` | Permission mode: baby-sitter (confirm all) \| teenager (auto edits, confirm commands) \| let-it-go (auto, blocks catastrophic). All block catastrophic moves. _(choices: baby-sitter, teenager, let-it-go)_ _(aliases: guard, allow, readonly, ask, babysitter, teen, letgo, letitgo, yolo, baby)_ |
| `AGENT_PROVIDER` | — | Default provider id to use from the config's [providers.<id>] tables (overrides [agent].default_provider). |
| `AGENT_REASONING` | `full` | Reasoning effort: full=provider default, fast=minimal, high/max=more. _(choices: full, fast, high, max)_ |
| `AGENT_ROOT` | — | Workspace root override (defaults to the current directory). |
| `AGENT_ROUTER` | `lexical` | Topic router: lexical (instant, no LLM) or llm (classifier round-trip). _(choices: lexical, llm)_ |
| `AGENT_SANDBOX` | `local` | Tool sandbox backend. _(choices: local, docker)_ |
| `AGENT_SELFCHECK_MAX` | `3` | Max grounded done-gate verification rounds before accepting 'done'. |
| `AGENT_SUBAGENT_DEPTH` | `1` | Max delegation depth for spawn_subagent/spawn_explore (0=off). |
| `AGENT_THINKING` | — | Set to 'off' to disable reasoning (alias for AGENT_REASONING=fast). |
| `AGENT_TOOL_TIMEOUT` | — | Per-tool wall-clock deadline in seconds (0/unset = off). |
| `AGENT_VERIFY_CMD` | — | Oracle verify command run after a turn (e.g. 'pytest -q'). |
| `AGENT_WEB` | `1` | Enable the web tools (fetch_url + web_search, DuckDuckGo, no key); set 0/off to disable network egress from the agent. |

## debug

| variable | default | description |
|---|---|---|
| `AGENT_EXPERIMENTAL_ALL` | — | Master switch: set truthy to enable ALL experimental flags (per-flag AGENT_EXPERIMENTAL_<ID> overrides). |
| `MEMAGENT_DEBUG_TRACE` | — | Set truthy to print tracebacks for parked/hook errors. |
| `MEMAGENT_NO_CLOSURE` | — | Debug flag: disable the turn closeout call. |
| `MEMAGENT_PROMPT_FILE` | — | A/B experiment seam: path to a full SYSTEM_PROMPT template (must keep the {{MEMORY_MODEL}} marker) to override the prompt for a measurement run (evals/prompt_ab). Unset → the production prompt. |

## memory

| variable | default | description |
|---|---|---|
| `MEMAGENT_CACHE_DIR` | — | Directory for the episodic cache / durable log. |
| `MEMAGENT_SKILLS_DIR` | — | Extra directory to discover skills from. |
| `MEMAGENT_VAULT` | — | memagent's STATE vault (episodic cache + task-state records). |
| `MEMEM_VAULT` | — | memem's lesson vault (markdown long-term memories), if memem is installed. |

## monitor

| variable | default | description |
|---|---|---|
| `AGENT_METRICS` | — | Set truthy to print per-turn fresh-token (moat) metrics at exit. |
| `AGENT_MONITOR` | — | Set truthy to enable the live monitor server. |
| `AGENT_MONITOR_PORT` | — | Port for the monitor server. |
| `MEMAGENT_MONITOR_DIR` | — | Directory the monitor writes slice snapshots to. |

## provider

| variable | default | description |
|---|---|---|
| `AGENT_PROXY` | — | Proxy override: 'off', a URL, or unset for auto (foreign→ClashX, CN→direct). |
| `LLM_API_KEY` 🔒 | — | API key for the LLM provider (REQUIRED). |
| `LLM_BASE_URL` | — | OpenAI-compatible endpoint (e.g. https://api.moonshot.cn/v1). |
| `LLM_TIMEOUT` | — | Per-request timeout in seconds. |
| `LLM_TIMEOUT_SEC` | — | Alias for LLM_TIMEOUT. |
| `MOONSHOT_API_KEY` 🔒 | — | Legacy alias for LLM_API_KEY (Moonshot). |
| `OPENAI_API_KEY` 🔒 | — | Legacy alias for LLM_API_KEY. |
| `OPENAI_BASE_URL` | — | Legacy alias for LLM_BASE_URL. |

## ui

| variable | default | description |
|---|---|---|
| `AGENT_TUI` | `rich` | UI mode: rich (default inline), live (pinned box), off (plain). _(choices: rich, live, off)_ _(aliases: 1, on, true, yes, 0, false, no)_ |
| `SHOW_SLICE` | — | Set truthy to print the rebuilt slice each turn (debug view). |
