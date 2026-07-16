# Configuration reference

_Auto-generated from `src/sliceagent/envspec.py` — do not edit by hand (`python scripts/gen_config_reference.py`)._

sliceagent reads **58** environment variables across **6** groups; every value is validated at startup (a misspelled enum warns instead of silently defaulting). Run `sliceagent config --list` to see the resolved value of each on your machine. Secrets (🔒) are read from the environment / config and never printed.

## agent

| variable | default | description |
|---|---|---|
| `AGENT_ADVANCED_AGENTS` | — | Enable writable/nested/named specialist delegation; default core mode exposes one-shot read-only explorers only. |
| `AGENT_ADVANCED_TOOLS` | — | Expose persistent process and interactive terminal tools; off by default in the demo kernel. |
| `AGENT_ALLOW_PLUGINS` | — | Set truthy to load project/user plugins. |
| `AGENT_BACKGROUND_REVIEW` | — | Set truthy to run an off-thread reviewer that consolidates lessons after each turn. |
| `AGENT_COMPLETION_TOKENS` | `8192` | Per-REQUEST completion cap (max output tokens); distinct from the AGENT_MAX_TOKENS turn budget. |
| `AGENT_CONTEXT_WINDOW` | — | Provider context window used for strict per-call capacity preflight when the model catalog cannot supply one (0/unset = explicit compatibility mode). |
| `AGENT_DELEGATION_TIMEOUT` | `900` | Hard ceiling for a child-agent wave in seconds; invalid/non-positive values use 900. |
| `AGENT_EXPLORER_NAV_STEPS` | `6` | Fast-navigation model-step ceiling for staged explorers. Values are clamped to 1..(the child max minus the reserved synthesis step). |
| `AGENT_EXPLORER_REASONING` | `staged` | Explorer profile: staged uses fast evidence navigation plus one full tool-free synthesis; fast/full/high/max keep a single-stage override. _(choices: staged, fast, full, high, max)_ |
| `AGENT_MAX_STEPS` | `60` | Per-turn step ceiling (runaway backstop); raise for deep analysis. |
| `AGENT_MAX_TOKENS` | — | Per-turn task token budget, including delegated child usage (parks when exhausted). |
| `AGENT_MINE` | `deterministic` | Lesson-mining mode for end-of-session consolidation. |
| `AGENT_MODEL` | — | LLM model id to drive the agent. REQUIRED — no default; set it here or pick a provider+model via `sliceagent init`. |
| `AGENT_MODEL_FALLBACK` | — | Larger-context model to switch to ONCE if the context overflows even after compaction (secondary net; the bounded slice is the primary). |
| `AGENT_PROVIDER` | — | Default provider id to use from the config's [providers.<id>] tables (overrides [agent].default_provider). |
| `AGENT_REASONING` | `full` | Reasoning effort: full=provider default, fast=minimal, high/max=more. _(choices: full, fast, high, max)_ |
| `AGENT_ROOT` | — | Workspace root override (defaults to the current directory). |
| `AGENT_ROUTER` | `lexical` | Topic router: lexical (instant, no LLM) or llm (classifier round-trip). _(choices: lexical, llm)_ |
| `AGENT_SANDBOX` | `local` | Tool sandbox backend; docker requires POSIX/WSL2 (native Windows: use local or run under WSL2). _(choices: local, docker)_ |
| `AGENT_SUBAGENT_DEPTH` | `1` | Max delegation depth for spawn_agent (0=off). |
| `AGENT_THINKING` | — | Set to 'off' to disable reasoning (alias for AGENT_REASONING=fast). |
| `AGENT_TOOL_TIMEOUT` | — | Outer deadline for declared pure-read tools in seconds (0/unset = off). |
| `AGENT_TOPIC_TOOLS` | — | Expose model-callable topic switching (off by default; host routing and slash commands remain available). |
| `AGENT_VERIFY_CMD` | — | Oracle verify command run after a turn (e.g. 'pytest -q'). |
| `AGENT_WEB` | `1` | Enable the web tools (fetch_url + web_search, DuckDuckGo, no key); set 0/off to disable network egress from the agent. |
| `SLICEAGENT_BASH` | — | Windows only: bash.exe that runs shell commands (default: auto-detect Git Bash). |

## debug

| variable | default | description |
|---|---|---|
| `AGENT_EXPERIMENTAL_ALL` | — | Master switch: set truthy to enable ALL experimental flags (per-flag AGENT_EXPERIMENTAL_<ID> overrides). |
| `SLICEAGENT_DEBUG_TRACE` | — | Set truthy to print tracebacks for parked/hook errors. |
| `SLICEAGENT_MEMORY_MODEL_FILE` | — | A/B experiment seam: path to content replacing only the {{MEMORY_MODEL}} operating-contract splice. An empty file is a valid no-contract arm; unset uses the production contract. |
| `SLICEAGENT_NO_CLOSURE` | — | Debug flag: disable the turn closeout call. |
| `SLICEAGENT_PROMPT_FILE` | — | A/B experiment seam: path to a full SYSTEM_PROMPT template (must keep the {{MEMORY_MODEL}} marker) to override the prompt for a measurement run (evals/prompt_ab). Unset → the production prompt. |

## memory

| variable | default | description |
|---|---|---|
| `MEMEM_VAULT` | — | memem's lesson vault (markdown long-term memories), if memem is installed. |
| `SLICEAGENT_AGENT_ID` | `sliceagent` | Stable agent scope key for typed CRAFT knowledge. |
| `SLICEAGENT_CACHE_DIR` | — | sliceagent state root for always-on checkpoints, immutable artifacts, recovery journals, and the optional episodic compatibility mirror. |
| `SLICEAGENT_KNOWLEDGE_DB` | — | Override the native typed-knowledge SQLite database path. |
| `SLICEAGENT_PROJECT_REGISTRY` | — | Override the private stable project-identity registry path. |
| `SLICEAGENT_SKILLS_DIR` | — | Extra directory to discover skills from. |
| `SLICEAGENT_USER_ID` | `local-user` | Stable local user scope key for typed USER knowledge. |
| `SLICEAGENT_VAULT` | — | Legacy episodic/task/roster compatibility vault. |

## monitor

| variable | default | description |
|---|---|---|
| `AGENT_METRICS` | — | Set truthy to print per-turn fresh-token (moat) metrics at exit. |
| `AGENT_MONITOR` | — | Set truthy to enable the live monitor server. |
| `AGENT_MONITOR_PORT` | — | Port for the monitor server. |
| `AGENT_TIMING` | — | Set truthy to print a per-turn latency breakdown (slice build vs model). |
| `SLICEAGENT_MONITOR_DIR` | — | Directory the monitor writes slice snapshots to. |

## provider

| variable | default | description |
|---|---|---|
| `AGENT_PROXY` | — | HTTP proxy URL for LLM calls; 'none'/'off' forces direct. Unset = direct (no proxy). |
| `LLM_API_KEY` 🔒 | — | API key for the LLM provider (REQUIRED). |
| `LLM_BASE_URL` | — | OpenAI-compatible endpoint (e.g. https://api.moonshot.cn/v1). |
| `LLM_HARD_TIMEOUT_SEC` | — | Absolute whole-request watchdog in seconds; unset derives a provider-agnostic ceiling from the completion-token budget (minimum 180 seconds). |
| `LLM_PROVIDER_MAX_INFLIGHT` | `4` | Process-wide physical request ceiling per provider account. Timed-out calls retain a slot until their transport actually closes; invalid or non-positive values use the default. |
| `LLM_STREAM_CLOSE_GRACE_SEC` | `2` | Seconds to confirm SSE connection closure after cancellation/deadline before reporting an indeterminate call. |
| `LLM_TIMEOUT` | — | Per-request timeout in seconds. |
| `LLM_TIMEOUT_SEC` | — | Alias for LLM_TIMEOUT. |
| `MOONSHOT_API_KEY` 🔒 | — | Legacy alias for LLM_API_KEY (Moonshot). |
| `OPENAI_API_KEY` 🔒 | — | Legacy alias for LLM_API_KEY. |
| `OPENAI_BASE_URL` | — | Legacy alias for LLM_BASE_URL. |

## ui

| variable | default | description |
|---|---|---|
| `AGENT_SPINNER` | `on` | Animated in-place status spinner during a turn (a Rich live region). Set off to drop just the spinner; all other Rich formatting stays. _(choices: on, off)_ _(aliases: 1, true, yes, 0, false, no)_ |
| `AGENT_TUI` | `rich` | UI mode: rich (default inline), live (pinned box), off (plain). _(choices: rich, live, off)_ _(aliases: 1, on, true, yes, 0, false, no)_ |
| `SHOW_SLICE` | — | Set truthy to print the rebuilt slice each turn (debug view). |
