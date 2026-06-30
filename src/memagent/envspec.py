"""Single source of truth for every memagent environment variable.

Before this, 28 env vars were scattered across llm.py / cli.py / config.py / hooks.py with no discovery and
no validation (a typo'd AGENT_POLICY silently used the default). This module centralizes them so:
  * `memagent config --list` can show every knob, its group, default, and current value;
  * `validate_env()` warns on a misspelled enum value at startup instead of silently defaulting;
  * a coverage test asserts no AGENT_*/LLM_*/MEMAGENT_* var is read in the code without being documented here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvVar:
    name: str
    group: str
    desc: str
    default: str = ""
    choices: tuple = ()          # if set AND validate=True, an out-of-set value warns at startup
    secret: bool = False         # value is masked in `config --list`
    validate: bool = False       # run startup validation against `choices`
    aliases: tuple = ()          # extra accepted values not shown as the canonical choice set


REGISTRY: list[EnvVar] = [
    # ── agent behaviour ───────────────────────────────────────────────────────────────────────
    EnvVar("AGENT_MODEL", "agent", "LLM model id to drive the agent.", "gpt-5.5"),
    EnvVar("AGENT_MODEL_FALLBACK", "agent", "Larger-context model to switch to ONCE if the context overflows "
           "even after compaction (secondary net; the bounded slice is the primary).", ""),
    EnvVar("AGENT_PROVIDER", "agent", "Default provider id to use from the config's [providers.<id>] tables "
           "(overrides [agent].default_provider).", ""),
    EnvVar("AGENT_POLICY", "agent", "Permission mode: baby-sitter (confirm all) | teenager (auto edits, "
           "confirm commands) | let-it-go (auto, blocks catastrophic). All block catastrophic moves.",
           "teenager", choices=("baby-sitter", "teenager", "let-it-go"),
           aliases=("guard", "allow", "readonly", "ask", "babysitter", "teen", "letgo", "letitgo", "yolo", "baby"),
           validate=True),
    EnvVar("AGENT_ROUTER", "agent", "Topic router: lexical (instant, no LLM) or llm (classifier round-trip).",
           "lexical", choices=("lexical", "llm"), validate=True),
    EnvVar("AGENT_REASONING", "agent", "Reasoning effort: full=provider default, fast=minimal, high/max=more.",
           "full", choices=("full", "fast", "high", "max"), validate=True),
    EnvVar("AGENT_THINKING", "agent", "Set to 'off' to disable reasoning (alias for AGENT_REASONING=fast).", ""),
    EnvVar("AGENT_MINE", "agent", "Lesson-mining mode for end-of-session consolidation.", "deterministic"),
    EnvVar("AGENT_SUBAGENT_DEPTH", "agent", "Max delegation depth for spawn_subagent/spawn_explore (0=off).", "1"),
    EnvVar("AGENT_EXPLORER_REASONING", "agent", "Reasoning effort for read-only explorer children.", "fast"),
    EnvVar("AGENT_AUTO_APPROVE", "agent", "Comma-separated globs of pre-approved safe commands (skip prompt).", ""),
    EnvVar("AGENT_VERIFY_CMD", "agent", "Oracle verify command run after a turn (e.g. 'pytest -q').", ""),
    EnvVar("AGENT_MAX_TOKENS", "agent", "Per-session token budget (parks the turn when exhausted).", ""),
    EnvVar("AGENT_COMPLETION_TOKENS", "agent", "Per-REQUEST completion cap (max output tokens); distinct from the AGENT_MAX_TOKENS turn budget.", "8192"),
    EnvVar("AGENT_MAX_STEPS", "agent", "Per-turn step ceiling (runaway backstop); raise for deep analysis.", "60"),
    EnvVar("AGENT_SELFCHECK_MAX", "agent", "Max grounded done-gate verification rounds before accepting 'done'.", "3"),
    EnvVar("AGENT_TOOL_TIMEOUT", "agent", "Per-tool wall-clock deadline in seconds (0/unset = off).", ""),
    EnvVar("AGENT_ROOT", "agent", "Workspace root override (defaults to the current directory).", ""),
    EnvVar("AGENT_ALLOW_PLUGINS", "agent", "Set truthy to load project/user plugins.", ""),
    EnvVar("AGENT_SANDBOX", "agent", "Tool sandbox backend.", "local", choices=("local", "docker"), validate=True),
    EnvVar("AGENT_WEB", "agent", "Enable the web tools (fetch_url + web_search, DuckDuckGo, no key); set "
           "0/off to disable network egress from the agent.", "1"),
    # ── provider / network ────────────────────────────────────────────────────────────────────
    EnvVar("LLM_API_KEY", "provider", "API key for the LLM provider (REQUIRED).", "", secret=True),
    EnvVar("LLM_BASE_URL", "provider", "OpenAI-compatible endpoint (e.g. https://api.moonshot.cn/v1).", ""),
    EnvVar("LLM_TIMEOUT", "provider", "Per-request timeout in seconds.", ""),
    EnvVar("LLM_TIMEOUT_SEC", "provider", "Alias for LLM_TIMEOUT.", ""),
    EnvVar("AGENT_PROXY", "provider", "Proxy override: 'off', a URL, or unset for auto (foreign→ClashX, CN→direct).", ""),
    EnvVar("OPENAI_API_KEY", "provider", "Legacy alias for LLM_API_KEY.", "", secret=True),
    EnvVar("MOONSHOT_API_KEY", "provider", "Legacy alias for LLM_API_KEY (Moonshot).", "", secret=True),
    EnvVar("OPENAI_BASE_URL", "provider", "Legacy alias for LLM_BASE_URL.", ""),
    # ── UI ────────────────────────────────────────────────────────────────────────────────────
    EnvVar("AGENT_TUI", "ui", "UI mode: rich (default inline), live (pinned box), off (plain).",
           "rich", choices=("rich", "live", "off"),
           aliases=("1", "on", "true", "yes", "0", "false", "no")),
    EnvVar("SHOW_SLICE", "ui", "Set truthy to print the rebuilt slice each turn (debug view).", ""),
    # ── memory ────────────────────────────────────────────────────────────────────────────────
    EnvVar("MEMAGENT_VAULT", "memory", "memagent's STATE vault (episodic cache + task-state records).", ""),
    EnvVar("MEMEM_VAULT", "memory", "memem's lesson vault (markdown long-term memories), if memem is installed.", ""),
    EnvVar("MEMAGENT_SKILLS_DIR", "memory", "Extra directory to discover skills from.", ""),
    EnvVar("AGENT_BACKGROUND_REVIEW", "agent", "Set truthy to run an off-thread reviewer that consolidates "
           "lessons after each turn.", ""),
    EnvVar("MEMAGENT_CACHE_DIR", "memory", "Directory for the episodic cache / durable log.", ""),
    EnvVar("AGENT_EXPERIMENTAL_ALL", "debug", "Master switch: set truthy to enable ALL experimental flags "
           "(per-flag AGENT_EXPERIMENTAL_<ID> overrides).", ""),
    # ── monitoring / debug ────────────────────────────────────────────────────────────────────
    EnvVar("AGENT_METRICS", "monitor", "Set truthy to print per-turn fresh-token (moat) metrics at exit.", ""),
    EnvVar("AGENT_MONITOR", "monitor", "Set truthy to enable the live monitor server.", ""),
    EnvVar("AGENT_MONITOR_PORT", "monitor", "Port for the monitor server.", ""),
    EnvVar("MEMAGENT_MONITOR_DIR", "monitor", "Directory the monitor writes slice snapshots to.", ""),
    EnvVar("MEMAGENT_DEBUG_TRACE", "debug", "Set truthy to print tracebacks for parked/hook errors.", ""),
    EnvVar("MEMAGENT_NO_CLOSURE", "debug", "Debug flag: disable the turn closeout call.", ""),
    EnvVar("MEMAGENT_PROMPT_FILE", "debug", "A/B experiment seam: path to a full SYSTEM_PROMPT template "
           "(must keep the {{MEMORY_MODEL}} marker) to override the prompt for a measurement run "
           "(evals/prompt_ab). Unset → the production prompt.", ""),
]

BY_NAME: dict[str, EnvVar] = {e.name: e for e in REGISTRY}
GROUPS = ("agent", "provider", "ui", "memory", "monitor", "debug")


def validate_env(env: dict | None = None) -> list[str]:
    """Return a list of human-readable warnings for any enum var set to an out-of-set value. Non-fatal:
    the caller prints them and continues on defaults (mature CLIs validate; they don't silently misbehave)."""
    env = env if env is not None else os.environ
    warnings = []
    for e in REGISTRY:
        if not (e.validate and e.choices):
            continue
        raw = env.get(e.name)
        if raw is None or raw == "":
            continue
        if raw.strip().lower() not in {c.lower() for c in e.choices} | {a.lower() for a in e.aliases}:
            warnings.append(f"{e.name}={raw!r} is not one of {{{', '.join(e.choices)}}} — using default "
                            f"{e.default!r}")
    return warnings


def current_value(name: str, env: dict | None = None) -> str:
    """The effective value of a var for display, masked if it is a secret."""
    env = env if env is not None else os.environ
    e = BY_NAME.get(name)
    raw = env.get(name)
    if raw is None:
        return ""
    if e and e.secret and raw:
        return f"*** ({len(raw)} chars)"     # never reveal any of a secret (not even the 'sk-' prefix)
    return raw
