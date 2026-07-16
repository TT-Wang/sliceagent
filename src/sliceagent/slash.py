"""Canonical public slash-command catalog shared by CLI help and TUI completion."""
from __future__ import annotations


PUBLIC_SLASH_COMMANDS = {
    "/config": "add / update LLM providers (the setup wizard, in-session) — then /model to switch",
    "/model": "switch model + reasoning — menu lists YOUR configured providers (switches endpoint too)",
    "/cwd": "show the workspace; /cwd <path> switches with a clean runtime handoff",
    "/learn": "turn what you just did into a reusable SKILL (/learn [name])",
    "/plan": "show the agent's current PLAN",
    "/cost": "show session token totals + $ saved vs full-history",
    "/update": "show how to update safely at the process boundary",
    "/threads": "list open/parked topics",
    "/resume": "resume a parked topic by task id (/resume <task_id>)",
    "/plugins": "list loaded plugins + their tools",
    "/mcp": "list configured MCP servers + connection status",
    "/skills": "list skills available to the agent",
    "/tools": "list tools currently available to the agent",
    "/agents": "list configured subagent profiles",
    "/help": "show commands  ·  Esc = undo last turn",
    "/exit": "quit",
}

# Accepted compatibility/power aliases that intentionally do not add palette clutter. `/model` owns the
# reasoning picker, Esc owns undo, and `/resume` is the public topic-navigation spelling.
HIDDEN_SLASH_ALIASES = {
    "/reasoning": "/model",
    "/undo": "Esc",
    "/switch": "/resume",
}

SUPPORTED_SLASH_COMMANDS = frozenset(PUBLIC_SLASH_COMMANDS) | frozenset(HIDDEN_SLASH_ALIASES)


def slash_help_line() -> str:
    """The exact public command list displayed by ``/help``."""
    return "commands: " + " · ".join(PUBLIC_SLASH_COMMANDS)
