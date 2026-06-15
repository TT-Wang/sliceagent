"""Config — layered settings from memagent.toml (Step ③.2).

Borrowed from Kimi/Hermes: a layered config file (user then project, project overriding)
that declares persistent settings AND extension surfaces (skills dirs, MCP servers,
plugin dirs). Precedence is ENV > project file > user file > default, so a quick
`AGENT_POLICY=allow memagent ...` still overrides the file and ALL prior env-driven
behavior is preserved (the file just makes settings persistent).

Read-only TOML via stdlib tomllib (Python 3.11+ — no new dependency).
"""
from __future__ import annotations

import os
import tomllib


def _read_toml(path: str) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _config_files() -> list[str]:
    # user first, then project (project overrides user)
    home = os.path.expanduser("~")
    cwd = os.getcwd()
    return [
        os.path.join(home, ".memagent", "config.toml"),
        os.path.join(cwd, "memagent.toml"),
        os.path.join(cwd, ".memagent", "config.toml"),
    ]


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


class Config:
    """Resolved settings. Each accessor checks ENV first, then the merged TOML, then a default."""

    def __init__(self, data: dict | None = None):
        self.data = data or {}

    @classmethod
    def load(cls) -> "Config":
        merged: dict = {}
        for f in _config_files():
            if os.path.isfile(f):
                merged = _deep_merge(merged, _read_toml(f))
        return cls(merged)

    def _get(self, section: str, key: str, env: str | None, default):
        if env and os.environ.get(env) is not None:
            return os.environ[env]
        sec = self.data.get(section, {})
        if isinstance(sec, dict) and key in sec:
            return sec[key]
        return default

    # --- agent ---
    @property
    def model(self) -> str:
        return self._get("agent", "model", "AGENT_MODEL", "gpt-5.5")

    @property
    def policy(self) -> str:
        return self._get("agent", "policy", "AGENT_POLICY", "guard")

    @property
    def mine(self) -> str:
        return self._get("agent", "mine", "AGENT_MINE", "deterministic")

    @property
    def subagent_depth(self) -> int:
        return int(self._get("agent", "subagent_depth", "AGENT_SUBAGENT_DEPTH", 1))

    @property
    def show_slice(self) -> bool:
        return _truthy(self._get("agent", "show_slice", "SHOW_SLICE", False))

    # --- oracle / budget ---
    @property
    def verify_cmd(self) -> str | None:
        return self._get("oracle", "verify_cmd", "AGENT_VERIFY_CMD", None)

    @property
    def max_tokens(self) -> int | None:
        v = self._get("budget", "max_tokens", "AGENT_MAX_TOKENS", None)
        return int(v) if v else None

    # --- extension surfaces ---
    @property
    def skills_roots(self) -> list[str] | None:
        sec = self.data.get("skills", {})
        dirs = sec.get("dirs") if isinstance(sec, dict) else None
        return [os.path.expanduser(d) for d in dirs] if dirs else None

    @property
    def mcp_servers(self) -> dict:
        """Declared MCP servers (consumed in ③.3). e.g. [mcp_servers.github] ..."""
        v = self.data.get("mcp_servers", {})
        return v if isinstance(v, dict) else {}

    @property
    def plugin_dirs(self) -> list[str]:
        """Extra plugin directories (consumed in ③.4)."""
        sec = self.data.get("plugins", {})
        dirs = sec.get("dirs", []) if isinstance(sec, dict) else []
        return [os.path.expanduser(d) for d in dirs]


def load_config() -> Config:
    return Config.load()
