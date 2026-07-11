"""Config — layered settings from sliceagent.toml (Step ③.2).

A layered config file (user then project, project overriding)
that declares persistent settings AND extension surfaces (skills dirs, MCP servers,
plugin dirs). Precedence is ENV > project file > user file > default, so a quick
`AGENT_POLICY=allow sliceagent ...` still overrides the file and ALL prior env-driven
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
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError, ValueError):
        return {}   # a corrupt / non-UTF-8 config must degrade to defaults, not crash startup


def _config_files(root: str | None = None) -> list[str]:
    # user first, then project (project overrides user)
    home = os.path.expanduser("~")
    cwd = os.path.realpath(root or os.getcwd())
    return [
        os.path.join(home, ".sliceagent", "config.toml"),
        os.path.join(cwd, "sliceagent.toml"),
        os.path.join(cwd, ".sliceagent", "config.toml"),
    ]


# ── runtime preferences (the /model switch persists here) ───────────────────────────────────────
# A tiny JSON sidecar, NOT config.toml: stdlib has no TOML WRITER (tomllib is read-only), so writing
# back to config.toml would need a new dep or a fragile hand-rolled serializer. JSON is safe + atomic.
# Precedence (resolved in cli): explicit env (AGENT_MODEL/AGENT_REASONING) > prefs > config.toml > default.
def _prefs_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".sliceagent", "prefs.json")


def load_prefs() -> dict:
    """The user's last /model + /reasoning choice (or {} if none/unreadable)."""
    try:
        import json
        with open(_prefs_path(), encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:  # noqa: BLE001 — missing/corrupt prefs must never break startup
        return {}


def save_prefs(updates: dict) -> None:
    """Merge non-empty `updates` into the prefs sidecar (atomic write); an explicit None DELETES the
    key (a stale `provider` pin must be removable — merge-only let an old endpoint pin resurrect at
    the next boot under a model it doesn't serve). Best-effort; never raises."""
    try:
        import json
        path = _prefs_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cur = load_prefs()
        for k, v in updates.items():
            if v is None:
                cur.pop(k, None)
        cur.update({k: v for k, v in updates.items() if v})
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cur, f, indent=2)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 — persistence is a nicety, not a hard requirement
        pass


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
    def load(cls, root: str | None = None) -> "Config":
        merged: dict = {}
        for f in _config_files(root):
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

    # --- provider (multi-provider; written by `sliceagent init`; ENV always wins) ---
    # Resolution order for api_key/base_url/model: ENV → the DEFAULT provider's [providers.<id>] table →
    # the legacy flat [provider]/[agent].model → default. So multiple named providers can coexist and
    # `sliceagent config --use <id>` switches between them, while old flat configs + env keep working.
    @property
    def default_provider(self) -> str:
        return self._get("agent", "default_provider", "AGENT_PROVIDER", "")

    def providers(self) -> dict:
        """All declared providers: {id: {api_key, base_url, model}}."""
        v = self.data.get("providers", {})
        return {k: val for k, val in v.items() if isinstance(val, dict)} if isinstance(v, dict) else {}

    def _provider_table(self) -> dict:
        """The active provider's table: the configured default, or the sole provider if exactly one exists."""
        provs = self.providers()
        pid = self.default_provider
        if pid and pid in provs:
            return provs[pid]
        if len(provs) == 1:
            return next(iter(provs.values()))
        return {}

    @property
    def api_key(self) -> str:
        env = os.environ.get("LLM_API_KEY")
        if env:   # empty string ("" exported) means UNSET → fall through to config, don't return ""
            return env
        return self._provider_table().get("api_key") or self._get("provider", "api_key", None, "")

    @property
    def base_url(self) -> str:
        env = os.environ.get("LLM_BASE_URL")
        if env:   # empty string → unset (use provider default), not a literal empty base_url
            return env
        return self._provider_table().get("base_url") or self._get("provider", "base_url", None, "")

    # --- agent ---
    @property
    def model(self) -> str:
        env = os.environ.get("AGENT_MODEL")
        if env:   # empty string → unset → fall through to config/default model, not ""
            return env
        # No built-in default model — the user chooses one (sliceagent init / AGENT_MODEL / config.toml).
        return self._provider_table().get("model") or self._get("agent", "model", None, "")

    @property
    def policy(self) -> str:
        return self._get("agent", "policy", "AGENT_POLICY", "teenager")

    @property
    def mine(self) -> str:
        return self._get("agent", "mine", "AGENT_MINE", "deterministic")

    @property
    def subagent_depth(self) -> int:
        v = self._get("agent", "subagent_depth", "AGENT_SUBAGENT_DEPTH", 1)
        try:
            return max(0, int(v))                 # 0 = off; a malformed value falls back to the default
        except (TypeError, ValueError):
            return 1

    @property
    def show_slice(self) -> bool:
        return _truthy(self._get("agent", "show_slice", "SHOW_SLICE", False))

    # --- sandbox ---
    @property
    def sandbox_backend(self) -> str:
        return self._get("sandbox", "backend", "AGENT_SANDBOX", "local")  # local | docker

    @property
    def sandbox_image(self) -> str:
        return self._get("sandbox", "image", None, "python:3.12-slim")

    @property
    def sandbox_network(self) -> str:
        return self._get("sandbox", "network", None, "none")

    @property
    def intent_gate(self) -> str:
        """"essential" (default) lifts the fail-closed turn-authority gate — it over-blocks ordinary local
        work and its errors mislead. The ESSENTIAL protections stay regardless (catastrophic-command blocking
        + confirm-mode prompts via the policy, plus the loop guard). "strict" restores the full v2 gate that
        denies task-state/external effects unless the exact turn authorizes them. Env: AGENT_INTENT_GATE."""
        return str(self._get("policy", "intent_gate", "AGENT_INTENT_GATE", "essential")
                   or "essential").strip().lower()

    # --- oracle / budget ---
    @property
    def verify_cmd(self) -> str | None:
        return self._get("oracle", "verify_cmd", "AGENT_VERIFY_CMD", None)

    @property
    def max_tokens(self) -> int | None:
        v = self._get("budget", "max_tokens", "AGENT_MAX_TOKENS", None)
        try:
            n = int(v) if v is not None else None
        except (TypeError, ValueError):
            return None                            # garbage budget → no budget (don't crash startup)
        return n if (n is not None and n > 0) else None   # discard a nonsensical <=0 budget

    @property
    def max_steps(self) -> int:
        # Per-turn step ceiling (runaway backstop). Default raised above the old hard 40 so deep
        # analysis/review turns aren't guillotined; overridable for heavier work.
        v = self._get("budget", "max_steps", "AGENT_MAX_STEPS", None)
        try:
            n = int(v) if v not in (None, "") else None
        except (TypeError, ValueError):
            return 60
        return n if (n is not None and n >= 1) else 60   # <=0 (incl. the env STRING "0") → default, consistent across env/TOML

    # --- extension surfaces ---
    @property
    def skills_roots(self) -> list[str] | None:
        sec = self.data.get("skills", {})
        dirs = sec.get("dirs") if isinstance(sec, dict) else None
        if isinstance(dirs, str):                  # a scalar `dirs = "..."` must not iterate char-by-char
            dirs = [dirs]
        if not isinstance(dirs, list):
            return None
        roots = [os.path.expanduser(d) for d in dirs if isinstance(d, str)]   # skip non-str entries (don't crash startup)
        return roots or None

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
        if isinstance(dirs, str):                  # scalar `dirs = "..."` → single entry, not char iteration
            dirs = [dirs]
        if not isinstance(dirs, list):
            return []
        return [os.path.expanduser(d) for d in dirs if isinstance(d, str)]   # skip non-str entries (don't crash startup)


def load_config(root: str | None = None) -> Config:
    """Load user config plus the selected workspace's config without changing process cwd."""
    return Config.load(root)
