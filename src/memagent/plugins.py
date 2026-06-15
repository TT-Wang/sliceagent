"""Plugins — packaging that feeds the existing seams (③.4).

Lift-adapted from Hermes' register(ctx) facade + manifest + multi-source discovery, with
Kimi's small-surface discipline. A plugin is a directory with a `plugin.toml` manifest and
an `__init__.py` exposing `register(ctx)`. Through `ctx` it contributes to the SAME seams
everything else uses — the tool registry, the skill manager, MCP servers, and hooks — so a
plugin gets NO privileged surface and reuses all the sandbox/policy/scheduler machinery.

A broken plugin logs and is skipped; it never crashes the host.
"""
from __future__ import annotations

import importlib.util
import os
import tomllib

from .access import AllAccess
from .registry import ToolEntry


def _default_dirs() -> list[str]:
    return [
        os.path.join(os.getcwd(), ".memagent", "plugins"),
        os.path.join(os.path.expanduser("~"), ".memagent", "plugins"),
    ]


class PluginContext:
    """The single facade a plugin's `register(ctx)` uses. Everything it contributes flows into
    existing seams; aggregated MCP servers + hooks are returned to the host to connect/compose."""

    def __init__(self, name: str, registry, skills, *, root: str, config):
        self.name = name
        self.registry = registry      # shared ToolRegistry
        self.skills = skills          # SkillManager
        self.root = root              # workspace root
        self.config = config          # Config
        self.mcp_servers: dict = {}   # collected; host connects them
        self.hooks: list = []         # collected Hooks instances; host composes them
        self.counts = {"tools": 0, "skills": 0, "mcp": 0, "hooks": 0}

    def register_tool(self, name: str, description: str, handler, *, parameters: dict | None = None,
                      accesses=None, check=None, override: bool = False) -> None:
        schema = {"type": "function", "function": {
            "name": name, "description": description,
            "parameters": parameters or {"type": "object", "properties": {}}}}
        self.registry.register(ToolEntry(
            name=name, schema=schema, handler=handler,
            accesses=accesses or (lambda _a: [AllAccess()]), check=check,
            source=f"plugin:{self.name}"), override=override)
        self.counts["tools"] += 1

    def register_skill(self, name: str, body: str, description: str = "") -> None:
        self.skills.add(name, body, description=description)
        self.counts["skills"] += 1

    def register_mcp_server(self, name: str, config: dict) -> None:
        self.mcp_servers[name] = config
        self.counts["mcp"] += 1

    def register_hook(self, hook) -> None:  # a Hooks instance
        self.hooks.append(hook)
        self.counts["hooks"] += 1

    def log(self, msg: str) -> None:
        print(f"  · plugin:{self.name}: {msg}")


def _load_one(pdir: str, registry, skills, *, root, config, on_log) -> PluginContext:
    try:
        with open(os.path.join(pdir, "plugin.toml"), "rb") as f:
            meta = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        meta = {}
    name = (meta.get("name") or os.path.basename(pdir)).strip()
    ctx = PluginContext(name, registry, skills, root=root, config=config)
    try:
        spec = importlib.util.spec_from_file_location(
            f"memagent_plugin_{name}", os.path.join(pdir, "__init__.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        register = getattr(mod, "register", None)
        if register is None:
            on_log(f"plugin:{name} has no register(ctx), skipped")
            return ctx
        register(ctx)
        c = ctx.counts
        on_log(f"plugin:{name} loaded (tools={c['tools']} skills={c['skills']} "
               f"mcp={c['mcp']} hooks={c['hooks']})")
    except Exception as e:  # a broken plugin must never crash the host
        on_log(f"plugin:{name} failed: {e}")
    return ctx


def load_plugins(registry, skills, dirs: list[str] | None = None, *, root: str, config,
                 on_log=lambda m: None) -> tuple[dict, list]:
    """Discover + load plugins from (provided dirs + defaults). Each contributes to the shared
    registry/skills; returns aggregated (mcp_servers, hooks) for the host to connect/compose."""
    search = list(dict.fromkeys((dirs or []) + _default_dirs()))
    mcp_servers: dict = {}
    hooks: list = []
    for rootdir in search:
        if not os.path.isdir(rootdir):
            continue
        for entry in sorted(os.listdir(rootdir)):
            pdir = os.path.join(rootdir, entry)
            if os.path.isfile(os.path.join(pdir, "plugin.toml")) and \
               os.path.isfile(os.path.join(pdir, "__init__.py")):
                ctx = _load_one(pdir, registry, skills, root=root, config=config, on_log=on_log)
                mcp_servers.update(ctx.mcp_servers)
                hooks.extend(ctx.hooks)
    return mcp_servers, hooks
