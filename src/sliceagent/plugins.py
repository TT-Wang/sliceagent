"""Plugins — packaging that feeds the existing seams (③.4).

A plugin is a directory with a `plugin.toml` manifest and
an `__init__.py` exposing `register(ctx)`. Through `ctx` it contributes tools, skills, and MCP
servers through the same registries as built-ins. Plugins cannot install tool-preflight hooks
or recreate a hidden permission gate.

A broken plugin logs and is skipped; it never crashes the host.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
import re
import sys
from dataclasses import fields
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

from .access import AllAccess
from .registry import ToolEntry


def _default_dirs(root: str | None = None) -> list[str]:
    return [
        os.path.join(os.path.realpath(root or os.getcwd()), ".sliceagent", "plugins"),
        os.path.join(os.path.expanduser("~"), ".sliceagent", "plugins"),
    ]


class PluginContext:
    """The single facade a plugin's `register(ctx)` uses. Everything it contributes flows into
    the shared registries; aggregated MCP servers are returned to the host to connect."""

    def __init__(self, name: str, registry, skills, *, root: str, config):
        self.name = name
        self.registry = registry      # shared ToolRegistry
        self.skills = skills          # SkillManager
        self.root = root              # workspace root
        self.config = config          # Config
        self.mcp_servers: dict = {}   # collected; host connects them
        self.counts = {"tools": 0, "skills": 0, "mcp": 0}

    def register_tool(self, name: str, description: str, handler, *, parameters: dict | None = None,
                      accesses=None, check=None, override: bool = False) -> None:
        """Register a plugin tool on the same host surface as built-in tools."""
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

    def log(self, msg: str) -> None:
        print(f"  · plugin:{self.name}: {msg}")


def _manifest_name(meta: dict, pdir: str, on_log) -> str:
    """Return one log/schema-safe plugin name, tolerating hand-edited manifest types."""
    fallback = os.path.basename(os.path.normpath(pdir)) or "plugin"
    raw = meta.get("name") if isinstance(meta, dict) else None
    if raw is not None and not isinstance(raw, str):
        on_log(f"plugin:{fallback} has a non-string manifest name; using its directory name")
        raw = ""
    candidate = str(raw or fallback).strip() or fallback
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate).strip("._-")[:80]
    if not normalized:
        normalized = "plugin"
    if normalized != candidate:
        on_log(f"plugin:{fallback} manifest name normalized to {normalized!r}")
    return normalized


def _load_one(pdir: str, registry, skills, *, root, config, on_log) -> PluginContext | None:
    """Load one plugin transactionally.

    Registration is an all-or-nothing startup operation. A plugin can override an existing tool and then
    fail, so removing only newly-added names is insufficient: restore the complete registry and skill maps
    to their exact pre-plugin values. ``KeyboardInterrupt`` still belongs to the user and escapes, but the
    ``finally`` rollback runs before it does; an accidental ``SystemExit`` is treated as a plugin failure.
    """
    tools_before = dict(registry._tools)
    # Keep the original object identities (other runtime owners may hold them), but also retain their field
    # values. A plugin receives the registry facade and can accidentally mutate an existing ToolEntry/Skill in
    # place before failing; a shallow map snapshot alone would preserve that half-applied mutation.
    tool_fields_before = {
        key: {
            field.name: (copy.deepcopy(getattr(entry, field.name))
                         if field.name == "schema" else getattr(entry, field.name))
            for field in fields(entry)
        }
        for key, entry in tools_before.items()
    }
    generation_before = registry.generation
    skills_before = (dict(skills._skills)
                     if skills is not None and isinstance(getattr(skills, "_skills", None), dict)
                     else None)
    skill_fields_before = ({
        key: {field.name: getattr(skill, field.name) for field in fields(skill)}
        for key, skill in skills_before.items()
    } if skills_before is not None else None)
    committed = False
    name = os.path.basename(os.path.normpath(pdir)) or "plugin"
    module_name = ""
    modules_before: dict[str, object] = {}
    try:
        with open(os.path.join(pdir, "plugin.toml"), "rb") as f:
            meta = tomllib.load(f)
        name = _manifest_name(meta, pdir, on_log)
        ctx = PluginContext(name, registry, skills, root=root, config=config)
        # Load __init__.py as a real, path-unique package. Merely executing a package spec without first
        # publishing its parent in sys.modules makes ordinary ``from .helper import ...`` fail. The path hash
        # also prevents two plugins with the same manifest name from sharing stale helper modules.
        module_stem = re.sub(r"[^A-Za-z0-9_]", "_", name)
        path_hash = hashlib.sha1(os.path.realpath(pdir).encode("utf-8", "surrogatepass")).hexdigest()[:10]
        module_name = f"sliceagent_plugin_{module_stem}_{path_hash}"
        prefix = module_name + "."
        modules_before = {
            key: value for key, value in sys.modules.items()
            if key == module_name or key.startswith(prefix)
        }
        for key in tuple(modules_before):
            sys.modules.pop(key, None)
        spec = importlib.util.spec_from_file_location(
            module_name, os.path.join(pdir, "__init__.py"),
            submodule_search_locations=[pdir],
        )
        if spec is None or spec.loader is None:
            raise ImportError("could not create an import loader")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        register = getattr(mod, "register", None)
        if register is None:
            on_log(f"plugin:{name} has no register(ctx), skipped")
            return None
        register(ctx)
        c = ctx.counts
        on_log(f"plugin:{name} loaded (tools={c['tools']} skills={c['skills']} mcp={c['mcp']})")
        committed = True
        return ctx
    except (Exception, SystemExit) as e:  # broken plugin code must not crash the host
        on_log(f"plugin:{name} failed: {e}")
        return None
    finally:
        if not committed:
            if module_name:
                prefix = module_name + "."
                for key in tuple(sys.modules):
                    if key == module_name or key.startswith(prefix):
                        sys.modules.pop(key, None)
                sys.modules.update(modules_before)
            for key, entry in tools_before.items():
                for field_name, value in tool_fields_before[key].items():
                    setattr(entry, field_name, value)
            if isinstance(getattr(registry, "_tools", None), dict):
                registry._tools.clear()
                registry._tools.update(tools_before)
            else:
                registry._tools = dict(tools_before)
            registry.generation = generation_before
            if skills_before is not None:
                for key, skill in skills_before.items():
                    for field_name, value in skill_fields_before[key].items():
                        setattr(skill, field_name, value)
                if isinstance(getattr(skills, "_skills", None), dict):
                    skills._skills.clear()
                    skills._skills.update(skills_before)
                else:
                    skills._skills = dict(skills_before)


def load_plugins(registry, skills, dirs: list[str] | None = None, *, root: str, config,
                 on_log=lambda m: None) -> dict:
    """Discover + load plugins from (provided dirs + defaults). Each contributes to the shared
    registry/skills and returns aggregated MCP server configuration."""
    search = list(dict.fromkeys((dirs or []) + _default_dirs(root)))
    mcp_servers: dict = {}
    def _ls(r):
        try:
            return sorted(os.listdir(r))
        except OSError:        # an unreadable plugin search dir must not crash the whole host
            return []
    found = [os.path.join(r, e) for r in search if os.path.isdir(r) for e in _ls(r)
             if os.path.isfile(os.path.join(r, e, "plugin.toml"))
             and os.path.isfile(os.path.join(r, e, "__init__.py"))]
    # SECURITY (#6/#7): a plugin's __init__.py executes with FULL host privileges. Loading is therefore
    # OPT-IN: require an explicit
    # AGENT_ALLOW_PLUGINS=1 before running ANY plugin code — otherwise a plugin dropped in ~/.sliceagent/
    # plugins (a default search dir) would auto-execute silently. Default off = safe.
    allowed = (os.environ.get("AGENT_ALLOW_PLUGINS") or "").strip().lower() in ("1", "true", "yes", "on")
    if not allowed:
        if found:
            on_log(f"{len(found)} plugin(s) present but NOT loaded — set AGENT_ALLOW_PLUGINS=1 to run them "
                   "(plugin code executes with HOST privileges)")
        return mcp_servers
    for pdir in found:
        on_log(f"⚠ loading TRUSTED plugin {os.path.basename(pdir)} — runs with host privileges")
        ctx = _load_one(pdir, registry, skills, root=root, config=config, on_log=on_log)
        if ctx is not None:
            mcp_servers.update(ctx.mcp_servers)
    return mcp_servers
