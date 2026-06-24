"""First-run onboarding + config discovery — the `memagent init` / `config` / `help` / `version` subcommands.

Turns the cold start from "copy .env.example, learn 28 env vars, hand-edit a TOML" into "run `memagent init`":
pick a provider, paste a key, we test it, and write ~/.memagent/config.toml so the next bare `memagent`
just works. `memagent config --list` makes every knob discoverable; `memagent help` shows the surface.

All entry points take injectable input/getpass/llm/home so the wizard is testable without a tty or a key.
"""
from __future__ import annotations

import os

from .envspec import GROUPS, REGISTRY, current_value

# provider presets: key → (label, base_url, default_model). 'custom' prompts for the base_url.
PROVIDERS = {
    "1": ("moonshot", "Moonshot (Kimi)", "https://api.moonshot.cn/v1", "kimi-k2.7-code"),
    "2": ("openai", "OpenAI", "", "gpt-5.5"),
    "3": ("deepseek", "DeepSeek", "https://api.deepseek.com/v1", "deepseek-chat"),
    "4": ("custom", "Custom OpenAI-compatible endpoint", "", ""),
}


def _version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:  # noqa: BLE001
        return "0.0.0"


def _toml_str(v: str) -> str:
    s = ((v or "").replace("\\", "\\\\").replace('"', '\\"')
         .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))
    return '"' + s + '"'


def _toml_key(k: str) -> str:
    """A TOML table-header key: bare if it's a simple identifier, else a quoted key (so a provider id with a
    dot/space/quote — e.g. 'my.host' — round-trips as one key instead of a nested table or a parse error)."""
    if k and all(c.isalnum() or c in "-_" for c in k):
        return k
    return '"' + k.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _toml_val(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_val(x) for x in v) + "]"
    return _toml_str(str(v))


def _emit_toml(data: dict) -> str:
    """Minimal TOML emitter for memagent's config shape (scalars, sections, and a table-of-tables for
    [providers.<id>]). Round-trips a tomllib-parsed dict so editing one provider preserves the rest."""
    lines = ["# memagent config — managed by `memagent init` / `config`. ENV overrides any value here."]
    for k, v in data.items():                                  # top-level scalars (rare)
        if not isinstance(v, dict):
            lines.append(f"{k} = {_toml_val(v)}")
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        if v and all(isinstance(x, dict) for x in v.values()):  # table-of-tables, e.g. [providers.moonshot]
            for sub, subv in v.items():
                lines.append(f"\n[{k}.{_toml_key(sub)}]")        # quote a non-bare provider id (e.g. "my.host")
                for kk, vv in subv.items():
                    lines.append(f"{kk} = {_toml_val(vv)}")
        elif v:                                                 # a normal [section]: scalars first, then nested
            lines.append(f"\n[{k}]")
            for kk, vv in v.items():
                if not isinstance(vv, dict):
                    lines.append(f"{kk} = {_toml_val(vv)}")
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    lines.append(f"\n[{k}.{_toml_key(kk)}]")
                    for k3, v3 in vv.items():
                        lines.append(f"{k3} = {_toml_val(v3)}")
    return "\n".join(lines) + "\n"


def _config_path(home: str | None = None) -> str:
    home = home or os.path.expanduser("~")
    return os.path.join(home, ".memagent", "config.toml")


def _read_config(path: str) -> dict:
    import tomllib
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _atomic_write(path: str, body: str) -> None:
    """ATOMIC 0600 write (the file holds an API key — never leave it half-written): write a temp in the same
    dir, fsync, then os.replace(); on ANY failure remove the temp so no key-bearing fragment is left behind."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".memagent-cfg-", suffix=".tmp")
    ok = False
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, body.encode("utf-8"))
        os.fsync(fd)
        os.close(fd)
        os.replace(tmp, path)
        ok = True
    finally:
        if not ok:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.remove(tmp)
            except OSError:
                pass


def _save_provider(path: str, *, pid: str, model: str, api_key: str, base_url: str) -> None:
    """Merge a provider into the config: add/update [providers.<pid>], set it as the default, keep the rest."""
    data = _read_config(path)
    provs = data.setdefault("providers", {})
    entry = {"api_key": api_key, "model": model}
    if base_url:
        entry["base_url"] = base_url
    provs[pid] = entry
    agent = data.setdefault("agent", {})
    agent["default_provider"] = pid
    agent["model"] = model                                     # keep top-level model in sync (back-compat)
    _atomic_write(path, _emit_toml(data))


def _test_key(model: str, api_key: str, base_url: str, llm_factory) -> tuple[bool, str]:
    """One cheap completion to confirm the key/endpoint work. Returns (ok, message)."""
    prev = {k: os.environ.get(k) for k in ("LLM_API_KEY", "LLM_BASE_URL")}
    try:
        os.environ["LLM_API_KEY"] = api_key
        if base_url:
            os.environ["LLM_BASE_URL"] = base_url
        llm = llm_factory(model)
        resp = llm.complete([{"role": "user", "content": "Reply with the single word: ok"}], [])
        txt = (getattr(resp, "content", "") or "").strip()
        return (True, txt[:40] or "(empty reply, but the call succeeded)")
    except Exception as e:  # noqa: BLE001
        return (False, f"{type(e).__name__}: {e}")
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_init(*, inp=input, getpw=None, llm_factory=None, home=None) -> int:
    """Interactive setup wizard. Returns a process exit code."""
    import getpass
    getpw = getpw or getpass.getpass
    if llm_factory is None:
        def llm_factory(model):
            from .llm import OpenAILLM
            return OpenAILLM(model=model)
    out = print
    path = _config_path(home)
    out("\n  memagent setup\n  ─────────────")
    if os.path.exists(path):
        try:
            ans = inp(f"  A config already exists at {path}. Add/update a provider in it? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            out("\n  cancelled."); return 1
        if ans in ("n", "no"):
            out("  Leaving the existing config unchanged. Run `memagent` to start."); return 0

    out("\n  Choose a provider:")
    for k, (_id, label, base, model) in PROVIDERS.items():
        out(f"    {k}. {label}" + (f"  ({model})" if model else ""))
    try:
        choice = inp("  > ").strip() or "1"
        pid, label, base_url, model = PROVIDERS.get(choice, PROVIDERS["1"])
        if pid == "custom":
            base_url = inp("  Base URL (OpenAI-compatible, e.g. https://host/v1): ").strip()
        key = getpw("  API key (hidden): ").strip()
        model = (inp(f"  Model [{model or 'required'}]: ").strip() or model)
    except (EOFError, KeyboardInterrupt):
        out("\n  cancelled."); return 1
    if not key:
        out("  No API key entered — aborting."); return 1
    if not model:
        out("  No model specified — aborting."); return 1

    out("\n  Testing the key with one request…")
    ok, msg = _test_key(model, key, base_url, llm_factory)
    out(f"  {'✓ works' if ok else '✗ failed'}: {msg}")
    if not ok:
        try:
            cont = inp("  Save the config anyway? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            cont = "n"
        if cont not in ("y", "yes"):
            out("  Not saved. Re-run `memagent init` to try again."); return 1

    _save_provider(path, pid=pid, model=model, api_key=key, base_url=base_url)
    out(f"\n  Saved provider '{pid}' (model {model}) → {path} (0600)")
    out("  Ready. Run:  memagent\n")
    return 0


def run_config(argv=None, *, home=None, env=None) -> int:
    """`memagent config` shows the resolved settings + config path; `--list` shows every env var."""
    argv = argv or []
    env = env if env is not None else os.environ
    out = print
    path = _config_path(home)
    if "--path" in argv:
        out(path); return 0
    if argv and argv[0] == "--use":
        pid = argv[1] if len(argv) > 1 else ""
        data = _read_config(path)
        provs = data.get("providers", {}) if isinstance(data.get("providers"), dict) else {}
        if not pid or pid not in provs:
            out(f"  usage: memagent config --use <provider>  "
                f"(configured: {', '.join(provs) or 'none — run `memagent init`'})")
            return 1
        data.setdefault("agent", {})["default_provider"] = pid
        if isinstance(provs[pid], dict) and provs[pid].get("model"):
            data["agent"]["model"] = provs[pid]["model"]
        _atomic_write(path, _emit_toml(data))
        out(f"  default provider → {pid} (model {(provs[pid] or {}).get('model', '?')})")
        return 0
    if "--list" in argv:
        out("\n  memagent environment variables (ENV overrides config file):")
        for g in GROUPS:
            out(f"\n  [{g}]")
            for e in [e for e in REGISTRY if e.group == g]:
                cur = current_value(e.name, env)
                shown = f" = {cur}" if cur else (f"  (default: {e.default})" if e.default else "")
                choices = f"  {{{', '.join(e.choices)}}}" if e.choices else ""
                out(f"    {e.name}{shown}{choices}")
                out(f"        {e.desc}")
        out("")
        return 0
    out(f"\n  memagent {_version()}")
    out(f"  config file: {path}  ({'exists' if os.path.exists(path) else 'not created — run `memagent init`'})")
    data = _read_config(path)
    provs = data.get("providers", {}) if isinstance(data.get("providers"), dict) else {}
    if provs:
        default = (data.get("agent", {}) or {}).get("default_provider", "")
        out("  providers (* = default · `config --use <id>` to switch):")
        for pid, p in provs.items():
            mark = "*" if pid == default else " "
            out(f"    {mark} {pid}  ({(p or {}).get('model', '?')})")
    out("  set values:")
    any_set = False
    for e in REGISTRY:
        cur = current_value(e.name, env)
        if cur:
            out(f"    {e.name} = {cur}")
            any_set = True
    if not any_set:
        out("    (none — all defaults)")
    out("\n  `memagent config --list` for all knobs · `memagent init` to (re)configure\n")
    return 0


def print_usage() -> int:
    out = print
    out(f"""
  memagent {_version()} — a memory-native coding agent (the slice/cache-not-log kernel)

  usage:
    memagent                 start the interactive agent (inline UI; AGENT_TUI=textual|live|off to switch)
    memagent init            interactive first-run setup (provider, key, model) → ~/.memagent/config.toml
    memagent config          show resolved settings, providers, and config path
    memagent config --list   list every environment variable, default, and current value
    memagent config --use <id>   switch the default provider
    memagent help            show this help
    memagent version         show the version

  first run:  memagent init   then   memagent
  docs: README.md · QUICKSTART.md
""")
    return 0


def dispatch(argv) -> int:
    """Route a recognized subcommand; return an exit code. cli.main() calls this before any key gate."""
    cmd = argv[0] if argv else ""
    if cmd in ("--version", "-V", "version"):
        print(f"memagent {_version()}"); return 0
    if cmd in ("help", "--help", "-h"):
        return print_usage()
    if cmd == "init":
        return run_init()
    if cmd == "config":
        return run_config(argv[1:])
    print(f"unknown command: {cmd!r}\n")
    print_usage()
    return 1                          # non-zero so shell scripts see the failure
