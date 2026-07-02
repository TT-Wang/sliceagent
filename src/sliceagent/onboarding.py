"""First-run onboarding + config discovery — the `sliceagent init` / `config` / `help` / `version` subcommands.

Turns the cold start from "copy .env.example, learn 28 env vars, hand-edit a TOML" into "run `sliceagent init`":
pick a provider, paste a key, we test it, and write ~/.sliceagent/config.toml so the next bare `sliceagent`
just works. `sliceagent config --list` makes every knob discoverable; `sliceagent help` shows the surface.

All entry points take injectable input/getpass/llm/home so the wizard is testable without a tty or a key.
"""
from __future__ import annotations

import os

from .envspec import GROUPS, REGISTRY, current_value

# provider presets: key → (label, base_url, default_model). 'custom' prompts for the base_url.
# The lineup: one AGGREGATOR door (OpenRouter — breadth, one key) + four first-party doors
# (OpenAI, Anthropic, DeepSeek, Moonshot). Anthropic rides its OpenAI-compatible endpoint, so all
# five share the single OpenAILLM adapter — no per-provider SDKs.
PROVIDERS = {
    "1": ("openrouter", "OpenRouter (hundreds of models, one key)", "https://openrouter.ai/api/v1", "openai/gpt-5.5"),
    "2": ("openai", "OpenAI", "", "gpt-5.5"),
    "3": ("anthropic", "Anthropic (Claude)", "https://api.anthropic.com/v1", "claude-sonnet-5"),
    "4": ("deepseek", "DeepSeek", "https://api.deepseek.com/v1", "deepseek-chat"),
    "5": ("moonshot", "Moonshot (Kimi)", "https://api.moonshot.cn/v1", "kimi-k2.7-code"),
    "6": ("custom", "Custom OpenAI-compatible endpoint", "", ""),
}

# wizard model MENU per provider — suggestions for the arrow picker, NOT a cap: "type another…"
# always accepts any model id (the catalog/pricing layers stay the capability source of truth).
MODEL_SUGGESTIONS = {
    "openrouter": ["openai/gpt-5.5", "anthropic/claude-sonnet-5", "deepseek/deepseek-chat",
                   "moonshotai/kimi-k2.7"],
    "openai": ["gpt-5.5", "gpt-5", "gpt-5-mini", "o3"],
    "anthropic": ["claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "moonshot": ["kimi-k2.7-code", "kimi-k2-0905-preview"],
    "custom": [],
}


def _tty() -> bool:
    import sys
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:  # noqa: BLE001
        return False


def _masked_input(prompt_text: str, fallback):
    """API-key entry with VISIBLE feedback — asterisks per keystroke (prompt_toolkit is_password)
    instead of getpass's fully-invisible field, which first-run users read as 'nothing is happening'.
    Falls back to getpass when the TUI stack is absent or anything goes wrong."""
    try:
        from prompt_toolkit import prompt as _ptk_prompt
        return _ptk_prompt(prompt_text, is_password=True)
    except (ImportError, OSError, EOFError):
        return fallback(prompt_text)


def _pick(options: list, default: int = 0):
    """Arrow-key selector for the wizard — the VERTICAL menu (_menu_select: one option per row), NOT
    the single-line _arrow_select, whose one-line redraw wraps and stacks with 6 long provider labels
    (live-repro'd). Returns the index; raises KeyboardInterrupt on Esc (→ the wizard's normal
    'cancelled' path); returns None when a selector can't safely run → caller falls back to typed."""
    try:
        from .tui import _menu_select
    except ImportError:
        return None
    idx = _menu_select(options, default=default)
    if idx == -1:
        raise KeyboardInterrupt
    return idx


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


def _emit_section(prefix: str, d: dict, lines: list) -> None:
    """Emit one TOML table RECURSIVELY. Scalars first (TOML requires them before any sub-table header),
    then each nested dict as a sub-table at ANY depth — so a nested dict like mcp_servers.<id>.env
    becomes a proper [mcp_servers.<id>.env] sub-table instead of being stringified into a corrupt value
    (the old code only handled two levels). A pure container of sub-tables (e.g. `providers`) gets no
    header of its own; the sub-table headers imply it."""
    scalars = [(k, v) for k, v in d.items() if not isinstance(v, dict)]
    subtables = [(k, v) for k, v in d.items() if isinstance(v, dict)]
    if prefix and (scalars or not subtables):
        lines.append(f"\n[{prefix}]")
    for k, v in scalars:
        lines.append(f"{_toml_key(k)} = {_toml_val(v)}")
    for k, v in subtables:
        child = f"{prefix}.{_toml_key(k)}" if prefix else _toml_key(k)
        _emit_section(child, v, lines)


def _emit_toml(data: dict) -> str:
    """Minimal TOML emitter for sliceagent's config shape (scalars, sections, table-of-tables for
    [providers.<id>], and arbitrarily-nested sub-tables like [mcp_servers.<id>.env]). Round-trips a
    tomllib-parsed dict so editing one provider/server preserves the rest."""
    lines = ["# sliceagent config — managed by `sliceagent init` / `config`. ENV overrides any value here."]
    _emit_section("", data, lines)
    return "\n".join(lines) + "\n"


def _config_path(home: str | None = None) -> str:
    home = home or os.path.expanduser("~")
    return os.path.join(home, ".sliceagent", "config.toml")


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
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".sliceagent-cfg-", suffix=".tmp")
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
    if not isinstance(provs, dict):              # a corrupt non-dict providers must not crash on provs[pid]=
        provs = data["providers"] = {}
    entry = {"api_key": api_key, "model": model}
    if base_url:
        entry["base_url"] = base_url
    provs[pid] = entry
    agent = data.setdefault("agent", {})
    if not isinstance(agent, dict):
        agent = data["agent"] = {}
    agent["default_provider"] = pid
    agent["model"] = model                                     # keep top-level model in sync (back-compat)
    _atomic_write(path, _emit_toml(data))


def _test_key(model: str, api_key: str, base_url: str, llm_factory) -> tuple[bool, str]:
    """One cheap completion to confirm the key/endpoint work. Returns (ok, message)."""
    prev = {k: os.environ.get(k) for k in ("LLM_API_KEY", "LLM_BASE_URL", "OPENAI_BASE_URL")}
    try:
        os.environ["LLM_API_KEY"] = api_key
        if base_url:
            os.environ["LLM_BASE_URL"] = base_url
        else:
            # empty preset base_url ⇒ provider default; clear BOTH aliases (OpenAILLM also resolves
            # OPENAI_BASE_URL) so a stale exported endpoint can't hijack the key-test probe.
            os.environ.pop("LLM_BASE_URL", None)
            os.environ.pop("OPENAI_BASE_URL", None)
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
    out("\n  sliceagent setup\n  ─────────────")
    if os.path.exists(path):
        try:
            ans = inp(f"  A config already exists at {path}. Add/update a provider in it? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            out("\n  cancelled."); return 1
        if ans in ("n", "no"):
            out("  Leaving the existing config unchanged. Run `sliceagent` to start."); return 0

    # arrow-key niceties only on a REAL interactive run with the default input hooks — injected
    # inp/getpw (tests, scripted runs) keep the typed flow byte-for-byte.
    fancy = inp is input and getpw is __import__("getpass").getpass and _tty()
    keys = sorted(PROVIDERS)
    try:
        out("\n  Choose a provider (↑/↓ + Enter):" if fancy else "\n  Choose a provider:")
        idx = None
        if fancy:
            labels = [f"{PROVIDERS[k][1]}" + (f"  — {PROVIDERS[k][3]}" if PROVIDERS[k][3] else "")
                      for k in keys]
            idx = _pick(labels)                       # Esc → KeyboardInterrupt → 'cancelled' below
        if idx is not None:
            choice = keys[idx]
        else:                                          # typed fallback (no tty / selector unavailable)
            for k in keys:
                _id, label, _base, model0 = PROVIDERS[k]
                out(f"    {k}. {label}" + (f"  ({model0})" if model0 else ""))
            choice = inp("  > ").strip() or "1"
        pid, label, base_url, model = PROVIDERS.get(choice, PROVIDERS["1"])
        if pid == "custom":
            base_url = inp("  Base URL (OpenAI-compatible, e.g. https://host/v1): ").strip()
        # RE-CONFIGURING an ALREADY-SAVED provider (re-running `sliceagent init` to update the model,
        # or just re-confirming) must not force a blind full-key retype: pressing Enter keeps the
        # existing key/model. A BRAND-NEW provider has no existing entry, so blank still means "no
        # key entered" and falls through to the abort below — same as before.
        existing = _read_config(path).get("providers")
        existing = existing.get(pid) if isinstance(existing, dict) else None
        existing = existing if isinstance(existing, dict) else {}
        existing_key = existing.get("api_key") or ""
        key_prompt = ("  API key (typed as ******, Enter to keep existing): " if existing_key
                      else "  API key (typed as ******): ") if fancy else (
                      "  API key (hidden, Enter to keep existing): " if existing_key
                      else "  API key (hidden): ")
        key = (_masked_input(key_prompt, getpw) if fancy else getpw(key_prompt)).strip() or existing_key
        picked_model = None
        if fancy:
            cur = existing.get("model") or model
            opts = [m for m in ([cur] if cur else []) + MODEL_SUGGESTIONS.get(pid, []) if m]
            opts = list(dict.fromkeys(opts)) + ["type another model id…"]
            out("  Model (↑/↓ + Enter):")
            midx = _pick(opts)
            if midx is not None:
                picked_model = None if opts[midx].startswith("type another") else opts[midx]
                if picked_model is None:
                    picked_model = inp("  Model id: ").strip()
        if picked_model:
            model = picked_model
        else:
            model = (inp(f"  Model [{existing.get('model') or model or 'required'}]: ").strip()
                     or existing.get("model") or model)
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
            out("  Not saved. Re-run `sliceagent init` to try again."); return 1

    _save_provider(path, pid=pid, model=model, api_key=key, base_url=base_url)
    out(f"\n  Saved provider '{pid}' (model {model}) → {path} (0600)")
    out("  Ready. Run:  sliceagent\n")
    return 0


def run_config(argv=None, *, home=None, env=None) -> int:
    """`sliceagent config` shows the resolved settings + config path; `--list` shows every env var."""
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
            out(f"  usage: sliceagent config --use <provider>  "
                f"(configured: {', '.join(provs) or 'none — run `sliceagent init`'})")
            return 1
        agent = data.setdefault("agent", {})
        if not isinstance(agent, dict):                 # a corrupt non-dict [agent] must not crash on item-assign
            agent = data["agent"] = {}
        agent["default_provider"] = pid
        if isinstance(provs[pid], dict) and provs[pid].get("model"):
            agent["model"] = provs[pid]["model"]
        _atomic_write(path, _emit_toml(data))
        out(f"  default provider → {pid} (model {provs[pid].get('model', '?') if isinstance(provs[pid], dict) else '?'})")
        return 0
    if "--list" in argv:
        out("\n  sliceagent environment variables (ENV overrides config file):")
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
    out(f"\n  sliceagent {_version()}")
    out(f"  config file: {path}  ({'exists' if os.path.exists(path) else 'not created — run `sliceagent init`'})")
    data = _read_config(path)
    provs = data.get("providers", {}) if isinstance(data.get("providers"), dict) else {}
    if provs:
        _agent = data.get("agent")
        default = _agent.get("default_provider", "") if isinstance(_agent, dict) else ""
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
    out("\n  `sliceagent config --list` for all knobs · `sliceagent init` to (re)configure\n")
    return 0


def print_usage() -> int:
    out = print
    out(f"""
  sliceagent {_version()} — a memory-native coding agent (the slice/cache-not-log kernel)

  usage:
    sliceagent                 start the interactive agent (inline UI; AGENT_TUI=live|off to switch)
    sliceagent init            interactive first-run setup (provider, key, model) → ~/.sliceagent/config.toml
    sliceagent config          show resolved settings, providers, and config path
    sliceagent config --list   list every environment variable, default, and current value
    sliceagent config --use <id>   switch the default provider
    sliceagent help            show this help
    sliceagent version         show the version

  first run:  sliceagent init   then   sliceagent
  docs: README.md · QUICKSTART.md
""")
    return 0


def dispatch(argv) -> int:
    """Route a recognized subcommand; return an exit code. cli.main() calls this before any key gate."""
    cmd = argv[0] if argv else ""
    if cmd in ("--version", "-V", "version"):
        print(f"sliceagent {_version()}"); return 0
    if cmd in ("help", "--help", "-h"):
        return print_usage()
    if cmd == "init":
        return run_init()
    if cmd == "config":
        return run_config(argv[1:])
    print(f"unknown command: {cmd!r}\n")
    print_usage()
    return 1                          # non-zero so shell scripts see the failure
