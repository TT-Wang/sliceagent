"""Onboarding & config: the envspec registry (coverage + validation) and the `sliceagent init` wizard
(driven headlessly with stubbed input/getpass/llm). No model, no pytest.
Run: PYTHONPATH=src python tests/test_onboarding.py
"""
import os
import re
import sys
import tempfile
import tomllib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent import envspec                                       # noqa: E402
from sliceagent import onboarding                                    # noqa: E402
from sliceagent.config import Config                                 # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# ---- A3: registry coverage (drift guard) ------------------------------------
@check
def every_env_var_read_in_code_is_registered():
    # scan src for our-namespace env-var string literals; each must be documented in envspec (or be a
    # known external/standard one). Prevents a new os.environ.get from going undocumented.
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src", "sliceagent")
    pat = re.compile(r'"(AGENT_[A-Z_]+|LLM_[A-Z_]+|SLICEAGENT_[A-Z_]+|SHOW_SLICE)"')
    found = set()
    for fn in os.listdir(src_dir):
        if fn.endswith(".py"):
            found |= set(pat.findall(open(os.path.join(src_dir, fn), encoding="utf-8").read()))
    # AGENT_EXPERIMENTAL_ is a DYNAMIC per-flag prefix (flags.py builds AGENT_EXPERIMENTAL_<ID>); the master
    # switch AGENT_EXPERIMENTAL_ALL is registered, the prefix itself is not a single var.
    allow_prefixes = ("AGENT_EXPERIMENTAL_",)
    missing = {v for v in found - set(envspec.BY_NAME)
               if not (any(v == p or v.startswith(p) for p in allow_prefixes) and v not in envspec.BY_NAME
                       and v.endswith("_"))}
    # the bare prefix literal "AGENT_EXPERIMENTAL_" is allowed; concrete vars must be registered
    missing -= {"AGENT_EXPERIMENTAL_"}
    assert not missing, f"env vars read but not in envspec.REGISTRY: {sorted(missing)}"


@check
def registry_groups_are_known():
    for e in envspec.REGISTRY:
        assert e.group in envspec.GROUPS, f"{e.name} has unknown group {e.group!r}"


# ---- A2: validation ---------------------------------------------------------
@check
def validate_flags_bad_enum_and_passes_good():
    assert envspec.validate_env({"AGENT_ROUTER": "lexical"}) == []
    assert any("AGENT_ROUTER" in w for w in envspec.validate_env({"AGENT_ROUTER": "magic"}))
    # AGENT_TUI accepts bool-ish aliases without warning
    assert envspec.validate_env({"AGENT_TUI": "1"}) == [], "AGENT_TUI alias should be accepted"
    assert envspec.validate_env({}) == []


@check
def secret_values_are_masked():
    assert "sk-" not in envspec.current_value("LLM_API_KEY", {"LLM_API_KEY": "sk-supersecret-1234"})
    assert envspec.current_value("AGENT_MODEL", {"AGENT_MODEL": "test-model"}) == "test-model"


# ---- provider config plumbing (key/base_url from TOML, ENV wins) ------------
@check
def config_reads_provider_from_toml_but_env_wins():
    c = Config({"provider": {"api_key": "FILEKEY", "base_url": "http://file/v1"}, "agent": {"model": "M"}})
    saved = {k: os.environ.pop(k, None) for k in ("LLM_API_KEY", "LLM_BASE_URL", "AGENT_MODEL", "AGENT_PROVIDER")}
    try:
        assert c.api_key == "FILEKEY" and c.base_url == "http://file/v1" and c.model == "M"
        os.environ["LLM_API_KEY"] = "ENVKEY"
        assert c.api_key == "ENVKEY", "ENV must override the config file"
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


@check
def config_resolves_from_default_provider():
    c = Config({"agent": {"default_provider": "p2", "model": "top"},
                "providers": {"p1": {"api_key": "K1", "model": "m1"},
                              "p2": {"api_key": "K2", "base_url": "http://p2/v1", "model": "m2"}}})
    saved = {k: os.environ.pop(k, None) for k in ("LLM_API_KEY", "LLM_BASE_URL", "AGENT_MODEL", "AGENT_PROVIDER")}
    try:
        assert c.api_key == "K2" and c.base_url == "http://p2/v1" and c.model == "m2", "default provider must resolve"
        assert set(c.providers()) == {"p1", "p2"}
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


@check
def emit_toml_handles_dotted_id_and_newlines():
    from sliceagent.onboarding import _emit_toml
    data = {"agent": {"default_provider": "my.host"},
            "providers": {"my.host": {"api_key": "k\nwith\nnewlines", "base_url": "http://x/v1", "model": "m"}}}
    rt = tomllib.loads(_emit_toml(data))     # must be VALID TOML and round-trip exactly
    assert rt["providers"]["my.host"]["api_key"] == "k\nwith\nnewlines", "newlines in a value must survive"
    assert rt["agent"]["default_provider"] == "my.host", "a dotted provider id must stay one key"


@check
def config_single_provider_resolves_without_default():
    c = Config({"providers": {"only": {"api_key": "K", "model": "m"}}})
    saved = {k: os.environ.pop(k, None) for k in ("LLM_API_KEY", "LLM_BASE_URL", "AGENT_MODEL", "AGENT_PROVIDER")}
    try:
        assert c.api_key == "K" and c.model == "m", "a sole provider should resolve even with no default set"
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# ---- A1: the init wizard (headless) -----------------------------------------
def _seq(*answers):
    it = iter(answers)
    return lambda *_a, **_k: next(it)


class _OkLLM:
    def complete(self, messages, tools):
        from types import SimpleNamespace
        return SimpleNamespace(content="ok", tool_calls=[], finish_reason="stop", usage={})


@check
def init_writes_a_valid_config_on_success():
    home = tempfile.mkdtemp(prefix="init-")
    rc = onboarding.run_init(
        inp=_seq("5", ""),                 # provider=moonshot (menu slot 5), model=default
        getpw=_seq("sk-test-key"),
        llm_factory=lambda model: _OkLLM(),
        home=home)
    assert rc == 0
    path = os.path.join(home, ".sliceagent", "config.toml")
    assert os.path.exists(path)
    data = tomllib.load(open(path, "rb"))
    assert data["providers"]["moonshot"]["api_key"] == "sk-test-key"
    assert data["providers"]["moonshot"]["base_url"] == "https://api.moonshot.cn/v1"
    assert data["providers"]["moonshot"]["model"] == "kimi-k2.7-code"
    assert data["agent"]["default_provider"] == "moonshot"
    assert data["agent"]["model"] == "kimi-k2.7-code"
    # 0600 perms (holds a key) — POSIX only: NTFS has no octal modes and the writer skips fchmod on win32
    if sys.platform != "win32":
        assert (os.stat(path).st_mode & 0o077) == 0, "config with a key must be 0600"
    # and Config resolves the key/model from the default provider
    c = Config(data)
    saved = {k: os.environ.pop(k, None) for k in ("LLM_API_KEY", "LLM_BASE_URL", "AGENT_MODEL", "AGENT_PROVIDER")}
    try:
        assert c.api_key == "sk-test-key" and c.model == "kimi-k2.7-code"
        assert c.base_url == "https://api.moonshot.cn/v1"
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


@check
def init_offers_to_save_after_a_failed_key_test():
    home = tempfile.mkdtemp(prefix="init-fail-")
    def _boom(model):
        raise RuntimeError("401 unauthorized")
    rc = onboarding.run_init(
        inp=_seq("2", "gpt-x", "y"),       # provider=openai, model=gpt-x, then "save anyway? y"
        getpw=_seq("bad-key"),
        llm_factory=_boom,
        home=home)
    assert rc == 0
    data = tomllib.load(open(os.path.join(home, ".sliceagent", "config.toml"), "rb"))
    assert data["providers"]["openai"]["api_key"] == "bad-key" and data["agent"]["model"] == "gpt-x"


@check
def init_custom_provider_prompts_base_url():
    home = tempfile.mkdtemp(prefix="init-custom-")
    rc = onboarding.run_init(
        inp=_seq("6", "https://my.host/v1", "my-model"),   # custom → base url → model (menu: 6 = custom)
        getpw=_seq("k"),
        llm_factory=lambda model: _OkLLM(),
        home=home)
    assert rc == 0
    data = tomllib.load(open(os.path.join(home, ".sliceagent", "config.toml"), "rb"))
    assert data["providers"]["custom"]["base_url"] == "https://my.host/v1"
    assert data["providers"]["custom"]["model"] == "my-model"


@check
def init_aborts_without_a_key():
    home = tempfile.mkdtemp(prefix="init-nokey-")
    rc = onboarding.run_init(inp=_seq("5", ""), getpw=_seq(""), llm_factory=lambda m: _OkLLM(), home=home)
    assert rc == 1, "no key → abort"
    assert not os.path.exists(os.path.join(home, ".sliceagent", "config.toml"))


@check
def write_config_is_atomic_and_cleans_up_on_failure():
    # if the write fails mid-way: the real config is untouched (atomic) AND no key-bearing temp leaks
    home = tempfile.mkdtemp(prefix="cfg-fail-")
    path = onboarding._config_path(home)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("EXISTING\n")
    orig = os.write
    os.write = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    try:
        raised = False
        try:
            onboarding._atomic_write(path, "[providers.x]\napi_key = \"sekret\"\n")
        except OSError:
            raised = True
    finally:
        os.write = orig
    assert raised, "a write failure must propagate"
    leftover = [f for f in os.listdir(os.path.dirname(path)) if f.startswith(".sliceagent-cfg-")]
    assert not leftover, f"temp file (holds the key!) leaked on failure: {leftover}"
    assert open(path).read() == "EXISTING\n", "the existing config must be intact (atomic write)"


@check
def init_merge_keeps_existing_providers():
    home = tempfile.mkdtemp(prefix="init-merge-")
    onboarding.run_init(inp=_seq("5", ""), getpw=_seq("k-moon"), llm_factory=lambda m: _OkLLM(), home=home)
    # 2nd init on the existing config: "Add/update? [Y/n]" → "" (yes) → provider 2 (openai) → model ""
    onboarding.run_init(inp=_seq("", "2", ""), getpw=_seq("k-oai"), llm_factory=lambda m: _OkLLM(), home=home)
    data = tomllib.load(open(os.path.join(home, ".sliceagent", "config.toml"), "rb"))
    assert set(data["providers"]) == {"moonshot", "openai"}, data["providers"]
    assert data["providers"]["moonshot"]["api_key"] == "k-moon", "the first provider must be preserved"
    assert data["agent"]["default_provider"] == "openai", "the newly-added provider becomes default"


@check
def reinit_blank_key_keeps_existing_key_for_the_same_provider():
    # Re-running `sliceagent init` on an ALREADY-CONFIGURED provider and pressing Enter at the key
    # prompt (no retype) must KEEP the saved key/model, not abort — the abort-on-blank behavior
    # (init_aborts_without_a_key) is only correct for a provider with no existing entry.
    home = tempfile.mkdtemp(prefix="init-reblank-")
    onboarding.run_init(inp=_seq("5", ""), getpw=_seq("k-moon"), llm_factory=lambda m: _OkLLM(), home=home)
    # 2nd init: "Add/update? [Y/n]" → "" (yes) → provider 1 (moonshot, already saved) → key "" (blank,
    # keep existing) → model "" (blank, keep existing)
    rc = onboarding.run_init(inp=_seq("", "5", ""), getpw=_seq(""), llm_factory=lambda m: _OkLLM(), home=home)
    assert rc == 0, "blank key on an already-configured provider must NOT abort"
    data = tomllib.load(open(os.path.join(home, ".sliceagent", "config.toml"), "rb"))
    assert data["providers"]["moonshot"]["api_key"] == "k-moon", "the existing key must be kept, not wiped"
    assert data["providers"]["moonshot"]["model"] == "kimi-k2.7-code"


@check
def config_use_switches_default_provider():
    home = tempfile.mkdtemp(prefix="cfg-use-")
    onboarding.run_init(inp=_seq("5", ""), getpw=_seq("k1"), llm_factory=lambda m: _OkLLM(), home=home)
    onboarding.run_init(inp=_seq("", "2", ""), getpw=_seq("k2"), llm_factory=lambda m: _OkLLM(), home=home)
    assert onboarding.run_config(["--use", "moonshot"], home=home) == 0
    data = tomllib.load(open(os.path.join(home, ".sliceagent", "config.toml"), "rb"))
    assert data["agent"]["default_provider"] == "moonshot"
    assert onboarding.run_config(["--use", "nope"], home=home) == 1, "unknown provider → exit 1"


@check
def dispatch_routes_known_subcommands():
    from unittest import mock

    assert onboarding.dispatch(["version"]) == 0
    assert onboarding.dispatch(["help"]) == 0
    assert onboarding.dispatch(["config", "--list"]) == 0
    assert onboarding.dispatch(["config", "--path"]) == 0
    with mock.patch("sliceagent.updater.run_update", return_value=0) as update:
        assert onboarding.dispatch(["update"]) == 0
        assert onboarding.dispatch(["upgrade"]) == 0
        assert update.call_count == 2
    assert onboarding.dispatch(["update", "--mystery"]) == 2
    assert onboarding.dispatch(["bogus"]) == 1   # unknown → usage + non-zero exit (shell-correct)


@check
def arrow_ok_off_without_termios():
    # Windows: termios is absent → the raw-mode arrow menu can't run → _arrow_ok() must be False EVEN on a
    # real tty, so the wizard uses the typed numbered menu and never prints a '↑/↓' hint it can't honor.
    saved_tty, saved_mod = onboarding._tty, sys.modules.get("termios", "MISS")
    onboarding._tty = lambda: True                       # pretend a real tty (like Windows PowerShell)
    sys.modules["termios"] = None                        # `import termios` now raises → simulate Windows
    try:
        assert onboarding._arrow_ok() is False, "no termios → arrows OFF even on a tty"
    finally:
        onboarding._tty = saved_tty
        if saved_mod == "MISS":
            sys.modules.pop("termios", None)
        else:
            sys.modules["termios"] = saved_mod


@check
def typed_menu_picks_by_number_without_advertising_arrows():
    # the Windows repro: a non-arrow run must (a) NOT print '↑/↓', and (b) let a typed number pick the
    # provider she wanted (DeepSeek = 4), instead of Enter silently defaulting to provider 1.
    import io
    home = tempfile.mkdtemp(prefix="init-typed-")
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        rc = onboarding.run_init(inp=_seq("4", ""), getpw=_seq("k"),      # 4 = DeepSeek in the menu order
                                 llm_factory=lambda m: _OkLLM(), home=home)
    finally:
        sys.stdout = old
    assert rc == 0
    out = buf.getvalue()
    assert "↑/↓" not in out, "a run without a working arrow menu must not advertise arrow keys"
    assert "type the number" in out, "it must tell the user to type the number"
    data = tomllib.load(open(os.path.join(home, ".sliceagent", "config.toml"), "rb"))
    assert "deepseek" in data["providers"], f"typing '4' must select DeepSeek, got {list(data['providers'])}"
    assert data["agent"]["default_provider"] == "deepseek"


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
