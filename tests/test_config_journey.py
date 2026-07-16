"""The clear config journey (user-designed): /config manages providers in-session, and /model shows
ONLY configured providers' models — picking one switches model + endpoint + key together.
Offline: stub objects, no network, no pytest. Run: PYTHONPATH=src python tests/test_config_journey.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.config import Config                      # noqa: E402
from sliceagent.llm import OpenAILLM                      # noqa: E402
from sliceagent.tui import _SLASH, _model_candidates, select_model_reasoning  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _Llm:
    model = "deepseek-chat"
    _base_url = "https://api.deepseek.com/v1"
    reasoning = "full"


_CFG = Config({"providers": {
    "deepseek": {"api_key": "k1", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "openrouter": {"api_key": "k2", "base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-5.5"},
    "keyless": {"model": "phantom-model"},               # no api_key → NOT usable → must not appear
}})


@check
def model_menu_lists_only_configured_providers():
    cands = _model_candidates(_Llm(), _CFG)
    pids = {pid for _, _, pid in cands if pid}
    models = [m for m, _, _ in cands]
    assert pids == {"deepseek", "openrouter"}, f"only keyed providers may appear, got {pids}"
    assert "phantom-model" not in models, "a provider without a key must not contribute models"
    assert "deepseek-chat" in models and "openai/gpt-5.5" in models
    assert "anthropic/claude-sonnet-5" in models, "configured providers contribute their suggestions too"


@check
def every_candidate_carries_its_provider_for_endpoint_switch():
    for m, _grp, pid in _model_candidates(_Llm(), _CFG):
        assert pid in ("deepseek", "openrouter"), f"{m} must map to a configured provider, got {pid!r}"


@check
def env_current_model_is_labeled_honestly_not_as_a_provider():
    class _EnvLlm:                                    # env model NOT in any configured provider's list
        model = "gpt-5.5"
        _base_url = "https://api.deepseek.com/v1"     # deepseek endpoint must NOT make it say "deepseek"
        reasoning = "full"
    cands = _model_candidates(_EnvLlm(), _CFG)
    m, grp, pid = cands[0]
    assert m == "gpt-5.5" and pid is None
    assert grp == "current (env)", f"env model must not masquerade as a configured provider, got {grp!r}"


@check
def no_providers_configured_falls_back_to_known_set():
    cands = _model_candidates(_Llm(), Config({}))
    assert cands, "env-only setups still need a menu"
    assert all(pid is None for _, _, pid in cands), "fallback entries have no provider to rebind to"
    assert any(m == "deepseek-chat" for m, _, _ in cands)


@check
def switch_rebinds_endpoint_and_key():
    saved = {k: os.environ.pop(k, None) for k in ("AGENT_PROXY", "HTTPS_PROXY", "HTTP_PROXY")}
    try:
        llm = OpenAILLM(model="deepseek-chat", api_key="k1", base_url="https://api.deepseek.com/v1")
        assert llm._base_url == "https://api.deepseek.com/v1"
        llm.switch(model="openai/gpt-5.5", base_url="https://openrouter.ai/api/v1", api_key="k2")
        assert llm.model == "openai/gpt-5.5"
        assert llm._base_url == "https://openrouter.ai/api/v1"
        assert str(llm.client.base_url).startswith("https://openrouter.ai/api/v1"), \
            "the OpenAI client itself must be rebuilt against the new endpoint"
        assert llm.client.api_key == "k2", "the key must follow the provider"
        # base_url="" → SDK default endpoint (OpenAI)
        llm.switch(model="gpt-5.5", base_url="", api_key="k3")
        assert llm._base_url == "" and "openrouter" not in str(llm.client.base_url)
        # model-only switch keeps the endpoint (the classic /model <name> path)
        llm.switch(model="gpt-5")
        assert llm.model == "gpt-5" and llm.client.api_key == "k3"
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


@check
def selector_returns_the_provider_triple():
    if __import__("sys").platform == "win32":
        return  # prompt_toolkit needs a real Windows console; CI's Git-Bash runner has none
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    with create_pipe_input() as pinp:
        pinp.send_text("\r\r")                     # Enter (model) + Enter (reasoning)
        got = select_model_reasoning(_Llm(), _CFG, pt_input=pinp, pt_output=DummyOutput())
    assert got is not None and len(got) == 3, f"expected (model, reasoning, pid), got {got!r}"
    model, _reasoning, pid = got
    assert pid in ("deepseek", "openrouter") and isinstance(model, str)


@check
def config_command_is_in_the_slash_palette():
    assert "/config" in _SLASH, "/config must be discoverable in the palette"


@check
def boot_binds_pinned_provider_as_a_unit_never_default_providers_base():
    """The cross-wiring bug: prefs pin openai (no base_url = SDK default) while default_provider is
    deepseek → boot must NOT pair the openai key with the deepseek base_url."""
    from sliceagent.cli import _env_from_config
    c = Config({"providers": {
        "openai": {"api_key": "sk-openai", "model": "gpt-5.5"},
        "deepseek": {"api_key": "sk-ds", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    }, "agent": {"default_provider": "deepseek"}})
    saved = {k: os.environ.pop(k, None) for k in ("LLM_API_KEY", "LLM_BASE_URL")}
    try:
        _env_from_config(c, "openai")
        assert os.environ.get("LLM_API_KEY") == "sk-openai"
        assert not os.environ.get("LLM_BASE_URL"), \
            f"openai pin must leave base at SDK default, got {os.environ.get('LLM_BASE_URL')!r}"
        os.environ.pop("LLM_API_KEY", None); os.environ.pop("LLM_BASE_URL", None)
        _env_from_config(c, "deepseek")                       # a pin WITH base_url binds its own pair
        assert os.environ.get("LLM_API_KEY") == "sk-ds"
        assert os.environ.get("LLM_BASE_URL") == "https://api.deepseek.com/v1"
        os.environ.pop("LLM_API_KEY", None); os.environ.pop("LLM_BASE_URL", None)
        _env_from_config(c, "gone")                           # deleted pin → default provider's own pair
        assert os.environ.get("LLM_API_KEY") == "sk-ds" and "deepseek" in os.environ.get("LLM_BASE_URL", "")
    finally:
        for k in ("LLM_API_KEY", "LLM_BASE_URL"):
            os.environ.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


@check
def save_prefs_none_deletes_a_stale_pin():
    import json
    import stat
    import tempfile
    import sliceagent.config as _cfgmod
    with tempfile.TemporaryDirectory() as tmp:
        orig = _cfgmod._prefs_path
        _cfgmod._prefs_path = lambda: os.path.join(tmp, "prefs.json")
        try:
            _cfgmod.save_prefs({"model": "a", "provider": "openrouter"})
            _cfgmod.save_prefs({"model": "b", "provider": None})       # typed /model → pin removed
            got = json.load(open(os.path.join(tmp, "prefs.json")))
            assert got.get("model") == "b" and "provider" not in got, f"stale pin survived: {got}"
            if os.name != "nt":
                assert stat.S_IMODE(os.stat(tmp).st_mode) == 0o700
                assert stat.S_IMODE(os.stat(os.path.join(tmp, "prefs.json")).st_mode) == 0o600
            assert not [name for name in os.listdir(tmp) if name.endswith(".tmp")], \
                "atomic preference writes must not leave fixed or unique temp fragments"
        finally:
            _cfgmod._prefs_path = orig


@check
def loading_user_config_repairs_modes_without_changing_project_config():
    if os.name == "nt":
        return
    import stat
    import tempfile
    import sliceagent.config as config_module
    with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as project:
        user_dir = os.path.join(home, ".sliceagent")
        os.makedirs(user_dir)
        user_path = os.path.join(user_dir, "config.toml")
        project_path = os.path.join(project, "sliceagent.toml")
        for path in (user_path, project_path):
            with open(path, "w", encoding="utf-8") as stream:
                stream.write('[agent]\nmodel = "m"\n')
            os.chmod(path, 0o644)
        original = config_module._config_files
        config_module._config_files = lambda _root=None: [user_path, project_path,
                                                           os.path.join(project, ".sliceagent", "config.toml")]
        try:
            config_module.Config.load(project)
        finally:
            config_module._config_files = original
        assert stat.S_IMODE(os.stat(user_dir).st_mode) == 0o700
        assert stat.S_IMODE(os.stat(user_path).st_mode) == 0o600
        assert stat.S_IMODE(os.stat(project_path).st_mode) == 0o644


@check
def openrouter_models_offer_high_reasoning_in_the_menu():
    from sliceagent.tui import _reasoning_levels
    names = [n for n, _ in _reasoning_levels("openai/gpt-5.5", "https://openrouter.ai/api/v1")]
    assert "high" in names, "openrouter's unified reasoning honors high — the menu must offer it"
    assert "max" not in names, "openrouter maps max→high; offering max would lie"
    assert [n for n, _ in _reasoning_levels("deepseek-chat", "https://api.deepseek.com/v1")] == ["fast", "full"]


@check
def switch_closes_the_replaced_client():
    saved = {k: os.environ.pop(k, None) for k in ("AGENT_PROXY", "HTTPS_PROXY", "HTTP_PROXY")}
    try:
        llm = OpenAILLM(model="deepseek-chat", api_key="k1", base_url="https://api.deepseek.com/v1")
        first = llm.client
        llm.switch(model="openai/gpt-5.5", base_url="https://openrouter.ai/api/v1", api_key="k2")
        hc = getattr(first, "_client", None)                  # the underlying httpx client
        assert hc is None or hc.is_closed, "the replaced client's connection pool must be closed"
        assert llm.client is not first
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


@check
def config_save_backs_up_a_corrupt_config_instead_of_erasing_keys():
    import tempfile
    from sliceagent.onboarding import _save_provider
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "config.toml")
        with open(p, "w") as f:
            f.write('[providers.deepseek]\napi_key = "sk-PRECIOUS"\nmodel = "deepseek-chat"\n')
        with open(p, "w") as f:
            f.write('[providers.deepseek\napi_key = "sk-PRECIOUS"\n')   # one-char TOML typo
        bak = _save_provider(p, pid="openai", model="gpt-5.5", api_key="sk-new", base_url="")
        assert bak and os.path.exists(bak), "an unparseable config must be moved aside, not erased"
        assert "sk-PRECIOUS" in open(bak).read(), "the old key must survive in the backup"
        assert "sk-new" in open(p).read(), "the new provider must still be written"


@check
def config_read_degrades_on_non_utf8_instead_of_crashing():
    import tempfile
    from sliceagent.onboarding import _read_config, run_config
    with tempfile.TemporaryDirectory() as d:
        cfgdir = os.path.join(d, ".sliceagent"); os.makedirs(cfgdir)
        p = os.path.join(cfgdir, "config.toml")
        with open(p, "wb") as f:
            f.write(b'[agent]\nmodel = "caf\xe9"\n')            # non-UTF-8 byte
        assert _read_config(p) == {}, "_read_config must degrade, not raise UnicodeDecodeError"
        assert run_config([], home=d) == 0, "`sliceagent config` must not crash on a non-utf8 file"


@check
def config_show_tolerates_a_scalar_under_providers():
    import tempfile
    from sliceagent.onboarding import run_config
    with tempfile.TemporaryDirectory() as d:
        cfgdir = os.path.join(d, ".sliceagent"); os.makedirs(cfgdir)
        with open(os.path.join(cfgdir, "config.toml"), "w") as f:
            f.write('[providers]\nfoo = "bar"\n')               # scalar, not a table
        assert run_config([], home=d) == 0, "a scalar under [providers] must not crash `config`"


@check
def wizard_typed_fallback_rejects_unknown_provider_choice():
    """A typed 'openai' must configure OPENAI (name accepted); pure garbage must abort, not silently
    become OpenRouter."""
    import tempfile
    from sliceagent.onboarding import run_init
    with tempfile.TemporaryDirectory() as tmp:
        feed = iter(["garbage", "junk", "nope"])              # 3 strikes → abort
        rc = run_init(inp=lambda *_: next(feed), getpw=lambda *_: "k", home=tmp)
        assert rc == 1, f"3 invalid choices must abort, got rc={rc}"
    with tempfile.TemporaryDirectory() as tmp:
        feed = iter(["openai", "my-model", ""])               # name accepted; blank key → abort path
        rc = run_init(inp=lambda *_: next(feed), getpw=lambda *_: "", home=tmp)
        assert rc == 1, "no key entered must abort"


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
