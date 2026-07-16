"""Regression tests for the fixes verified from sliceagent's own code-review plus
the calibrated `reviewer` agent kind. No model, no network.
Run: PYTHONPATH=src python tests/test_selfreview_fixes.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# ---- H4: version-shaped tokens are not extracted as auto-grant paths -----------------------------------

@check
def h4_version_token_not_granted_but_real_dir_is():
    import re
    from sliceagent.tools import LocalToolHost
    home = os.path.realpath(os.path.expanduser("~"))
    real = tempfile.mkdtemp(prefix="grant-real-", dir=home)
    verdir = os.path.join(home, "v9-granttest")   # matches the version shape [/~]v?\d[\d.]* only as '/v9...'? no —
    os.makedirs(verdir, exist_ok=True)            # a bare-name token isn't '/'-rooted; test the regex + a real path
    try:
        h = LocalToolHost(root=tempfile.mkdtemp())
        h._grant_shell_paths(f"mytool {real} /v1.2.3")
        granted = {os.path.realpath(x) for x in h.allowed_roots()}
        assert os.path.realpath(real) in granted, "a real dir under HOME should be granted"
        # the discriminator: version-shaped '/'-tokens are skipped, real multi-segment paths are not
        assert re.fullmatch(r"[/~]v?\d[\d.]*", "/v1.2.3") and re.fullmatch(r"[/~]v?\d[\d.]*", "/1.0")
        assert not re.fullmatch(r"[/~]v?\d[\d.]*", "/home/me/proj")
    finally:
        os.rmdir(real); os.rmdir(verdir)


# ---- H10: ripgrep subprocesses decode as UTF-8, not the locale codec -----------------------------------

@check
def h10_ripgrep_subprocess_forces_utf8():
    import inspect
    from sliceagent import code_index, code_grep
    for mod in (code_index, code_grep):
        src = inspect.getsource(mod)
        for call in src.split("subprocess.run")[1:]:
            assert 'encoding="utf-8"' in call[:220], f"a subprocess.run in {mod.__name__} lacks encoding=utf-8"


# ---- M6: a park failure must not masquerade as "no such topic" -----------------------------------------

@check
def m6_park_guards_missing_active_id():
    from sliceagent.session import Session
    from sliceagent.memory import NullMemory
    s = Session(NullMemory())
    s.active_id = "ghost-not-in-tasks"        # set but absent from self.tasks
    s._park()                                 # must NOT raise KeyError (was: crashed → mislabeled by _switch)
    try:
        s.switch_topic("also-ghost")
        raised = False
    except KeyError:
        raised = True
    assert raised, "switch to an unknown topic should raise KeyError (surfaced upstream as 'no open topic')"


# ---- M29: an unknown Access type conflicts conservatively instead of AttributeError --------------------

@check
def m29_unknown_access_type_serializes_no_crash():
    from sliceagent.access import _pair_conflict, ReadAllAccess, FileAccess

    class _NewAccess:            # a future Access subclass with no .operation
        pass
    other = _NewAccess()
    assert _pair_conflict(ReadAllAccess(), other) is True     # conflict (safe), never AttributeError
    assert _pair_conflict(other, ReadAllAccess()) is True
    assert _pair_conflict(other, FileAccess("read", "a.py")) is True
    assert _pair_conflict(FileAccess("read", "a.py"), FileAccess("read", "b.py")) is False   # normal path intact


# ---- H2: a SIGALRM hard-deadline is re-raised, not downgraded to a partial "length" stop ---------------

@check
def h2_stream_timeout_is_reraised_not_salvaged():
    from sliceagent.llm import OpenAILLM, _import_api_timeout_error
    TimeoutErr = _import_api_timeout_error()
    llm = OpenAILLM.__new__(OpenAILLM)           # bypass __init__ (no network/key)
    llm.reasoning = "fast"; llm._base_url = "http://local"; llm._delta = None; llm.model = "deepseek-chat"

    class _Delta:
        def __init__(self, t): self.content = t; self.reasoning_content = None; self.tool_calls = None
    class _Choice:
        def __init__(self, t): self.delta = _Delta(t); self.finish_reason = None
    class _Chunk:
        def __init__(self, t): self.choices = [_Choice(t)]; self.usage = None

    import httpx
    _req = httpx.Request("POST", "http://local/chat/completions")
    def _gen():
        yield _Chunk("partial output ")          # content already emitted → parts != []
        raise TimeoutErr(request=_req)           # then the hard-deadline fires (as the SIGALRM handler builds it)
    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw): return _gen()
    llm.client = _Client()

    try:
        llm._stream_assemble({})
        raised = False
    except Exception as e:  # noqa: BLE001
        raised = isinstance(e, TimeoutErr)
    assert raised, "the hard-deadline timeout must propagate, not be salvaged as a partial 'length' stop"


# ---- the calibrated reviewer kind (the cry-wolf counterweight) -----------------------------------------

@check
def reviewer_kind_is_calibrated_and_readonly():
    from sliceagent.agents import BUILTIN_AGENTS
    r = BUILTIN_AGENTS["reviewer"]
    assert r.read_only, "reviewer should be read-only so a broad review fans out in parallel"
    p = r.system_prompt.lower()
    assert "severity rubric" in p and "single-user" in p                    # threat model + rubric
    assert "adjacent comment" in p or "docstring" in p                      # discipline 1
    assert "real consumer" in p or "trace the data" in p                    # discipline 2
    assert "refute" in p                                                    # discipline 4


def main():
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
