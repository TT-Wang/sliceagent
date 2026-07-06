"""The agent must know its OWN model — the harness has the fact (llm.model), so the seed surfaces it as
OBSERVED ground truth. Without it the agent confabulates a self-identity (DeepSeek models, trained on
Claude/GPT output, claim to BE Claude). No model, no pytest. Run: PYTHONPATH=src python tests/test_model_identity.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.pfc import Slice          # noqa: E402
from sliceagent.seed import make_build_slice  # noqa: E402
from sliceagent.tools import LocalToolHost     # noqa: E402
from sliceagent.code_index import make_code_index  # noqa: E402
from sliceagent.memory import NullMemory         # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _system(model_id=""):
    wd = tempfile.mkdtemp()
    s = Slice(); s.reset("hi")
    host = LocalToolHost(root=wd)
    build = make_build_slice(s, host, make_code_index(wd), NullMemory(), "hi", model_id=model_id)
    return build()[0]["content"]


@check
def identity_line_present_when_model_given():
    sysmsg = _system("deepseek-reasoner")
    assert "deepseek-reasoner" in sysmsg, "the actual model id must be in the system prompt"
    assert "which model" in sysmsg.lower() and "do not guess" in sysmsg.lower(), \
        "must explicitly tell the agent to state THIS model, not guess"


@check
def identity_line_absent_when_no_model():
    # backward-compatible: eval harnesses / subagents that don't pass model_id get no identity line
    sysmsg = _system("")
    assert "which model / LLM you are" not in sysmsg


@check
def identity_reflects_a_switch():
    # /model switch re-calls make_build_slice with the new llm.model → the line updates
    assert "gpt-5.5" in _system("gpt-5.5")
    assert "claude-opus-4-8" in _system("claude-opus-4-8")


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
