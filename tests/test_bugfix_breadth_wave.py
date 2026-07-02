"""Regression tests for the breadth wave: #55 UsageRecorder journals the typed cache breakdown (from
StepEnd) + total_usage aggregates it, #57 catalog covers o5/o6/gpt-6 reasoning models, #58 agent
frontmatter accepts an inline YAML `[list]` of tools. No model, no pytest.
Run: PYTHONPATH=src python tests/test_bugfix_breadth_wave.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.records import Journal, UsageRecorder, total_usage  # noqa: E402
from sliceagent.model_catalog import capability  # noqa: E402
from sliceagent.agents import _parse_agent_md  # noqa: E402
from sliceagent.events import StepEnd, TurnEnd, TurnInterrupted  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def usage_recorder_journals_cache_breakdown():  # #55
    j = Journal("s1", root=tempfile.mkdtemp(prefix="rec-"))
    rec = UsageRecorder(j, model="m1")
    rec(StepEnd(1, {"input_other": 100, "input_cache_read": 900, "completion_tokens": 50,
                    "prompt_tokens": 1000}, "tool_use"))
    rec(TurnEnd("end_turn", 1, {"prompt_tokens": 1000, "completion_tokens": 50}))
    # a parked turn also records (accumulator path)
    rec(StepEnd(2, {"input_other": 30, "input_cache_read": 0, "completion_tokens": 5}, "max_tokens"))
    rec(TurnInterrupted("max_tokens", message="cut"))
    tot = total_usage(j)["m1"]
    assert tot["input_other"] == 130, tot           # 100 + 30 (was always 0 before #55)
    assert tot["input_cache_read"] == 900, tot
    assert tot["output"] == 55, tot                 # 50 + 5 from completion_tokens fallback
    assert tot["turns"] == 2, tot


@check
def catalog_covers_future_reasoning_models():  # #57
    for m in ("o5-preview", "o6-mini", "gpt-6", "o4-mini", "gpt-5.5"):
        assert capability(m).supports_reasoning_effort, m
        assert capability(m).tokens_param == "max_completion_tokens", m
    assert not capability("kimi-k2.7-code").supports_reasoning_effort


@check
def agent_frontmatter_accepts_yaml_list():  # #58
    d = tempfile.mkdtemp(prefix="agent-")
    for body, label in (("tools: [read_file, grep]", "bracket"),
                        ("tools: read_file, grep", "comma"),
                        ('tools: ["read_file", grep]', "quoted")):
        p = os.path.join(d, f"a_{label}.md")
        with open(p, "w") as f:
            f.write(f"---\nname: ex\ndescription: x\n{body}\n---\nyou are ex\n")
        spec = _parse_agent_md(p)
        assert spec.tools == ("read_file", "grep"), (label, spec.tools)


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
