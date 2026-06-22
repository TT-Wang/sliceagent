"""Skills borrows (from Kimi agent-core/skill): when-to-use routing in the catalog, $ARGUMENTS / $N
parameter expansion, and a scan-depth bound. No model, no pytest.
Run: PYTHONPATH=src python tests/test_skill_metadata.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.skills import SkillManager, expand_skill_args, make_skill_tool  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _skills_root(files: dict) -> str:
    d = tempfile.mkdtemp(prefix="skills-")
    root = os.path.join(d, "skills")
    os.makedirs(root)
    for name, text in files.items():
        with open(os.path.join(root, name), "w", encoding="utf-8") as f:
            f.write(text)
    return root


@check
def expand_arguments_and_positionals():
    body = "Run the suite: pytest $1 then report $ARGUMENTS"
    out = expand_skill_args(body, "tests/ -v")
    assert out == "Run the suite: pytest tests/ then report tests/ -v", out
    # no placeholders → unchanged
    assert expand_skill_args("no params here", "x y") == "no params here"
    # unbalanced quotes → graceful fallback, no crash
    assert "$1" not in expand_skill_args("$1", 'a"b')


@check
def when_to_use_parsed_and_rendered():
    root = _skills_root({"deploy.md": "---\nname: deploy\ndescription: ship it\n"
                                       "when-to-use: when the user asks to release\n---\nstep 1: build\n"})
    mgr = SkillManager([root])
    assert mgr._skills["deploy"].when_to_use == "when the user asks to release"
    tool = make_skill_tool(mgr)
    assert "when: when the user asks to release" in tool.schema["function"]["description"]
    assert "arguments" in tool.schema["function"]["parameters"]["properties"]


@check
def skill_tool_expands_on_invoke():
    root = _skills_root({"greet.md": "---\nname: greet\ndescription: greet\n---\nSay hello to $1 ($ARGUMENTS)\n"})
    tool = make_skill_tool(SkillManager([root]))
    out = tool.handler({"name": "greet", "arguments": "alice extra"})
    assert out.strip() == "Say hello to alice (alice extra)", out


@check
def deep_tree_does_not_explode():
    root = _skills_root({"a.md": "---\nname: a\ndescription: d\n---\nbody\n"})
    deep = root
    for i in range(20):                       # nest well past the depth bound
        deep = os.path.join(deep, f"d{i}")
    os.makedirs(deep)
    with open(os.path.join(deep, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write("---\nname: buried\ndescription: x\n---\nb\n")
    mgr = SkillManager([root])               # must not hang / error; shallow skill still found
    assert "a" in mgr.names() and "buried" not in mgr.names()


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
