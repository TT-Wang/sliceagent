"""Named-agent registry (file-defined subagent KINDS, borrowed from Kimi Code / Claude Code):
AgentSpec + load_agents + the generic spawn_agent surface. No model, no pytest.
Run: PYTHONPATH=src python tests/test_agents.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.agents import AgentSpec, BUILTIN_AGENTS, load_agents  # noqa: E402
from memagent.subagent import SubagentHost                          # noqa: E402
from memagent.access import AllAccess, ReadAllAccess                # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def builtins_explorer_readonly_general_writable():
    assert BUILTIN_AGENTS["explorer"].read_only is True
    assert BUILTIN_AGENTS["explorer"].reasoning == "fast"
    assert BUILTIN_AGENTS["general"].read_only is False   # tools=None → inherit full → writable


@check
def read_only_derivation():
    assert AgentSpec("x", tools=("read_file", "grep")).read_only is True
    assert AgentSpec("x", tools=("read_file", "edit_file")).read_only is False   # has a write tool
    assert AgentSpec("x", tools=None).read_only is False                          # inherit-all = writable


@check
def load_agents_builtins_only_when_no_dirs():
    assert set(load_agents([])) == {"explorer", "general"}


@check
def load_agents_parses_user_file_and_overrides_by_name():
    d = tempfile.mkdtemp(prefix="agents-")
    ad = os.path.join(d, "agents"); os.makedirs(ad)
    with open(os.path.join(ad, "reviewer.md"), "w", encoding="utf-8") as f:
        f.write("---\nname: reviewer\ndescription: code review\ntools: read_file, grep\nreasoning: full\n---\n"
                "You are a code reviewer. Find bugs; do not edit.")
    with open(os.path.join(ad, "explorer.md"), "w", encoding="utf-8") as f:    # override a built-in by name
        f.write("---\nname: explorer\ntools: read_file\n---\ncustom explorer prompt")
    reg = load_agents([d])
    rv = reg["reviewer"]
    assert rv.tools == ("read_file", "grep") and rv.reasoning == "full" and rv.read_only is True
    assert "code reviewer" in rv.system_prompt
    assert reg["explorer"].system_prompt == "custom explorer prompt", "user file overrides built-in"
    assert reg["explorer"].tools == ("read_file",)
    assert reg["general"].read_only is False, "untouched built-in remains"


class _Inner:
    def schemas(self):
        return [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    def accesses(self, n, a):
        return []
    def run(self, n, a):
        return "inner"


@check
def spawn_agent_schema_lists_the_roster():
    host = SubagentHost(_Inner(), llm=None, retriever=None, memory=None, policy=None, max_depth=1, depth=0)
    names = [s["function"]["name"] for s in host.schemas()]
    assert "spawn_agent" in names and "spawn_explore" in names and "spawn_subagent" in names
    sa = next(s for s in host.schemas() if s["function"]["name"] == "spawn_agent")
    assert "explorer" in sa["function"]["description"] and "general" in sa["function"]["description"]


@check
def spawn_agent_unknown_is_graceful():
    host = SubagentHost(_Inner(), llm=None, retriever=None, memory=None, policy=None, max_depth=1, depth=0)
    out = host.run("spawn_agent", {"agent": "nope", "task": "x"})
    assert out.startswith("Error: unknown agent"), out


@check
def spawn_agent_access_readonly_vs_writable():
    host = SubagentHost(_Inner(), llm=None, retriever=None, memory=None, policy=None, max_depth=1, depth=0)
    assert isinstance(host.accesses("spawn_agent", {"agent": "explorer"})[0], ReadAllAccess)
    assert isinstance(host.accesses("spawn_agent", {"agent": "general"})[0], AllAccess)


@check
def custom_agent_child_tools_restricted_to_allowlist():
    # a child host built for a custom read-only spec exposes ONLY its allowlist.
    spec = AgentSpec("reviewer", tools=("read_file", "grep"))
    child = SubagentHost(_Inner(), llm=None, retriever=None, memory=None, policy=None,
                         max_depth=1, depth=1, spec=spec)
    names = [s["function"]["name"] for s in child.schemas()]
    assert names == ["read_file"], names   # _Inner only offers read_file; grep absent there but allowlisted


class _InnerWithAsk:
    def schemas(self):
        return [{"type": "function", "function": {"name": n, "parameters": {}}}
                for n in ("read_file", "edit_file", "ask_user")]
    def accesses(self, n, a):
        return []
    def run(self, n, a):
        return f"inner:{n}"


@check
def subagent_cannot_ask_user_but_parent_can():
    # CHILD (any spec) must not be offered ask_user; a general child keeps its other (writable) tools.
    child = SubagentHost(_InnerWithAsk(), llm=None, retriever=None, memory=None, policy=None,
                         max_depth=1, depth=1, spec=BUILTIN_AGENTS["general"])
    names = [s["function"]["name"] for s in child.schemas()]
    assert "ask_user" not in names, names
    assert "edit_file" in names, "a general child keeps its writable tools — only ask_user is barred"
    # defense-in-depth: even a hallucinated ask_user call is barred, not executed (no user prompt → no stall)
    out = child.run("ask_user", {"question": "?"})
    assert out.startswith("Error: a subagent cannot ask the user"), out
    # the top-level agent (parent host, spec=None) CAN still ask the user
    parent = SubagentHost(_InnerWithAsk(), llm=None, retriever=None, memory=None, policy=None,
                          max_depth=1, depth=0)
    assert "ask_user" in [s["function"]["name"] for s in parent.schemas()]


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
