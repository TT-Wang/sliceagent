"""Subagent artifact archive + recall (subagents/ virtual FS). A child seals a structured artifact; the
parent recalls its FULL detail via read_file("subagents/sub-N.md") — the refinement handle behind the
bounded digest. Race-safe sequential ids (parallel explorers archive concurrently). No model, no network.
Run: PYTHONPATH=src python tests/test_subagent_artifacts.py
"""
import os
import sys
import tempfile
import threading

os.environ["SLICEAGENT_VAULT"] = tempfile.mkdtemp(prefix="subidx-")   # hermetic: FTS index stays in tmp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.hippocampus import HippocampusMixin, SubagentFS  # noqa: E402
from sliceagent.memory import NullMemory  # noqa: E402
from sliceagent.subagent import run_subagent, SubagentHost  # noqa: E402
from sliceagent.agents import BUILTIN_AGENTS  # noqa: E402
from sliceagent.retriever import NullRetriever  # noqa: E402
from sliceagent.registry import ToolIntentEffect  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _Mem(HippocampusMixin, NullMemory):
    """Real archive (HippocampusMixin over a temp vault) + NullMemory's inert read side (recall/manifest) so
    make_build_slice has every method it calls — MRO puts the real subagent-archive methods first."""
    is_durable = False
    def __init__(self, vault):
        self._vault = vault
        self._idx_lock = threading.Lock()


def _mem():
    return _Mem(tempfile.mkdtemp(prefix="subvault-"))


def _art(kind, report, findings=(), status="ok", steps=3):
    return {"kind": kind, "task": "t", "status": status, "steps": steps,
            "report": report, "findings": list(findings), "change_set": [], "files": ["a.py"], "coverage": ""}


@check
def archive_roundtrip_assigns_sequential_ids():
    m = _mem()
    ids = [m.append_subagent_artifact("s1", _art("explorer", f"report {i}")) for i in range(3)]
    assert ids == ["sub-1", "sub-2", "sub-3"], ids
    arts = m.read_subagent_artifacts("s1")
    assert [a["id"] for a in arts] == ["sub-1", "sub-2", "sub-3"]
    assert arts[1]["artifact"]["report"] == "report 1"


@check
def ids_are_race_safe_under_parallel_appends():
    m = _mem()
    N = 20
    def _append(i):
        m.append_subagent_artifact("s1", _art("explorer", f"r{i}"))
    threads = [threading.Thread(target=_append, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ids = [a["id"] for a in m.read_subagent_artifacts("s1")]
    assert len(ids) == N, f"expected {N}, got {len(ids)}"
    assert sorted(ids) == sorted(f"sub-{i}" for i in range(1, N + 1)), f"non-unique/gapped ids: {sorted(ids)}"


@check
def launch_order_survives_reverse_completion_order():
    """sub-N remains archive/completion ordered, while user ordinals retain spawn order."""
    m = _mem()
    host = SubagentHost(
        _ToolsHost(), llm=_FakeLLM("unused"), retriever=NullRetriever(), memory=m,
        policy=None, session_id="s1",
    )
    first_id, first = host._next_artifact_identity(task_id="task-1", parent_id="turn-1")
    second_id, second = host._next_artifact_identity(task_id="task-1", parent_id="turn-1")
    assert (first_id, second_id) == ("", "")
    assert (first, second) == (1, 2)

    later = _art("explorer", "second child completed first")
    later["launch_ordinal"] = second
    earlier = _art("explorer", "first child completed second")
    earlier["launch_ordinal"] = first
    assert m.append_subagent_artifact("s1", later) == "sub-1"
    assert m.append_subagent_artifact("s1", earlier) == "sub-2"

    fs = SubagentFS(m, "s1")
    assert "launch-order: 2" in fs.read_file("subagents/sub-1.md")
    assert "launch-order: 1" in fs.read_file("subagents/sub-2.md")


@check
def delegation_effect_metadata_distinguishes_readers_from_writers():
    host = SubagentHost(
        _ToolsHost(), llm=_FakeLLM("unused"), retriever=NullRetriever(), memory=NullMemory(),
        policy=None, agents=BUILTIN_AGENTS,
    )
    assert host.resolve_intent_effect("spawn_agent", {"agent": "explorer"}) \
        is ToolIntentEffect.OBSERVE
    assert host.resolve_intent_effect("spawn_agent", {"agent": "general"}) \
        is ToolIntentEffect.EXTERNAL
    assert host.resolve_intent_effect("spawn_agent", {"agent": "not-real"}) \
        is ToolIntentEffect.UNKNOWN


@check
def subagentfs_recalls_full_detail():
    m = _mem()
    m.append_subagent_artifact("s1", _art("reviewer", "FULL REPORT with detail X", findings=["bug in feishu.ts"]))
    fs = SubagentFS(m, "s1")
    full = fs.read_file("subagents/sub-1.md")
    assert "FULL REPORT with detail X" in full and "bug in feishu.ts" in full, full
    idx = fs.read_file("subagents/index.md")
    assert "sub-1.md" in idx and "reviewer" in idx, idx


@check
def subagentfs_missing_and_bad_paths_are_helpful():
    m = _mem()
    m.append_subagent_artifact("s1", _art("explorer", "r"))
    fs = SubagentFS(m, "s1")
    assert "no such subagent report" in fs.read_file("subagents/sub-99.md")
    assert "not a subagent report" in fs.read_file("subagents/whatever.txt")
    # a fresh session with no delegated work
    assert "no subagent reports yet" in SubagentFS(m, "empty").read_file("subagents/index.md")


@check
def nullmemory_is_inert():
    n = NullMemory()
    assert n.append_subagent_artifact("s1", _art("explorer", "r")) == ""
    assert n.read_subagent_artifacts("s1") == []


# ---- end-to-end: bounded return AND full recall (the moat + no-detail-loss, in one check) --------------

class _Resp:
    def __init__(self, content):
        self.content, self.tool_calls, self.finish_reason, self.usage = content, [], "stop", {}


class _FakeLLM:
    """Ends the child turn in one step with a final assistant text (its report)."""
    def __init__(self, text):
        self._text = text
        self.reasoning = "fast"
    def complete(self, messages, schemas):
        return _Resp(self._text)


class _ToolsHost:
    def schemas(self): return []
    def root(self): return "/tmp/ws"
    def accesses(self, name, args): return []
    def run(self, name, args): return ""
    def read_text(self, path): return ""


@check
def run_subagent_returns_bounded_digest_and_archives_FULL_report():
    # a LONG child report with a distinctive TAIL conclusion (~800 chars)
    report = ("Detailed step-by-step analysis of the outreach flow. " * 15
              + "FINAL CONCLUSION: the bug is at feishu.ts:109 (falsy timestamp).")
    assert len(report) > 500
    mem = _mem()
    llm = _FakeLLM(report)
    out = run_subagent("investigate the outreach flow", tools=_ToolsHost(), llm=llm,
                       retriever=NullRetriever(), memory=mem, policy=None, max_steps=2,
                       read_only=True, session_id="s1")

    # (1) MOAT: the parent's tool result is a BOUNDED digest + a recall handle — NOT the full transcript.
    assert 'read_file("subagents/sub-1.md")' in out, out
    # The digest now carries two explicitly distinct layers: a bounded child-interpretation excerpt and one
    # bounded primary observation (or an explicit unavailable marker). The larger ceiling buys provenance at
    # synthesis time while remaining O(1) per child; the full report still stays behind the handle.
    assert len(out) < 1400, f"parent return not bounded: {len(out)}"
    assert "presentation-truncated" in out
    assert "primary observation: unavailable" in out
    assert "FINAL CONCLUSION" in out, "the bounded head+tail view must preserve the report's conclusion"
    assert "presentation omitted" in out, "the missing middle must be explicit and refinable"

    # (2) NO DETAIL LOSS: the FULL report is archived and recallable — the tail survives the seal.
    arts = mem.read_subagent_artifacts("s1")
    assert len(arts) == 1 and arts[0]["artifact"]["report"] == report
    full = SubagentFS(mem, "s1").read_file("subagents/sub-1.md")
    assert "FINAL CONCLUSION: the bug is at feishu.ts:109" in full, "recall must return the full detail"


@check
def run_subagent_without_session_stays_inline_backcompat():
    # no session_id → not archived; falls back to the pre-artifact inline summary (no recall handle).
    out = run_subagent("do a thing", tools=_ToolsHost(), llm=_FakeLLM("did the thing"),
                       retriever=NullRetriever(), memory=_mem(), policy=None, max_steps=2,
                       read_only=True, session_id="")
    assert "read_file(\"subagents/" not in out and out.startswith("[explore "), out


@check
def child_cannot_read_parent_reserved_namespaces():
    # ISOLATION (bug-hunt #2): a CHILD shares the base host, so without a guard it could page the parent's
    # trajectory (history/) or a sibling's sealed artifact (subagents/). Both must be blocked; real files pass.
    child = SubagentHost(_ToolsHost(), llm=None, retriever=None, memory=None, policy=None,
                         max_depth=1, depth=1, spec=BUILTIN_AGENTS["explorer"])
    for p in ("subagents/sub-1.md", "subagents/index.md", "history/turn-1.md", "./history/"):
        r = child.run("read_file", {"path": p})
        assert "private namespace" in r, f"child reached reserved ns {p!r}: {r!r}"
    # a real project file is NOT blocked (delegates to inner → "")
    assert child.run("read_file", {"path": "pkg/mod.py"}) == ""
    # a PARENT host (spec=None) is not a child → no isolation block (it OWNS these namespaces)
    parent = SubagentHost(_ToolsHost(), llm=None, retriever=None, memory=None, policy=None,
                          max_depth=1, depth=0, spec=None)
    assert "private namespace" not in parent.run("read_file", {"path": "subagents/sub-1.md"})


@check
def archived_artifact_redacts_secrets():
    # round-2 #1: a child that quotes a secret into its report must NOT persist it verbatim on disk —
    # append_subagent_artifact must redact like append_episode.
    m = _mem()
    secret = "AKIAIOSFODNN7EXAMPLE"
    m.append_subagent_artifact("s1", _art("explorer", f"config uses AWS key {secret} for access; rest of analysis"))
    stored = m.read_subagent_artifacts("s1")[0]["artifact"]["report"]
    assert secret not in stored, f"secret persisted verbatim: {stored!r}"
    assert "rest of analysis" in stored, "non-secret text must survive redaction"
    assert secret not in SubagentFS(m, "s1").read_file("subagents/sub-1.md")


@check
def child_cannot_search_the_parents_history():
    # round-2 #2: search_history is bound to the PARENT session (its this-session mode previews the parent's
    # turns) → a child using it leaks the parent's trajectory. Block in run() AND drop from the child schemas.
    class _InnerWithSearch:
        def schemas(self):
            return [{"type": "function", "function": {"name": n, "parameters":
                     {"type": "object", "properties": {}, "required": []}}} for n in ("read_file", "search_history")]
        def root(self): return "/tmp/ws"
        def accesses(self, n, a): return []
        def run(self, n, a): return "RAN:" + n
        def read_text(self, p): return ""
    child = SubagentHost(_InnerWithSearch(), llm=None, retriever=None, memory=None, policy=None,
                         max_depth=1, depth=1, spec=BUILTIN_AGENTS["explorer"])
    names = [s.get("function", {}).get("name") for s in child.schemas()]
    assert "search_history" not in names and "read_file" in names, names   # dropped from schemas, reads kept
    assert "private namespace" in child.run("search_history", {"query": "auth flow"})   # blocked at runtime too


@check
def read_subagent_artifacts_tolerates_non_dict_lines():
    # bug-hunt #6: a scalar/list JSONL line must not reach SubagentFS's .get() → AttributeError.
    m = _mem()
    m.append_subagent_artifact("s1", _art("explorer", "good report"))
    path = os.path.join(m._vault, "subagents", "s1.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write("5\n"); f.write('"a scalar string"\n'); f.write("[1,2,3]\n")   # corrupt/non-dict lines
    arts = m.read_subagent_artifacts("s1")
    assert len(arts) == 1 and arts[0]["id"] == "sub-1", arts   # scalars dropped, dict kept
    # every SubagentFS path must survive the corrupt file without raising
    fs = SubagentFS(m, "s1")
    assert "good report" in fs.read_file("subagents/sub-1.md")
    assert "sub-1.md" in fs.read_file("subagents/index.md")
    assert isinstance(fs.grep("report"), str)


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
