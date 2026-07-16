"""Roster (v3): instance identity + verbatim-brief provenance on subagent seals. A named delegation
('auth-explorer') is addressable as subagents/<name>.md (latest job by that identity); every artifact
carries the VERBATIM brief so the question travels with the answer. No model, no network.
Run: PYTHONPATH=src python tests/test_subagent_roster.py
"""
import os
import sys
import tempfile
import threading
from types import SimpleNamespace

os.environ["SLICEAGENT_VAULT"] = tempfile.mkdtemp(prefix="rosteridx-")   # hermetic: FTS index stays in tmp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.hippocampus import HippocampusMixin, SubagentFS, render_artifact  # noqa: E402
from sliceagent.execution import ToolStatus  # noqa: E402
from sliceagent.memory import NullMemory  # noqa: E402
from sliceagent.subagent import SubagentHost, run_subagent, _valid_instance_name  # noqa: E402
from sliceagent.retriever import NullRetriever  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _Mem(HippocampusMixin, NullMemory):
    is_durable = False
    def __init__(self, vault):
        self._vault = vault
        self._idx_lock = threading.Lock()


def _mem():
    return _Mem(tempfile.mkdtemp(prefix="rostervault-"))


def _art(kind, report, name="", task="t", status="ok"):
    return {"kind": kind, "name": name, "task": task, "brief": {"task": task}, "status": status,
            "steps": 3, "report": report, "findings": [], "change_set": [], "files": ["a.py"],
            "coverage": "", "refs": []}


class _Resp:
    def __init__(self, content):
        self.content, self.tool_calls, self.finish_reason, self.usage = content, [], "stop", {}


class _FakeLLM:
    def __init__(self, text):
        self._text = text
        self.reasoning = "fast"
        self._calls = [0]
    def complete(self, messages, schemas):
        self._calls[0] += 1
        if schemas and not any(message.get("role") == "tool" for message in messages):
            response = _Resp("inspect source")
            response.finish_reason = "tool_calls"
            response.tool_calls = [SimpleNamespace(
                name="read_file", args={"path": "a.py"}, id="read-1",
            )]
            return response
        return _Resp(self._text)


class _ToolsHost:
    def schemas(self):
        return [{"type": "function", "function": {
            "name": "read_file", "parameters": {"type": "object", "properties": {}},
        }}]
    def root(self): return "/tmp/ws"
    def accesses(self, name, args): return []
    def run(self, name, args): return "observed implementation"
    def read_text(self, path): return ""


@check
def name_validation_rules():
    for good in ("auth-explorer", "a", "A1_b-c", "x" * 40):
        assert _valid_instance_name(good), good
    for bad in ("", "../evil", "sub-3", "Sub-12", "index", "history", "subagents", "roster",
                "3abc", "-lead", "a b", "x" * 41, "a/b", "a.md"):
        assert not _valid_instance_name(bad), bad


@check
def named_alias_resolves_to_latest_job():
    m = _mem()
    m.append_subagent_artifact("s1", _art("explorer", "first survey", name="auth-explorer"))
    m.append_subagent_artifact("s1", _art("explorer", "anonymous job"))                       # unnamed between
    m.append_subagent_artifact("s1", _art("explorer", "second survey", name="auth-explorer"))
    fs = SubagentFS(m, "s1")
    latest = fs.read_file("subagents/auth-explorer.md")
    assert "second survey" in latest and "first survey" not in latest, latest
    # canonical per-job handles still exact
    assert "first survey" in fs.read_file("subagents/sub-1.md")
    # a name alias can never shadow a canonical handle (sub-N matched first)
    assert "anonymous job" in fs.read_file("subagents/sub-2.md")
    # unknown name → helpful error pointing at the roster
    assert "no subagent named" in fs.read_file("subagents/nobody.md")
    # listing shows the alias ONCE alongside per-job handles
    listing = fs.listing()
    assert listing.count("auth-explorer.md") == 1 and "sub-3.md" in listing, listing


@check
def brief_provenance_travels_with_the_answer():
    rec = {"id": "sub-1", "artifact": _art("explorer", "the report body", name="auth-explorer",
                                           task="Map ONLY the login flow; ignore token refresh.")}
    md = render_artifact(rec)
    assert "auth-explorer · explorer" in md.splitlines()[0], md.splitlines()[0]   # WHO leads the header
    assert "## brief (verbatim task this agent was given)" in md
    assert "Map ONLY the login flow; ignore token refresh." in md                 # verbatim, not gist
    assert md.index("## brief") < md.index("## report"), "the question must precede the answer"


@check
def render_backcompat_for_pre_v3_artifacts():
    # an OLD record (no name/brief keys) must still render: header from kind, brief section from task
    rec = {"id": "sub-1", "artifact": {"kind": "explorer", "task": "old task", "status": "ok",
                                       "steps": 2, "report": "old report"}}
    md = render_artifact(rec)
    assert md.splitlines()[0].startswith("# sub-1 — explorer"), md.splitlines()[0]
    assert "old task" in md and "old report" in md


@check
def index_is_roster_style():
    m = _mem()
    m.append_subagent_artifact("s1", _art("explorer", "r1", name="auth-explorer"))
    m.append_subagent_artifact("s1", _art("general", "r2"))
    idx = SubagentFS(m, "s1").read_file("subagents/index.md")
    assert "sub-1.md · auth-explorer · explorer" in idx, idx
    assert "sub-2.md · general" in idx and "sub-2.md · · " not in idx, idx        # unnamed → no empty column
    assert 'subagents/<name>.md' in idx                                            # alias advertised


@check
def host_rejects_invalid_names_and_threads_valid_ones():
    mem = _mem()
    host = SubagentHost(_ToolsHost(), llm=_FakeLLM("child report"), retriever=NullRetriever(),
                        memory=mem, max_depth=1, session_id="s1")
    r = host.run("spawn_explore", {"task": "t", "name": "../evil"})
    assert r.status is ToolStatus.STEERED and "invalid subagent name" in r, r
    r = host.run("spawn_explore", {"task": "t", "name": "sub-7"})    # canonical-handle spoof
    assert r.status is ToolStatus.STEERED and "invalid subagent name" in r, r
    out = host.run("spawn_explore", {"task": "investigate x", "name": "auth-explorer"})
    assert "[auth-explorer (explore) ok" in out, out                              # identity leads the digest
    # S11: the returned handle is the CANONICAL immutable sub-N.md (not the subagents/<name>.md alias, which
    # retargets to the latest same-name job). The "who" is still in the digest head; the alias still resolves
    # via SubagentFS (tested above) — only the sealed handle the parent stores is now immutable.
    assert 'read_file("subagents/sub-1.md")' in out, out
    art = mem.read_subagent_artifacts("s1")[-1]["artifact"]
    assert art["name"] == "auth-explorer" and art["brief"]["task"] == "investigate x"


@check
def unnamed_spawn_is_unchanged_backcompat():
    mem = _mem()
    out = run_subagent("do x", tools=_ToolsHost(), llm=_FakeLLM("done"), retriever=NullRetriever(),
                       memory=mem, max_steps=2, read_only=True, session_id="s1")
    assert out.startswith("[explore ok") and 'read_file("subagents/sub-1.md")' in out, out
    assert mem.read_subagent_artifacts("s1")[-1]["artifact"]["name"] == ""


@check
def grep_reaches_name_aliases():
    m = _mem()
    m.append_subagent_artifact("s1", _art("explorer", "needle in the latest", name="auth-explorer"))
    fs = SubagentFS(m, "s1")
    hits = fs.grep("needle", path="subagents/auth-explorer.md")
    assert "auth-explorer.md" in hits and "needle" in hits, hits


# ---- W2: capability grants — the governed handle channel --------------------------------------------

from sliceagent.agents import BUILTIN_AGENTS  # noqa: E402


class _MarkerHost(_ToolsHost):
    """Inner host that marks pass-through so a test can tell 'allowed' from 'blocked'."""
    def run(self, name, args): return f"RAN:{name}:{(args or {}).get('path', '')}"


@check
def grants_allow_exact_reads_only():
    child = SubagentHost(_MarkerHost(), llm=None, retriever=None, memory=None,
                         max_depth=1, depth=1, spec=BUILTIN_AGENTS["explorer"],
                         grants=frozenset({"subagents/sub-1.md"}))
    assert child.run("read_file", {"path": "subagents/sub-1.md"}).startswith("RAN:")      # granted
    assert child.run("read_file", {"path": "./subagents/sub-1.md"}).startswith("RAN:")    # normalized
    assert child.run("grep", {"pattern": "x", "path": "subagents/sub-1.md"}).startswith("RAN:")
    for blocked in ({"path": "subagents/sub-2.md"}, {"path": "subagents/index.md"},
                    {"path": "subagents"}, {"path": "history/turn-1.md"}):
        r = child.run("read_file", blocked)
        assert "private namespaces" in r, (blocked, r)
    assert "private namespaces" in child.run("list_files", {"path": "subagents/sub-1.md"})  # never list
    assert "private namespaces" in child.run("search_history", {"query": "q"})              # still blocked
    # the deny message ADVERTISES what IS granted (a grant the child can't see is a grant it never uses)
    assert "subagents/sub-1.md" in child.run("read_file", {"path": "subagents/sub-2.md"})


@check
def spawn_validates_grants_against_existing_seals():
    mem = _mem()
    mem.append_subagent_artifact("s1", _art("explorer", "r1", name="auth-explorer"))
    host = SubagentHost(_ToolsHost(), llm=_FakeLLM("synth"), retriever=NullRetriever(),
                        memory=mem, max_depth=1, session_id="s1")
    for bad in (["subagents/sub-99.md"],        # nonexistent job
                ["subagents/nobody.md"],        # nonexistent name
                ["subagents/index.md"],         # the manifest is not grantable
                ["subagents/"], ["subagents"],  # never a directory
                ["history/turn-1.md"],          # other namespaces can't be granted
                ["subagents/a/b.md"]):          # no nesting
        r = host.run("spawn_explore", {"task": "t", "grants": bad})
        assert r.startswith("Error: cannot grant"), (bad, r)
    r = host.run("spawn_explore", {"task": "t", "grants": ["x"] * 17})
    assert "too many grants" in r, r
    r = host.run("spawn_explore", {"task": "t", "grants": "subagents/sub-1.md"})   # not a list
    assert "'grants' must be a list" in r, r
    # valid: canonical handle, name alias, and a bare leaf all normalize + pass
    out = host.run("spawn_explore", {"task": "use the input", "name": "synth",
                                     "grants": ["subagents/sub-1.md", "auth-explorer.md"]})
    assert "[synth (explore) ok" in out, out
    art = mem.read_subagent_artifacts("s1")[-1]["artifact"]
    # Mutable aliases resolve ONCE at grant validation; the brief/refinement map stores only the immutable
    # canonical job handle. Both requested paths named sub-1 here, so they dedupe to one dependency.
    assert art["brief"]["grants"] == ["subagents/sub-1.md"], art["brief"]


@check
def granted_inputs_are_advertised_in_the_childs_brief():
    mem = _mem()
    mem.append_subagent_artifact("s1", _art("explorer", "r1"))
    seen = {}
    class _SpyLLM(_FakeLLM):
        def complete(self, messages, schemas):
            seen["prompt"] = "\n".join(str(m.get("content", "")) for m in messages)
            return super().complete(messages, schemas)
    host = SubagentHost(_ToolsHost(), llm=_SpyLLM("done"), retriever=NullRetriever(),
                        memory=mem, max_depth=1, session_id="s1")
    host.run("spawn_explore", {"task": "t", "grants": ["subagents/sub-1.md"]})
    assert 'read_file("subagents/sub-1.md")' in seen["prompt"], "grant not advertised to the child"


@check
def children_cannot_regrant_one_hop_only():
    # a GENERAL child with depth left still may not mint grants for a grandchild
    child = SubagentHost(_ToolsHost(), llm=_FakeLLM("x"), retriever=NullRetriever(), memory=_mem(),
                         max_depth=2, depth=1, spec=BUILTIN_AGENTS["general"],
                         session_id="s1", grants=frozenset({"subagents/sub-1.md"}))
    r = child.run("spawn_explore", {"task": "t", "grants": ["subagents/sub-1.md"]})
    assert "cannot re-grant" in r, r


# ---- W3: synthesiser = a child granted all N handles; refs = the seal's refinement map ---------------

@check
def synthesiser_is_a_readonly_kind_not_machinery():
    sp = BUILTIN_AGENTS["synthesiser"]
    assert sp.read_only, "synthesiser must classify read-only (parallel-safe, no writes)"
    assert sp.summary_is_deliverable, "its summary IS the synthesis"
    assert "CITE" in sp.system_prompt and "CONFLICT" in sp.system_prompt.upper()


@check
def explorer_prompt_separates_primary_observation_from_inference():
    sp = BUILTIN_AGENTS["explorer"]
    assert sp.read_only
    assert sp.reasoning == "full"
    assert "separate exact observation from inference" in sp.system_prompt
    assert "does not prove it is executed" in sp.system_prompt
    assert "global 'unkillable' claim" in sp.system_prompt
    assert "most certain concrete failure first" in sp.system_prompt


@check
def synthesis_seal_ships_its_refinement_map():
    mem = _mem()
    mem.append_subagent_artifact("s1", _art("explorer", "auth findings", name="auth-explorer"))
    mem.append_subagent_artifact("s1", _art("explorer", "db findings"))
    host = SubagentHost(_ToolsHost(), llm=_FakeLLM("merged synthesis"), retriever=NullRetriever(),
                        memory=mem, max_depth=1, session_id="s1")
    out = host.run("spawn_agent", {"agent": "synthesiser", "task": "merge the two surveys",
                                   "grants": ["subagents/sub-1.md", "subagents/sub-2.md"]})
    assert "[synthesiser" in out and 'read_file("subagents/sub-3.md")' in out, out
    rec = mem.read_subagent_artifacts("s1")[-1]
    art = rec["artifact"]
    assert art["refs"] == ["subagents/sub-1.md", "subagents/sub-2.md"], art["refs"]   # drillable to inputs
    md = render_artifact(rec)
    assert "built on: subagents/sub-1.md, subagents/sub-2.md" in md, md               # rendered provenance


# ---- W4': durable roster — hire once, wake many ------------------------------------------------------

from sliceagent.hippocampus import RosterFS  # noqa: E402


def _staff_host(mem, llm=None):
    return SubagentHost(_ToolsHost(), llm=llm or _FakeLLM("job done"), retriever=NullRetriever(),
                        memory=mem, max_depth=1, session_id="s1")


@check
def first_named_spawn_hires_then_wakes():
    mem = _mem()
    host = _staff_host(mem)
    out = host.run("spawn_explore", {"task": "survey auth", "name": "auth-explorer"})
    assert "hired standing specialist 'auth-explorer'" in out, out
    p = mem.roster_get("auth-explorer")
    assert p and p["kind"] == "explorer" and p["jobs"] == 1, p          # profile minted + career started
    assert [r["id"] for r in mem.roster_read_jobs("auth-explorer")] == ["job-1"]
    out2 = host.run("spawn_explore", {"task": "survey tokens", "name": "auth-explorer"})
    assert "hired" not in out2, out2                                     # second time = WAKE, not re-hire
    assert mem.roster_get("auth-explorer")["jobs"] == 2
    assert [r["id"] for r in mem.roster_read_jobs("auth-explorer")] == ["job-1", "job-2"]


@check
def wake_is_kind_stable():
    mem = _mem()
    host = _staff_host(mem)
    host.run("spawn_explore", {"task": "t", "name": "auth-explorer"})
    r = host.run("spawn_agent", {"agent": "general", "task": "t", "name": "auth-explorer"})
    assert r.status is ToolStatus.STEERED and "standing 'explorer' specialist" in r, r
    assert mem.roster_get("auth-explorer")["jobs"] == 1                  # the refused wake added no job


@check
def wake_seed_carries_identity_lessons_absent_and_abstention():
    mem = _mem()
    seen = {}
    class _SpyLLM(_FakeLLM):
        def complete(self, messages, schemas):
            seen["prompt"] = "\n".join(str(m.get("content", "")) for m in messages)
            return super().complete(messages, schemas)
    host = _staff_host(mem, llm=_SpyLLM("mapped the login flow end to end"))
    host.run("spawn_explore", {"task": "first job", "name": "auth-explorer"})
    seen.clear()
    host.run("spawn_explore", {"task": "follow-up job", "name": "auth-explorer"})
    p = seen["prompt"]
    assert "YOUR STANDING IDENTITY" in p and "'auth-explorer'" in p, p[:400]
    assert "memories are ONLY what your sealed reports say" in p         # the abstention self-model
    assert 'read_file("roster/auth-explorer/job-<N>.md")' in p           # career manifest with handles
    assert "job-1" in p and "mapped the login flow" in p                 # last-K one-liners = the CONCLUSIONS
    assert "LESSONS" not in p                                            # none yet (W5')


@check
def roster_is_uncapped_a_dormant_specialist_costs_nothing():
    # the roster has NO hire cap — a dormant specialist is just files on disk; hiring the Nth always
    # succeeds. The bound is on the VIEW (roster_recent surfaces top-K), not the STORE.
    mem = _mem()
    host = _staff_host(mem)
    for i in range(40):                                              # comfortably past the old cap of 32
        out = host.run("spawn_explore", {"task": "t", "name": f"spec-{i:02d}"})
        assert "hired standing specialist" in out, out
    assert len(mem.roster_list()) == 40                             # all 40 stand
    # the per-turn manifest read is BOUNDED: roster_recent parses only the top-K, and reports the true total
    from sliceagent.regions import ROSTER_MANIFEST_K
    profs, total = mem.roster_recent(ROSTER_MANIFEST_K)
    assert total == 40 and len(profs) == ROSTER_MANIFEST_K, (total, len(profs))
    # most-recently-active first (spec-39 was hired last)
    assert profs[0]["name"] == "spec-39", profs[0]["name"]


@check
def own_namespace_carveout_self_memory_not_a_channel():
    mem = _mem()
    child = SubagentHost(_MarkerHost(), llm=None, retriever=None, memory=mem,
                         max_depth=1, depth=1, spec=BUILTIN_AGENTS["explorer"],
                         instance_name="auth-explorer")
    for own in ("roster/auth-explorer", "roster/auth-explorer/profile.md",
                "roster/auth-explorer/job-1.md", "roster/auth-explorer/lessons.md"):
        assert child.run("read_file", {"path": own}).startswith("RAN:"), own
    assert child.run("list_files", {"path": "roster/auth-explorer"}).startswith("RAN:")
    for other in ("roster/other-agent/profile.md", "roster/index.md", "roster",
                  "roster/auth-explorer-2/profile.md"):   # prefix spoof must not pass
        r = child.run("read_file", {"path": other})
        assert "private namespaces" in r, (other, r)
    assert "roster/auth-explorer/" in child.run("read_file", {"path": "roster/index.md"})  # hint advertised
    # a TEMP (no identity) gets no roster reach at all
    temp = SubagentHost(_MarkerHost(), llm=None, retriever=None, memory=mem,
                        max_depth=1, depth=1, spec=BUILTIN_AGENTS["explorer"])
    assert "private namespaces" in temp.run("read_file", {"path": "roster/auth-explorer/job-1.md"})
    # TRAVERSAL: '..' inside an own-namespace (or granted) path must never reach a sibling — the guard
    # normalizes exactly like the mounted FS does, so the prefix check sees the CANONICAL target.
    granted = SubagentHost(_MarkerHost(), llm=None, retriever=None, memory=mem,
                           max_depth=1, depth=1, spec=BUILTIN_AGENTS["explorer"],
                           instance_name="auth-explorer", grants=frozenset({"subagents/sub-1.md"}))
    for sneaky in ("roster/auth-explorer/../other-agent/job-1.md",
                   "roster/auth-explorer/../../history/turn-1.md",
                   "subagents/sub-1.md/../sub-2.md",
                   "./roster/auth-explorer/./../victim/lessons.md"):
        r = granted.run("read_file", {"path": sneaky})
        assert "private namespaces" in r, (sneaky, r)
    # normalization helps, never hurts: dotted forms of LEGIT paths still pass
    assert granted.run("read_file", {"path": "roster/auth-explorer/./job-1.md"}).startswith("RAN:")
    assert granted.run("read_file", {"path": "./subagents/sub-1.md"}).startswith("RAN:")


@check
def rosterfs_browsing_and_grep():
    mem = _mem()
    host = _staff_host(mem)
    host.run("spawn_explore", {"task": "map the auth flow", "name": "auth-explorer"})
    fs = RosterFS(mem)
    idx = fs.read_file("roster/index.md")
    assert "auth-explorer · explorer · 1 job(s)" in idx, idx
    prof = fs.read_file("roster/auth-explorer/profile.md")
    assert "standing explorer specialist" in prof and "job-1.md" in prof, prof
    job = fs.read_file("roster/auth-explorer/job-1.md")
    assert "job done" in job and "## brief" in job, job                  # career job renders w/ provenance
    assert "(no lessons recorded yet.)" in fs.read_file("roster/auth-explorer/lessons.md")
    assert "no standing specialist named 'ghost'" in fs.read_file("roster/ghost/profile.md")
    assert "profile.md" in fs.listing("roster/auth-explorer")
    hits = fs.grep("auth flow")
    assert "auth-explorer" in hits, hits
    # empty roster renders guidance, not a crash
    assert "none hired yet" in RosterFS(_mem()).read_file("roster/index.md")


@check
def roster_storage_edges():
    mem = _mem()
    assert mem.roster_get("../evil") is None                              # path guard (defense in depth)
    assert mem.roster_hire("../evil", "explorer") == {}
    assert mem.roster_append_job("never-hired", _art("explorer", "r")) == ""   # temps have no careers
    assert NullMemory().roster_get("x") is None and NullMemory().roster_list() == []
    # named spawn on a NullMemory (headless) degrades to a session temp, no crash
    host = SubagentHost(_ToolsHost(), llm=_FakeLLM("d"), retriever=NullRetriever(),
                        memory=NullMemory(), max_depth=1, session_id="")
    out = host.run("spawn_explore", {"task": "t", "name": "auth-explorer"})
    assert "[auth-explorer (explore) ok" in out and "hired" not in out, out


# ---- W5': lessons — seal-time reflection + curated tier + seed injection -----------------------------

@check
def lesson_marker_is_lifted_into_the_seal():
    mem = _mem()
    host = _staff_host(mem, llm=_FakeLLM("Did the survey.\nLESSON: the auth config lives in env, not code"))
    host.run("spawn_explore", {"task": "t", "name": "auth-explorer"})
    art = mem.read_subagent_artifacts("s1")[-1]["artifact"]
    assert art["lesson"] == "the auth config lives in env, not code", art["lesson"]
    assert "LESSON:" in art["report"]                                   # the seal stays verbatim-honest
    p = mem.roster_get("auth-explorer")
    L = p["lessons"]
    assert len(L) == 1 and L[0]["text"] == art["lesson"] and L[0]["job"] == "job-1" and L[0]["ts"], L


@check
def lesson_curation_dedupes_and_caps():
    mem = _mem()
    mem.roster_hire("x", "explorer")
    for i in range(12):
        mem.roster_append_job("x", _art("explorer", "r") | {"lesson": f"lesson {i}"})
    L = mem.roster_get("x")["lessons"]
    assert len(L) == 8 and L[0]["text"] == "lesson 4" and L[-1]["text"] == "lesson 11", L   # cap, newest kept
    # an exact re-learned lesson collapses to ONE entry with refreshed provenance
    mem.roster_append_job("x", _art("explorer", "r") | {"lesson": "LESSON 11"})
    L = mem.roster_get("x")["lessons"]
    assert sum(1 for e in L if e["text"].lower() == "lesson 11") == 1 and L[-1]["job"] == "job-13", L
    # a no-lesson job changes nothing
    mem.roster_append_job("x", _art("explorer", "r"))
    assert len(mem.roster_get("x")["lessons"]) == 8


@check
def wake_seed_injects_lessons_as_advisory_priors():
    mem = _mem()
    seen = {}
    class _SpyLLM(_FakeLLM):
        def complete(self, messages, schemas):
            seen["prompt"] = "\n".join(str(m.get("content", "")) for m in messages)
            return super().complete(messages, schemas)
    host = _staff_host(mem, llm=_SpyLLM("ok\nLESSON: never trust the cached schema"))
    host.run("spawn_explore", {"task": "job one", "name": "db-explorer"})
    seen.clear()
    host.run("spawn_explore", {"task": "job two", "name": "db-explorer"})
    p = seen["prompt"]
    assert "LESSONS from your past jobs" in p and "never trust the cached schema" in p, p[:600]
    assert "advisory priors" in p and "(job-1" in p                      # framed advisory + provenance
    # and the reflection instruction is offered to NAMED children only
    assert 'end your summary with ONE line: "LESSON:' in p
    seen.clear()
    host.run("spawn_explore", {"task": "temp job"})                      # unnamed temp
    # The fake navigator itself returns a LESSON line, which legitimately appears in the staged handoff;
    # what an unnamed temp must not receive is the standing-specialist reflection instruction.
    assert "end your summary with ONE line" not in seen["prompt"]


@check
def lessons_md_renders_the_curated_tier():
    mem = _mem()
    host = _staff_host(mem, llm=_FakeLLM("ok\nLESSON: check the feature flag first"))
    host.run("spawn_explore", {"task": "t", "name": "flag-explorer"})
    md = RosterFS(mem).read_file("roster/flag-explorer/lessons.md")
    assert "check the feature flag first" in md and "(job-1" in md and "advisory priors" in md, md


# ---- W6': trace archiving + FTS5 dual-write ----------------------------------------------------------

from sliceagent.subagent import _TraceSink, _TRACE_MAX_LINES  # noqa: E402
from sliceagent.events import ToolResult  # noqa: E402
from sliceagent.hippocampus import render_search  # noqa: E402
from sliceagent.search_index import fts5_available  # noqa: E402


@check
def trace_sink_is_bounded_and_marks_failures():
    t = _TraceSink()
    t(ToolResult(name="read_file", args={"path": "a.py"}, output="...", failing=False))
    t(ToolResult(name="run_command", args={"command": "pytest -x"}, output="boom", failing=True))
    assert t.lines[0] == "read_file a.py" and t.lines[1].endswith(" ✗"), t.lines
    for i in range(_TRACE_MAX_LINES + 7):
        t(ToolResult(name="grep", args={"pattern": f"p{i}"}, output="", failing=False))
    assert len(t.lines) == _TRACE_MAX_LINES and "more action(s) not recorded" in t.text()


@check
def trace_is_sealed_and_rendered():
    rec = {"id": "sub-1", "artifact": _art("explorer", "found it") | {"trace": "read_file a.py\ngrep auth"}}
    md = render_artifact(rec)
    assert "## trace (actions taken)" in md and "grep auth" in md, md
    assert md.index("## report") < md.index("## trace"), "conclusions first, path second"


def _fresh_fts_mem():
    """A _Mem whose FTS index is a PRISTINE db (its own SLICEAGENT_VAULT) — the offline suite's test
    doubles otherwise share one index.db and leak connections, so a lock-contending stale writer can make
    the silent-failing index_subagent_artifact drop rows. Production has ONE memory closed per session."""
    os.environ["SLICEAGENT_VAULT"] = tempfile.mkdtemp(prefix="ftsidx-")
    return _mem()


@check
def delegated_seals_are_content_searchable_without_polluting_turns():
    if not fts5_available():
        return   # environment without FTS5 → the mirror degrades to no-op by design
    m = _fresh_fts_mem()
    art = _art("explorer", "the refresh token rotates hourly via cron", name="auth-explorer",
               task="investigate token refresh")
    h = m.append_subagent_artifact("s-fts", art)
    m.index_subagent_artifact("s-fts", h, art)
    hits = m.search_episodes("refresh token rotates", only_session="s-fts")
    assert hits and str(hits[0].get("task_id")) == f"subagent:{h}", hits
    assert "[delegated] auth-explorer" in str(hits[0].get("title")), hits[0]
    # the episodic JSONL (the turn timeline) got NOTHING — a delegation seal is not a turn
    assert m.read_episodes("s-fts") == []
    # and the search renderer points at the SEAL, never a bogus turn file
    from sliceagent.pagetable import PageTable
    refs = PageTable(memory=m, session_id="s-fts").lookup("refresh token rotates",
                                                          kind="episode-search-thissession", k=3)
    assert refs and refs[0].handle == h, refs
    out = render_search(refs, [])
    assert f'read_file("subagents/{h}.md")' in out and "history/turn-" not in out, out
    m.close()


# ---- bug-hunt round 1 fixes: concurrent hire race + FTS mirror per-handle key ------------------------

import concurrent.futures as cf  # noqa: E402


@check
def concurrent_same_name_hire_is_race_safe():
    # HIGH (bug-hunt r1): parallel first-spawns of the same name must NOT double-hire or corrupt kind.
    mem = _mem()

    def _hire(kind):
        return mem.roster_hire("scout", kind)

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        kinds = ["explorer", "synthesiser", "general", "reviewer"] * 8
        profs = list(ex.map(_hire, kinds))
    # exactly ONE identity exists, ONE kind, and every caller got that SAME kind back (idempotent winner)
    assert mem.roster_get("scout") is not None
    won = mem.roster_get("scout")["kind"]
    assert all(p.get("kind") == won for p in profs), [p.get("kind") for p in profs]
    assert len([n for n in os.listdir(os.path.join(mem._vault, "roster"))
                if n == "scout"]) == 1
    # the loser at the SPAWN layer gets a clean kind-mismatch, never a wrong-kind run
    host = _staff_host(mem)
    r = host.run("spawn_agent", {"agent": ("general" if won != "general" else "explorer"),
                                 "task": "t", "name": "scout"})
    assert r.status is ToolStatus.STEERED and "standing" in r, r


@check
def concurrent_career_appends_keep_every_job():
    # profile.json is rewritten atomically (tmp+replace) under the lock → no torn read drops a job
    mem = _mem()
    mem.roster_hire("worker", "explorer")
    def _job(i):
        return mem.roster_append_job("worker", _art("explorer", f"job {i}"))
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        jids = list(ex.map(_job, range(24)))
    assert len(set(jids)) == 24, f"duplicate/lost job ids: {sorted(jids)}"
    assert mem.roster_get("worker")["jobs"] == 24
    assert len(mem.roster_read_jobs("worker")) == 24


@check
def roster_recent_bounds_the_manifest_work_not_the_store():
    # roster_recent parses only the top-K (bounded per-turn work) while the store is unbounded; it reports
    # the true total, and ranks most-recently-active first, tolerating a null-date record without crashing.
    mem = _mem()
    for i in range(20):
        mem.roster_hire(f"w{i:02d}", "explorer")
    # bump one older specialist so it becomes most-recent (a job seal rewrites its profile → dir mtime)
    mem.roster_append_job("w05", _art("explorer", "fresh work"))
    profs, total = mem.roster_recent(5)
    assert total == 20 and len(profs) == 5, (total, len(profs))
    assert profs[0]["name"] == "w05", profs[0]["name"]           # recently-active surfaces first
    # a present-but-null last_active must not crash the recency parse
    import json as _json
    d = mem._roster_dir("nulldate"); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "profile.json"), "w") as f:
        _json.dump({"name": "nulldate", "kind": "explorer", "jobs": 0, "last_active": None}, f)
    profs2, total2 = mem.roster_recent(25)
    assert total2 == 21 and any(p["name"] == "nulldate" for p in profs2), total2


@check
def delegated_fts_rows_do_not_evict_each_other():
    # HIGH (self-found): every delegated seal was indexed at turn=0 → each evicted the prior (idempotent
    # per session_id+turn). Keyed by HANDLE now, so ALL delegated seals stay searchable.
    if not fts5_available():
        return
    m = _fresh_fts_mem()
    for i, kw in enumerate(("alpha-marker", "bravo-marker", "charlie-marker"), 1):
        art = _art("explorer", f"finding about {kw}", name=f"agent{i}")
        h = m.append_subagent_artifact("s-multi", art)
        m.index_subagent_artifact("s-multi", h, art)
    # all three findable (turn=0 collision would have kept only 'charlie')
    for kw, want in (("alpha-marker", "sub-1"), ("bravo-marker", "sub-2"), ("charlie-marker", "sub-3")):
        hits = m.search_episodes(kw, only_session="s-multi")
        assert hits and str(hits[0]["task_id"]) == f"subagent:{want}", (kw, hits)
    # re-indexing the SAME handle still replaces only itself (idempotent per handle)
    art1b = _art("explorer", "alpha-marker REVISED", name="agent1")
    m.index_subagent_artifact("s-multi", "sub-1", art1b)
    assert len(m.search_episodes("alpha-marker", only_session="s-multi")) == 1
    assert len(m.search_episodes("bravo-marker", only_session="s-multi")) == 1   # untouched
    m.close()


# ---- bug-hunt round 2 fixes -------------------------------------------------------------------------

@check
def only_the_creating_caller_announces_the_hire():
    # MED (r2 #1/#4): under a same-name race both callers saw jobs==0 and double-announced. The creator now
    # carries an ephemeral _created marker; the idempotent-return loser does not.
    mem = _mem()
    p1 = mem.roster_hire("dup", "explorer")          # winner (create)
    p2 = mem.roster_hire("dup", "explorer")          # loser (idempotent return)
    assert p1.get("_created") is True and "_created" not in p2, (p1, p2)
    # and _created is EPHEMERAL — never persisted
    import json as _json
    on_disk = _json.load(open(os.path.join(mem._vault, "roster", "dup", "profile.json")))
    assert "_created" not in on_disk, on_disk
    # end-to-end: two sequential first-spawns → exactly ONE 'hired' announcement
    mem2 = _mem()
    host = _staff_host(mem2)
    a = host.run("spawn_explore", {"task": "t", "name": "solo"})
    b = host.run("spawn_explore", {"task": "t", "name": "solo"})
    assert a.count("hired standing specialist") == 1 and b.count("hired standing specialist") == 0, (a, b)


@check
def cross_process_empty_profile_window_is_retried():
    # LOW (r2 #2): a peer's O_EXCL create leaves profile.json momentarily EMPTY; roster_hire must re-read
    # and return the peer's profile rather than {} (which would degrade the spawn to a temp). Simulate by
    # pre-creating an EMPTY profile.json (as if a peer is mid-write), then filling it on the first re-read.
    mem = _mem()
    d = mem._roster_dir("peer"); os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "profile.json"), "w").close()          # empty file (peer between O_EXCL and write)
    import json as _json
    real_get = mem.roster_get
    calls = {"n": 0}
    def _flaky_get(name):
        calls["n"] += 1
        if name == "peer" and calls["n"] <= 2:                   # first couple reads see the empty file
            return real_get(name)
        if name == "peer" and calls["n"] == 3:                   # peer finishes its write
            with open(os.path.join(d, "profile.json"), "w") as f:
                _json.dump({"v": 1, "name": "peer", "kind": "explorer", "jobs": 0}, f)
        return real_get(name)
    mem.roster_get = _flaky_get
    got = mem.roster_hire("peer", "explorer")                   # O_EXCL → FileExistsError → retry loop
    assert got.get("name") == "peer" and got.get("kind") == "explorer", got   # returned theirs, not {}


@check
def validate_grants_no_crash_when_memory_none():
    # LOW/certain (r2 #3): a parent host with memory=None + a session_id must not AttributeError on grants.
    host = SubagentHost(_ToolsHost(), llm=None, retriever=None, memory=None,
                        max_depth=1, depth=0, spec=None, session_id="s1")
    err, grants = host._validate_grants(["subagents/sub-1.md"])
    assert err.startswith("Error: cannot grant") and grants == frozenset(), (err, grants)   # clean error, no crash


# ---- bug-hunt round 3 fixes: null-field robustness + hire-suffix on error ----------------------------

from sliceagent.subagent import _render_wake_block  # noqa: E402
from sliceagent.hippocampus import render_profile  # noqa: E402


@check
def corrupt_null_date_fields_do_not_crash_render_or_roster():
    # LOW/certain (r3 #1): a hand-edited/legacy profile.json with present-but-null created/last_active/ts
    # must not TypeError the wake seed, roster/index.md, profile.md, or roster_list's sort.
    bad = {"name": "x", "kind": "explorer", "jobs": 1, "created": None, "last_active": None}
    assert "standing explorer" in _render_wake_block(bad, [], "x")            # None[:10] would crash
    assert "explorer" in render_profile(bad, [{"id": "job-1", "ts": None, "artifact": _art("explorer", "r")}])
    # a null-lesson-ts and null-job-ts render clean
    badp = dict(bad, lessons=[{"text": "L", "job": "job-1", "ts": None}])
    assert "L" in _render_wake_block(badp, [{"id": "job-1", "ts": None, "artifact": _art("explorer", "r")}], "x")
    # roster_list must survive a null last_active without a sort TypeError (None < str)
    mem = _mem()
    mem.roster_hire("good", "explorer")
    d = mem._roster_dir("bad2"); os.makedirs(d, exist_ok=True)
    import json as _json
    with open(os.path.join(d, "profile.json"), "w") as f:
        _json.dump({"name": "bad2", "kind": "explorer", "jobs": 0, "last_active": None}, f)
    names = [p.get("name") for p in mem.roster_list()]              # would raise on the mixed None/str sort
    assert "good" in names and "bad2" in names, names
    assert "bad2" in RosterFS(mem).read_file("roster/index.md")     # index renders both, no crash


@check
def hire_suffix_not_appended_to_a_failed_childs_error():
    # LOW (r3 #2): a freshly-hired child that FAILS returns 'Error: ...'; the hire announcement must not be
    # concatenated onto it (garbled error tier). The hire still really happened (career accrues).
    class _TC:
        def __init__(self): self.name, self.args, self.id = "read_file", {"path": "x.py"}, "1"
    class _LoopLLM:                                 # never yields end_turn → the writable child fails
        reasoning = "fast"
        def complete(self, messages, schemas):
            r = _Resp(""); r.tool_calls = [_TC()]; r.finish_reason = "tool_calls"; return r
    mem = _mem()
    host = SubagentHost(_ToolsHost(), llm=_LoopLLM(), retriever=NullRetriever(),
                        memory=mem, max_depth=1, max_steps=1, session_id="s1")
    out = host.run("spawn_agent", {"agent": "general", "task": "do x", "name": "flaky"})
    assert out.startswith("Error:") and "hired standing specialist" not in out, out
    assert mem.roster_get("flaky") is not None                      # the hire is real regardless of outcome


# ---- roster VISIBILITY: the standing roster must be surfaced in the slice (the discovery bug) ---------

from sliceagent.regions import render_roster, REGION_ORDER, ROSTER_MANIFEST_K  # noqa: E402


@check
def roster_is_surfaced_in_the_slice_manifest():
    # The bug the user hit: the roster was MOUNTED for tool routing but never ADVERTISED in the slice, so a
    # fresh session couldn't discover @sliceagent/roster/index.md and spelunked the raw vault instead. The
    # STANDING SPECIALISTS region is the fix (the visible-cache-manifest lesson, applied to the roster).
    assert render_roster([]) == ""                                          # empty → nothing
    profs = [{"name": f"spec-{i}", "kind": "explorer", "jobs": i, "last_active": "2026-07-09T00:00:00Z"}
             for i in range(ROSTER_MANIFEST_K + 5)]
    profs.append({"name": "nulldate", "kind": "explorer", "jobs": 0, "last_active": None})  # r3 null-safety
    man = render_roster(profs)
    assert "spec-0 · explorer · 0 job(s)" in man                            # locators: name · kind · jobs
    assert man.count("\n") == ROSTER_MANIFEST_K                             # K lines + one overflow line
    assert "+6 more" in man and 'read_file("@sliceagent/roster/index.md")' in man
    # the region is wired into REGION_ORDER, emits the discovery affordances, and self-suppresses when empty
    lam = next(e for e in REGION_ORDER if e[0] == "roster")[2]
    assert lam({"roster": ""}) == ""
    blk = lam({"roster": man})
    assert "# STANDING SPECIALISTS" in blk
    assert "spawn_agent(agent=<kind>, name=<name>" in blk
    assert 'read_file("@sliceagent/roster/index.md")' in blk


@check
def real_profile_json_name_resolves_through_the_virtual_fs():
    # papercut: the on-disk files are profile.json, but the virtual FS served only profile.md — a model that
    # saw the real name got "not a roster file". Both names now resolve.
    mem = _mem()
    mem.roster_hire("worker", "explorer")
    fs = RosterFS(mem)
    assert fs.read_file("roster/worker/profile.json").startswith("# worker")
    assert fs.read_file("roster/worker/profile.md").startswith("# worker")


# ---- system-prompt wiring: the delegation mental model must actually reach the model -----------------

@check
def delegation_guidance_is_spliced_when_spawn_agent_is_offered():
    # Guidance is compiled from the live spawn_agent schema, so it is present iff the capability is offered
    # and cannot advertise advanced fields/kinds in core mode.
    from sliceagent.pfc import Slice
    from sliceagent.seed import make_build_slice

    def _sys(tools):
        st = Slice(); st.reset("review the repo")
        msgs = make_build_slice(st, tools, NullRetriever(), NullMemory(), "review the repo")()
        return "\n".join(m.get("content", "") if isinstance(m.get("content"), str) else ""
                         for m in msgs if m.get("role") == "system")

    # a parent host offers spawn_agent → the delegation guidance (and the new mental model) is in the prompt
    parent = SubagentHost(_ToolsHost(), llm=None, retriever=None, memory=NullMemory(),
                          max_depth=1, depth=0)
    sysp = _sys(parent)
    assert "<delegation>" in sysp, "delegation block missing from the system prompt"
    assert "LIVE DELEGATION CAPABILITY" in sysp and "standing specialist" in sysp
    assert "Available agent kinds:" in sysp and "general" in sysp
    assert "spawn_explore" not in sysp, "stale spawn_explore leaked into the guidance"

    # a host at the depth floor offers NO spawn tool → no delegation block (don't advertise a tool it lacks)
    floor = SubagentHost(_ToolsHost(), llm=None, retriever=None, memory=NullMemory(),
                         max_depth=1, depth=1)
    assert "<delegation>" not in _sys(floor)


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
