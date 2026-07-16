"""Regression tests for the subagent-review fixes: S8 (blob/reserved-ns isolation from children),
S5 (child tokens charged to the parent budget + sealed usage), S12 (0600/0700 private state).
S7 (child llm isolation) and S11 (canonical handle) are locked by the updated test_readonly_subagent.py
and test_subagent_roster.py. No model, no network. Run: PYTHONPATH=src python tests/test_subagent_review_fixes.py
"""
import os
import stat
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# ---- S8: a child must not read the host's private .sliceagent/ (paged-out blobs live there) --------------

@check
def s8_reserved_ns_blocks_sliceagent_private_dir():
    from sliceagent.subagent import _targets_reserved_ns
    for p in (".sliceagent", ".sliceagent/blobs/out-abc123.txt", ".sliceagent/config.toml",
              "subagents", "history/turn-1.md", "roster/spec/profile.json"):
        assert _targets_reserved_ns({"path": p}), f"child should be blocked from {p!r}"
    for p in ("src/main.py", "README.md", "notes.txt", "sliceagentfoo.py"):
        assert not _targets_reserved_ns({"path": p}), f"a normal repo path must NOT be blocked: {p!r}"


# ---- S5: one typed usage path; no second callback accountant --------------------------------------------

@check
def s5_budget_hook_accounts_every_usage_record_through_one_method():
    from sliceagent.hooks import BudgetHook
    b = BudgetHook(1000)
    assert b.record_step_usage({"prompt_tokens": 100, "completion_tokens": 100}) is None   # 200, under cap
    assert b.record_step_usage({"prompt_tokens": 800, "completion_tokens": 100}) == {"stop_turn": True}
    assert not hasattr(b, "record_external"), "child usage must not have a second accounting API"


@check
def s5_subagent_host_has_no_parallel_budget_sink_state():
    from sliceagent.subagent import SubagentHost
    host = SubagentHost(_Inner(), llm=None, retriever=None, memory=None,
                        max_depth=1, depth=0)
    assert "budget_sink" not in vars(host)


# ---- S12: durable subagent/roster state is 0600 (files) / 0700 (dirs), not umask 0644 -------------------

@check
def s12_private_state_is_0600():
    if os.name != "posix":
        return
    os.environ.setdefault("SLICEAGENT_VAULT", tempfile.mkdtemp())
    from sliceagent.hippocampus import HippocampusMixin
    from sliceagent.memory import NullMemory

    class _Mem(HippocampusMixin, NullMemory):
        is_durable = True
        def __init__(self, vault): self._vault = vault

    v = tempfile.mkdtemp()
    m = _Mem(v)
    art = {"kind": "explorer", "name": "", "task": "t", "report": "r", "findings": [], "change_set": [],
           "files": [], "trace": "", "coverage": "", "refs": [], "lesson": "", "brief": {}}
    m.append_subagent_artifact("s1", art)
    m.roster_hire("spec1", "explorer")
    m.roster_append_job("spec1", {**art, "name": "spec1", "status": "ok", "steps": 1})

    def mode(p):
        return stat.S_IMODE(os.stat(p).st_mode)
    assert mode(os.path.join(v, "subagents", "s1.jsonl")) == 0o600
    assert mode(os.path.join(v, "roster", "spec1", "profile.json")) == 0o600   # the atomic-update downgrade
    assert mode(os.path.join(v, "roster", "spec1", "episodes.jsonl")) == 0o600
    assert mode(os.path.join(v, "subagents")) == 0o700
    assert mode(os.path.join(v, "roster", "spec1")) == 0o700


class _Inner:
    def schemas(self):
        return [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    def accesses(self, n, a):
        return []
    def run(self, n, a):
        return "inner"


def main():
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
