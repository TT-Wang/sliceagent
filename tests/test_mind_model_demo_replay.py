"""Deterministic replay of the mind-model demo failures. No model, network, or pytest required.

This intentionally crosses the public seams the real host uses: discourse interpretation produces one
TurnAdmission, TurnReceipt reduces canonical execution journals, and the turn-contract renderer projects
only the selected durable evidence back into the next slice.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.discourse import extract_pending_proposal, interpret_turn  # noqa: E402
from sliceagent.events import ToolResult, TurnEnd  # noqa: E402
from sliceagent.pfc import Slice, record_user, slice_sink  # noqa: E402
from sliceagent.receipts import TurnReceipt  # noqa: E402
from sliceagent.regions import render_evidence_detail, render_evidence_result  # noqa: E402
from sliceagent.session import apply_turn_continuation  # noqa: E402


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


HUNTER_PATH = "/Users/tongtao/Desktop/hunter"


def _sliceagent_focus():
    oriented = interpret_turn("Review the Hunter project", ())
    counterfactual = interpret_turn(
        "If you were to improve it, what would you do?", (), focus=oriented.focus,
    )
    repaired = interpret_turn("I mean SliceAgent", (), focus=counterfactual.focus)
    return oriented, counterfactual, repaired


@check
def hunter_directive_and_adjacent_yes_compile_navigation_scope():
    directive = interpret_turn("go to Hunter workspace", ())
    assert directive.admission.effect_authority == "explicit"
    assert directive.admission.target.label == "Hunter"
    assert directive.admission.effect_grants[0].operation == "workspace.navigate"

    assistant = (
        "The Hunter workspace appears to be outside the current project. "
        f"Could you confirm the exact path? Is it {HUNTER_PATH}?"
    )
    proposal = extract_pending_proposal(assistant)
    assert proposal and proposal["action"] == {
        "tool": "change_workspace", "args": {"path": HUNTER_PATH},
    }

    adjacent_yes = interpret_turn("yes", (), pending_proposal=proposal)
    assert adjacent_yes.admission.effect_authority == "continuation"
    assert dict(adjacent_yes.admission.effect_grants[0].exact_args) == {"path": HUNTER_PATH}

    stale_yes = interpret_turn("yes", ())
    assert stale_yes.admission.effect_authority == "uncertain"
    assert not stale_yes.admission.effect_grants, \
        "the same word outside its adjacent proposal must not retain an action scope"


@check
def project_focus_flows_into_counterfactual_then_explicitly_repairs_to_sliceagent():
    oriented, counterfactual, repaired = _sliceagent_focus()
    assert oriented.admission.target.label == "Hunter"
    assert oriented.admission.target.source == "explicit"

    assert counterfactual.admission.effect_authority == "none"
    assert "recommend" in counterfactual.admission.requested_modes
    assert counterfactual.admission.target.label == "Hunter"
    assert counterfactual.admission.target.source == "focus"

    assert repaired.admission.target.label == "SliceAgent"
    assert repaired.admission.target.source == "repair"
    assert repaired.admission.focus_repairs[0].field == "target"
    assert repaired.focus[-1]["entity"]["label"] == "SliceAgent"

    after_repair = interpret_turn("What would you improve first?", (), focus=repaired.focus)
    assert after_repair.admission.target.label == "SliceAgent"
    assert after_repair.admission.target.source == "focus"


@check
def failure_questions_and_self_audits_do_not_become_open_user_reports():
    state = Slice(); state.reset("Review this project")
    record_user(state, "Review this project", source_artifact="turn-review")
    slice_sink(state)(TurnEnd("end_turn", 1, {}))
    assert state.task.objective_status == "provisionally_satisfied"

    request = "Reflect on your performance this session — what failed or went badly?"
    audit = interpret_turn(request, (), task_id="task")
    assert "audit" in audit.admission.requested_modes
    apply_turn_continuation(state, request, admission=audit.admission)
    assert not state.open_report
    assert state.task.objective_status == "provisionally_satisfied", \
        "a leading question is not evidence that the completed objective became broken"

    real_report = "The review is still broken and the command fails."
    live = interpret_turn(real_report, (), task_id="task")
    apply_turn_continuation(state, real_report, admission=live.admission)
    assert state.open_report == real_report
    assert state.task.objective_status == "active"

    state.open_report = ""
    state.task.mark_objective_provisional()
    mixed_report = "What went wrong? The app still crashes on launch."
    mixed = interpret_turn(mixed_report, (), task_id="task")
    assert "audit" not in mixed.admission.requested_modes
    assert mixed.admission.quality_evidence_query is None
    assert mixed.admission.grounding == "live_present"
    apply_turn_continuation(state, mixed_report, admission=mixed.admission)
    assert state.open_report == mixed_report
    assert state.task.objective_status == "active", \
        "a live failure assertion remains a blocker even when the same turn also asks for reflection"

    state.open_report = ""
    deployment = "The app crashed. What went wrong with the deployment?"
    diagnosed = interpret_turn(deployment, (), task_id="task")
    assert diagnosed.admission.evidence_query is None
    assert diagnosed.admission.quality_evidence_query is None
    apply_turn_continuation(state, deployment, admission=diagnosed.admission)
    assert state.open_report == deployment, "a product failure question must not be mistaken for agent self-audit"


@check
def open_report_closes_only_after_edit_then_real_verification_success():
    def repaired_state(report="tests still fail"):
        state = Slice(); state.reset("repair it"); state.open_report = report
        sink = slice_sink(state)
        sink(ToolResult(
            "str_replace", {"path": "app.py", "old_string": "bad", "new_string": "good"},
            "updated", False, status="succeeded",
        ))
        assert state.open_report and state.runtime.report_repair_observed
        return state, sink

    for name, args, report in (
        ("run_command", {"command": "PYTHONPATH=src pytest -q"}, "tests still fail"),
        ("run_command", {"command": "uv run pytest -q"}, "pytest still fails"),
        ("run_command", {"command": "bash scripts/run_tests.sh"}, "the test suite still fails"),
        ("run_command", {"command": "npm run build"}, "the build still fails"),
        ("run_command", {"command": "ruff check ."}, "lint still fails"),
        ("run_command", {"command": "mypy src"}, "the type check still fails"),
        ("run_command", {"command": "pytest -q && ruff check ."}, "tests and lint still fail"),
        ("run_command", {"command": "ruff check ."}, "the verification command failed"),
    ):
        state, sink = repaired_state(report)
        sink(ToolResult(name, args, "passed", False, status="succeeded"))
        assert not state.open_report, (name, args)

    for command in (
        "echo pytest", "printf 'npm test'", "python -c \"print('pytest')\"",
        "pytest || true", "pytest; echo done", "pytest | tee test.log", "pytest & wait",
        "python -c \"print('ok')\" tests.py", "bash -c \"true\" run_tests.sh",
        "pytest --collect-only", "npm run test --if-present", "make -n test",
        "bash -n scripts/run_tests.sh", "cargo test --no-run", "tsc --showConfig",
        "env --help pytest", "pytest -qh", "npm run test --if-present=true",
        "ruff check . --exit-zero", "bash --rcfile scripts/run_tests.sh", "mvn test -DskipTests",
        "go test -list .", "ctest -N", "nox -l", "eslint --print-config app.js",
    ):
        state, sink = repaired_state()
        sink(ToolResult(
            "run_command", {"command": command}, "mentioned", False, status="succeeded",
        ))
        assert state.open_report, command

    for code in (
        "import pytest\npytest.main(['-q'])",
        "import unittest\nunittest.TextTestRunner().run(suite)",
        "import subprocess\nsubprocess.call(['pytest', '-q'])",
        "import subprocess\nsubprocess.Popen(['pytest', '-q'])",
        "import os\nos.system('pytest -q')",
        "import subprocess\nsubprocess.run(['pytest', '-q'])",
        "run_command('pytest -q')",
        "try:\n    subprocess.check_call(['pytest'])\nexcept Exception:\n    pass",
        "def verify():\n    raise SystemExit(pytest.main(['-q']))\nverify()",
        "import subprocess\nsubprocess.run(['pytest', '--collect-only'], check=True)",
        "import subprocess\nsubprocess.run(['pytest', '-q'], check=True)",
        "import subprocess\nsubprocess.check_call(['pytest', '-q'])",
        "import subprocess\nsubprocess.check_call(['npm', 'run', 'test', '--if-present'])",
        "import pytest\nraise SystemExit(pytest.main(['-q']))",
        "import pytest\nraise SystemExit(pytest.main(['--collect-only']))",
        "import pytest, sys\nsys.exit(pytest.main(['--help']))",
        "import pytest\nraise SystemExit(pytest.main(args=['--collect-only']))",
        "import subprocess\nmode = '--collect-only'\nsubprocess.check_call(['pytest', mode])",
        "import pytest\nmode = '--help'\nraise SystemExit(pytest.main([mode]))",
    ):
        state, sink = repaired_state()
        sink(ToolResult(
            "execute_code", {"code": code}, "script completed", False, status="succeeded",
        ))
        assert state.open_report, code

    for report, command in (
        ("pytest tests still fail", "ruff check ."),
        ("lint still fails", "pytest -q"),
        ("the type check still fails", "npm run build"),
        ("the build and tests still fail", "npm run build"),
    ):
        state, sink = repaired_state(report)
        sink(ToolResult(
            "run_command", {"command": command}, "passed", False, status="succeeded",
        ))
        assert state.open_report, (report, command)

    sequential, sequential_sink = repaired_state("tests and lint still fail")
    sequential_sink(ToolResult(
        "run_command", {"command": "pytest -q"}, "passed", False, status="succeeded",
    ))
    assert sequential.open_report
    assert sequential.runtime.report_verification_families == {"test"}
    sequential_sink(ToolResult(
        "run_command", {"command": "ruff check ."}, "passed", False, status="succeeded",
    ))
    assert not sequential.open_report

    invalidated, invalidated_sink = repaired_state("tests and lint still fail")
    invalidated_sink(ToolResult(
        "run_command", {"command": "pytest -q"}, "passed", False, status="succeeded",
    ))
    invalidated_sink(ToolResult(
        "str_replace", {"path": "app.py", "old_string": "good", "new_string": "better"},
        "updated", False, status="succeeded",
    ))
    invalidated_sink(ToolResult(
        "run_command", {"command": "ruff check ."}, "passed", False, status="succeeded",
    ))
    assert invalidated.open_report
    assert invalidated.runtime.report_verification_families == {"lint"}

    contradicted, contradicted_sink = repaired_state("tests and lint still fail")
    contradicted_sink(ToolResult(
        "run_command", {"command": "pytest -q"}, "passed", False, status="succeeded",
    ))
    contradicted_sink(ToolResult(
        "run_command", {"command": "pytest -q"}, "failed", True, status="failed",
    ))
    contradicted_sink(ToolResult(
        "run_command", {"command": "ruff check ."}, "passed", False, status="succeeded",
    ))
    assert contradicted.open_report
    assert contradicted.runtime.report_verification_families == {"lint"}

    # Verification before the repair is not proof about the later bytes.
    state = Slice(); state.reset("repair it"); state.open_report = "tests still fail"
    sink = slice_sink(state)
    sink(ToolResult(
        "run_command", {"command": "pytest -q"}, "passed", False, status="succeeded",
    ))
    sink(ToolResult(
        "str_replace", {"path": "app.py", "old_string": "bad", "new_string": "good"},
        "updated", False, status="succeeded",
    ))
    assert state.open_report

    for code in (
        'print("write_file(\\\'app.py\\\', \\\'fixed\\\')")',
        "# write_file('app.py', 'fixed')",
        "if False:\n    write_file('app.py', 'fixed')",
        "def write_file(path, content): return None\nwrite_file('app.py', 'fixed')",
    ):
        lexical = Slice(); lexical.reset("repair it"); lexical.open_report = "tests still fail"
        lexical_sink = slice_sink(lexical)
        lexical_sink(ToolResult(
            "execute_code", {"code": code}, "completed", False, status="succeeded",
        ))
        assert not lexical.runtime.report_repair_observed
        lexical_sink(ToolResult(
            "run_command", {"command": "pytest -q"}, "passed", False, status="succeeded",
        ))
        assert lexical.open_report, "source text alone is not a physical repair receipt"

    tracked = Slice(); tracked.reset("repair it")
    slice_sink(tracked)(ToolResult(
        "execute_code", {"code": "write_file('app.py', 'fixed')"},
        "completed", False, status="succeeded",
    ))
    assert "app.py" in tracked.edited_files, "code-as-action residency remains best-effort tracked"

    ui = Slice(); ui.reset("repair it"); ui.open_report = "the message panel is still too crowded"
    ui_sink = slice_sink(ui)
    ui_sink(ToolResult(
        "str_replace", {"path": "app.py", "old_string": "dense", "new_string": "spacious"},
        "updated", False, status="succeeded",
    ))
    ui_sink(ToolResult(
        "run_command", {"command": "pytest -q"}, "passed", False, status="succeeded",
    ))
    assert ui.open_report, "a passing code suite does not verify a visual/UX report"


@check
def successful_retry_of_same_edit_clears_only_its_stale_error():
    state = Slice(); state.reset("edit")
    sink = slice_sink(state)
    args = {"path": "app.py", "old_string": "missing", "new_string": "fixed"}
    sink(ToolResult("str_replace", args, "Error: old string not found", True, status="failed"))
    assert state.last_error
    sink(ToolResult("str_replace", args, "updated", False, status="succeeded"))
    assert not state.last_error

    unrelated = Slice(); unrelated.reset("edit")
    unrelated_sink = slice_sink(unrelated)
    unrelated_sink(ToolResult(
        "str_replace", args, "Error: old string not found", True, status="failed",
    ))
    original = unrelated.last_error
    unrelated_sink(ToolResult(
        "run_command", {"command": "pwd"}, "/tmp/project", False, status="succeeded",
    ))
    assert unrelated.last_error == original, "an unrelated shell success does not resolve an edit failure"

    other_edit = {"path": "app.py", "old_string": "different", "new_string": "also fixed"}
    unrelated_sink(ToolResult(
        "str_replace", other_edit, "updated another snippet", False, status="succeeded",
    ))
    assert unrelated.last_error == original, "a different edit in the same file is not the failed retry"
    unrelated_sink(ToolResult(
        "str_replace", args, "updated intended snippet", False, status="succeeded",
    ))
    assert not unrelated.last_error

@check
def quoted_navigation_transcript_is_attributed_context_not_current_action():
    quoted = interpret_turn(
        'The transcript says: "go to Hunter workspace" and then "yes". '
        "Explain why that behavior was wrong.",
        (),
    )
    assert quoted.admission.attributed_spans, "the quoted operative language must be marked as data"
    assert quoted.admission.effect_authority in {"none", "uncertain"}
    assert not quoted.admission.effect_grants


def _event(event_type: str, payload: dict) -> dict:
    return {"type": event_type, "payload": payload}


def _spawn_receipt(count: int, *, rejected: bool, turn_id: str) -> TurnReceipt:
    events = []
    for index in range(count):
        identity = f"spawn-{'denied' if rejected else 'ok'}-{index}"
        base = {
            "invocation_id": identity,
            "name": "spawn_agent",
            "args": {
                "agent": "explorer",
                "name": f"demo-{'denied' if rejected else 'ok'}-{index}",
            },
            "provider_index": index,
        }
        events.append(_event("tool-requested", base))
        if rejected:
            reason = f"demo extension rejection {index}"
            events.append(_event("tool-rejected", {**base, "reason": reason}))
            events.append(_event("tool-settled", {
                **base,
                "outcome": {
                    "status": "cancelled",
                    "text": f"Not run: {reason}",
                    "effects": [],
                },
            }))
        else:
            events.append(_event("tool-execution-started", base))
            events.append(_event("tool-settled", {
                **base,
                "outcome": {"status": "succeeded", "text": "sealed", "effects": []},
            }))
    return TurnReceipt.from_events(events, turn_id=turn_id, turn_status="end_turn")


def _receipt_artifact(artifact_id: str, timestamp: str, receipt: TurnReceipt, lie: str):
    return SimpleNamespace(
        id=artifact_id,
        kind="turn",
        timestamp=timestamp,
        task_id="demo-task",
        summary=lie,
        structured_body={"assistant": lie, "turn_receipt": receipt.to_dict()},
    )


def _render(request: str, preview) -> str:
    state = SimpleNamespace(intent=SimpleNamespace(
        current_request=request, turn_contract=preview.admission,
    ))
    return render_evidence_result(state) + "\n" + render_evidence_detail(state)


@check
def mixed_spawn_counts_and_failure_self_reflection_come_only_from_receipts():
    denied = _spawn_receipt(13, rejected=True, turn_id="turn-denied")
    succeeded = _spawn_receipt(11, rejected=False, turn_id="turn-succeeded")
    assert denied.counts["requested"] == 13
    assert denied.counts["rejected_before_execution"] == 13
    assert denied.counts["execution_started"] == 0
    assert denied.counts["failed"] == 0, "pre-handler rejection is not physical execution failure"
    assert succeeded.counts["requested"] == 11
    assert succeeded.counts["execution_started"] == 11
    assert succeeded.counts["succeeded"] == 11

    artifacts = (
        _receipt_artifact(
            "turn-denied", "2026-07-11T00:00:00Z", denied,
            "PROSE LIE: all 13 denied agents physically started and failed.",
        ),
        _receipt_artifact(
            "turn-succeeded", "2026-07-11T00:01:00Z", succeeded,
            "PROSE LIE: no agent succeeded.",
        ),
    )
    _, _, repaired = _sliceagent_focus()

    count_request = (
        "Across this task, how many explorer agents were requested, rejected, started, and succeeded?"
    )
    count_preview = interpret_turn(
        count_request, artifacts, task_id="demo-task", focus=repaired.focus,
    )
    assert count_preview.admission.evidence_query.family == "delegation"
    assert count_preview.admission.evidence_query.predicate == "aggregate"
    aggregate = next(
        ref for ref in count_preview.admission.referents
        if isinstance(ref, dict) and ref.get("kind") == "execution_receipt_aggregate"
    )
    expected = {
        "requested": 24, "rejected_before_execution": 13, "execution_started": 11,
        "settled": 24, "succeeded": 11, "failed": 0, "cancelled": 0,
        "indeterminate": 0, "not_started": 0, "unknown": 0,
    }
    assert {key: aggregate["counts"][key] for key in expected} == expected
    count_rendered = _render(count_request, count_preview)
    assert "requested=24" in count_rendered
    assert "rejected-before-execution=13" in count_rendered
    assert "execution-started=11" in count_rendered
    assert "succeeded=11" in count_rendered and "failed=0" in count_rendered
    assert "PROSE LIE" not in count_rendered

    reflection_request = "Own up to your failures: why were the explorer agents rejected or failed?"
    reflection = interpret_turn(
        reflection_request, artifacts, task_id="demo-task", focus=repaired.focus,
    )
    assert reflection.admission.target.label == "SliceAgent"
    assert reflection.admission.evidence_query.family == "delegation"
    assert reflection.admission.evidence_query.predicate == "failure_detail"
    detail_refs = [
        ref for ref in reflection.admission.referents
        if isinstance(ref, dict) and ref.get("kind") == "execution_receipt"
    ]
    assert [ref["artifact_id"] for ref in detail_refs] == ["turn-denied"]
    operations = detail_refs[0]["operations"]
    assert len(operations) == 13
    assert all(operation["rejected_before_execution"] for operation in operations)
    assert all(not operation["execution_started"] for operation in operations)
    assert all(operation["disposition"] == "rejected" for operation in operations)

    reflected = _render(reflection_request, reflection)
    assert "rejected-before-execution=13" in reflected
    assert "execution-started=11" in reflected and "failed=0" in reflected
    assert detail_refs[0]["counts"]["execution_started"] == 0, \
        "the rejected-turn detail remains distinct from the task-wide aggregate"
    assert "recorded reason excerpt=demo extension rejection 0" in reflected
    assert "recorded reason excerpt=demo extension rejection 12" in reflected
    assert "PROSE LIE" not in reflected


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 — standalone replay reports every contract independently
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
