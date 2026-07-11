"""Authoritative typed-intent seam. No model/network; standalone suite convention."""
import os
import sys
import tempfile
from types import MappingProxyType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.events import ToolResult, TurnEnd  # noqa: E402
from sliceagent.intent import IntentEntry, IntentState  # noqa: E402
from sliceagent.memory import NullMemory, _now_iso, _parse_task_md, _render_task_md  # noqa: E402
from sliceagent.pfc import Slice, record_user, slice_sink  # noqa: E402
from sliceagent.seed import make_build_slice  # noqa: E402
from sliceagent.slice_state import ProgressSignal  # noqa: E402
from sliceagent.taskstate import slice_to_task_state, task_state_to_slice  # noqa: E402
from sliceagent.tools import LocalToolHost  # noqa: E402

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _emit(s, name, text):
    slice_sink(s)(ToolResult(name=name, args={"text": text}, output="ok", failing=False))


@check
def exact_user_clause_gets_provenance_range():
    request = "Refactor auth; keep the public API stable; support Python 3.11."
    clause = "keep the public API stable"
    s = Slice(); s.reset(request)
    _emit(s, "require", clause)
    entry = s.intent.entries[0]
    start = request.index(clause)
    assert entry.authority == "user"
    assert entry.verbatim_clause == clause
    assert entry.source_range == (start, start + len(clause))


@check
def paraphrase_is_task_state_not_forged_user_text():
    s = Slice(); s.reset("Make the parser safer")
    _emit(s, "require", "do not change parse_date's return type")
    entry = s.intent.entries[0]
    assert entry.authority == "task" and entry.source_range is None


@check
def current_request_and_stable_objective_have_distinct_render_authority():
    s = Slice(); s.reset("old task label")
    record_user(s, "the live request to render")
    host = LocalToolHost(tempfile.mkdtemp(prefix="intent-seed-"))
    user = make_build_slice(s, host, None, NullMemory(), "fallback task")()[1]["content"]
    assert user.count("the live request to render") == 1, "the live request renders exactly once"
    assert "# STABLE TASK OBJECTIVE" in user and "old task label" in user


@check
def stable_objective_survives_beyond_the_recent_conversation_ring():
    objective = "Implement retry scheduling and never modify config.py"
    s = Slice(); s.reset(objective)
    record_user(s, objective, source_artifact="turn-initial")
    for index in range(1, 6):
        record_user(s, f"continue phase {index}", source_artifact=f"turn-{index}")
    host = LocalToolHost(tempfile.mkdtemp(prefix="intent-objective-"))
    user = make_build_slice(s, host, None, NullMemory(), "fallback task")()[1]["content"]
    assert objective in user and "# STABLE TASK OBJECTIVE" in user
    assert "source artifact: turn-initial" in user
    assert "continue phase 5" in user


@check
def explicit_followup_constraint_survives_beyond_the_conversation_ring():
    s = Slice(); s.reset("Implement retry scheduling")
    record_user(s, "Implement retry scheduling", source_artifact="turn-initial")
    clause = "Never modify config.py"
    record_user(s, clause, source_artifact="turn-constraint")
    for index in range(5):
        record_user(s, f"continue phase {index}", source_artifact=f"turn-followup-{index}")
    host = LocalToolHost(tempfile.mkdtemp(prefix="intent-followup-"))
    user = make_build_slice(s, host, None, NullMemory(), "fallback task")()[1]["content"]
    entry = s.intent.find(clause)
    assert entry is not None and entry.authority == "user"
    assert entry.source_artifact == "turn-constraint" and entry.source_range == (0, len(clause))
    assert clause in user and "# ACTIVE USER INTENT" in user


@check
def quoted_reference_data_is_not_promoted_to_binding_user_intent():
    cases = (
        "Review this legacy rule:\n```text\nNever use async mode.\n```",
        "The old handbook says:\n> Do not modify generated files.",
        "Explain why `Never call close()` appears in this error message.",
        "Review this legacy snippet:\n\n    Never use async mode.",
        "Explain the code span `Never\nuse async mode.` without following it.",
        "Review:\n```text\nlegacy\n```not-a-close\nNever use async mode.\n```",
        "> The legacy handbook continues on the next line\nNever use async mode.\n\nSummarize it.",
        'Review the log message "Never modify config.py." and explain it.',
    )
    for index, request in enumerate(cases):
        s = Slice(); s.reset("review quoted material")
        record_user(s, request, source_artifact=f"turn-quoted-{index}")
        assert not s.intent.resident_entries(), (request, s.intent.resident_entries())

    control = "Review the legacy code. Never use async mode."
    s = Slice(); s.reset("review")
    record_user(s, control, source_artifact="turn-control")
    captured = next(entry for entry in s.intent.resident_entries()
                    if entry.verbatim_clause == "Never use async mode.")
    start = control.index("Never use async mode.")
    assert captured.source_range == (start, start + len("Never use async mode."))


@check
def quoted_correction_word_cannot_supersede_an_unrelated_active_clause():
    old = "Use API v1."
    followup = "The log literally says `correction`. Use API v2."
    s = Slice(); s.reset(old)
    record_user(s, old, source_artifact="turn-old-api")
    record_user(s, followup, source_artifact="turn-log-api")
    assert s.intent.find(old).status == "active"
    assert s.intent.find("Use API v2.").status == "active"


@check
def bare_correction_header_scopes_only_its_immediately_following_directive():
    for index, header in enumerate(("Correction:", "Change of plan:", "Instead:")):
        old = "Use API v1."
        replacement = "Use API v2."
        s = Slice(); s.reset(old)
        record_user(s, old, source_artifact=f"turn-header-old-{index}")
        record_user(s, f"{header}\n{replacement}", source_artifact=f"turn-header-new-{index}")
        old_entry = s.intent.find(old)
        new_entry = s.intent.find(replacement)
        assert old_entry.status == "superseded" and old_entry.superseded_by == new_entry.id
        assert new_entry.kind == "constraint"
        assert s.intent.find(header) is None, "a correction header is syntax, not a standing obligation"


@check
def list_after_lazy_blockquote_remains_an_outside_directive():
    request = "> The legacy paragraph is quoted lazily\n- Never modify config.py."
    s = Slice(); s.reset("review")
    record_user(s, request, source_artifact="turn-lazy-list")
    entry = s.intent.find("Never modify config.py.")
    assert entry is not None and entry.authority == "user"


@check
def modal_question_is_not_promoted_but_a_polite_directive_is():
    s = Slice(); s.reset("discuss")
    record_user(s, "Do we really need to never use async?", source_artifact="turn-question")
    assert not s.intent.resident_entries()

    directive = "Could you please ensure the public API stays stable?"
    record_user(s, directive, source_artifact="turn-polite-directive")
    entry = s.intent.find(directive)
    assert entry is not None and entry.authority == "user"


@check
def one_shot_actions_stay_current_without_becoming_standing_constraints():
    cases = (
        "Review this project.",
        ("Review this project: spawn exactly 3 parallel explorer subagents — one each for app.py, "
         "auth.py and util.py — each reporting its top bug. Then give me a combined 3-line summary."),
        "Fix auth.py.",
        "Spawn exactly 3 explorer subagents.",
        "Run the test suite.",
        "Switch to the Hunter workspace.",
        "Implement the receipt upgrade.",
        "Refactor parser.py.",
        "Write me a summary.",
    )
    for index, request in enumerate(cases):
        s = Slice(); s.reset("task")
        record_user(s, request, source_artifact=f"turn-one-shot-{index}")
        assert s.intent.current_request == request
        assert not s.intent.resident_entries(), (request, s.intent.resident_entries())

    completed = Slice(); completed.reset("Review the project")
    record_user(completed, cases[1], source_artifact="turn-completed-one-shot")
    slice_sink(completed)(TurnEnd("end_turn", 1, {}))
    assert completed.task.objective_status == "provisionally_satisfied", \
        "a clean one-shot turn must not be kept active by a phantom standing constraint"

    constrained = Slice(); constrained.reset("Repair authentication")
    record_user(constrained, "Only modify auth.py.", source_artifact="turn-open-constraint")
    slice_sink(constrained)(TurnEnd("end_turn", 1, {}))
    assert constrained.task.objective_status == "active", \
        "a real still-binding constraint must continue to hold the task open"


@check
def format_configuration_and_explicit_corrections_remain_durable():
    durable = (
        "Use Python 3.12.",
        "Target PostgreSQL 15.",
        "Return exactly JSON.",
        "Output YAML.",
        "Format the response as a table.",
        "Keep the public API stable.",
        "Only modify auth.py.",
    )
    for index, request in enumerate(durable):
        s = Slice(); s.reset("task")
        record_user(s, request, source_artifact=f"turn-durable-{index}")
        entry = s.intent.find(request)
        assert entry is not None and entry.authority == "user", request

    s = Slice(); s.reset("Fix app.py.")
    record_user(s, "Fix app.py.", source_artifact="turn-action-old")
    assert not s.intent.resident_entries()
    correction = "Fix auth.py."
    record_user(s, f"Correction:\n{correction}", source_artifact="turn-action-correction")
    entry = s.intent.find(correction)
    assert entry is not None and entry.kind == "correction"


@check
def explicit_correction_supersedes_the_named_clause_and_prevents_goal_resurrection():
    old = "Keep the public API stable."
    correction = "Correction: do not keep the public API stable; breaking it is required."
    s = Slice(); s.reset(old)
    record_user(s, old, source_artifact="turn-old")
    record_user(s, correction, source_artifact="turn-correction")
    for index in range(5):
        record_user(s, f"continue phase {index}", source_artifact=f"turn-later-{index}")
    old_entry = s.intent.find(old)
    assert old_entry is not None and old_entry.status == "superseded" and old_entry.superseded_by
    resident = [entry.verbatim_clause for entry in s.intent.resident_entries()]
    assert old not in resident and any("do not keep" in clause for clause in resident)
    host = LocalToolHost(tempfile.mkdtemp(prefix="intent-correction-"))
    user = make_build_slice(s, host, None, NullMemory(), "fallback task")()[1]["content"]
    assert "# STABLE TASK OBJECTIVE" not in user, "the superseded original must not regain authority"


@check
def equivalent_wording_correction_cannot_resurrect_the_stable_objective():
    old = "You must preserve backwards compatibility."
    correction = "Actually, backwards compatibility is no longer required."
    s = Slice(); s.reset(old)
    record_user(s, old, source_artifact="turn-old-equivalent")
    record_user(s, correction, source_artifact="turn-correction-equivalent")
    for index in range(5):
        record_user(s, f"continue equivalent phase {index}",
                    source_artifact=f"turn-equivalent-later-{index}")
    old_entry = s.intent.find(old)
    assert old_entry is not None and old_entry.status == "superseded" and old_entry.superseded_by
    resident = [entry.verbatim_clause for entry in s.intent.resident_entries()]
    assert old not in resident and correction in resident
    host = LocalToolHost(tempfile.mkdtemp(prefix="intent-equivalent-correction-"))
    user = make_build_slice(s, host, None, NullMemory(), "fallback task")()[1]["content"]
    assert old not in user
    assert "# STABLE TASK OBJECTIVE" not in user, "equivalent correction must retire the old objective"


@check
def ordinary_version_correction_cannot_disappear_or_resurrect_the_old_objective():
    old = "Use Python 3.10."
    correction = "Actually, target Python 3.12 instead."
    s = Slice(); s.reset(old)
    record_user(s, old, source_artifact="turn-old-version")
    record_user(s, correction, source_artifact="turn-correction-version")
    for index in range(5):
        record_user(s, f"continue version phase {index}", source_artifact=f"turn-version-{index}")
    old_entry = s.intent.find(old)
    correction_entry = s.intent.find(correction)
    assert old_entry is not None and old_entry.status == "superseded"
    assert correction_entry is not None and correction_entry.status == "active"
    host = LocalToolHost(tempfile.mkdtemp(prefix="intent-version-correction-"))
    user = make_build_slice(s, host, None, NullMemory(), "fallback task")()[1]["content"]
    assert old not in user and correction in user
    assert "# STABLE TASK OBJECTIVE" not in user


@check
def actually_corrections_are_retained_and_retire_parallel_directives():
    cases = (
        ("Use Python 3.10.", "Actually, Python 3.12."),
        ("Never modify config.py.", "Actually, modifying config.py is allowed."),
        ("Return JSON.", "Actually, return YAML."),
    )
    for case_index, (old, correction) in enumerate(cases):
        s = Slice(); s.reset(old)
        record_user(s, old, source_artifact=f"turn-actually-old-{case_index}")
        record_user(s, correction, source_artifact=f"turn-actually-new-{case_index}")
        for index in range(5):
            record_user(s, f"continue actually phase {case_index}-{index}",
                        source_artifact=f"turn-actually-{case_index}-{index}")
        old_entry = s.intent.find(old)
        correction_entry = s.intent.find(correction)
        assert old_entry is not None and correction_entry is not None and correction_entry.status == "active"
        host = LocalToolHost(tempfile.mkdtemp(prefix="intent-actually-correction-"))
        user = make_build_slice(s, host, None, NullMemory(), "fallback task")()[1]["content"]
        if case_index < 2:
            assert old_entry.status == "superseded" and old not in user
        else:
            assert old_entry.status == "active" and correction_entry.kind == "correction"
            assert user.index(old) < user.index(correction)
        assert correction in user


@check
def one_correction_cannot_retire_sibling_requirements_with_the_same_action():
    cases = (
        (
            "Use Python 3.10. Use PostgreSQL 15.",
            "Actually, use Python 3.12.",
            "Use Python 3.10.", "Use PostgreSQL 15.",
        ),
        (
            "Return JSON for users. Return XML for legacy.",
            "Actually, return YAML for users.",
            "Return JSON for users.", "Return XML for legacy.",
        ),
    )
    for case_index, (goal, correction, replaced, retained) in enumerate(cases):
        s = Slice(); s.reset(goal)
        record_user(s, goal, source_artifact=f"turn-siblings-old-{case_index}")
        record_user(s, correction, source_artifact=f"turn-siblings-new-{case_index}")
        for index in range(5):
            record_user(s, f"continue sibling phase {case_index}-{index}",
                        source_artifact=f"turn-sibling-{case_index}-{index}")
        assert s.intent.find(replaced).status == "superseded"
        assert s.intent.find(retained).status == "active"
        host = LocalToolHost(tempfile.mkdtemp(prefix="intent-sibling-correction-"))
        user = make_build_slice(s, host, None, NullMemory(), "fallback task")()[1]["content"]
        assert replaced not in user and retained in user and correction in user


@check
def token_disjoint_actually_reversal_is_retained_without_guessing_a_target():
    old = "Keep the public API stable."
    correction = "Actually, breaking changes are allowed."
    s = Slice(); s.reset(old)
    record_user(s, old, source_artifact="turn-reversal-old")
    record_user(s, correction, source_artifact="turn-reversal-new")
    for index in range(5):
        record_user(s, f"continue reversal phase {index}", source_artifact=f"turn-reversal-{index}")
    assert s.intent.find(old).status == "active"
    assert s.intent.find(correction).status == "active"
    assert s.intent.find(correction).kind == "correction"
    host = LocalToolHost(tempfile.mkdtemp(prefix="intent-reversal-correction-"))
    user = make_build_slice(s, host, None, NullMemory(), "fallback task")()[1]["content"]
    assert old in user and correction in user and user.index(old) < user.index(correction)
    assert "newer exact wording overrides conflicting older objective text" in user


@check
def ambiguous_corrections_never_delete_an_unrelated_unique_constraint():
    cases = (
        ("Use Python 3.10.", "Actually, use Docker too."),
        ("Use Python 3.10.", "Actually, file deletion is allowed."),
    )
    for case_index, (old, correction) in enumerate(cases):
        s = Slice(); s.reset(old)
        record_user(s, old, source_artifact=f"turn-ambiguous-old-{case_index}")
        record_user(s, correction, source_artifact=f"turn-ambiguous-new-{case_index}")
        for index in range(5):
            record_user(s, f"continue ambiguous phase {case_index}-{index}",
                        source_artifact=f"turn-ambiguous-{case_index}-{index}")
        assert s.intent.find(old).status == "active"
        assert s.intent.find(correction).kind == "correction"


@check
def explicitly_additive_corrections_never_retire_existing_scopes():
    cases = (
        ("Use Python 3.10 for the runtime.", "Actually, use Python 3.12 for the docs too."),
        ("Return JSON for users.", "Actually, return JSON for admins too."),
    )
    for case_index, (old, addition) in enumerate(cases):
        s = Slice(); s.reset(old)
        record_user(s, old, source_artifact=f"turn-additive-old-{case_index}")
        record_user(s, addition, source_artifact=f"turn-additive-new-{case_index}")
        for index in range(5):
            record_user(s, f"continue additive phase {case_index}-{index}",
                        source_artifact=f"turn-additive-{case_index}-{index}")
        assert s.intent.find(old).status == "active"
        assert s.intent.find(addition).kind == "correction"


@check
def numeric_actually_facts_are_clarifications_not_standing_obligations():
    cases = (
        ("Fix the failing tests.", "Actually, there are 2 failing tests."),
        ("Use Python 3.10.", "Actually, Python 3.12 is installed."),
    )
    for case_index, (goal, clarification) in enumerate(cases):
        s = Slice(); s.reset(goal)
        record_user(s, goal, source_artifact=f"turn-fact-old-{case_index}")
        record_user(s, clarification, source_artifact=f"turn-fact-new-{case_index}")
        for index in range(5):
            record_user(s, f"continue fact phase {case_index}-{index}",
                        source_artifact=f"turn-fact-{case_index}-{index}")
        entry = s.intent.find(clarification)
        assert entry is not None and entry.kind == "correction"
        assert clarification not in [row["text"] for row in s.requirements]
        host = LocalToolHost(tempfile.mkdtemp(prefix="intent-fact-clarification-"))
        user = make_build_slice(s, host, None, NullMemory(), "fallback task")()[1]["content"]
        assert "not unchecked acceptance requirements" in user


@check
def correction_linkage_does_not_fuzzily_match_a_related_new_constraint():
    old = "You must preserve backwards compatibility."
    related = "Actually, backwards compatibility tests are required."
    s = Slice(); s.reset(old)
    record_user(s, old, source_artifact="turn-old-related")
    record_user(s, related, source_artifact="turn-related")
    old_entry = s.intent.find(old)
    assert old_entry is not None and old_entry.status == "active"
    assert related in [entry.verbatim_clause for entry in s.intent.resident_entries()]


@check
def reconciliation_gate_is_mandatory_context():
    s = Slice(); s.reset("repair")
    s.reconciliation_required = "run_command call-7 may still write"
    host = LocalToolHost(tempfile.mkdtemp(prefix="intent-reconcile-"))
    plan = make_build_slice(s, host, None, NullMemory(), "repair")()
    rendered = plan[1]["content"]
    block = next(item for item in plan.blocks if item.item_id == "region:reconciliation")
    assert block.mandatory and "EXECUTION RECONCILIATION REQUIRED" in rendered


@check
def raw_historical_requests_do_not_accumulate_in_seed():
    s = Slice(); s.reset("start")
    for i in range(12):
        record_user(s, f"unique-old-request-{i}")
    host = LocalToolHost(tempfile.mkdtemp(prefix="intent-history-"))
    user = make_build_slice(s, host, None, NullMemory(), "fallback")()[1]["content"]
    assert "# USER INSTRUCTIONS" not in user
    assert "unique-old-request-0" not in user, "old raw messages must not remain resident context"
    # The current request and bounded recent ring remain available.
    assert "unique-old-request-11" in user and "unique-old-request-9" in user


@check
def provisional_and_user_supersession_are_explicit():
    state = IntentState(current_request="keep API v1")
    old = state.add_from_current_request("keep API v1")
    assert old is not None
    state.mark_provisional("KEEP api V1", evidence_refs=("invocation:verify-1",))
    assert state.entries[0].status == "provisionally_satisfied"
    new = state.supersede_from_user("keep API v1", "API v2 is allowed")
    assert new is not None
    assert state.entries[0].status == "superseded"
    assert state.entries[0].superseded_by == new.id
    assert [e.verbatim_clause for e in state.resident_entries()] == ["API v2 is allowed"]


@check
def typed_record_roundtrip_preserves_all_fields():
    entry = IntentEntry(
        id="intent-9", verbatim_clause="exact clause", source_artifact="turn-4",
        source_range=(3, 15), authority="user", kind="correction", status="provisionally_satisfied",
        evidence_refs=("tool-7",),
    )
    state = IntentState.from_records([entry.to_dict()], current_request="now", next_id=10)
    assert state.entries == [entry] and state.current_request == "now" and state.next_id == 10


@check
def authoritative_records_reject_ambiguous_or_broken_supersession_graphs():
    def row(intent_id, *, status="active", superseded_by=None, text=None):
        return IntentEntry(
            id=intent_id, verbatim_clause=text or intent_id,
            status=status, superseded_by=superseded_by,
        ).to_dict()

    malformed = (
        [row("", text="missing id")],
        [row("intent-1"), row("intent-1", text="duplicate")],
        [row("intent-1", status="superseded", superseded_by="intent-missing")],
        [row("intent-1", status="superseded", superseded_by="intent-1")],
        [row("intent-1", superseded_by="intent-2"), row("intent-2")],
        [row("intent-1", status="superseded", superseded_by="intent-2"),
         row("intent-2", status="superseded", superseded_by="intent-1")],
    )
    for records in malformed:
        try:
            IntentState.from_records(records)
            assert False, f"malformed authoritative records were accepted: {records!r}"
        except ValueError:
            pass

    valid = IntentState.from_records([
        row("intent-1", status="superseded", superseded_by="intent-2"),
        row("intent-2"),
    ])
    assert valid.entries[0].superseded_by == "intent-2"


@check
def frozen_mapping_records_are_valid_deserializer_inputs():
    entry = IntentEntry(
        id="intent-4", verbatim_clause="only modify auth.py", source_artifact="turn-2",
        source_range=(3, 22), authority="user", kind="constraint",
        status="provisionally_satisfied", evidence_refs=("invocation:read-1",),
    )
    assert IntentEntry.from_dict(MappingProxyType(entry.to_dict())) == entry

    signal = ProgressSignal("read", "Inspected auth.py", 3)
    assert ProgressSignal.from_dict(MappingProxyType(signal.to_dict())) == signal

    legacy = IntentState()
    legacy.load_legacy_requirements((MappingProxyType({
        "text": "preserve compatibility", "done": True,
    }),))
    assert len(legacy.entries) == 1
    assert legacy.entries[0].verbatim_clause == "preserve compatibility"
    assert legacy.entries[0].authority == "legacy"
    assert legacy.entries[0].status == "provisionally_satisfied"


@check
def task_state_v2_and_legacy_v1_both_restore():
    s = Slice(); s.reset("implement it")
    _emit(s, "require", "keep compatibility")
    ts = slice_to_task_state(s, "t-intent", session_id="s-intent")
    restored = task_state_to_slice(ts)
    assert restored.intent.current_request == "implement it"
    assert restored.intent.entries[0].verbatim_clause == "keep compatibility"

    # Write/read the real markdown shape too.
    tmp = tempfile.mkdtemp(prefix="intent-state-")
    path = os.path.join(tmp, "t-intent.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_render_task_md(ts, created=_now_iso(), updated=_now_iso()))
    parsed = _parse_task_md(path)
    assert parsed is not None and parsed.intent_entries
    assert task_state_to_slice(parsed).intent.entries[0].verbatim_clause == "keep compatibility"

    # A real v1 markdown checkpoint with only Requirements imports them at legacy authority.
    legacy_path = os.path.join(tmp, "legacy.md")
    with open(legacy_path, "w", encoding="utf-8") as f:
        f.write(
            "---\ntype: task-state\nv: 1\nsession_id: old\ntask_id: old-task\n"
            "title: old\nstatus: active\nsince_edit: 0\nlinks: \ntags: \n---\n"
            "## Goal\nlegacy goal\n"
            "## Requirements\n- {\"text\": \"legacy exact\", \"done\": true}\n"
        )
    legacy_ts = _parse_task_md(legacy_path)
    assert legacy_ts is not None and not legacy_ts.intent_entries
    legacy = task_state_to_slice(legacy_ts)
    assert legacy.intent.entries[0].authority == "legacy"
    assert legacy.intent.entries[0].status == "provisionally_satisfied"


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
