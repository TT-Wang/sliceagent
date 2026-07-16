from __future__ import annotations

from types import SimpleNamespace as NS

from sliceagent.deliverables import (
    DeliverableRequirement,
    assess_deliverable,
    requirement_for_contract,
)
from sliceagent.events import AssistantText, ToolResult, TurnEnd
from sliceagent.execution import ToolInvocation, ToolStatus
from sliceagent.hooks import CompositeHooks, DeliverableCompletionHook, Hooks
from sliceagent.loop import run_turn
from sliceagent.memory import _parse_task_md, _render_task_md
from sliceagent.pfc import Slice, record_user, slice_sink
from sliceagent.registry import finalize_tool_outcome
from sliceagent.skills import SkillManager, make_skill_tool
from sliceagent.taskstate import slice_to_task_state, task_state_to_slice


INCIDENT_META = (
    "All 6 sealed reports were opened and cross-checked against the source. "
    "The findings above are the consolidated result; fan-in is complete."
)
VALID_REPORT = """## Findings

- **P1 · confirmed issue** — `src/sliceagent/loop.py:1570` accepts a terminal placeholder as the final
  response. The user receives no actual review. Recommendation: validate the declared response envelope before
  publication and preserve the rejected candidate only in the private model trajectory.

## Coverage and gaps

Inspected `src/sliceagent/loop.py`, `src/sliceagent/hooks.py`, and `src/sliceagent/skills.py`. Excluded `tests/fixtures/`;
the report does not make claims about provider-specific rendering.
"""
NO_FINDINGS_REPORT = """## Findings

No confirmed findings.

## Coverage and gaps

Inspected the entire repository, including `src/` and `tests/`. Generated files under `dist/` were excluded.
"""
PLAIN_REPORT = (
    "One confirmed issue is at src/sliceagent/loop.py:1570: a private progress placeholder can become the "
    "terminal answer. The practical fix is a one-shot response reminder. I inspected loop.py and hooks.py; "
    "provider-specific behavior remains outside this review."
)


class _Host:
    def schemas(self):
        return [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "search",
                "parameters": {"type": "object", "properties": {}},
            },
        }]

    def run(self, _name, _args):
        return "ok"


class _SequenceLLM:
    model = "deliverable-test-model"

    def __init__(self, *contents):
        self.contents = list(contents)
        self.calls = 0
        self.schemas = []

    def complete(self, _messages, schemas):
        self.schemas.append(list(schemas))
        content = self.contents[min(self.calls, len(self.contents) - 1)]
        self.calls += 1
        return NS(content=content, tool_calls=[], finish_reason="stop", usage={})


def _requirement(logical_id: str = "logical-review") -> DeliverableRequirement:
    requirement = requirement_for_contract(
        "code-review-report/v1", logical_id=logical_id, source="skill:review",
    )
    assert requirement is not None
    return requirement


def _run(llm, hooks, *, max_steps=4):
    events = []
    result = run_turn(
        build_slice=lambda: [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "review the code"},
        ],
        llm=llm,
        tools=_Host(),
        dispatch=events.append,
        hooks=hooks,
        max_steps=max_steps,
    )
    return result, events


def test_first_tool_free_response_publishes_without_a_host_rewrite():
    requirement = _requirement()
    hook = DeliverableCompletionHook(lambda: (requirement, requirement.logical_id))
    llm = _SequenceLLM(INCIDENT_META, VALID_REPORT)

    result, events = _run(llm, hook)

    published = [event.content for event in events if isinstance(event, AssistantText)]
    assert result.stop_reason == "end_turn"
    assert llm.calls == 1
    assert len(llm.schemas[0]) == 1
    assert published == [INCIDENT_META]
    assert len([event for event in events if isinstance(event, TurnEnd)]) == 1


def test_normal_reports_need_no_template_and_finish_in_one_call():
    for report in (VALID_REPORT, NO_FINDINGS_REPORT, PLAIN_REPORT, "No confirmed findings in the inspected code."):
        requirement = _requirement()
        llm = _SequenceLLM(report)
        result, events = _run(
            llm, DeliverableCompletionHook(lambda r=requirement: (r, r.logical_id)),
        )
        assert result.stop_reason == "end_turn"
        assert llm.calls == 1
        assert [event.content for event in events if isinstance(event, AssistantText)] == [report]


def test_contract_assessment_never_creates_a_hidden_second_model_call():
    requirement = _requirement()
    llm = _SequenceLLM(INCIDENT_META, VALID_REPORT)

    result, events = _run(
        llm, DeliverableCompletionHook(lambda: (requirement, requirement.logical_id)),
    )

    assert result.stop_reason == "end_turn"
    assert llm.calls == 1
    assert [event.content for event in events if isinstance(event, AssistantText)] == [INCIDENT_META]
    assert len([event for event in events if isinstance(event, TurnEnd)]) == 1


def test_contract_assessment_does_not_consume_the_last_available_step():
    requirement = _requirement()
    llm = _SequenceLLM(INCIDENT_META, "a closeout that must never be requested")
    result, events = _run(
        llm,
        DeliverableCompletionHook(lambda: (requirement, requirement.logical_id)),
        max_steps=1,
    )
    assert result.stop_reason == "end_turn"
    assert llm.calls == 1
    assert [event.content for event in events if isinstance(event, AssistantText)] == [INCIDENT_META]


def test_contract_scope_isolated_and_workspace_transport_can_own_its_edge():
    ordinary = _SequenceLLM(INCIDENT_META, PLAIN_REPORT)
    result, events = _run(ordinary, DeliverableCompletionHook(lambda: (None, "logical")))
    assert result.stop_reason == "end_turn" and ordinary.calls == 1
    assert [event.content for event in events if isinstance(event, AssistantText)] == [INCIDENT_META]

    class _Handoff(Hooks):
        def __init__(self):
            self.continued = False

        def should_continue_after_stop(self, _stop_reason):
            if self.continued:
                return None
            self.continued = True
            return {
                "continue": True,
                "exclusive": True,
                "feedback": "The workspace transport requires one continuation.",
            }

    requirement = _requirement()
    transport = _SequenceLLM(INCIDENT_META, PLAIN_REPORT)
    result, transport_events = _run(
        transport,
        CompositeHooks(
            _Handoff(),
            DeliverableCompletionHook(lambda: (requirement, requirement.logical_id)),
        ),
    )
    assert result.stop_reason == "end_turn" and transport.calls == 2
    assert [event.content for event in transport_events if isinstance(event, AssistantText)] == [PLAIN_REPORT]


def test_assessor_only_nudges_private_pointers_and_deferred_progress_updates():
    requirement = _requirement()
    fake = """## Findings

The findings above include an issue at `src/sliceagent/loop.py:1570`. Recommendation: see above for details.

## Coverage and gaps

Coverage is complete for `src/`.
"""
    vague = """## Findings

No confirmed findings.

## Coverage and gaps

Coverage is complete and there are no gaps.
"""
    deferred = "Good, I have the Wave 1 findings. Let me confirm the Wave 2 reports before answering."
    assert not assess_deliverable(requirement, fake).complete
    assert not assess_deliverable(requirement, deferred).complete
    assert assess_deliverable(requirement, vague).complete
    assert assess_deliverable(requirement, PLAIN_REPORT).complete


def test_skill_metadata_emits_typed_activation_without_changing_workgraph_revision(tmp_path):
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text(
        "---\nname: review\ndescription: review code\n"
        "completion-contract: code-review-report/v1\n---\nReview it.\n",
        encoding="utf-8",
    )
    manager = SkillManager([str(tmp_path)])
    assert manager.get("review").completion_contract == "code-review-report/v1"
    tool = make_skill_tool(manager)
    assert tool is not None
    invocation = ToolInvocation("skill-call", "skill", {"name": "review"}, 0)
    outcome = finalize_tool_outcome(
        invocation, tool.handler(dict(invocation.args)), entry=tool,
    )
    assert outcome.status is ToolStatus.SUCCEEDED
    assert outcome.effects[0].kind == "skill_activated"

    state = Slice()
    state.reset("review the code")
    record_user(
        state,
        "review the code",
        source_event_id="event:user:review",
        logical_id="logical-review",
    )
    revision = state.active_work.revision
    slice_sink(state)(ToolResult(
        name="skill",
        args=dict(invocation.args),
        output=outcome.text,
        failing=False,
        status=outcome.status.value,
        invocation_id=invocation.id,
        outcome=outcome,
    ))

    requirement = state.task.deliverable_requirement
    assert requirement is not None and requirement.logical_id == "logical-review"
    assert state.active_work.revision == revision


def test_deliverable_roundtrip_and_new_logical_request_retirement(tmp_path):
    state = Slice()
    state.reset("review")
    record_user(state, "review", source_event_id="event:1", logical_id="logical-1")
    state.task.bind_deliverable(_requirement("logical-1"))

    restored = task_state_to_slice(slice_to_task_state(state, "task", session_id="session"))
    assert restored.task.deliverable_requirement == state.task.deliverable_requirement

    task_state = slice_to_task_state(state, "task", session_id="session")
    # The legacy human-readable task projection remains a supported restart/compatibility surface.
    # Keep the typed envelope there too; older files simply omit the additive section.
    task_path = tmp_path / "task.md"
    task_path.write_text(
        _render_task_md(task_state, created="now", updated="now"), encoding="utf-8",
    )
    parsed = _parse_task_md(str(task_path))
    assert parsed is not None
    assert parsed.deliverable_requirement == task_state.deliverable_requirement

    record_user(restored, "new task", source_event_id="event:2", logical_id="logical-2")
    assert restored.task.deliverable_requirement is None
