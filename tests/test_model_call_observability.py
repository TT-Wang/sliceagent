"""Every physical provider request is observable without fabricating turn/step boundaries."""
from __future__ import annotations

import copy
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.context import (  # noqa: E402
    ContextBlock,
    Fidelity,
    FreshnessClass,
    InstructionClass,
    RepresentationLoss,
    SeedPlan,
)
from sliceagent.context_overflow import ContextOverflow  # noqa: E402
from sliceagent.events import ApiRetry, ModelCallPrepared, SliceBuilt  # noqa: E402
from sliceagent.hooks import Hooks  # noqa: E402
from sliceagent.loop import run_turn  # noqa: E402


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _block(item: str, content: str, *, fidelity=Fidelity.FULL,
           loss=RepresentationLoss.NONE, handles=()):
    return ContextBlock(
        block_id=f"{item}:{fidelity.value}", item_id=item, alternative_group=item,
        priority=2, instruction_class=InstructionClass.DATA, freshness=FreshnessClass.LIVE,
        fidelity=fidelity, representation_loss=loss, content=content, handles=tuple(handles),
    )


def _plan(*blocks):
    return SeedPlan(
        system="system", blocks=blocks,
        render_blocks=lambda selection: "".join(block.content for block in selection.blocks),
        request_block="CURRENT REQUEST: preserve exactly\n", now_block="NOW",
    )


class Host:
    def schemas(self):
        return []

    def run(self, _name, _args):
        return "file contents"


@check
def reactive_unknown_window_reports_each_exact_physical_attempt():
    full = _block("workspace", "FULL:" + "x" * 180)
    locator = _block(
        "workspace", "LOCATOR:read_file(a.py)", fidelity=Fidelity.LOCATOR,
        loss=RepresentationLoss.POINTER_ONLY, handles=("a.py",),
    )
    plan = _plan(full, locator)

    class LLM:
        model = "uncatalogued-observability-model"

        def __init__(self):
            self.seen = []

        def complete(self, messages, _schemas):
            self.seen.append(copy.deepcopy(messages))
            if "FULL:" in messages[1]["content"]:
                raise ContextOverflow(RuntimeError("provider context_length_exceeded"))
            return NS(content="done", tool_calls=[], finish_reason="stop", usage={})

    llm, events = LLM(), []
    result = run_turn(
        build_slice=lambda: plan, llm=llm, tools=Host(), dispatch=events.append,
        hooks=Hooks(), max_steps=1,
    )
    prepared = [event for event in events if isinstance(event, ModelCallPrepared)]
    assert result.stop_reason == "end_turn" and len(llm.seen) >= 2
    assert len([event for event in events if isinstance(event, SliceBuilt)]) == 1
    assert [event.step for event in prepared] == [1] * len(llm.seen)
    assert [event.attempt for event in prepared] == list(range(1, len(llm.seen) + 1))
    assert [event.messages for event in prepared] == llm.seen
    assert all(event.preflight_mode == "compatibility-unknown" for event in prepared)
    assert "FULL:" in prepared[0].messages[1]["content"]
    assert "LOCATOR:" in prepared[-1].messages[1]["content"]


@check
def later_same_turn_calls_get_their_own_step_scoped_observation():
    plan = _plan(_block("workspace", "small live context"))

    class LLM:
        model = "uncatalogued-observability-model"

        def __init__(self):
            self.seen = []

        def complete(self, messages, _schemas):
            self.seen.append(copy.deepcopy(messages))
            if len(self.seen) == 1:
                call = NS(id="read-1", name="read_file", args={"path": "a.py"})
                return NS(content="", tool_calls=[call], finish_reason="tool_calls", usage={})
            return NS(content="done", tool_calls=[], finish_reason="stop", usage={})

    llm, events = LLM(), []
    result = run_turn(
        build_slice=lambda: plan, llm=llm, tools=Host(), dispatch=events.append,
        hooks=Hooks(), max_steps=3,
    )
    prepared = [event for event in events if isinstance(event, ModelCallPrepared)]
    assert result.stop_reason == "end_turn" and len(llm.seen) == 2
    assert len([event for event in events if isinstance(event, SliceBuilt)]) == 1
    assert [(event.step, event.attempt) for event in prepared] == [(1, 1), (2, 1)]
    assert [event.messages for event in prepared] == llm.seen
    assert any(message.get("role") == "tool" for message in prepared[1].messages)


@check
def sdk_retry_attempts_are_observed_before_each_provider_io():
    from sliceagent import errors

    plan = _plan(_block("workspace", "small live context"))

    class LLM:
        model = "uncatalogued-observability-model"

        def __init__(self):
            self.seen = []

        def complete(self, messages, _schemas):
            self.seen.append(copy.deepcopy(messages))
            if len(self.seen) == 1:
                raise TimeoutError("temporary provider timeout")
            return NS(content="done", tool_calls=[], finish_reason="stop", usage={})

    old_sleep = errors.time.sleep
    errors.time.sleep = lambda _seconds: None
    try:
        llm, events = LLM(), []
        result = run_turn(
            build_slice=lambda: plan, llm=llm, tools=Host(), dispatch=events.append,
            hooks=Hooks(), max_steps=1,
        )
    finally:
        errors.time.sleep = old_sleep

    prepared = [event for event in events if isinstance(event, ModelCallPrepared)]
    assert result.stop_reason == "end_turn" and len(llm.seen) == 2
    assert [(event.step, event.attempt) for event in prepared] == [(1, 1), (1, 2)]
    assert [event.messages for event in prepared] == llm.seen
    assert len([event for event in events if isinstance(event, ApiRetry)]) == 1


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
