"""Canonical context-block and global elasticity contracts. No model, no pytest."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.context import (  # noqa: E402
    ContextBlock,
    ContextUnfitError,
    ElasticityController,
    Fidelity,
    FreshnessClass,
    InstructionClass,
    PressureLevel,
    RepresentationLoss,
    SeedPlan,
)

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _block(name, text, *, fidelity=Fidelity.FULL, loss=RepresentationLoss.NONE,
           priority=5, mandatory=False, handles=(), reobservable=False, order=0):
    return ContextBlock(
        block_id=f"{name}:{fidelity.value}", item_id=name, alternative_group=name,
        priority=priority, instruction_class=InstructionClass.TASK_STATE,
        freshness=FreshnessClass.DERIVED, fidelity=fidelity, representation_loss=loss,
        content=text, handles=handles, mandatory=mandatory, reobservable=reobservable,
        order=order,
    )


@check
def one_alternative_per_semantic_item():
    full = _block("report", "x" * 100)
    digest = _block("report", "summary", fidelity=Fidelity.DIGEST,
                    loss=RepresentationLoss.SUMMARY, handles=("artifact:1",))
    selected = ElasticityController().select([full, digest])
    assert len(selected.blocks) == 1 and selected.blocks[0] == full


@check
def global_pressure_degrades_low_priority_first():
    low_full = _block("history", "h" * 100, priority=1)
    low_ptr = _block("history", "history:1", fidelity=Fidelity.LOCATOR,
                     loss=RepresentationLoss.POINTER_ONLY, priority=1, handles=("history:1",))
    high = _block("failure", "f" * 70, priority=9, order=1)
    selected = ElasticityController().select([low_full, low_ptr, high], capacity_chars=90)
    assert {b.block_id for b in selected.blocks} == {low_ptr.block_id, high.block_id}
    assert selected.pressure in (PressureLevel.ELEVATED, PressureLevel.TIGHT, PressureLevel.CRITICAL)


@check
def incomplete_representation_requires_recovery():
    try:
        _block("x", "summary", fidelity=Fidelity.DIGEST, loss=RepresentationLoss.SUMMARY)
        assert False, "lossy block without a recovery path must be rejected"
    except ValueError as exc:
        assert "recovery" in str(exc)


@check
def revision_tagged_live_excerpt_can_be_reobserved():
    excerpt = _block("file:a", "lines 10-20", fidelity=Fidelity.EXCERPT,
                     loss=RepresentationLoss.SELECTION, reobservable=True)
    assert ElasticityController().select([excerpt]).blocks == (excerpt,)


@check
def mandatory_meaning_never_degrades_lossily():
    exact = _block("intent", "do exactly this", mandatory=True, priority=100)
    try:
        _block("intent", "do this", fidelity=Fidelity.DIGEST, loss=RepresentationLoss.SUMMARY,
               mandatory=True, handles=("turn:1",))
        assert False, "mandatory lossy representation must be rejected"
    except ValueError:
        pass
    try:
        ElasticityController().select([exact], capacity_chars=3)
        assert False, "mandatory state that cannot fit must fail honestly"
    except ContextUnfitError as exc:
        assert exc.mandatory_items == ("intent",)


@check
def request_sandwich_drops_to_one_exact_copy_before_declaring_unfit():
    request = "CURRENT REQUEST (verbatim):\nline one\n    exact  spacing\n"
    plan = SeedPlan(
        system="system", blocks=(), render_blocks=lambda _selection: "",
        request_block=request, now_block="NOW",
    )
    one_copy_capacity = plan._fixed_user_chars(1)
    user = plan.project(one_copy_capacity)[1]["content"]
    assert user.count(request) == 1 and "    exact  spacing" in user
    assert plan.last_request_copies == 1


@check
def authority_freshness_priority_and_fidelity_are_independent():
    live_data = ContextBlock(
        block_id="file:full", item_id="file", alternative_group="file", priority=8,
        instruction_class=InstructionClass.DATA, freshness=FreshnessClass.LIVE,
        fidelity=Fidelity.FULL, representation_loss=RepresentationLoss.NONE, content="bytes",
    )
    assert live_data.instruction_class is InstructionClass.DATA
    assert live_data.freshness is FreshnessClass.LIVE
    assert live_data.priority == 8 and live_data.fidelity is Fidelity.FULL


@check
def model_runner_reprojects_same_seed_as_trajectory_grows():
    from types import SimpleNamespace as NS

    from sliceagent.hooks import Hooks
    from sliceagent.loop import run_turn

    full = _block("workspace", "FULL:" + "x" * 280, priority=2)
    locator = _block(
        "workspace", "LOCATOR:read_file(a.py)", fidelity=Fidelity.LOCATOR,
        loss=RepresentationLoss.POINTER_ONLY, priority=2, handles=("a.py",),
    )
    plan = SeedPlan(
        system="system", blocks=(full, locator),
        render_blocks=lambda selection: "".join(block.content for block in selection.blocks),
        request_block="CURRENT REQUEST: preserve-me\n", now_block="NOW",
    )

    class LLM:
        context_window = 700
        max_tokens = 20

        def __init__(self):
            self.seen = []

        def complete(self, messages, _schemas):
            import copy
            self.seen.append(copy.deepcopy(messages))
            if len(self.seen) == 1:
                return NS(content="", tool_calls=[NS(id="c1", name="read_file", args={"path": "a.py"})],
                          finish_reason="tool_calls", usage={})
            return NS(content="done", tool_calls=[], finish_reason="stop", usage={})

    class Host:
        def schemas(self):
            return []

        def accesses(self, _name, _args):
            return []

        def run(self, _name, _args):
            return "o" * 240

    llm = LLM()
    result = run_turn(build_slice=lambda: plan, llm=llm, tools=Host(),
                      dispatch=lambda _event: None, hooks=Hooks(), max_steps=3)
    first = llm.seen[0][1]["content"]
    second = llm.seen[1][1]["content"]
    assert "FULL:" in first and "LOCATOR:" in second and "FULL:" not in second
    assert "preserve-me" in first and "preserve-me" in second
    assert result.stop_reason == "end_turn"


@check
def hook_injected_messages_participate_in_capacity_projection():
    from types import SimpleNamespace as NS

    from sliceagent.hooks import Hooks
    from sliceagent.loop import run_turn

    full = _block("workspace", "FULL:" + "x" * 220, priority=2)
    locator = _block(
        "workspace", "LOCATOR", fidelity=Fidelity.LOCATOR,
        loss=RepresentationLoss.POINTER_ONLY, priority=2, handles=("a.py",),
    )
    plan = SeedPlan(
        system="system", blocks=(full, locator),
        render_blocks=lambda selection: "".join(block.content for block in selection.blocks),
        request_block="CURRENT REQUEST: preserve-me\n", now_block="NOW",
    )

    class Inject(Hooks):
        def __init__(self):
            self.calls = 0

        def prepare_messages(self, messages):
            self.calls += 1
            return [*messages, {"role": "user", "content": "H" * 300}]

    class LLM:
        context_window = 700
        max_tokens = 20

        def __init__(self):
            self.seen = []

        def complete(self, messages, _schemas):
            self.seen.append(messages)
            return NS(content="done", tool_calls=[], finish_reason="stop", usage={})

    class Host:
        def schemas(self):
            return []

    llm = LLM()
    hook = Inject()
    events = []
    result = run_turn(build_slice=lambda: plan, llm=llm, tools=Host(),
                      dispatch=events.append, hooks=hook, max_steps=1)
    assert result.stop_reason == "end_turn" and len(llm.seen) == 1
    assert "LOCATOR" in llm.seen[0][1]["content"] and "FULL:" not in llm.seen[0][1]["content"]
    assert hook.calls == 1, f"one provider call replayed prepare_messages {hook.calls} times"
    from sliceagent.events import SliceBuilt
    built = next(event for event in events if isinstance(event, SliceBuilt))
    assert built.messages == llm.seen[0], "SliceBuilt and provider must observe the same prepared value"


@check
def in_place_policy_injection_survives_post_hook_elastic_tightening():
    from types import SimpleNamespace as NS

    from sliceagent.hooks import Hooks
    from sliceagent.loop import run_turn

    full = _block("workspace", "FULL:" + "x" * 220, priority=2)
    locator = _block(
        "workspace", "LOCATOR", fidelity=Fidelity.LOCATOR,
        loss=RepresentationLoss.POINTER_ONLY, priority=2, handles=("a.py",),
    )
    plan = SeedPlan(
        system="system", blocks=(full, locator),
        render_blocks=lambda selection: "".join(block.content for block in selection.blocks),
        request_block="CURRENT REQUEST: preserve-me\n", now_block="NOW",
    )

    class InjectPolicy(Hooks):
        def __init__(self):
            self.calls = 0

        def prepare_messages(self, messages):
            self.calls += 1
            messages[0]["content"] += "\nPOLICY:" + "H" * 300
            return None

    class LLM:
        context_window = 700
        max_tokens = 20

        def __init__(self):
            self.seen = []

        def complete(self, messages, _schemas):
            self.seen.append(messages)
            return NS(content="done", tool_calls=[], finish_reason="stop", usage={})

    class Host:
        def schemas(self):
            return []

    hook = InjectPolicy(); llm = LLM()
    result = run_turn(build_slice=lambda: plan, llm=llm, tools=Host(),
                      dispatch=lambda _event: None, hooks=hook, max_steps=1)
    assert result.stop_reason == "end_turn" and hook.calls == 1
    assert "POLICY:" in llm.seen[0][0]["content"], "elasticity discarded a hook's policy injection"
    assert "LOCATOR" in llm.seen[0][1]["content"] and "FULL:" not in llm.seen[0][1]["content"]


@check
def real_unknown_window_overflow_tightens_graded_seed_instead_of_parking():
    from types import SimpleNamespace as NS

    from sliceagent.context_overflow import ContextOverflow
    from sliceagent.hooks import Hooks
    from sliceagent.loop import run_turn

    full = _block("workspace", "FULL:" + "x" * 180, priority=2)
    locator = _block(
        "workspace", "LOCATOR:read_file(a.py)", fidelity=Fidelity.LOCATOR,
        loss=RepresentationLoss.POINTER_ONLY, priority=2, handles=("a.py",),
    )
    plan = SeedPlan(
        system="system", blocks=(full, locator),
        render_blocks=lambda selection: "".join(block.content for block in selection.blocks),
        request_block="CURRENT REQUEST: preserve exactly\n", now_block="NOW",
    )

    class UnknownWindowLLM:
        model = "uncatalogued-test-model"

        def __init__(self):
            self.seen = []

        def complete(self, messages, _schemas):
            self.seen.append(messages)
            if "FULL:" in messages[1]["content"]:
                raise ContextOverflow(RuntimeError("provider context_length_exceeded"))
            return NS(content="done", tool_calls=[], finish_reason="stop", usage={})

    class Host:
        def schemas(self):
            return []

    llm = UnknownWindowLLM()
    result = run_turn(build_slice=lambda: plan, llm=llm, tools=Host(),
                      dispatch=lambda _event: None, hooks=Hooks(), max_steps=1)
    assert result.stop_reason == "end_turn", result.stop_reason
    assert "LOCATOR:" in llm.seen[-1][1]["content"] and len(llm.seen) >= 2


@check
def provider_overflow_can_correct_a_stale_positive_window_estimate():
    from types import SimpleNamespace as NS

    from sliceagent.context_overflow import ContextOverflow
    from sliceagent.hooks import Hooks
    from sliceagent.loop import run_turn

    full = _block("workspace", "FULL:" + "x" * 180, priority=2)
    locator = _block(
        "workspace", "LOCATOR", fidelity=Fidelity.LOCATOR,
        loss=RepresentationLoss.POINTER_ONLY, priority=2, handles=("a.py",),
    )
    plan = SeedPlan(
        system="system", blocks=(full, locator),
        render_blocks=lambda selection: "".join(block.content for block in selection.blocks),
        request_block="CURRENT REQUEST: preserve exactly\n", now_block="NOW",
    )

    class StaleEstimateLLM:
        model = "known-but-stale"
        context_window = 10_000
        max_tokens = 20

        def __init__(self):
            self.seen = []

        def complete(self, messages, _schemas):
            self.seen.append(messages)
            if "FULL:" in messages[1]["content"]:
                raise ContextOverflow(RuntimeError("provider counted more than the local estimate"))
            return NS(content="done", tool_calls=[], finish_reason="stop", usage={})

    class Host:
        def schemas(self):
            return []

    llm = StaleEstimateLLM()
    result = run_turn(build_slice=lambda: plan, llm=llm, tools=Host(),
                      dispatch=lambda _event: None, hooks=Hooks(), max_steps=1)
    assert result.stop_reason == "end_turn"
    assert "LOCATOR" in llm.seen[-1][1]["content"]


@check
def fallback_model_reprojects_without_the_primary_models_reactive_hint():
    from types import SimpleNamespace as NS

    from sliceagent.context_overflow import ContextOverflow
    from sliceagent.hooks import Hooks
    from sliceagent.loop import run_turn

    full = _block("workspace", "FULL:" + "x" * 180, priority=2)
    locator = _block(
        "workspace", "LOCATOR", fidelity=Fidelity.LOCATOR,
        loss=RepresentationLoss.POINTER_ONLY, priority=2, handles=("a.py",),
    )
    request = "CURRENT REQUEST: exact\n"
    plan = SeedPlan(
        system="system", blocks=(full, locator),
        render_blocks=lambda selection: "".join(block.content for block in selection.blocks),
        request_block=request, now_block="NOW",
    )

    class RoutedLLM:
        model = "primary-small"
        max_tokens = 20

        def __init__(self):
            self.seen = []

        @property
        def context_window(self):
            return 10_000 if self.model == "fallback-large" else 0

        def complete(self, messages, _schemas):
            self.seen.append((self.model, messages))
            if self.model == "primary-small":
                raise ContextOverflow(RuntimeError("primary rejected every representation"))
            return NS(content="done", tool_calls=[], finish_reason="stop", usage={})

    class Host:
        def schemas(self):
            return []

    saved = os.environ.get("AGENT_MODEL_FALLBACK")
    os.environ["AGENT_MODEL_FALLBACK"] = "fallback-large"
    try:
        llm = RoutedLLM()
        result = run_turn(build_slice=lambda: plan, llm=llm, tools=Host(),
                          dispatch=lambda _event: None, hooks=Hooks(), max_steps=1)
    finally:
        if saved is None:
            os.environ.pop("AGENT_MODEL_FALLBACK", None)
        else:
            os.environ["AGENT_MODEL_FALLBACK"] = saved
    assert result.stop_reason == "end_turn"
    fallback_user = next(messages[1]["content"] for model, messages in llm.seen
                         if model == "fallback-large")
    assert "FULL:" in fallback_user and fallback_user.count(request) == 2


@check
def findings_offer_a_durable_locator_under_global_pressure():
    import tempfile

    from sliceagent.memory import NullMemory
    from sliceagent.pfc import Slice
    from sliceagent.seed import make_build_slice
    from sliceagent.tools import LocalToolHost

    state = Slice(); state.reset("review")
    state.findings = ["observed: " + "x" * 30_000]
    plan = make_build_slice(
        state, LocalToolHost(tempfile.mkdtemp(prefix="findings-elastic-")), None,
        NullMemory(), "review",
    )()
    alternatives = [block for block in plan.blocks if block.item_id == "region:findings"]
    assert {block.fidelity for block in alternatives} >= {Fidelity.FULL, Fidelity.LOCATOR}
    locator = next(block for block in alternatives if block.fidelity is Fidelity.LOCATOR)
    selected = plan.controller.select(alternatives, capacity_chars=len(locator.content))
    assert selected.blocks == (locator,) and "artifacts/index.md" in locator.content


@check
def closeout_reprojects_seed_for_its_own_trajectory():
    from types import SimpleNamespace as NS

    from sliceagent.hooks import Hooks
    from sliceagent.loop import run_turn

    full = _block("workspace", "FULL:" + "x" * 220, priority=2)
    locator = _block(
        "workspace", "LOCATOR", fidelity=Fidelity.LOCATOR,
        loss=RepresentationLoss.POINTER_ONLY, priority=2, handles=("a.py",),
    )
    plan = SeedPlan(
        system="system", blocks=(full, locator),
        render_blocks=lambda selection: "".join(block.content for block in selection.blocks),
        request_block="CURRENT REQUEST: preserve-me\n", now_block="NOW",
    )

    class LLM:
        context_window = 800
        max_tokens = 20

        def __init__(self):
            self.seen = []

        def complete(self, messages, _schemas):
            self.seen.append(messages)
            return NS(content="best summary", tool_calls=[], finish_reason="stop", usage={})

    class Host:
        def schemas(self):
            return []

    llm = LLM()
    result = run_turn(build_slice=lambda: plan, llm=llm, tools=Host(),
                      dispatch=lambda _event: None, hooks=Hooks(), max_steps=0)
    assert result.stop_reason == "max_steps" and len(llm.seen) == 1
    assert "LOCATOR" in llm.seen[0][1]["content"] and "FULL:" not in llm.seen[0][1]["content"]


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
