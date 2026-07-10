"""The reproducible benchmark must exercise the upgraded lifecycle, not a pre-upgrade approximation."""
from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace as NS

ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

spec = importlib.util.spec_from_file_location("sliceagent_benchmark_run", os.path.join(ROOT, "benchmarks", "run.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

from sliceagent.interfaces import AssistantMessage  # noqa: E402
import sliceagent.config as config_mod  # noqa: E402
import sliceagent.llm as llm_mod  # noqa: E402
import sliceagent.pfc as pfc_mod  # noqa: E402


class FakeLLM:
    def __init__(self):
        self.model = "fake-model"

    def set_cache_key(self, _key):
        return None

    def complete(self, _messages, _tools):
        return AssistantMessage(
            content="done", tool_calls=[], finish_reason="stop",
            usage={"prompt_tokens": 3, "completion_tokens": 1},
        )


def scenario():
    def setup(root):
        with open(os.path.join(root, "sentinel.txt"), "w", encoding="utf-8") as stream:
            stream.write("ready")

    def verify(root):
        return os.path.isfile(os.path.join(root, "sentinel.txt")), "sentinel"

    return {
        "name": "lifecycle-probe", "meta": {"max_steps_per_turn": 2},
        "prompts": ["Initial stable task", "Follow-up detail"],
        "setup": setup, "verify": verify,
    }


def config_fallback_uses_the_initialized_provider():
    old_load, old_prefs, old_llm = config_mod.load_config, config_mod.load_prefs, llm_mod.OpenAILLM
    saved = {key: os.environ.pop(key, None) for key in ("AGENT_MODEL", "LLM_API_KEY", "LLM_BASE_URL")}
    captured = {}

    class Capture:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    try:
        config_mod.load_config = lambda: NS(model="configured-model", api_key="configured-key",
                                            base_url="https://configured.invalid/v1", providers=lambda: {})
        config_mod.load_prefs = lambda: {}
        llm_mod.OpenAILLM = Capture
        bench._configured_llm()
        assert captured["model"] == "configured-model"
        assert captured["api_key"] == "configured-key"
        assert captured["base_url"] == "https://configured.invalid/v1"
    finally:
        config_mod.load_config, config_mod.load_prefs, llm_mod.OpenAILLM = old_load, old_prefs, old_llm
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def each_turn_seals_without_replacing_the_stable_objective():
    old_factory, old_slice = bench._configured_llm, pfc_mod.Slice
    seals = []

    class SpySlice(old_slice):
        def seal(self):
            seals.append(self.goal)
            super().seal()

    try:
        bench._configured_llm = FakeLLM
        pfc_mod.Slice = SpySlice
        result = bench.run(scenario())
    finally:
        bench._configured_llm, pfc_mod.Slice = old_factory, old_slice
    assert result["passed"] and [turn["stop"] for turn in result["per_turn"]] == ["end_turn", "end_turn"]
    assert seals == ["Initial stable task", "Initial stable task"]


def usage_tap_preserves_the_model_runner_contract():
    inner = NS(model="known-model", _base_url="https://provider.invalid/v1", max_tokens=8192,
               is_retryable=lambda _exc: True)
    tap = bench._Tap(inner)
    assert tap.model == inner.model and tap._base_url == inner._base_url and tap.max_tokens == 8192
    assert tap.is_retryable(RuntimeError("retry"))


def reducer_failure_fails_the_eval_instead_of_becoming_observer_noise():
    old_factory, old_sink = bench._configured_llm, pfc_mod.slice_sink
    try:
        bench._configured_llm = FakeLLM

        def broken_sink(_state):
            def sink(_event):
                raise RuntimeError("required benchmark reducer failed")
            return sink

        pfc_mod.slice_sink = broken_sink
        result = bench.run(scenario())
    finally:
        bench._configured_llm, pfc_mod.slice_sink = old_factory, old_sink
    assert not result["passed"] and "required benchmark reducer failed" in result["detail"]


def abnormal_turn_stop_fails_acceptance_even_when_repo_verifier_passes():
    class TruncatedLLM(FakeLLM):
        def complete(self, _messages, _tools):
            return AssistantMessage(
                content="partial", tool_calls=[], finish_reason="length",
                usage={"prompt_tokens": 3, "completion_tokens": 1},
            )

    old_factory = bench._configured_llm
    try:
        bench._configured_llm = TruncatedLLM
        result = bench.run(scenario())
    finally:
        bench._configured_llm = old_factory
    assert not result["passed"]
    assert "turn 1 stopped abnormally: max_tokens" in result["detail"]


def command_exit_status_reflects_failed_scenarios():
    old_run, old_load = bench.run, bench.load_scenario
    try:
        bench.load_scenario = lambda name: {"name": name}
        bench.run = lambda scn: {
            "scenario": scn["name"], "passed": False, "detail": "verification failed",
            "steps": 0, "peak_in": 0, "in_total": 0, "out_total": 0, "in_cached": 0,
            "wall_s": 0, "per_turn": [],
        }
        assert bench.main(["--scenario", "forced-failure"]) == 1
    finally:
        bench.run, bench.load_scenario = old_run, old_load


if __name__ == "__main__":
    config_fallback_uses_the_initialized_provider()
    print("PASS config_fallback_uses_the_initialized_provider")
    each_turn_seals_without_replacing_the_stable_objective()
    print("PASS each_turn_seals_without_replacing_the_stable_objective")
    usage_tap_preserves_the_model_runner_contract()
    print("PASS usage_tap_preserves_the_model_runner_contract")
    reducer_failure_fails_the_eval_instead_of_becoming_observer_noise()
    print("PASS reducer_failure_fails_the_eval_instead_of_becoming_observer_noise")
    abnormal_turn_stop_fails_acceptance_even_when_repo_verifier_passes()
    print("PASS abnormal_turn_stop_fails_acceptance_even_when_repo_verifier_passes")
    command_exit_status_reflects_failed_scenarios()
    print("PASS command_exit_status_reflects_failed_scenarios")
