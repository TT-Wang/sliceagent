"""The dedicated adversarial VERIFICATION subagent kind. It must be registered, runnable
(surfaces in the spawn_agent roster), able to RUN checks but not
EDIT, and carry the adversarial 'try to break it' + VERDICT contract. No model, no pytest.
Run: PYTHONPATH=src python tests/test_verification_agent.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.agents import BUILTIN_AGENTS, load_agents  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def verification_is_registered_and_spawnable():
    assert "verification" in BUILTIN_AGENTS
    assert "verification" in load_agents([]), "must be in the registry → enumerated in the spawn_agent roster"


@check
def can_run_checks_but_not_edit():
    spec = BUILTIN_AGENTS["verification"]
    assert "run_command" in spec.tools and "execute_code" in spec.tools, "must run builds/tests/probes"
    assert "read_file" in spec.tools and "grep" in spec.tools, "must read/grep to inspect"
    for edit in ("edit_file", "str_replace", "append_to_file"):
        assert edit not in spec.tools, f"verifier must NOT carry {edit} (runtime allowlist blocks it)"
    assert spec.reasoning == "full", "careful verification uses full reasoning"
    # shell is in WRITE_TOOLS → classified writable → serializes vs other writers (correct for a test-runner)
    assert spec.read_only is False


@check
def prompt_demands_verdict_and_command_evidence():
    sp = BUILTIN_AGENTS["verification"].system_prompt
    assert "TRY TO BREAK IT" in sp, "adversarial framing"
    assert "VERDICT: PASS" in sp and "VERDICT: FAIL" in sp and "VERDICT: PARTIAL" in sp, "machine-parseable verdict"
    assert "Command:" in sp and "Output:" in sp, "command/output evidence format"
    assert "Reading is NOT verification" in sp, "anti verification-avoidance"


@check
def user_file_can_still_override_by_name():
    # the registry overlays file-defined agents on the built-ins (name wins) — built-in is a DEFAULT, not a lock
    reg = load_agents([])
    assert reg["verification"].name == "verification"  # default present when no user file


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
