"""Experimental feature flags: precedence master > per-flag env > default, live env read,
typo-safe unknowns. No model, no pytest. Run: PYTHONPATH=src python tests/test_flags.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sliceagent.flags as flags  # noqa: E402
from sliceagent.flags import Flag  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _clear_env(*names):
    for n in names:
        os.environ.pop(n, None)


@check
def default_governs_when_no_env():
    flags.register(Flag("feat_off", "x", default=False))
    flags.register(Flag("feat_on", "x", default=True))
    _clear_env("AGENT_EXPERIMENTAL_ALL", "AGENT_EXPERIMENTAL_FEAT_OFF", "AGENT_EXPERIMENTAL_FEAT_ON")
    assert flags.enabled("feat_off") is False
    assert flags.enabled("feat_on") is True


@check
def per_flag_env_overrides_default_both_ways():
    flags.register(Flag("feat_off", "x", default=False))
    flags.register(Flag("feat_on", "x", default=True))
    _clear_env("AGENT_EXPERIMENTAL_ALL")
    os.environ["AGENT_EXPERIMENTAL_FEAT_OFF"] = "1"     # force a default-off flag ON
    os.environ["AGENT_EXPERIMENTAL_FEAT_ON"] = "off"    # force a default-on flag OFF
    try:
        assert flags.enabled("feat_off") is True
        assert flags.enabled("feat_on") is False
    finally:
        _clear_env("AGENT_EXPERIMENTAL_FEAT_OFF", "AGENT_EXPERIMENTAL_FEAT_ON")


@check
def master_switch_forces_all_on():
    flags.register(Flag("feat_off", "x", default=False))
    _clear_env("AGENT_EXPERIMENTAL_FEAT_OFF")
    os.environ["AGENT_EXPERIMENTAL_ALL"] = "true"
    try:
        assert flags.enabled("feat_off") is True
    finally:
        _clear_env("AGENT_EXPERIMENTAL_ALL")


@check
def unknown_flag_is_off():
    assert flags.enabled("does_not_exist") is False


@check
def live_env_read_no_cache():
    flags.register(Flag("live", "x", default=False))
    _clear_env("AGENT_EXPERIMENTAL_ALL")
    os.environ.pop("AGENT_EXPERIMENTAL_LIVE", None)
    assert flags.enabled("live") is False
    os.environ["AGENT_EXPERIMENTAL_LIVE"] = "yes"       # change mid-process → reflected immediately
    try:
        assert flags.enabled("live") is True
    finally:
        _clear_env("AGENT_EXPERIMENTAL_LIVE")


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
