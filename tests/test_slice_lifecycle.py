"""Slice field-lifecycle enforcement (#1). Closes the silent carry-by-omission trap: the flat-peak moat
and cross-topic isolation depend on reset()/seal() handling EVERY Slice field correctly, but that was kept
in sync by hand. This suite fails the moment a field is added without a conscious lifecycle decision —
(a) every field is classified in _SLICE_SEAL_POLICY, (b) reset() wipes all fields to default (a forgotten
field leaks across tasks), (c) seal() RESETS every transient field and CARRIES every durable one (a forgotten
transient field silently accumulates across turns → breaks the moat). No model, no network.
Run: PYTHONPATH=src python tests/test_slice_lifecycle.py
"""
import os
import sys
from dataclasses import fields

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.pfc import Slice, _SLICE_SEAL_POLICY  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _sentinel(default):
    """A non-default value of the same shape — proves a lifecycle method actually touched the field."""
    if isinstance(default, bool):   return not default          # bool BEFORE int (bool is an int subclass)
    if isinstance(default, int):    return default + 12345
    if isinstance(default, str):    return "SENTINEL"
    if isinstance(default, set):    return {"SENTINEL"}
    if isinstance(default, dict):   return {"SENTINEL": 1}
    if isinstance(default, list):   return ["SENTINEL"]
    return "SENTINEL"


_FIELDS = [f.name for f in fields(Slice)]


@check
def every_field_is_classified_exactly_once():
    classified, actual = set(_SLICE_SEAL_POLICY), set(_FIELDS)
    missing, stale = actual - classified, classified - actual
    assert not missing, f"unclassified Slice field(s) — add to _SLICE_SEAL_POLICY: {sorted(missing)}"
    assert not stale, f"_SLICE_SEAL_POLICY names field(s) that no longer exist: {sorted(stale)}"
    assert set(_SLICE_SEAL_POLICY.values()) <= {"carry", "reset", "custom"}, "unknown lifecycle policy value"


@check
def reset_wipes_every_field_to_default():
    s = Slice()
    for name in _FIELDS:                                        # dirty EVERY field
        setattr(s, name, _sentinel(getattr(s, name)))
    s.reset("newgoal")
    fresh = Slice()
    for name in _FIELDS:
        expected = "newgoal" if name == "goal" else getattr(fresh, name)
        assert getattr(s, name) == expected, f"reset() left field {name!r} dirty → leaks across tasks"


@check
def seal_resets_transient_and_carries_durable():
    for name, policy in _SLICE_SEAL_POLICY.items():
        if policy == "custom":
            continue                                            # cap/filter logic has its own dedicated tests
        default = getattr(Slice(), name)
        s = Slice()
        setattr(s, name, _sentinel(default))                    # isolate: dirty ONLY this field
        s.seal()
        if policy == "reset":
            assert getattr(s, name) == default, \
                f"seal() must RESET transient field {name!r} (else it accumulates across turns → moat break)"
        else:  # carry
            assert getattr(s, name) == _sentinel(default), \
                f"seal() must CARRY durable field {name!r} (else it's lost across turns)"


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
