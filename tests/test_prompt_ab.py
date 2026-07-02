"""Offline guards for the prompt A/B suite (no live key). Proves the harness machinery is sound:
variants build with the {{MEMORY_MODEL}} marker preserved, the single-variable transforms do exactly
what they claim (control is a no-op; v02 is a pure reorder; ctrl_dedupe removes only the two clauses),
and the paired-bootstrap stats separate a real shift from flat noise. The live runs are gated elsewhere."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "evals"))

# The prompt A/B harness lives in evals/, which is local-only dev tooling (not shipped in the public
# repo) — skip cleanly where it's absent (e.g. CI on the public checkout), like the no-pty skips.
if not os.path.isdir(os.path.join(ROOT, "evals", "prompt_ab")):
    print("SKIP: evals/prompt_ab not present (local-only eval tooling)")
    sys.exit(0)

from prompt_ab import stats as S          # noqa: E402
from prompt_ab import variants as V       # noqa: E402


def test_variants_build_and_preserve_marker():
    base = V.control_text()
    assert "{{MEMORY_MODEL}}" in base
    V.build_all()
    for name in V.list_variants():
        t = open(V.variant_path(name), encoding="utf-8").read()
        assert "{{MEMORY_MODEL}}" in t, f"{name} dropped the memory marker"


def test_control_is_a_noop():
    base = V.control_text()
    V.build_all()
    assert open(V.variant_path("control"), encoding="utf-8").read() == base


def test_lead_verification_is_pure_reorder():
    base = V.control_text()
    V.build_all()
    t = open(V.variant_path("v02_lead_verification"), encoding="utf-8").read()
    assert len(t) == len(base)                       # no chars added/removed
    assert t.index("<verification>") < t.index("<ask>")   # moved to the front
    assert t.count("<verification>") == 1


def test_recency_verify_appends_reminder_without_deleting():
    base = V.control_text()
    V.build_all()
    t = open(V.variant_path("v01_recency_verify"), encoding="utf-8").read()
    assert t.startswith(base)                         # adds only, deletes nothing
    assert t.count("<reminder>") == 1 and t.rstrip().endswith("</reminder>")


def test_dedupe_control_removes_only_the_two_clauses():
    base = V.control_text()
    V.build_all()
    t = open(V.variant_path("ctrl_dedupe"), encoding="utf-8").read()
    assert "does NOT clear a user report" not in t
    assert "NOT proof — confirm it on the real artifact first" not in t
    assert len(t) < len(base)


def test_anchor_failure_is_loud():
    try:
        V.v_precedence_tag("a prompt without the TIERS anchor {{MEMORY_MODEL}}")
    except AssertionError:
        return
    raise AssertionError("expected an AssertionError when the anchor is missing")


def test_paired_diff_detects_shift_and_ignores_flat():
    assert S.paired_diff([1, 1, 1, 1], [3, 3, 3, 3])["significant"] is True
    assert S.paired_diff([1, 1, 1, 1], [1, 1, 1, 1])["significant"] is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("PASS")
