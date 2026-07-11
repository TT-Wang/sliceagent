"""I1 PROVENANCE — findings come from the WORLD, never the model's prose; mined lessons are
titled by the PITFALL, not the goal; self-inflicted/harness errors mine nothing. No model, no
pytest. Run: PYTHONPATH=src python3 tests/test_provenance.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.events import AssistantText, ToolResult  # noqa: E402
from sliceagent.neocortex import is_self_inflicted, pitfall_signature  # noqa: E402
from sliceagent.pfc import Slice, record_user, slice_sink  # noqa: E402
from sliceagent.seed import render_slice  # noqa: E402
from sliceagent.prompt import SYSTEM_PROMPT  # noqa: E402
from sliceagent.regions import is_done_claim, record_note, render_findings  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def system_prompt_forbids_narration_in_replies():
    # REGRESSION (the reasoning-leak quality bug): the model was dumping its planning monologue into the
    # visible reply ("Let me draft…", "Final response coming up", "I should…"). The <communication> contract
    # must explicitly forbid narrating the process — replies are the answer, not a scratchpad.
    comm = SYSTEM_PROMPT[SYSTEM_PROMPT.index("<communication>"):SYSTEM_PROMPT.index("</communication>")]
    low = comm.lower()
    assert "scratchpad" in low, "communication rule must say replies are not a scratchpad"
    assert "silently" in low or "not shown" in low, "must tell the model to think silently"
    assert "let me" in low and "narrate" in low, "must name the narration anti-pattern explicitly"
    assert "preamble" in low, "must forbid preamble/postamble (lead with the answer)"


# --- findings come ONLY from notes on real tool calls (NOT assistant narration) ----------

@check
def assistant_text_is_archived_for_continuity_not_promoted_to_evidence():
    # Generic answer prose is not typed evidence. It remains in the bounded conversation ring and sealed
    # artifact, while load-bearing conclusions must come through an observed/tool-note path.
    s = Slice(); s.reset("build it")
    record_user(s, "build it")
    sink = slice_sink(s)
    answer = "The aggregator is built and its tests pass"
    sink(AssistantText(answer))
    assert s.findings == [] and s.conversation[-1]["assistant"] == answer
    assert render_findings(s.findings, s.finding_source) == ""


@check
def distinct_assistant_status_turns_do_not_form_a_shadow_transcript():
    s = Slice(); s.reset("long task")
    sink = slice_sink(s)
    for index in range(25):
        record_user(s, f"continue {index}")
        sink(AssistantText(f"status update {index}: still working"))
        s.seal()
    assert len(s.conversation) == 4
    assert s.findings == [] and s.finding_source == {}


@check
def narration_note_arg_dropped():
    # even via the note arg, a pure intent/narration note carries no durable fact → dropped
    s = Slice(); s.reset("t")
    for narration in ("Let me run it", "I'll check the file", "Now I will edit x.py",
                      "Next, run the tests", "okay, let's start"):
        record_note(s, narration)
    assert s.findings == [], f"narration folded as findings: {s.findings}"


@check
def real_fact_note_is_kept():
    s = Slice(); s.reset("t")
    record_note(s, "root cause: the lexer drops trailing newlines")
    assert s.findings == ["root cause: the lexer drops trailing newlines"]


# --- source tags + trust framing ---------------------------------------------------------

@check
def note_on_successful_call_is_tool_note():
    s = Slice(); s.reset("t")
    sink = slice_sink(s)
    sink(ToolResult("read_file", {"path": "a.py", "note": "uses a queue, not a list"},
                    "contents", False))
    assert s.findings == ["uses a queue, not a list"]
    assert s.finding_source["uses a queue, not a list"] == "tool-note"


@check
def note_on_failing_call_is_claim():
    s = Slice(); s.reset("t")
    sink = slice_sink(s)
    sink(ToolResult("run_command", {"command": "x", "note": "the parser handles unicode"},
                    "Error: boom", True))
    assert s.findings == ["the parser handles unicode"]
    assert s.finding_source["the parser handles unicode"] == "claim"  # no observation backed it


@check
def done_claim_downgraded_even_on_success():
    # a "done" assertion is NOT durable just because the call succeeded — it needs an observation.
    s = Slice(); s.reset("t")
    sink = slice_sink(s)
    sink(ToolResult("run_command", {"command": "ls", "note": "the task is done"}, "ok", False))
    assert s.finding_source["the task is done"] == "claim"


@check
def is_done_claim_detects_completion_language():
    assert is_done_claim("Done — built it") and is_done_claim("the feature works now")
    assert is_done_claim("task complete") and is_done_claim("already implemented")
    assert not is_done_claim("the lexer drops newlines")


@check
def render_findings_frames_by_source_no_re_derive():
    findings = ["uses a queue", "the task is done"]
    sources = {"uses a queue": "tool-note", "the task is done": "claim"}
    out = render_findings(findings, sources)
    assert "verify against OPEN FILES" in out          # tool-note framing
    assert "UNVERIFIED" in out                          # claim framing
    assert "do not re-derive" not in out.lower()        # the dangerous framing is GONE
    assert "UNVERIFIED" in render_findings(["legacy"], {"legacy": "unknown-source"}), \
        "unknown persisted provenance must fail closed rather than render as observed"


@check
def delegated_fan_in_remains_recallable_without_becoming_observed_workspace_truth():
    s = Slice(); s.reset("review the project")
    sink = slice_sink(s)
    mixed = (
        "child report: review completed; parser.py is unsafe | primary observation [obs:abc123]: return value | "
        'full report: read_file("subagents/sub-1.md")'
    )
    sink(ToolResult(
        "spawn_agent", {"agent": "explorer", "task": "inspect parser.py"}, mixed, False,
    ))
    assert mixed in s.findings, "bounded child fan-in must remain available for cross-turn recall"
    assert s.finding_source[mixed] == "delegated"
    rendered = render_findings(s.findings, s.finding_source)
    assert "delegated testimony" in rendered and "UNVERIFIED" in rendered
    assert 'read_file("subagents/sub-1.md")' in rendered, "the refinement handle must survive"


@check
def slice_tier_reframed_away_from_do_not_rederive():
    s = Slice(); s.reset("t")
    record_note(s, "root cause: off-by-one in tokenize()")
    user = render_slice(s, "(no files opened yet)")
    assert "do not re-derive" not in user.lower(), "the 'do not re-derive' ratchet framing survived"
    assert "verify against OPEN FILES" in user or "ground truth" in user.lower()


# --- mining behaviour (pitfall-titling, self-inflicted filtering, multi-error pick) moved to
# test_consolidate.py: distillation is now CACHE-only (the per-turn slice-reading LessonMiner was
# removed); promote_episodes owns it. The pure helpers below still live in mining.py. ---------

@check
def is_self_inflicted_and_signature_helpers():
    assert is_self_inflicted("Error: path escapes the boundary (/r): /etc/x")
    assert is_self_inflicted("Error: Permission denied: /root")
    assert not is_self_inflicted("Error: ImportError: no module named foo")
    # signature strips the host's 'Error:' prefix so the title leads with the real failure
    assert pitfall_signature("Error: ImportError: no module named foo").startswith("ImportError")


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
