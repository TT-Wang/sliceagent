"""I1 PROVENANCE — findings come from the WORLD, never the model's prose; mined lessons are
titled by the PITFALL, not the goal; self-inflicted/harness errors mine nothing. No model, no
pytest. Run: PYTHONPATH=src python3 tests/test_provenance.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.events import AssistantText, ToolResult, TurnEnd  # noqa: E402
from memagent.mining import LessonMiner, is_self_inflicted, pitfall_signature  # noqa: E402
from memagent.slice import (  # noqa: E402
    SYSTEM_PROMPT, Slice, is_done_claim, record_note, render_findings, render_slice, slice_sink,
)

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


class _Mem:
    is_durable = True
    def __init__(self):
        self.saved = []
    def remember(self, content, *, title="", scope="default", tags=""):
        self.saved.append((title, content))


# --- findings come ONLY from notes on real tool calls (NOT assistant narration) ----------

@check
def assistant_text_folded_as_unverified_claim():
    # root-cause revision of F1/C3: assistant reasoning IS carried forward (anti-re-derivation),
    # but as an UNVERIFIED claim — never an established fact — and pure narration is still dropped.
    s = Slice(); s.reset("build it")
    sink = slice_sink(s)
    sink(AssistantText("The aggregator is built and its tests pass"))  # substantive → folded as claim
    sink(AssistantText("Let me run it now"))                            # pure narration → dropped
    assert s.findings == ["The aggregator is built and its tests pass"], s.findings
    assert s.finding_source[s.findings[0]] == "claim", s.finding_source
    rendered = render_findings(s.findings, s.finding_source)
    assert "UNVERIFIED" in rendered and "do not re-derive" not in rendered.lower(), rendered


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


@check
def slice_tier_reframed_away_from_do_not_rederive():
    s = Slice(); s.reset("t")
    record_note(s, "root cause: off-by-one in tokenize()")
    user = render_slice(s, "(no files opened yet)")
    assert "do not re-derive" not in user.lower(), "the 'do not re-derive' ratchet framing survived"
    assert "verify against OPEN FILES" in user or "ground truth" in user.lower()


# --- mining: title from PITFALL not goal; self-inflicted errors mine nothing --------------

def _run_miner(state, *, fail_out, clear_error=True):
    mem = _Mem()
    miner = LessonMiner(mem, state, mode="deterministic", scope="test")
    miner(ToolResult("run_command", {"command": "x"}, fail_out, True))
    if clear_error:
        state.last_error = ""
    miner(TurnEnd("end_turn", 1, {}))
    return mem


@check
def mined_lesson_titled_by_pitfall_not_goal():
    s = Slice(); s.reset("ok lets create a simple news aggregator project")
    s.edited_files = {"news_agg.py"}
    mem = _run_miner(s, fail_out="Error: ModuleNotFoundError: No module named 'feedparser'")
    assert len(mem.saved) == 1
    title, content = mem.saved[0]
    assert "feedparser" in title or "ModuleNotFoundError" in title, f"title lost the pitfall: {title!r}"
    assert "create a simple news aggregator" not in title, f"goal leaked into the title: {title!r}"
    # the goal survives only as inline context, never as the headline
    assert "Pitfall:" in content and "news_agg.py" in content
    assert content.startswith("Pitfall:"), "the lesson body must LEAD with the pitfall, not the goal"


@check
def confinement_error_mines_nothing():
    # the agent hit its OWN sandbox (D2) — that is not an engineering pitfall → mine NOTHING
    s = Slice(); s.reset("write a file outside the workspace")
    s.edited_files = {"x.py"}
    err = ("Error: path escapes workspace (/repo): ~/Desktop/x.py — File tools are confined "
           "to the workspace.")
    mem = _run_miner(s, fail_out=err)
    assert mem.saved == [], f"a self-inflicted confinement error mined a junk lesson: {mem.saved}"


@check
def real_pitfall_after_self_inflicted_still_mines():
    # a turn with BOTH a confinement error AND a real error must mine the real one
    s = Slice(); s.reset("fix the importer")
    s.edited_files = {"imp.py"}
    mem = _Mem()
    miner = LessonMiner(mem, s, mode="deterministic", scope="test")
    miner(ToolResult("read_file", {"path": "/etc/x"}, "Error: path escapes workspace (/repo): /etc/x", True))
    miner(ToolResult("run_command", {"command": "python imp.py"}, "Error: ImportError: cannot import bar", True))
    s.last_error = ""
    miner(TurnEnd("end_turn", 1, {}))
    assert len(mem.saved) == 1
    title, _ = mem.saved[0]
    assert "ImportError" in title or "import bar" in title, f"mined the wrong pitfall: {title!r}"


@check
def is_self_inflicted_and_signature_helpers():
    assert is_self_inflicted("Error: path escapes workspace (/r): /etc/x")
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
