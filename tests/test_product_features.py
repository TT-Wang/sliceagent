"""Feature-parity additions: the edit journal / undo (B4), the code_review tool (D1), and the live cost
meter (D4). No model, no pytest. Run: PYTHONPATH=src python tests/test_product_features.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.tools import LocalToolHost                          # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# ---- B4: edit journal / undo ------------------------------------------------
@check
def undo_reverts_then_removes_then_empties():
    wd = tempfile.mkdtemp(prefix="undo-")
    host = LocalToolHost(root=wd)
    host._t_edit_file({"path": "a.py", "content": "x = 1\n"})        # create (prev=None)
    host._t_str_replace({"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"})  # edit (prev="x = 1")
    assert host.read_text("a.py").strip() == "x = 2"
    assert "Undid" in host.undo_last()                              # revert the str_replace
    assert host.read_text("a.py").strip() == "x = 1"
    assert "removed" in host.undo_last()                            # revert the create → file gone
    assert not os.path.exists(os.path.join(wd, "a.py"))
    assert "Nothing to undo" in host.undo_last()


@check
def undo_handles_append():
    wd = tempfile.mkdtemp(prefix="undo-app-")
    host = LocalToolHost(root=wd)
    host._t_edit_file({"path": "log.txt", "content": "line1\n"})
    host._t_append({"path": "log.txt", "content": "line2\n"})
    assert "line2" in host.read_text("log.txt")
    host.undo_last()                                                # undo the append
    assert host.read_text("log.txt") == "line1\n"


# ---- D1: code_review tool ----------------------------------------------------
def _git(wd, *args):
    subprocess.run(["git", "-C", wd, *args], capture_output=True, text=True, check=False)


@check
def code_review_returns_the_diff():
    if not shutil.which("git"):
        print("  (skip: git not installed)"); return
    wd = tempfile.mkdtemp(prefix="cr-")
    _git(wd, "init", "-q")
    _git(wd, "config", "user.email", "t@t.dev")
    _git(wd, "config", "user.name", "t")
    open(os.path.join(wd, "f.py"), "w").write("a = 1\n")
    _git(wd, "add", "-A"); _git(wd, "commit", "-qm", "init")
    open(os.path.join(wd, "f.py"), "w").write("a = 2\n")            # modify after commit
    host = LocalToolHost(root=wd)
    out = host._t_code_review({"ref": "HEAD"})
    assert "f.py" in out and "+a = 2" in out, out
    # after committing, the tree matches HEAD → "no changes"
    _git(wd, "commit", "-qam", "change")
    assert "No changes" in host._t_code_review({"ref": "HEAD"})


@check
def code_review_inventory_does_not_hide_untracked_files():
    if not shutil.which("git"):
        print("  (skip: git not installed)"); return
    wd = tempfile.mkdtemp(prefix="cr-untracked-")
    _git(wd, "init", "-q")
    _git(wd, "config", "user.email", "t@t.dev")
    _git(wd, "config", "user.name", "t")
    open(os.path.join(wd, "tracked.py"), "w").write("x = 1\n")
    _git(wd, "add", "-A"); _git(wd, "commit", "-qm", "init")
    open(os.path.join(wd, "new.py"), "w").write("created = True\n")
    out = LocalToolHost(root=wd)._t_code_review({"ref": "HEAD"})
    assert out.startswith("[code review: tracked + untracked inventory]")
    assert "?? new.py" in out and "no untracked files exist" not in out.lower(), out


@check
def code_review_inventory_does_not_hide_ignored_late_effects():
    if not shutil.which("git"):
        print("  (skip: git not installed)"); return
    wd = tempfile.mkdtemp(prefix="cr-ignored-")
    _git(wd, "init", "-q")
    _git(wd, "config", "user.email", "t@t.dev")
    _git(wd, "config", "user.name", "t")
    open(os.path.join(wd, ".gitignore"), "w").write("build/\n*.late\n")
    open(os.path.join(wd, "tracked.py"), "w").write("x = 1\n")
    _git(wd, "add", "-A"); _git(wd, "commit", "-qm", "init")
    os.makedirs(os.path.join(wd, "build", "nested"))
    open(os.path.join(wd, "build", "nested", "result.bin"), "wb").write(b"late")
    open(os.path.join(wd, "effect.late"), "w").write("landed")
    out = LocalToolHost(root=wd)._t_code_review({"ref": "HEAD", "include_ignored": True})
    assert out.startswith("[workspace observation: tracked + untracked + ignored inventory complete]")
    assert "!! effect.late" in out and "!! build/nested/result.bin" in out, out


@check
def code_review_errors_outside_a_repo():
    wd = tempfile.mkdtemp(prefix="cr-nogit-")
    host = LocalToolHost(root=wd)
    out = host._t_code_review({"ref": "HEAD"})
    assert "Error" in out, out                                     # not a git repo → graceful error
    assert getattr(out, "ok", True) is False


@check
def code_review_is_registered_as_a_tool():
    host = LocalToolHost(root=tempfile.mkdtemp(prefix="cr-reg-"))
    names = [s["function"]["name"] for s in host.schemas()]
    assert "code_review" in names


# ---- D4: cost meter ----------------------------------------------------------
@check
def cost_accrues_for_known_model_only():
    from sliceagent.tui import _accrue_cost, _price
    assert _price("kimi-k2.7-code") is not None
    assert _price("some-unknown-model") is None
    stats = {"model": "kimi-k2.7-code"}
    _accrue_cost(stats, {"input_other": 1_000_000, "input_cache_read": 0, "output": 0})
    assert abs(stats["cost"] - 0.60) < 1e-9, stats                 # 1M fresh input × $0.60/1M
    _accrue_cost(stats, {"output": 1_000_000})                     # + 1M output × $2.50/1M
    assert abs(stats["cost"] - (0.60 + 2.50)) < 1e-9, stats
    created = {"model": "kimi-k2.7-code"}
    _accrue_cost(created, {"input_cache_creation": 1_000_000})
    assert abs(created["cost"] - 0.60) < 1e-9, created             # cache writes are billed input, not free
    assert created["_transcript_tok"] == 1_000_000, created       # and become later transcript history
    unknown = {"model": "mystery-llm"}
    _accrue_cost(unknown, {"input_other": 1_000_000})
    assert "cost" not in unknown                                   # unknown price → no $ shown


# ---- D3: model fallback on overflow -----------------------------------------
@check
def model_fallback_swaps_once_when_configured():
    from types import SimpleNamespace
    from sliceagent.loop import _try_model_fallback
    os.environ.pop("AGENT_MODEL_FALLBACK", None)
    llm = SimpleNamespace(model="small-ctx")
    assert _try_model_fallback(llm) is False                       # nothing configured → no swap
    os.environ["AGENT_MODEL_FALLBACK"] = "big-ctx"
    try:
        assert _try_model_fallback(llm) is True and llm.model == "big-ctx"
        assert _try_model_fallback(llm) is False                   # only once (sticky)
    finally:
        os.environ.pop("AGENT_MODEL_FALLBACK", None)


# ---- B5: plain-mode sink readability ----------------------------------------
@check
def plain_sink_is_readable_and_quiet_on_reads():
    import contextlib
    import io
    from sliceagent.cli import cli_sink
    from sliceagent.events import AssistantText, ToolResult
    s = cli_sink()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        s(ToolResult("run_command", {"command": "pytest -q"}, "3 passed", False))
        s(ToolResult("read_file", {"path": "secret.py"}, "TOP_SECRET_CONTENT", False))
        s(ToolResult("run_command", {"command": "boom"}, "Traceback", True))
        s(AssistantText("here is the answer"))
    out = buf.getvalue()
    assert "✓ run_command pytest -q" in out and "3 passed" in out, out
    assert "TOP_SECRET_CONTENT" not in out, "read content must not be dumped in plain mode"
    assert "✗ run_command boom" in out and "Traceback" in out, "failures show output"
    assert "here is the answer" in out


@check
def plain_sink_survives_none_usage():
    # TurnEnd is the model-loop boundary, not durable completion; a later commit owns the saved claim.
    import contextlib
    import io
    from sliceagent.cli import cli_sink
    from sliceagent.events import TurnCommitted, TurnEnd
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sink = cli_sink()
        sink(TurnEnd("end_turn", 5, None))
        assert "done" not in buf.getvalue()
        sink(TurnCommitted(True, "end_turn"))
    assert "turn saved" in buf.getvalue()


# ---- streaming-to-content (RichSink) ----------------------------------------
@check
def richsink_streams_content_then_finalizes_once():
    import io
    from rich.console import Console
    from sliceagent.events import AssistantText, TurnCommitted, TurnEnd, TurnStarted
    from sliceagent.tui import make_rich_sink
    for force in (False, True):                    # non-tty (fallback) AND tty-like (Live) must both be safe
        buf = io.StringIO()
        c = Console(file=buf, force_terminal=force, width=80, soft_wrap=False)
        sink = make_rich_sink(c, {"model": "test-model"}, await_commit=True)
        sink(TurnStarted("req"))                   # starts truthful progress before slice construction
        sink.on_delta("content", "Hello ")         # deltas change semantic phase (no preview)
        sink.on_delta("content", "world — the fix is X.")
        assert sink.progress.snapshot().phase.value == "writing"
        sink(AssistantText("Hello world — the fix is X."))
        assert "fix is X" not in buf.getvalue(), "terminal answer must wait for completion + commit"
        sink(TurnEnd("end_turn", 1, {}))
        sink(TurnCommitted(True, "end_turn"))
        out = buf.getvalue()
        assert "fix is X" in out, f"(force_terminal={force}) reply missing: {out!r}"
        assert out.count("fix is X") == 1, out


@check
def richsink_ondelta_noop_when_idle():
    # a content delta with no active step (e.g. during routing) must not crash or start a live region
    import io
    from rich.console import Console
    from sliceagent.tui import make_rich_sink
    sink = make_rich_sink(Console(file=io.StringIO(), force_terminal=False), {})
    sink.on_delta("content", "stray")
    assert sink._status is None


@check
def streaming_reply_shows_a_fixed_single_line_status_not_a_growing_region():
    # Reported live THREE times: stacked "assistant streaming…" panels instead of one region updating in
    # place. Every earlier fix that still showed the GROWING reply — a Markdown rich.live.Live panel, then a
    # plain-text bounded panel, then a bounded reply-tail INSIDE this status line — shared one root cause: a
    # multi-row region whose height grows/wraps eventually reaches the bottom of the terminal and forces a
    # scroll, and ANSI erase codes cannot un-scroll content already in scrollback, so stale frames pile up.
    # Root fix: while streaming, show a FIXED single-line "writing…" status (the exact shape of the
    # "thinking…" spinner, which has never misbehaved) — no reply preview at all. The full reply still prints
    # once via AssistantText. Pin: on_delta creates no Live/Panel and embeds no reply text; the status render
    # stays a short constant-length line no matter how long the reply gets.
    import io
    from rich.console import Console
    from sliceagent.tui import make_rich_sink, _LiveStatus
    from sliceagent.events import TurnStarted
    sink = make_rich_sink(Console(file=io.StringIO(), force_terminal=True, width=80), {"model": "x"})
    sink(TurnStarted("req"))                            # arms the status before slice construction
    try:
        assert not hasattr(sink, "_live"), "on_delta must not create a separate rich.live.Live panel"
        before = _LiveStatus(sink).__rich__().plain
        sink.on_delta("content", "z" * 5000)           # a very long reply must not enlarge the status line
        after = _LiveStatus(sink).__rich__().plain
        assert "Writing" in after, f"the status must reflect the writing phase, got {after!r}"
        assert "z" * 40 not in after, "the status line must NOT embed the growing reply text"
        assert len(after) < 120 and abs(len(after) - len(before)) < 20, \
            f"the status must stay a short, fixed-length single line regardless of reply length ({len(after)})"
    finally:
        sink._stop()   # a Status left running past the test would dangle a live region / refresh thread


# ---- image input (vision) ----------------------------------------------------
def _slice_msgs(host, goal="do the thing"):
    from sliceagent.memory import NullMemory
    from sliceagent.retriever import NullRetriever
    from sliceagent.pfc import Slice
    from sliceagent.seed import make_build_slice
    s = Slice(); s.reset(goal)
    return make_build_slice(s, host, NullRetriever(), NullMemory(), goal)()


@check
def build_user_content_is_a_string_without_images():
    # THE moat invariant: a text-only turn's user content stays a plain string (no multimodal list).
    host = LocalToolHost(root=tempfile.mkdtemp(prefix="img-none-"))
    msgs = _slice_msgs(host)
    assert msgs[1]["role"] == "user" and isinstance(msgs[1]["content"], str), "text turn must stay a string"


@check
def build_attaches_pending_images_as_parts_and_consumes():
    host = LocalToolHost(root=tempfile.mkdtemp(prefix="img-parts-"))
    host.pending_images = [{"path": "a.png", "b64": "QUJD", "mime": "image/png"}]
    content = _slice_msgs(host)[1]["content"]
    assert isinstance(content, list), "with images, content becomes a multimodal parts list"
    assert content[0]["type"] == "text" and isinstance(content[0]["text"], str)
    assert content[1]["type"] == "image_url"
    assert "data:image/png;base64,QUJD" in content[1]["image_url"]["url"]
    assert host.pending_images == [], "images are consumed into the seed (cleared in place)"


@check
def attach_image_encodes_and_errors_cleanly():
    wd = tempfile.mkdtemp(prefix="att-")
    open(os.path.join(wd, "x.png"), "wb").write(b"\x89PNG\r\n\x1a\n" + b"payload")
    host = LocalToolHost(root=wd)
    msg = host.attach_image("x.png")
    assert "attached image" in msg and len(host.pending_images) == 1
    assert host.pending_images[0]["mime"] == "image/png" and host.pending_images[0]["b64"]
    assert "Error" in host.attach_image("missing.png")


@check
def attach_image_rejects_spoofed_and_sniffs_real_type():
    wd = tempfile.mkdtemp(prefix="sniff-")
    open(os.path.join(wd, "fake.png"), "wb").write(b"this is not really a png")    # .png ext, wrong bytes
    open(os.path.join(wd, "real.jpg"), "wb").write(b"\xff\xd8\xff\xe0" + b"jpegbody")
    host = LocalToolHost(root=wd)
    assert "not a recognized image" in host.attach_image("fake.png"), "spoofed extension must be rejected"
    assert host.pending_images == []
    assert "image/jpeg" in host.attach_image("real.jpg")            # sniffed from magic bytes
    assert host.pending_images and host.pending_images[0]["mime"] == "image/jpeg"


@check
def vision_capability_is_gated_by_model_name():
    from sliceagent.model_catalog import capability
    assert capability("kimi-k2.7-code").supports_vision is False, "the default code model is text-only"
    assert capability("moonshot-v1-8k-vision").supports_vision is True
    assert capability("gpt-4o").supports_vision is True
    assert capability("claude-sonnet-4").supports_vision is True
    assert capability("deepseek-chat").supports_vision is False


# ---- subagent activity → ONE dynamic line (not a line per child tool call) ---
@check
def subagent_activity_is_compact_counting_not_json_spam():
    from sliceagent.events import SubagentProgress, ToolStarted
    from sliceagent.subagent import _nested_sink
    updates = []
    sink = _nested_sink(
        updates.append, depth=1, agent_id="child-7", parent_turn_id="turn-4",
        launch_ordinal=2, kind="explorer", name="ui-audit",
    )
    for i in range(5):
        sink(ToolStarted("read_file", {"path": f"f{i}.py"}))
    assert all(isinstance(update, SubagentProgress) for update in updates), updates
    latest = updates[-1]
    assert latest.agent_id == "child-7" and latest.parent_turn_id == "turn-4"
    assert latest.tool_count == 5 and latest.sequence == 5 and "f4.py" in latest.detail, latest
    assert "{" not in latest.detail, "must show the primary arg, not raw JSON args"


@check
def subagent_activity_does_not_leave_the_last_tool_pinned_during_report_synthesis():
    from sliceagent.events import AssistantText, StepBegin, ToolStarted
    from sliceagent.subagent import _nested_sink
    updates = []
    sink = _nested_sink(
        updates.append, depth=1, agent_id="child-8", parent_turn_id="turn-4",
        launch_ordinal=3, kind="explorer",
    )
    sink(StepBegin(2))
    sink(ToolStarted("grep", {"pattern": "receipt"}))
    sink(AssistantText("private child report prose", final=True))
    assert [update.phase for update in updates] == ["starting", "running_tool", "writing"]
    assert [update.detail for update in updates] == ["pass 2", "grep receipt", ""]
    assert updates[1].tool_name == "grep"
    assert updates[-1].tool_count == 1 and updates[-1].sequence == 3


@check
def richsink_subagent_notify_is_safe_and_quiet():
    import io
    from rich.console import Console
    from sliceagent.tui import make_rich_sink
    sink = make_rich_sink(Console(file=io.StringIO(), force_terminal=False, width=80), {})
    for i in range(40):                              # 40 child tool calls → one updating line, never a crash
        sink.subagent_notify(f"↳ read_file f{i}.py · {i + 1} calls")
    sink._stop()


# ---- edit-result echo: post-edit region rides the tool result (within-turn ground truth) ----
@check
def str_replace_echoes_post_edit_region():
    # The critique's case: model edits `a - b` -> `a + b` but the old result was just a byte count, so it
    # never saw `a + b`. Now the post-edit region (with line numbers) rides the result.
    wd = tempfile.mkdtemp(prefix="echo-sr-")
    host = LocalToolHost(root=wd)
    host._t_edit_file({"path": "m.py", "content": "def add(a, b):\n    return a - b\n"})
    out = host._t_str_replace({"path": "m.py", "old_string": "a - b", "new_string": "a + b"})
    assert out.startswith("Replaced 1 occurrence in"), out                 # back-compat prefix kept
    assert "Updated region (lines" in out, out
    assert "return a + b" in out, "post-edit content must be in the result, not just a byte count"
    assert "\t" in out, "echo carries cat -n line numbers (match read_file)"


@check
def edit_file_echoes_head_bounded():
    wd = tempfile.mkdtemp(prefix="echo-ef-")
    host = LocalToolHost(root=wd)
    body = "".join(f"line{i}\n" for i in range(50))
    out = host._t_edit_file({"path": "big.py", "content": body})
    assert out.startswith("Wrote") and "Head:" in out, out                 # back-compat prefix kept
    assert "line0" in out and "more lines" in out, "a 50-line write echoes a bounded head + (+N more)"
    assert "line49" not in out, "content beyond the head window must be elided (bounded)"


@check
def append_echoes_tail():
    wd = tempfile.mkdtemp(prefix="echo-ap-")
    host = LocalToolHost(root=wd)
    host._t_edit_file({"path": "log.txt", "content": "a\nb\nc\n"})
    out = host._t_append({"path": "log.txt", "content": "NEW_TAIL\n"})
    assert out.startswith("Appended") and "File tail:" in out and "NEW_TAIL" in out, out


@check
def edit_echo_is_best_effort_and_writes_land():
    # the echo must NEVER prevent the write; the result is always a string and the file is correct
    wd = tempfile.mkdtemp(prefix="echo-be-")
    host = LocalToolHost(root=wd)
    out = host._t_edit_file({"path": "x.py", "content": "y = 1\n"})
    assert isinstance(out, str) and host.read_text("x.py") == "y = 1\n", out


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
