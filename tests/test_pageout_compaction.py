"""Context hygiene: (#74) page-out large tool output to a blob, (#77) micro-compaction
clears old tool-result bodies before dropping whole exchanges, (#76) configurable max_steps.
No model, no pytest. Run: PYTHONPATH=src python tests/test_pageout_compaction.py
"""
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.tools import LocalToolHost                          # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


# ---- #74: page-out large tool output --------------------------------------------------------
@check
def page_out_keeps_small_output_inline():
    host = LocalToolHost(root=tempfile.mkdtemp(prefix="po-small-"))
    small = "hello world\n" * 5
    assert host._page_out(small, label="command output") == small, "small output must be untouched"


@check
def page_out_pages_large_output_to_a_readable_blob():
    host = LocalToolHost(root=tempfile.mkdtemp(prefix="po-big-"))
    big = "".join(f"row {i}\n" for i in range(5000))                # > 16k chars
    assert len(big) > 16000
    out = host._page_out(big, label="command output")
    assert len(out) < len(big), "inline view must be BOUNDED"
    assert "paged out" in out and "read_file('.memagent/blobs/" in out, out
    assert big[:40] in out and big[-40:] in out, "head + tail must be inlined"
    # the reference pages the FULL output back via read_file (L1→L2 recall, not a cut)
    rel = re.search(r"read_file\('([^']+)'\)", out).group(1)
    back = host._t_read_file({"path": rel})
    assert "row 0" in back and "row 4999" in back, "full output must be recoverable via the blob ref"


@check
def page_out_with_control_chars_pages_back_as_text_not_hexdump():
    # REGRESSION (review MAJOR): a paged output with a NUL / control char in the first 8KB used to read
    # back as a 256-byte hexdump (read_file's binary gate). The paged path strips control bytes, so the
    # blob is plain text and the FULL output is recoverable through the advertised read_file channel.
    host = LocalToolHost(root=tempfile.mkdtemp(prefix="po-ctrl-"))
    big = "HEAD\x00\x1b[31m" + "".join(f"line {i}\n" for i in range(5000)) + "END-UNIQUE-MARKER"
    out = host._page_out(big, label="command output")
    assert "\x00" not in out, "a NUL must not ride the transcript (API-safe)"
    rel = re.search(r"read_file\('([^']+)'\)", out).group(1)
    back = host._t_read_file({"path": rel})
    assert "hexdump" not in back.lower(), "the blob must read back as TEXT, not a binary hexdump"
    assert "END-UNIQUE-MARKER" in back and "line 4999" in back, "full output must be recoverable via the ref"


@check
def page_out_never_fails_the_tool_on_write_error():
    host = LocalToolHost(root=tempfile.mkdtemp(prefix="po-err-"))
    host._mkparent = lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full"))   # force blob write to fail
    big = "x" * 20000
    out = host._page_out(big, label="command output")
    assert len(out) < len(big) and "paged out" in out, "must still bound the inline view when paging fails"


# ---- #77: micro-compaction (clear old tool bodies, keep reasoning + recent) ------------------
@check
def micro_compact_clears_old_tool_bodies_keeps_reasoning_and_recent():
    from memagent.loop import MICRO_MARKER, _micro_compact
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]   # seed (len 2)
    for i in range(6):
        msgs.append({"role": "assistant", "content": f"thinking {i}", "tool_calls": [{"id": f"c{i}"}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"big result {i}"})
    assert _micro_compact(msgs, floor=2, keep_recent=4) is True
    assert any(m["content"] == MICRO_MARKER for m in msgs), "old tool bodies must be cleared"
    assert all(m["content"].startswith("thinking") for m in msgs if m["role"] == "assistant"), "reasoning kept"
    assert msgs[-1]["content"] == "big result 5", "the recent window's tool result must be kept"
    # tool_call↔reply pairing stays valid (cleared bodies keep their tool_call_id)
    assert all(m.get("tool_call_id") for m in msgs if m["role"] == "tool"), "tool pairings intact"
    assert _micro_compact(msgs, floor=2, keep_recent=4) is False, "second pass clears nothing new"


@check
def micro_compact_returns_false_when_nothing_to_clear():
    from memagent.loop import _micro_compact
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "thinking"}]   # no tool messages
    assert _micro_compact(msgs, floor=2, keep_recent=4) is False


# ---- #76: configurable max_steps ------------------------------------------------------------
@check
def max_steps_is_configurable_default_60():
    from memagent.config import load_config
    os.environ.pop("AGENT_MAX_STEPS", None)
    assert load_config().max_steps == 60, "default ceiling is 60 (raised from the old hard 40)"
    os.environ["AGENT_MAX_STEPS"] = "120"
    try:
        assert load_config().max_steps == 120
        os.environ["AGENT_MAX_STEPS"] = "not-a-number"
        assert load_config().max_steps == 60, "a bad value falls back to the default, never crashes"
    finally:
        os.environ.pop("AGENT_MAX_STEPS", None)


@check
def agent_max_steps_is_documented_in_envspec():
    from memagent.envspec import REGISTRY
    assert any(v.name == "AGENT_MAX_STEPS" for v in REGISTRY), "AGENT_MAX_STEPS must be in the env registry"


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
