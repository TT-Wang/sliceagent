"""MCP output bounding: a single tool result can't blow up the slice. The PRIMARY bound is now the host
page-out (big MCP result → blob + head/tail view + read_file ref, nothing lost); _result_to_text keeps
only a last-resort OOM cap. No model, no pytest. Run: PYTHONPATH=src python tests/test_mcp_output_cap.py
"""
import os
import sys
import tempfile
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.mcp_client import _MCP_SAFETY_CAP, _result_to_text, _mcp_handler  # noqa: E402
from memagent.tools import LocalToolHost                                         # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _result(text, is_error=False):
    return NS(content=[NS(text=text, type="text")], isError=is_error)


# ── _result_to_text: text extraction + OOM safety cap ────────────────────────────────────────
@check
def small_result_passes_through():
    assert _result_to_text(_result("hello world")) == "hello world"


@check
def error_flag_is_prefixed():
    assert _result_to_text(_result("boom", is_error=True)).startswith("Error: boom")


@check
def empty_content_is_safe():
    assert _result_to_text(NS(content=[], isError=False)) == "(no content)"


@check
def oversized_result_hits_oom_safety_cap():
    out = _result_to_text(_result("x" * (_MCP_SAFETY_CAP + 5000)))
    assert len(out) < _MCP_SAFETY_CAP + 200 and "truncated" in out


# ── _mcp_handler: page a large result OUT to a blob (the Context-Mode borrow) ─────────────────
class _FakeServer:
    def __init__(self, text):
        self._text = text
    def call(self, tool, args):
        return _result_to_text(_result(self._text))


@check
def large_mcp_result_is_paged_out_not_inlined():
    root = tempfile.mkdtemp(prefix="mcp-po-")
    host = LocalToolHost(root=root)
    big = "RESULT_LINE data payload\n" * 4000          # ~96 KB, well over the 16 KB inline cap
    handler = _mcp_handler(_FakeServer(big), "browse", host._page_out)
    out = handler({})
    assert len(out) < len(big) / 5, f"should be a bounded view, got {len(out)}"
    assert "paged out" in out and "read_file(" in out and "mcp-browse" in out, out
    # the FULL payload is preserved on disk (read_file pages it back; here we read the blob directly)
    import re
    m = re.search(r"read_file\('([^']+)'\)", out)
    assert m, out
    blob = os.path.join(root, m.group(1))
    assert os.path.exists(blob), f"blob not written: {blob}"
    assert big.strip() in open(blob, encoding="utf-8").read(), "blob must hold the full MCP output"


@check
def small_mcp_result_unchanged_with_pageout():
    host = LocalToolHost(root=tempfile.mkdtemp(prefix="mcp-po-s-"))
    handler = _mcp_handler(_FakeServer("tiny output"), "ping", host._page_out)
    assert handler({}) == "tiny output"


@check
def no_pageout_returns_raw():
    handler = _mcp_handler(_FakeServer("raw text"), "ping", None)
    assert handler({}) == "raw text"


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
