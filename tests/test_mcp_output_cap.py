"""MCP robustness (from Kimi mcp/client-stdio): bound a single tool result so a runaway payload can't
blow up the slice. No model, no pytest. Run: PYTHONPATH=src python tests/test_mcp_output_cap.py
"""
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.mcp_client import _MCP_MAX_OUTPUT, _result_to_text  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _result(text, is_error=False):
    return NS(content=[NS(text=text, type="text")], isError=is_error)


@check
def oversized_result_is_capped_with_notice():
    out = _result_to_text(_result("x" * (_MCP_MAX_OUTPUT + 5000)))
    assert len(out) < _MCP_MAX_OUTPUT + 200, len(out)
    assert "truncated" in out and "MCP output" in out


@check
def small_result_passes_through():
    assert _result_to_text(_result("hello world")) == "hello world"


@check
def error_flag_is_prefixed():
    assert _result_to_text(_result("boom", is_error=True)).startswith("Error: boom")


@check
def empty_content_is_safe():
    assert _result_to_text(NS(content=[], isError=False)) == "(no content)"


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
