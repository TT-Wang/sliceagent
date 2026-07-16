"""Context-overflow classification — pure stdlib, no model, no pytest.
Run: python tests/test_context_overflow.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.context_overflow import (  # noqa: E402
    ContextOverflow,
    classify,
    is_context_overflow,
    normalize_http_status,
)

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


class _StatusErr(Exception):
    """Fake SDK exception carrying an HTTP status_code attr."""
    def __init__(self, msg, status_code=None, status=None, code=None, body=None):
        super().__init__(msg)
        if status_code is not None:
            self.status_code = status_code
        if status is not None:
            self.status = status
        if code is not None:
            self.code = code
        if body is not None:
            self.body = body


@check
def overflow_phrase_true():
    assert is_context_overflow(Exception("This model's maximum context length is 8192 tokens"))
    assert is_context_overflow(Exception("prompt is too long for the context window"))
    assert is_context_overflow(Exception("input exceeds the max_model_len of the engine"))


@check
def rate_limit_false():
    # rate-limit wording is NOT overflow
    assert not is_context_overflow(Exception("Rate limit reached for requests"))
    assert not is_context_overflow(Exception("429 Too Many Requests"))


@check
def status_413_true():
    # 413 surfaced as a status attr => overflow (compress)
    assert is_context_overflow(_StatusErr("Bad Request", status_code=413))
    # 413 phrased in the message text (no status attr)
    assert is_context_overflow(Exception("Request Entity Too Large"))
    assert is_context_overflow(Exception("error code: 413"))


@check
def numeric_string_statuses_are_normalized():
    assert normalize_http_status(" 413 ") == 413
    assert normalize_http_status("HTTP 413") is None
    assert normalize_http_status(True) is None
    assert is_context_overflow(_StatusErr("Bad Request", status_code="413"))
    assert classify(_StatusErr("upstream", status="503")) == {
        "retryable": True, "is_context_overflow": False, "status": 503,
    }


@check
def bare_400_404_not_overflow():
    # a bare 400/404 with no overflow text/code is NOT overflow
    assert not is_context_overflow(_StatusErr("Bad Request", status_code=400))
    assert not is_context_overflow(_StatusErr("Not Found", status_code=404))


@check
def structured_code_true():
    assert is_context_overflow(_StatusErr("invalid request", code="context_length_exceeded"))
    assert is_context_overflow(_StatusErr("oops", body={"error": {"code": "max_tokens_exceeded"}}))


@check
def cause_walk_true():
    # status code lives on the __cause__, message text says nothing
    inner = _StatusErr("upstream", status_code=413)
    try:
        raise ValueError("wrapper says nothing useful") from inner
    except ValueError as outer:
        assert is_context_overflow(outer)


@check
def cause_walk_status_extracted():
    # classify pulls the status out of the cause chain
    inner = _StatusErr("upstream", status_code=413)
    try:
        raise RuntimeError("opaque") from inner
    except RuntimeError as outer:
        out = classify(outer)
        assert out["status"] == 413 and out["is_context_overflow"] is True


@check
def classify_overflow_not_retryable():
    out = classify(Exception("This model's maximum context length is 4096 tokens"))
    assert out["is_context_overflow"] is True
    assert out["retryable"] is False                # tighten the slice, don't blind-retry


@check
def classify_timeout_retryable():
    out = classify(Exception("Connection timed out while reading response"))
    assert out["is_context_overflow"] is False
    assert out["retryable"] is True


@check
def classify_5xx_retryable_not_overflow():
    out = classify(_StatusErr("Internal Server Error", status_code=503))
    assert out == {"retryable": True, "is_context_overflow": False, "status": 503}


@check
def classify_400_not_retryable():
    out = classify(_StatusErr("Bad Request", status_code=400))
    assert out == {"retryable": False, "is_context_overflow": False, "status": 400}


@check
def exception_wraps_original_and_status():
    orig = _StatusErr("ctx too big", status_code=413)
    e = ContextOverflow(orig, status_code=413)
    assert e.original is orig
    assert e.status_code == 413
    assert "ctx too big" in str(e)
    assert isinstance(e, Exception)
    assert ContextOverflow(orig, status_code="413").status_code == 413


@check
def never_imports_openai():
    import sliceagent.context_overflow as mod
    src = open(mod.__file__).read()
    assert "import openai" not in src and "from openai" not in src


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
