"""Context-overflow classification — pure stdlib, never imports openai.

Ported from Hermes `agent/error_classifier.py`:
  - `_CONTEXT_OVERFLOW_PATTERNS` table (error_classifier.py:208)
  - the cause-walk `_extract_status_code` (error_classifier.py:1266, max depth 5)
  - the 400 / 413 / `context_length_exceeded` overflow rules
    (error_classifier.py:1016, :1125, :838)

The moat: memagent always has a slice it can TIGHTEN, so there is NO
session-size / token-count heuristic here (Hermes' generic-400 + large-session
proxy at :1054 is deliberately dropped). Overflow is decided purely from the
error's text and HTTP status.

Public surface (pinned in adopt_plan.md sec 1):
    class ContextOverflow(Exception)
    def is_context_overflow(error: Exception) -> bool
    def classify(error: Exception) -> dict  # {retryable, is_context_overflow, status}
"""

from __future__ import annotations

from typing import Optional

# ── Pattern table (Hermes error_classifier.py:208) ──────────────────────────
# Matched against str(error).lower(). Copied verbatim from the port source.
_CONTEXT_OVERFLOW_PATTERNS = (
    "context length",
    "context size",
    "maximum context",
    "token limit",
    "too many tokens",
    "reduce the length",
    "exceeds the limit",
    "context window",
    "prompt is too long",
    "prompt exceeds max length",
    "maximum number of tokens",
    # vLLM / local inference server patterns
    "exceeds the max_model_len",
    "max_model_len",
    "prompt length",  # "engine prompt length X exceeds"
    "input is too long",
    "maximum model length",
    # Ollama patterns
    "context length exceeded",
    "truncating input",
    # llama.cpp / llama-server patterns
    "slot context",  # "slot context: N tokens, prompt N tokens"
    "n_ctx_slot",
    # Chinese error messages (some providers return these)
    "超过最大长度",
    "上下文长度",
    # AWS Bedrock Converse API error patterns
    "input is too long",
    "max input token",
    "exceeds the maximum number of input tokens",
    # NOTE: bare "input token" was removed — it matched OpenAI's TPM rate-limit text
    # ("Limit: 30000 input tokens per minute"), misclassifying a 429 as a hard overflow.
)

# Structured error codes that unambiguously mean overflow
# (Hermes error_classifier.py:1125).
_CONTEXT_OVERFLOW_CODES = frozenset({
    "context_length_exceeded",
    "max_tokens_exceeded",
})

# Payload-too-large message patterns (Hermes error_classifier.py:158) — a 413
# surfaced in the message text when no status_code attr is present.
_PAYLOAD_TOO_LARGE_PATTERNS = (
    "request entity too large",
    "payload too large",
    "error code: 413",
)


# A parameter / validation error (e.g. "unsupported parameter 'max_tokens'") is NOT a context
# overflow even though it may name a token param — reading it as overflow would wrongly trigger the
# slice-tighten/rebuild loop. Root-cause guard: exclude param errors regardless of which param.
# (Kept SPECIFIC: a real OpenAI overflow is type invalid_request_error / code context_length_exceeded,
# so we must NOT exclude on those — only on explicit "unsupported/invalid parameter" wording.)
_NOT_OVERFLOW_MARKERS = (
    "unsupported parameter",
    "unsupported_parameter",
    "is not supported with this model",
    "unknown parameter",
    "invalid parameter",
    "parameter is invalid",   # "the prompt length parameter is invalid" — a validation error, NOT overflow
    "invalid value",
    "invalid input token",    # "invalid input token format" — not a context-size overflow
)


def _error_text(error: Exception) -> str:
    """Lowercased message text for pattern matching."""
    return str(error).lower()


def _extract_status_code(error: Exception) -> Optional[int]:
    """Walk the error and its cause chain to find an HTTP status code.

    Port of Hermes `_extract_status_code` (error_classifier.py:1266); max depth
    5 to bound the walk. Checks `.status_code` (int) then `.status` (sane int)
    on each node, following `__cause__`/`__context__`.
    """
    current: Optional[BaseException] = error
    for _ in range(5):  # max depth to prevent infinite loops
        if current is None:
            break
        code = getattr(current, "status_code", None)
        if isinstance(code, int):
            return code
        # Some SDKs use .status instead of .status_code
        code = getattr(current, "status", None)
        if isinstance(code, int) and 100 <= code < 600:
            return code
        # Walk cause chain
        cause = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if cause is None or cause is current:
            break
        current = cause
    return None


def _extract_error_code(error: Exception) -> str:
    """Best-effort structured error-code string from a `.code`/`.body` attr.

    Defensive and pure-stdlib: never imports an SDK, never raises. Returns ''
    when no usable code is found.
    """
    # Direct attribute (many SDK exceptions expose .code)
    code = getattr(error, "code", None)
    if isinstance(code, str) and code.strip():
        return code.strip().lower()

    body = getattr(error, "body", None)
    if isinstance(body, dict):
        err_obj = body.get("error")
        if isinstance(err_obj, dict):
            nested = err_obj.get("code") or err_obj.get("type") or ""
            if isinstance(nested, str) and nested.strip():
                return nested.strip().lower()
        top = body.get("code") or body.get("error_code") or ""
        if isinstance(top, str) and top.strip():
            return top.strip().lower()
    return ""


def is_context_overflow(error: Exception) -> bool:
    """True when `error` is (or wraps) a context-overflow signal.

    An overflow is recognised when ANY of the following hold:
      - the lowercased message matches `_CONTEXT_OVERFLOW_PATTERNS`;
      - a structured error code is `context_length_exceeded`/`max_tokens_exceeded`;
      - the cause chain carries HTTP 413 (payload-too-large => compress);
      - a 413 phrased in the message text (`_PAYLOAD_TOO_LARGE_PATTERNS`).

    A bare 400/404 is NOT treated as overflow unless its TEXT matches the
    overflow table — many non-overflow errors also use 400/404, and memagent
    drops Hermes' session-size proxy (we always have a slice to tighten).
    """
    msg = _error_text(error)
    if "rate limit" in msg or "too many requests" in msg or "tokens per min" in msg:
        return False  # a TPM/RPM RATE LIMIT (429) must back off + retry, NOT trigger slice-destroying overflow handling
    if any(m in msg for m in _NOT_OVERFLOW_MARKERS):
        return False  # a parameter/validation error is never a context overflow
    if any(p in msg for p in _CONTEXT_OVERFLOW_PATTERNS):
        return True
    if any(p in msg for p in _PAYLOAD_TOO_LARGE_PATTERNS):
        return True

    code = _extract_error_code(error)
    if code in _CONTEXT_OVERFLOW_CODES:
        return True

    status = _extract_status_code(error)
    if status == 413:
        return True

    return False


def classify(error: Exception) -> dict:
    """Classify `error` for the retry/rebuild loop.

    Returns `{retryable, is_context_overflow, status}`:
      - `is_context_overflow`: result of `is_context_overflow(error)`;
      - `status`: HTTP status from the cause-walk, or None;
      - `retryable`: True for transient transport errors (5xx, 408, 429) and
        for timeout/connection wording; False for context overflow (the slice
        must be TIGHTENED, not blindly retried) and for non-transient 4xx.

    Overflow is intentionally `retryable=False` here: the caller rebuilds a
    smaller slice (see W5 errors.classify glue / loop overflow-rebuild loop)
    rather than re-sending the identical oversized request.
    """
    overflow = is_context_overflow(error)
    status = _extract_status_code(error)

    if overflow:
        return {"retryable": False, "is_context_overflow": True, "status": status}

    msg = _error_text(error)
    retryable = False
    if status is not None:
        # 5xx server errors + 408 request-timeout + 429 rate-limit are transient.
        if status >= 500 or status in (408, 429):
            retryable = True
    else:
        # No status code: fall back to transient transport wording.
        if any(
            p in msg
            for p in ("timeout", "timed out", "connection", "temporarily unavailable")
        ):
            retryable = True

    return {"retryable": retryable, "is_context_overflow": False, "status": status}


class ContextOverflow(Exception):
    """Raised by the LLM adapter when a request overflows the context window.

    Carries the original provider error so the retry/rebuild loop can inspect
    it. `status_code` is the HTTP status if one was extractable, else None.
    """

    def __init__(self, original: Exception, *, status_code: Optional[int] = None):
        self.original = original
        self.status_code = status_code
        super().__init__(str(original))
