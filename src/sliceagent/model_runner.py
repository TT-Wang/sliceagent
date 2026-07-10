"""Single provider-call seam for capacity preflight and retry policy."""
from __future__ import annotations

from .errors import with_retry
from .execution import preflight_model_call


def complete_model_call(
    llm,
    messages: list[dict],
    schemas: list[dict],
    *,
    dispatch=None,
    retry: bool = True,
    allow_unknown: bool | None = None,
    on_attempt=None,
):
    """Preflight and execute one model call through the shared retry boundary.

    Usage/budget ownership stays with the calling lifecycle because routing, onboarding, background
    consolidation, and an active turn have different accounting scopes. None bypasses physical validation.
    """
    if allow_unknown is None:
        allow_unknown = not bool(getattr(llm, "require_known_context", False))

    physical_attempt = 0

    def invoke():
        nonlocal physical_attempt
        report = preflight_model_call(llm, messages, schemas, allow_unknown=allow_unknown)
        physical_attempt += 1
        if on_attempt is not None:
            try:
                # Observation must never turn a valid provider request into an execution failure.
                on_attempt(physical_attempt, messages, report)
            except Exception:  # noqa: BLE001 - best-effort observer boundary
                pass
        return llm.complete(messages, schemas)

    if not retry:
        return invoke()
    return with_retry(
        invoke, is_retryable=getattr(llm, "is_retryable", None), dispatch=dispatch,
    )


__all__ = ["complete_model_call"]
