"""Model capability catalog (borrowed from Kimi agent-core/services/modelCatalog).

Maps a model name (+ base URL) to its capabilities and wire quirks so provider-specific knowledge lives
in ONE place instead of scattered `startswith` checks. Pattern-matched with a safe UNKNOWN default. Pure
data + lookup; the llm adapter consults it (it is the source of truth for the tokens-param rename and the
reasoning_effort capability — previously duplicated inline in llm.py).

context_window is left 0 (unknown) unless genuinely known — memagent's overflow is reactive, so no caller
relies on a fabricated number; the field is informational for any future context-window-aware feature.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCapability:
    family: str = "unknown"
    # OpenAI gpt-5 / o-series renamed the completion cap to `max_completion_tokens` and REJECT `max_tokens`.
    tokens_param: str = "max_tokens"
    # accepts the OpenAI `reasoning_effort` param (gpt-5 / o-series). NOT deepseek (uses extra_body.thinking)
    # nor moonshot/anthropic — those map "fast" to their own knobs in llm._reasoning_kwargs.
    supports_reasoning_effort: bool = False
    supports_tools: bool = True
    context_window: int = 0          # 0 = unknown (no fabricated values)


_UNKNOWN = ModelCapability()


def capability(model: str, base_url: str = "") -> ModelCapability:
    """Resolve the capability record for a model (first matching rule wins; specific before general)."""
    m = (model or "").lower()
    b = (base_url or "").lower()
    if m.startswith(("o1", "o3", "o4", "gpt-5")):
        return ModelCapability("openai-reasoning", tokens_param="max_completion_tokens",
                               supports_reasoning_effort=True)
    if "deepseek" in m or "deepseek" in b:
        return ModelCapability("deepseek")                 # reasoning via extra_body.thinking, not reasoning_effort
    if "kimi" in m or "moonshot" in b:
        return ModelCapability("moonshot")
    if "claude" in m or "anthropic" in b:
        return ModelCapability("anthropic")
    if m.startswith("gpt-") or "openai" in b:
        return ModelCapability("openai")
    return _UNKNOWN
