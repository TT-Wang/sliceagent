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
    supports_stream_options: bool = True   # OpenAI stream_options={include_usage}; set False if a provider 400s
    supports_vision: bool = False    # accepts image content parts (multimodal); gates @image attachment
    context_window: int = 0          # 0 = unknown (no fabricated values)


_UNKNOWN = ModelCapability()

# USD per 1M tokens: (input_fresh, input_cached, output). SINGLE SOURCE for the cost meter — keyed by a
# name/family substring, first match wins. Update HERE when a provider changes pricing. (Context windows stay
# 0/unknown by design: memagent's overflow is reactive, so nothing fabricates a window — see ModelCapability.)
_PRICES = {
    "gpt-5": (1.25, 0.125, 10.0), "gpt-4": (2.50, 1.25, 10.0), "o3": (2.0, 0.5, 8.0),
    "deepseek": (0.27, 0.07, 1.10), "kimi": (0.60, 0.15, 2.50), "moonshot": (0.60, 0.15, 2.50),
    "claude": (3.0, 0.30, 15.0),
}


def pricing(model: str, base_url: str = "") -> "tuple | None":
    """USD/1M (input, cached_input, output) for a model, or None if unknown. The cost meter's single source."""
    s = (model or "").lower() + " " + (base_url or "").lower()
    for k, v in _PRICES.items():
        if k in s:
            return v
    return None

# Vision is keyed off the MODEL name (not the family) — kimi-k2.7-code is text-only but moonshot-*-vision is
# not; gpt-4o/gpt-5/claude-3+/gemini/`*-vl`/anything with 'vision' is multimodal. Conservative allowlist.
_VISION_HINTS = ("vision", "gpt-4o", "gpt-4.1", "gpt-5", "gpt-6", "claude-3", "claude-4",
                 "claude-opus", "claude-sonnet", "gemini", "-vl", "qwen-vl")


def _is_openai_endpoint(base_url: str) -> bool:
    """True only when `base_url` is OpenAI's real API — the default (unset → the SDK's own default) or an
    explicit api.openai.com. reasoning_effort + the /v1/responses route are OpenAI-ONLY wire features; a
    model literally NAMED "gpt-5.5"/"o3" served by a DIFFERENT endpoint (DeepSeek, Moonshot, a local proxy —
    /model only switches the model string, never the endpoint) does NOT speak that protocol. Routing to
    /v1/responses there 404s (openai.NotFoundError — the route doesn't exist on that server), which used to
    surface as a cryptic 'internal error ended the turn'; gating on the endpoint keeps it on the universal
    chat/completions path instead — degrade gracefully, never assume a wire feature from the name alone."""
    b = (base_url or "").strip().lower()
    return b == "" or "api.openai.com" in b


def capability(model: str, base_url: str = "") -> ModelCapability:
    """Resolve the capability record for a model (first matching rule wins; specific before general)."""
    m = (model or "").lower()
    b = (base_url or "").lower()
    vis = any(h in m for h in _VISION_HINTS)
    if m.startswith(("o1", "o3", "o4", "o5", "o6", "gpt-5", "gpt-6")) and _is_openai_endpoint(b):
        return ModelCapability("openai-reasoning", tokens_param="max_completion_tokens",
                               supports_reasoning_effort=True, supports_vision=vis)
    if "deepseek" in m or "deepseek" in b:
        return ModelCapability("deepseek", supports_vision=vis)   # reasoning via extra_body.thinking
    if "kimi" in m or "moonshot" in b:
        return ModelCapability("moonshot", supports_vision=vis)
    if "claude" in m or "anthropic" in b:
        return ModelCapability("anthropic", supports_vision=vis)
    if m.startswith("gpt-") or "openai" in b:
        return ModelCapability("openai", supports_vision=vis)
    return ModelCapability(supports_vision=vis)
