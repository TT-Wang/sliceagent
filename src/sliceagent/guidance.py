"""Stable guidance for hard work-budget ceilings.

Tool cancellation messages belong to the typed execution outcome that produced them. Keeping this module
focused on budget closeout avoids reintroducing generic permission prose into durable state.
"""

# Maps a budget kind to the concrete ceiling that was hit, so the message can name
# the right limit. Unknown kinds fall back to a generic "work budget".
_BUDGET_CEILINGS = {
    "max_steps": "the maximum number of steps for this turn",
    "token_budget": "the token budget for this turn",
}


def BUDGET_EXHAUSTED(kind: str) -> str:
    """Guidance for a hard ceiling hit (kind in {"max_steps", "token_budget"}).

    Names the ceiling that was reached, then asks the model to wrap up usefully
    instead of silently looping: summarize progress and give the single most
    useful next action. Returns a stable string for a given ``kind``.
    """
    ceiling = _BUDGET_CEILINGS.get(kind, "the work budget for this turn")
    return (
        f"You have reached {ceiling} and cannot continue this turn. "
        "Do not silently retry or keep working past the limit. Instead, summarize "
        "the progress you have made and state the single most useful next action so "
        "the work can resume cleanly."
    )
