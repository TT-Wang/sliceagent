"""Action-oriented guidance strings injected at denial / ceiling boundaries.

Strings and pure functions only — no state, no imports. Wording is borrowed from
Hermes `coding_context.py` ("re-read before retrying; don't repeat a stale call;
after repeated failure switch approach") and Kimi `profile/default/system.md`
("default to taking action with tools"; "determine your next action") and tuned to
be ACTIONABLE so the model changes approach instead of spinning on the identical
call.

These land in a DURABLE tier, not a transcript:
  - DENIAL_NO_PROMPT / DENIAL_USER flow through
    `loop.run_tool_batch` -> `Error: blocked by policy: <reason>` -> `ToolResult`
    -> `slice.record_action` -> `s.last_error` (CURRENT ERROR), re-derivable each
    turn from the durable action record.
  - BUDGET_EXHAUSTED(kind) is the message carried on a `TurnInterrupted` event.

No per-session/per-turn computation here: every value is a module-level constant or
a pure function of its argument, so the system prefix stays byte-stable (cache-warm).
"""

# Permission was required but there is no interactive channel to ask the user.
# Action-oriented: do not re-issue the same blocked call; pick a different,
# permitted route or surface what you need from the user.
DENIAL_NO_PROMPT: str = (
    "This call needs permission but no approval channel is available, so it was "
    "blocked. Do NOT retry the identical call — it will be blocked again. Instead, "
    "either accomplish the goal with a tool that does not require approval, or stop "
    "and tell the user exactly which action you need them to authorize and why."
)

# The user explicitly declined the call.
# Action-oriented: treat the denial as final for this exact call; change approach
# or ask what the user would prefer rather than re-issuing it.
DENIAL_USER: str = (
    "The user declined this action. Do NOT retry the identical call — the answer is "
    "no. Instead, take a different approach that respects that decision, or ask the "
    "user how they would like to proceed before trying anything similar."
)

# Maps a budget kind to the concrete ceiling that was hit, so the message can name
# the right limit. Unknown kinds fall back to a generic "work budget".
_BUDGET_CEILINGS = {
    "max_steps": "the maximum number of steps for this turn",
    "token_budget": "the token budget for this turn",
}


# The anti-spin floor: the turn was stopped after repeated loop-blocks (the agent kept hitting the
# same wall instead of asking). Carried on a TurnInterrupted("stuck") event → shown to the user, who
# regains control. The proactive path is ask_user; this is the backstop when the model won't self-stop.
STUCK: str = (
    "Stopped: this turn hit the loop guard repeatedly without making progress. When you are unsure "
    "or blocked, call ask_user with a concise question instead of retrying — asking is better than "
    "spinning. Control is back with the user; clarify or rephrase the task to continue."
)


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
