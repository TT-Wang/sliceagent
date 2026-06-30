"""Experimental feature flags (borrowed from Kimi agent-core/flags).

A tiny env-driven registry so a not-yet-default feature can ship gated and OFF by default, then be
flipped on by setting its `default=True` once proven. State is read LIVE from the environment on every
call (nothing cached) so tests and process-env changes take effect immediately.

Precedence (Kimi resolver):
  1. master switch AGENT_EXPERIMENTAL_ALL truthy  → every flag ON
  2. per-flag env AGENT_EXPERIMENTAL_<ID>         → forces ON/OFF when set to a recognized bool
  3. the flag's registered `default`

Usage: `flags.register(Flag("cron", "Scheduled tasks"))` once at import, then gate with
`if flags.enabled("cron"): ...`. An unknown id resolves False (typo-safe).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

MASTER_ENV = "AGENT_EXPERIMENTAL_ALL"


@dataclass(frozen=True)
class Flag:
    id: str
    description: str = ""
    default: bool = False

    @property
    def env(self) -> str:
        return "AGENT_EXPERIMENTAL_" + self.id.upper()


_FLAGS: dict[str, Flag] = {}


def register(flag: Flag) -> Flag:
    """Register (or replace) a flag. Returns it so a module can `MY = register(Flag(...))`."""
    _FLAGS[flag.id] = flag
    return flag


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    v = value.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off", ""):
        return False
    return None   # unrecognized → defer to the next precedence level


def enabled(flag_id: str) -> bool:
    f = _FLAGS.get(flag_id)
    if f is None:
        return False                                   # unknown flag → off (typo-safe)
    if _parse_bool(os.environ.get(MASTER_ENV)) is True:
        return True                                    # master switch forces all on
    per = _parse_bool(os.environ.get(f.env))
    if per is not None:
        return per                                     # explicit per-flag override
    return f.default
