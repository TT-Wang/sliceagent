"""Sandbox — the command-execution backend (borrowed pattern: Hermes environments/base.py).

One `run()` seam over swappable backends, so the ToolHost never changes when the
isolation level does. v1 ships LocalSandbox (subprocess with cwd confinement, a
timeout, output capping, and secret-env scrubbing). A container/VM backend
(Docker, gVisor, Firecracker, an OpenHands runtime) is a drop-in that implements
the same `run()` — that's the P2 hardening, not a rewrite.

Secret scrubbing matters: run_command executes model-proposed shell, often against
untrusted/generated code. By default the child process does NOT inherit API keys or
proxy creds, so a stray `env`/exfil can't read them.
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import Protocol, runtime_checkable

# env var names whose values are secrets the child shouldn't see by default
_SECRET_RE = re.compile(
    r"(API_KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|ACCESS_KEY|PRIVATE_KEY|"
    r"_PROXY$|^HTTPS?_PROXY$|^ALL_PROXY$)",
    re.IGNORECASE,
)

_OUTPUT_CAP = 100_000  # chars; head+tail kept, middle elided


@runtime_checkable
class Sandbox(Protocol):
    """Execute a shell command, return (exit_code, combined_output)."""
    def run(self, command: str, *, cwd: str, timeout: float) -> tuple[int, str]: ...


def _scrub_env() -> dict:
    return {k: v for k, v in os.environ.items() if not _SECRET_RE.search(k)}


def _cap(out: str) -> str:
    if len(out) <= _OUTPUT_CAP:
        return out
    keep = _OUTPUT_CAP // 2
    return out[:keep] + f"\n…[{len(out) - _OUTPUT_CAP} chars elided]…\n" + out[-keep:]


class LocalSandbox:
    """Local subprocess backend. Confines cwd, enforces a timeout, caps output,
    and (by default) hides secret env vars from the child."""

    def __init__(self, *, scrub_secrets: bool = True):
        self.scrub_secrets = scrub_secrets

    def run(self, command: str, *, cwd: str, timeout: float) -> tuple[int, str]:
        env = _scrub_env() if self.scrub_secrets else None
        try:
            r = subprocess.run(
                command, shell=True, cwd=cwd, env=env,
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return 124, f"Command timed out after {timeout:g}s"
        except OSError as e:
            return 127, f"Could not run command: {e}"
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode, _cap(out)
