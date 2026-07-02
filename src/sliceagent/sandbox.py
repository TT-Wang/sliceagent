"""Sandbox — the command-execution backend.

`BaseSandbox` owns the cross-cutting concern (output capping); each backend implements only
`_exec()`. So swapping the isolation level never touches the ToolHost or the loop. Ships
`LocalSandbox` (subprocess) and `DockerSandbox` (container) behind the same seam; gVisor /
Firecracker / a remote runtime are further drop-ins.

Secret scrubbing matters: run_command executes model-proposed shell, often against
untrusted/generated code. By default the child does NOT inherit API keys or proxy creds, so
a stray `env`/exfil can't read them (Local scrubs its subprocess env; Docker only passes
explicitly-configured env into the container).

`python_cmd` lets code-as-action stay backend-portable: Local runs the venv interpreter
(so workspace code can import installed packages); Docker runs the container's `python3`.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import uuid
from typing import Protocol, runtime_checkable

# env var names whose values are secrets the child shouldn't see by default
_SECRET_RE = re.compile(
    r"(API_KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|ACCESS_KEY|PRIVATE_KEY|"
    r"_PROXY$|^HTTPS?_PROXY$|^ALL_PROXY$)",
    re.IGNORECASE,
)

_OUTPUT_CAP = 1_000_000  # chars; head+tail kept, middle elided. Sized ABOVE realistic logs/diffs so the
#                          page-out blob (the recall-on-demand promise) captures the FULL output for normal
#                          large results; this is only the last-resort OOM/disk ceiling for pathological dumps.


@runtime_checkable
class Sandbox(Protocol):
    """Execute a shell command, return (exit_code, combined_output)."""
    python_cmd: str
    def run(self, command: str, *, cwd: str, timeout: float) -> tuple[int, str]: ...


def _scrub_env() -> dict:
    return {k: v for k, v in os.environ.items() if not _SECRET_RE.search(k)}


def _cap(out: str) -> str:
    if len(out) <= _OUTPUT_CAP:
        return out
    keep = _OUTPUT_CAP // 2
    return out[:keep] + f"\n…[{len(out) - _OUTPUT_CAP} chars elided]…\n" + out[-keep:]


class BaseSandbox:
    """Template: run() caps output; subclasses implement _exec(). `python_cmd` is how
    code-as-action invokes Python in this backend."""
    python_cmd: str = "python3"

    def __init__(self, *, scrub_secrets: bool = True):
        self.scrub_secrets = scrub_secrets

    def run(self, command: str, *, cwd: str, timeout: float) -> tuple[int, str]:
        code, out = self._exec(command, cwd=cwd, timeout=timeout)
        return code, _cap(out)

    def _exec(self, command: str, *, cwd: str, timeout: float) -> tuple[int, str]:
        raise NotImplementedError


class LocalSandbox(BaseSandbox):
    """Local subprocess backend. cwd-confined, timeout, secret-env scrubbed. Runs the
    current (venv) interpreter for code-as-action so workspace imports resolve."""
    python_cmd = sys.executable

    def _exec(self, command: str, *, cwd: str, timeout: float) -> tuple[int, str]:
        env = _scrub_env() if self.scrub_secrets else None
        try:
            r = subprocess.run(command, shell=True, cwd=cwd, env=env,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124, f"Command timed out after {timeout:g}s"
        except OSError as e:
            return 127, f"Could not run command: {e}"
        return r.returncode, (r.stdout or "") + (r.stderr or "")


class DockerSandbox(BaseSandbox):
    """Container backend: run each command in `docker run --rm`, with the workspace bind-
    mounted at the SAME path (so workspace-relative and -absolute paths match host↔container)
    and the network off by default. Only explicitly-configured env enters the container."""
    python_cmd = "python3"

    def __init__(self, image: str, *, network: str = "none", docker: str = "docker",
                 env: dict | None = None, scrub_secrets: bool = True):
        super().__init__(scrub_secrets=scrub_secrets)
        self.image = image
        # fail CLOSED: blank/whitespace network → "none" (no networking), not "drop the flag" (which gives
        # the container default bridge networking — an isolation hole).
        self.network = (network or "none").strip() or "none"
        self.docker = docker
        self.env = env or {}

    def docker_args(self, command: str, *, cwd: str, name: str | None = None) -> list[str]:
        args = [self.docker, "run", "--rm", "-v", f"{cwd}:{cwd}", "-w", cwd]
        if name:
            args += ["--name", name]
        if self.network:
            args += ["--network", self.network]
        for k, v in self.env.items():
            args += ["-e", f"{k}={v}"]
        args += [self.image, "sh", "-c", command]
        return args

    def _exec(self, command: str, *, cwd: str, timeout: float) -> tuple[int, str]:
        # Name the container so a timeout can reap it: subprocess.run only SIGKILLs the local `docker run`
        # CLI; the daemon-side container keeps running. With a name we can `docker kill` it (and --rm then
        # removes it), instead of leaking an orphan container per timeout.
        name = f"sliceagent-{uuid.uuid4().hex[:12]}"
        try:
            r = subprocess.run(self.docker_args(command, cwd=cwd, name=name),
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                subprocess.run([self.docker, "kill", name], capture_output=True, timeout=10)
            except Exception:  # noqa: BLE001 — best-effort reap; never mask the timeout result
                pass
            return 124, f"Command timed out after {timeout:g}s"
        except OSError as e:
            return 127, f"Could not run docker: {e}"
        return r.returncode, (r.stdout or "") + (r.stderr or "")


def make_sandbox(backend: str = "local", *, image: str = "python:3.12-slim",
                 network: str = "none", scrub_secrets: bool = True) -> BaseSandbox:
    """Factory: 'local' (default) or 'docker'."""
    b = (backend or "local").lower()
    if b == "docker":
        return DockerSandbox(image, network=network, scrub_secrets=scrub_secrets)
    if b == "local":
        return LocalSandbox(scrub_secrets=scrub_secrets)
    # #27: a typo'd backend (e.g. "dokcer") must NOT silently fall back to the unisolated host — fail loud.
    raise ValueError(f"unknown sandbox backend {backend!r} (expected 'local' or 'docker')")
