"""Security screen for user-configured MCP server entries (ported from Hermes hermes_cli/mcp_security.py).

MCP stdio transports intentionally allow ARBITRARY local commands — a configured server is remote-code-
execution by design. We don't try to sandbox that. We refuse two high-signal ABUSE SHAPES that a real MCP
server never has, so a hand-edited or pre-planted config.toml is caught BEFORE `connect_mcp_servers` spawns it:

  1. a shell interpreter whose inline script performs NETWORK EGRESS (curl/wget/nc/socat, /dev/tcp,
     PowerShell web clients) — the exfiltration shape;
  2. a shell interpreter whose inline script writes to an OS PERSISTENCE surface (SSH keys, PAM, sudoers,
     cron, init units, shell rc files) — the backdoor shape.

This is intentionally NOT a whitelist: legitimate local MCPs using npx / uvx / python / a custom binary all
pass. General + task-agnostic — only the shell-interpreter-plus-egress/persistence combination is refused.
"""
from __future__ import annotations

import os
import re
import shlex

_SHELL_INTERPRETERS = frozenset({
    "bash", "sh", "zsh", "dash", "fish", "ksh",
    "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe",
})

_EGRESS = re.compile(
    r"(?<![\w.-])(?:curl|wget|nc|ncat|socat)(?![\w.-])"
    r"|/dev/tcp/"
    r"|\bInvoke-WebRequest\b|\bInvoke-RestMethod\b|\bSystem\.Net\.WebClient\b",
    re.IGNORECASE,
)

_PERSISTENCE = re.compile(
    r"authorized_keys|\.ssh/|/etc/ssh\b"
    r"|/etc/pam\.d\b|pam_[\w-]+\.so|/etc/sudoers"
    r"|/etc/cron|crontab\b|/etc/rc\.local|/etc/systemd"
    r"|\.bashrc\b|\.bash_profile\b|\.profile\b|\.zshrc\b",
    re.IGNORECASE,
)


def _basename(command) -> str:
    text = str(command or "").strip()
    if not text:
        return ""
    try:
        parts = shlex.split(text, posix=(os.name != "nt"))
    except ValueError:
        parts = text.split()
    return os.path.basename(parts[0] if parts else text).lower()


def _script(args) -> str:
    if args is None:
        return ""
    if isinstance(args, (list, tuple)):
        return " ".join(str(a) for a in args)
    return str(args)


def validate_mcp_server_entry(name: str, conf) -> list[str]:
    """Return a list of security objections to spawning this MCP entry (empty list = clean).

    Only a shell interpreter (bash/sh/pwsh/…) carrying an inline script with network-egress OR
    OS-persistence content is refused; everything else (npx/uvx/python/custom binaries) passes.
    """
    if not isinstance(conf, dict):
        return []
    # Tokenize command + args together and check whether ANY token is a shell interpreter — so a wrapped
    # interpreter (env bash -c …, /usr/bin/timeout 5 sh -c …, a full path) is screened, not just a bare
    # `command: bash`. Then scan the FULL command+args text for the egress/persistence shapes.
    full = (str(conf.get("command") or "") + " " + _script(conf.get("args"))).strip()
    if not full:
        return []
    try:
        tokens = shlex.split(full, posix=(os.name != "nt"))
    except ValueError:
        tokens = full.split()
    if not any(os.path.basename(t).lower() in _SHELL_INTERPRETERS for t in tokens):
        return []
    issues: list[str] = []
    if _EGRESS.search(full):
        issues.append(f"MCP server '{name}': a shell interpreter with network-egress arguments "
                      "(exfiltration shape — not a real MCP server)")
    if _PERSISTENCE.search(full):
        issues.append(f"MCP server '{name}': a shell interpreter writing to an OS persistence surface "
                      "(SSH keys / PAM / sudoers / cron / shell rc — backdoor shape, not a real MCP server)")
    return issues


def is_suspicious(name: str, conf) -> bool:
    return bool(validate_mcp_server_entry(name, conf))
