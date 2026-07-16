# Security Policy

## Reporting a vulnerability

Please email **tongtao.wang@gmail.com** with details and a reproduction. **Do not** open a public issue
for security problems. We aim to acknowledge within 7 days and will credit reporters who wish it.

## Supported versions

Pre-1.0: only the latest release (and `main`) receives security fixes.

## Threat model — what to know before running sliceagent

sliceagent runs **model-authored shell commands and edits files in your workspace.** Treat it like any
other program to which you gave your local user account's shell access. The default local command backend
starts commands in the workspace, but it is **not an OS security boundary**: a command can access anything
your operating-system user can access. The relevant containment and safety mechanisms are:

- **File-tool boundary and command hygiene** (`sandbox.py`) — built-in file tools reject paths outside the
  authorized workspace roots. Model-run commands receive a scrubbed child environment (common
  `*_API_KEY`, `*_TOKEN`, and proxy variables are removed), timeouts, and output caps. These measures do
  not confine a local shell process. `AGENT_SANDBOX=docker` provides an optional container backend with
  network access off by default on POSIX and WSL2. Native Windows rejects that backend explicitly because
  its host path cannot preserve the Linux container's same-path workspace contract; use `local` or run
  SliceAgent inside WSL2.
- **Catastrophic-command safeguard** (`safeguards.py`) — a deliberately narrow, high-confidence check
  refuses direct commands such as formatting a device, powering off the machine, or recursively deleting
  `/` or the user's home. It is not a general permission system, confirmation mode, shell allowlist, or
  substitute for isolation; ordinary requested work runs without a host policy prompt.
- **MCP** (`mcp_security.py`) — configured stdio servers are RCE by design; each entry is **screened before
  spawn** and refused if it's a shell interpreter performing network egress or writing OS-persistence
  surfaces (SSH keys / PAM / sudoers / cron / shell rc).
- **Untrusted content** (`safety.py`) — retrieved memory, skills, related code, and subdirectory hints are
  injection-scanned and secret-redacted before they enter the model's context.

## Hardening tips

- On POSIX/WSL2, use **`AGENT_SANDBOX=docker`** (or another disposable environment) for untrusted
  repositories or tasks. On native Windows, run SliceAgent inside WSL2 to use that backend.
- On the local backend, run sliceagent under an OS account that has access only to data you are willing to
  expose to model-authored commands.
- Keep secrets in `.env` (gitignored) or the config file (written `0600`) — never commit them.
- Review MCP server entries before adding them to your config.
