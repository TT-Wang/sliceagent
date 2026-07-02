# Security Policy

## Reporting a vulnerability

Please email **tongtao.wang@gmail.com** with details and a reproduction. **Do not** open a public issue
for security problems. We aim to acknowledge within 7 days and will credit reporters who wish it.

## Supported versions

Pre-1.0: only the latest release (and `main`) receives security fixes.

## Threat model — what to know before running sliceagent

sliceagent runs **model-authored shell commands and edits files in your workspace.** Treat it like any
agent with shell access. Four independent layers contain that:

- **Sandbox** (`sandbox.py`) — commands run **cwd-confined**, with **secret-env scrubbing** (your
  `*_API_KEY` / `*_TOKEN` / proxy vars are stripped from the child environment so a model-run command
  can't read them), a timeout, and output capping. File ops are confined to the workspace root; path
  traversal out of it is rejected. `AGENT_SANDBOX=docker` adds a container with **network off by default**.
- **Policy** (`policy.py`) — every mutating/exec tool passes a `PolicyChain`. All three modes
  (`baby-sitter` / `teenager` / `let-it-go`) enforce a **catastrophic-command floor** (`rm -rf /`, `sudo`,
  `curl … | sh`, writes to `/etc`, credential reads, force-push). Default is **`teenager`** (auto-applies
  edits, asks before shell commands). A non-interactive run downgrades a confirm-mode to auto-run (still
  catastrophic-gated) and prints a notice.
- **MCP** (`mcp_security.py`) — configured stdio servers are RCE by design; each entry is **screened before
  spawn** and refused if it's a shell interpreter performing network egress or writing OS-persistence
  surfaces (SSH keys / PAM / sudoers / cron / shell rc).
- **Untrusted content** (`safety.py`) — retrieved memory, skills, related code, and subdirectory hints are
  injection-scanned and secret-redacted before they enter the model's context.

## Hardening tips

- Use **`AGENT_SANDBOX=docker`** for untrusted repositories.
- Use **`AGENT_POLICY=baby-sitter`** to confirm every edit and command.
- Keep secrets in `.env` (gitignored) or the config file (written `0600`) — never commit them.
- Review MCP server entries before adding them to your config.
