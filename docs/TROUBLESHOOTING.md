# Troubleshooting

## "No API key found"
Run `memagent init` (guided setup), or export `LLM_API_KEY` (and `LLM_BASE_URL` for non-OpenAI providers).
Check what's configured with `memagent config`. The config file lives at `~/.memagent/config.toml`.

## A setting seems ignored / "config warning: AGENT_X=… is not one of {…}"
You set an enum var to an unrecognized value, so the default is used. The warning names the valid set.
List every knob, its default, and current value: `memagent config --list`.

## Requests hang or fail to connect (region/proxy)
memagent auto-routes the network: foreign endpoints (OpenAI) go through a local proxy (ClashX at
`127.0.0.1:7890`), CN-direct providers (Moonshot/DeepSeek) go direct. The chosen route is shown on startup as
`net=…`. Override explicitly:
- `AGENT_PROXY=off` — force a direct connection.
- `AGENT_PROXY=http://host:port` — use a specific proxy.

Slow but valid large-context calls timing out? Raise `LLM_TIMEOUT_SEC` (default 60).

## No code search / "RELATED CODE" tier empty
The code index uses **ripgrep**. Install it (`brew install ripgrep` / `apt install ripgrep`); without `rg`
memagent falls back to `NullRetriever` (no auto code discovery, everything else works).

## Copy/paste doesn't work
- **Default (`AGENT_TUI=rich`) and `live`**: stay in the normal terminal buffer — native select + copy/paste/scroll work everywhere, including macOS Terminal.app.

## An MCP server fails to load
memagent logs the failure and continues without that server (it never crashes the session). Check the server
command in `[mcp_servers.*]` and that its stdio binary is on PATH. A stuck server can't freeze exit — shutdown
is bounded by a timeout.

## A turn aborted with "stuck" / "max_steps" / "overflow"
- **stuck** — the model repeated blocked actions; rephrase or narrow the task.
- **max_steps** — raise the step budget or break the task up.
- **overflow** — too many files/too large a context; reduce the files in play.

## Permission prompts are blocking automation
- `AGENT_POLICY=allow` — permissive (use with care).
- `AGENT_AUTO_APPROVE="git status*,ls *,cat *"` — pre-approve safe command globs.
- `AGENT_POLICY=readonly` — no writes/exec at all.

## A plugin or tool is misbehaving
A failing plugin **hook** degrades gracefully (it's logged, the turn continues); a failing **permission**
hook fails *closed* (the tool is denied). Set `MEMAGENT_DEBUG_TRACE=1` to see tracebacks for parked/hook errors.
Disable plugins by leaving `AGENT_ALLOW_PLUGINS` unset.

## A custom tool hangs
Set `AGENT_TOOL_TIMEOUT=<seconds>` to put a wall-clock deadline on each tool call (off by default).
