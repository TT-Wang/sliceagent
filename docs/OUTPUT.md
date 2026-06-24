# Reading memagent's output

A quick guide to what you see while the agent runs (the default inline `rich` UI).

## The banner

On start you get a boxed logo and a status line:

```
model=kimi-k2.7-code · net=direct · policy=guard · sandbox=local · code=RipgrepCodeIndex · memory=… · …
```

- **model** — the LLM driving the agent.
- **net** — the network route: `direct`, or a proxy URL when one is in use (auto for foreign endpoints; set `AGENT_PROXY=off` to force direct).
- **policy** — permission mode: `guard` (blocks catastrophic commands, normal dev passes), `readonly`, `ask`, `allow`.
- **sandbox** — `local` or `docker` for shell/code execution.

## Your turn

```
▌ you  add a --json flag and a test
```

Your message is echoed the instant you press Enter (before any routing/LLM work), then the agent starts.

## While it works

- **`thinking…` / `writing… <tail>`** — a live spinner; the tail shows the reply streaming in real time.
- **Tool cards** — one line per action, set off by a dim `┊` gutter:
  ```
  ┊ ✓ 📖 read parser.py
  ┊ ✓ ⚡ run pytest -q
    ┊   3 passed in 0.4s
  ┊ ✗ ⚡ run python x.py
    ┊   Traceback: NameError: foo
  ```
  `✓`/`✗` = success/failure. Read/list cards show just the action (the content shows up in the reply); commands and failures show their output.
- **Inline diffs** — an edit shows `- old` / `+ new` lines under the card.
- **Plan checklist** — when the agent plans, you get a live panel: `✓ done · ▶ in-progress · ○ pending`.
- **🎯 mission** — the agent's one-line statement of what it's doing.
- **💡 learned** — a lesson mined from a successful, error-then-fixed episode (written to long-term memory).

## The reply

The assistant's answer renders as Markdown in a bordered box (`assistant` label, code fences highlighted).

## The status bar (bottom)

```
 ◆ kimi-k2.7-code  │  guard  │  <topic>  │  Σ 12k tok · 1.2k fresh  │  ⏲ 8s
```

- **Σ tok** — total tokens this session.
- **fresh** — the moat metric: *non-cached* input tokens. memagent rebuilds a bounded slice each turn instead of growing a transcript, so this stays roughly flat as the session grows (a transcript loop's grows every turn). If you want to *see* the moat, watch `fresh` over a long session.
- **⏲** — wall-clock of the last turn.

## Done

```
  ✓ done · 6 steps · 1843 tokens
```

A clean turn ends with a summary line. On exit, memagent consolidates the session's lessons into long-term memory (`· consolidated session memory`).
