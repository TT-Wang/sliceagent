"""The agent loop — the moat. ONE model call per turn, no accumulated history.

Each turn: reconstruct the slice from deterministic tiers + retrieval -> one LLM call ->
run tools -> fold results into the tiers -> repeat. The full record streams to a durable
on-disk log (the event store), never into the model's context.
"""
from __future__ import annotations

import json
import os

from .interfaces import LLMClient, Oracle, Retriever, ToolHost
from .slice import (
    DISCOVERY_K,
    MAX_ARTIFACT_CHARS,
    SYSTEM_PROMPT,
    Slice,
    record_action,
    render_slice,
    touch_file,
)


def build_artifacts(s: Slice, tools: ToolHost) -> str:
    """Re-read the active files FRESH (the working set) — head+tail capped to stay bounded."""
    if not s.active_files:
        return "(no files opened yet)"
    parts = []
    for p in s.active_files:
        try:
            body = tools.read_text(p)
        except Exception:
            parts.append(f"### {p}\n(not created yet)")
            continue
        if len(body) > MAX_ARTIFACT_CHARS:
            shown = body[: MAX_ARTIFACT_CHARS - 500] + "\n…[middle truncated]…\n" + body[-500:]
        else:
            shown = body
        parts.append(f"### {p} ({len(body)} bytes — current contents)\n```\n{shown}\n```")
    return "\n\n".join(parts)


def build_discovery(s: Slice, retriever: Retriever, query: str) -> str:
    snippets = retriever.retrieve(query, k=DISCOVERY_K)
    if not snippets:
        return ""
    return "\n\n".join(
        f"### {sn.path} (score {sn.score:.2f})\n```\n{sn.text[:MAX_ARTIFACT_CHARS]}\n```" for sn in snippets
    )


def _log(path: str, entry: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def run_task(
    task: str,
    s: Slice,
    llm: LLMClient,
    tools: ToolHost,
    retriever: Retriever,
    oracle: Oracle | None = None,
    *,
    max_steps: int = 40,
    log_path: str = "scratch/durable-log.jsonl",
    emit=print,
    show_slice: bool = False,
) -> dict:
    """Run one task to completion (or max_steps). Returns usage stats."""
    s.reset(task)
    system = (
        SYSTEM_PROMPT
        + "\n\n# TASK (your checklist — do the next item that OPEN FILES shows is not done)\n"
        + task
    )
    stats = {"calls": 0, "in": 0, "out": 0}

    for _ in range(max_steps):
        artifacts = build_artifacts(s, tools)
        discovery = build_discovery(s, retriever, task)
        user = render_slice(s, artifacts, discovery)
        if show_slice:
            emit("\n  ┌─ slice ─────────────\n" + "\n".join("  │ " + ln for ln in user.splitlines()) + "\n  └─────────────")

        reply = llm.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tools.schemas(),
        )
        stats["calls"] += 1
        if reply.usage:
            stats["in"] += reply.usage.get("prompt_tokens", 0)
            stats["out"] += reply.usage.get("completion_tokens", 0)
        _log(log_path, {"role": "assistant", "content": reply.content,
                        "tool_calls": [{"name": c.name, "args": c.args} for c in reply.tool_calls]})

        if reply.content:
            emit(f"\nAssistant: {reply.content}")
        if not reply.tool_calls:
            return stats  # done — nothing was ever appended to a history array

        for tc in reply.tool_calls:
            out = str(tools.run(tc.name, tc.args))
            emit(f"  → {tc.name}({json.dumps(tc.args, ensure_ascii=False)})")
            emit(f"  ← {out[:120]}")
            if tc.args.get("path"):
                touch_file(s, tc.args["path"])
            record_action(s, tc.name, tc.args, out)
            _log(log_path, {"role": "tool", "name": tc.name, "args": tc.args, "full": out})

    emit(f"\n[stopped: hit max_steps={max_steps}]")
    return stats
