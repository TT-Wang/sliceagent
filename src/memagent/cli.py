"""memagent CLI — a thin REPL over the slice core. The real product surfaces (TUI,
channels, IDE bridge) sit over the same engine via interfaces; this is the minimal one.
"""
from __future__ import annotations

import os
import sys


def _load_env(path: str = ".env") -> None:
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass


def main() -> None:
    _load_env()
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")):
        print("Set OPENAI_API_KEY (or MOONSHOT_API_KEY) — e.g. in a .env file.")
        sys.exit(1)

    from .llm import OpenAILLM
    from .loop import run_task
    from .retriever import NullRetriever
    from .slice import Slice
    from .tools import LocalToolHost

    llm = OpenAILLM()
    tools = LocalToolHost()
    retriever = NullRetriever()  # discovery tier off until memem is plugged in
    s = Slice()
    show_slice = bool(os.environ.get("SHOW_SLICE"))

    print(f"memagent · slice core · model={llm.model}")
    print('type a task, or "exit" to quit\n')
    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line in ("exit", "quit"):
            break
        if line:
            run_task(line, s, llm, tools, retriever, show_slice=show_slice)


if __name__ == "__main__":
    main()
