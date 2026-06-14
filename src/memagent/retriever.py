"""Retriever implementations.

NullRetriever is the empty discovery tier — the deterministic working-set of active
files carries the context (exactly like the validated prototype). It's the test/eval
seam (deterministic) and the graceful fallback when no code search is available.

The real RELATED CODE tier is `code_index.RipgrepCodeIndex` (ripgrep over the working
tree). Construct it via `code_index.make_code_index(root)`, which returns a NullRetriever
when ripgrep isn't on PATH. (memem is NOT a Retriever — it's the Memory tier; see
memory.py. memem indexes a lesson vault, not source code.)
"""
from __future__ import annotations

from .interfaces import Snippet


class NullRetriever:
    def retrieve(self, query: str, k: int = 6) -> list[Snippet]:
        return []
