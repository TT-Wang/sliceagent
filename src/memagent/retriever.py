"""Retriever implementations.

NullRetriever is the v0.1 default (no discovery tier — the deterministic working-set
of active files carries the context, exactly like the validated prototype).

MememRetriever is the planned plug: memem's hybrid BM25+embeddings+RRF+MMR pipeline,
called in-process each turn to fill the discovery tier, plus cross-session recall/mining.
"""
from __future__ import annotations

from .interfaces import Snippet


class NullRetriever:
    def retrieve(self, query: str, k: int = 6) -> list[Snippet]:
        return []


# class MememRetriever:
#     """TODO(P1): wrap memem. retrieve() -> top-k snippets for the discovery tier.
#     Keep memem behind this interface so the moat never imports it directly."""
#     def __init__(self, vault=None): ...
#     def retrieve(self, query: str, k: int = 6) -> list[Snippet]: ...
