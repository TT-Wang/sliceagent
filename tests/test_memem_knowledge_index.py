"""Adversarial checks for Memem as a canonical SliceAgent L2 projection."""
from __future__ import annotations

import hashlib

from sliceagent.knowledge import (
    KnowledgeHit,
    KnowledgeQuery,
    KnowledgeRecord,
    KnowledgeRepository,
    KnowledgeScope,
    KnowledgeSensitivity,
    KnowledgeSourceRef,
)
from sliceagent.knowledge_index import MememKnowledgeIndex, NativeKnowledgeIndex


NOW = "2026-07-13T00:00:00Z"


def _source(identity: str, text: str) -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        namespace="event", record_id=identity,
        digest=hashlib.sha256(text.encode()).hexdigest(),
        observer="test", observed_at=NOW,
    )


def _active(identity: str, project: str, content: str, **fields) -> KnowledgeRecord:
    return KnowledgeRecord(
        id=identity, kind="fact", scopes=KnowledgeScope(project_id=project),
        content=content, source_refs=(_source(identity + "-source", content),),
        status="active", **fields,
    )


def test_memem_projection_uses_stable_identity_harmonic_representation_and_canonical_scope(tmp_path):
    repository = KnowledgeRepository(str(tmp_path / "knowledge.db"), use_fts=False)
    hunter = _active(
        "hunter-parser", "hunter",
        "Verified parser bounds fix. Incidentalpayload belongs only in the full evidence value.",
        metadata={
            "title": "Parser bounds regression",
            "tags": "validation, parser",
            "paths": ["src/parser.py"],
            "cues": ["request boundary"],
        },
    )
    other = _active("other-parser", "other", "Verified parser bounds fix in another project.")
    secret = _active(
        "hunter-secret", "hunter", "Verified secret detail.",
        sensitivity=KnowledgeSensitivity.SECRET,
    )
    candidate = KnowledgeRecord(
        id="hunter-candidate", kind="fact", scopes=KnowledgeScope(project_id="hunter"),
        content="Unverified parser theory.", status="candidate",
    )
    for record in (hunter, other, secret, candidate):
        repository.put(record)

    projected: dict[str, dict] = {}
    removed: list[str] = []

    def upsert(external_id, value, **kwargs):
        projected[external_id] = {"value": value, **kwargs}
        return {"external_id": external_id}

    def remove(external_id):
        removed.append(external_id)
        return True

    calls: list[dict] = []

    def retrieve(_query, **kwargs):
        calls.append(kwargs)
        if kwargs["scope_id"] != "sliceagent.project:hunter":
            return []
        # A buggy/malicious backend returns both an out-of-scope canonical id
        # and an orphan. Canonical resolution must reject both.
        return [
            {"external_id": "sliceagent:other-parser", "score": 99.0},
            {"external_id": "sliceagent:missing", "score": 100.0},
            {"external_id": "sliceagent:hunter-parser", "score": 0.6,
             "primary_index": "Parser bounds regression"},
        ]

    backend = MememKnowledgeIndex(
        repository, fallback=NativeKnowledgeIndex(repository),
        retrieve_fn=retrieve, upsert_fn=upsert, remove_fn=remove,
    )
    backend.rebuild()

    item = projected["sliceagent:hunter-parser"]
    assert item["value"] == hunter.content  # complete, readable value retained
    assert "Parser bounds regression" in item["primary_index"]
    assert "Incidentalpayload" not in item["primary_index"]
    assert {"request boundary", "src/parser.py", "parser.py"}.issubset(item["cues"])
    assert item["scope_id"] == "sliceagent.project:hunter"
    assert "sliceagent:hunter-secret" in removed
    assert "sliceagent:hunter-candidate" in removed

    hits = backend.search(KnowledgeQuery(
        text="parser bounds", user_id="local-user", project_id="hunter",
        agent_id="sliceagent", paths_context=("src/parser.py",), limit=5,
    ))
    assert [hit.record.id for hit in hits] == ["hunter-parser"]
    assert all(call["scope_mode"] == "hard" and call["writeback"] is False for call in calls)
    assert calls[0]["paths_context"] == ["src/parser.py"]
    assert backend.health()["orphan_hits_dropped"] == 1
    repository.close()


class _SpyFallback:
    is_active = True

    def __init__(self, hit: KnowledgeHit) -> None:
        self.hit = hit
        self.search_calls = 0

    def index(self, records):
        return None

    def remove(self, record_ids):
        return None

    def search(self, query):
        self.search_calls += 1
        return [self.hit]

    def rebuild(self, repository=None):
        return None

    def health(self):
        return {"active": True, "backend": "spy"}


def test_successful_empty_memem_result_does_not_reintroduce_full_body_noise_but_failure_falls_back(tmp_path):
    repository = KnowledgeRepository(str(tmp_path / "knowledge.db"), use_fts=False)
    noisy = _active(
        "old-review", "hunter",
        "A historical bug report contains billingtoken only as incidental detail.",
    )
    repository.put(noisy)
    spy = _SpyFallback(KnowledgeHit(record=noisy, score=9.0, snippet=noisy.content))

    backend = MememKnowledgeIndex(
        repository, fallback=spy,
        retrieve_fn=lambda *_args, **_kwargs: [],
        upsert_fn=lambda *_args, **_kwargs: {},
        remove_fn=lambda *_args, **_kwargs: True,
    )
    query = KnowledgeQuery(
        text="billingtoken", user_id="local-user", project_id="hunter", agent_id="sliceagent",
    )
    assert backend.search(query) == []
    assert spy.search_calls == 0

    def broken(*_args, **_kwargs):
        raise TimeoutError("semantic backend unavailable")

    backend._retrieve = broken
    assert [hit.record.id for hit in backend.search(query)] == ["old-review"]
    assert spy.search_calls == 1
    assert backend.health()["state"] == "degraded"
    repository.close()


def test_failed_projection_forces_native_failover_instead_of_false_empty(tmp_path):
    repository = KnowledgeRepository(str(tmp_path / "knowledge.db"), use_fts=False)
    record = _active("parser-fix", "hunter", "Verified parser boundary fix.")
    repository.put(record)
    spy = _SpyFallback(KnowledgeHit(record=record, score=4.0, snippet=record.content))

    def unavailable(*_args, **_kwargs):
        raise OSError("vault unavailable")

    backend = MememKnowledgeIndex(
        repository, fallback=spy,
        retrieve_fn=lambda *_args, **_kwargs: [],
        upsert_fn=unavailable,
        remove_fn=lambda *_args, **_kwargs: True,
    )
    try:
        backend.index((record,))
    except OSError:
        pass
    else:
        raise AssertionError("projection failure must be surfaced")

    hits = backend.search(KnowledgeQuery(
        text="parser boundary", project_id="hunter", user_id="local-user", agent_id="sliceagent",
    ))
    assert [hit.record.id for hit in hits] == ["parser-fix"]
    assert spy.search_calls == 1
    assert backend.health()["projection_degraded"] is True
    repository.close()
