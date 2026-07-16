"""Always-on @sliceagent ContextFS contract. No model, network, or pytest dependency."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.contextfs import (  # noqa: E402
    CONTEXTFS_SCHEMA_MARKER,
    ArtifactContextProvider,
    ArtifactHistoryProvider,
    CapabilityStatus,
    ContextFS,
    ContextNotFoundError,
    ContextPathError,
    ContextReadOnlyError,
    ContextStatus,
    LedgerContextProvider,
    LegacyMountProvider,
    MappingContextProvider,
    is_context_path,
    normalize_context_path,
    schemas_advertise_contextfs,
)


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _providers():
    return {
        "evidence/turns": MappingContextProvider({
            "index.md": "# TURN EVIDENCE\n- turn-1.md",
            "turn-1.md": "# turn one\nreceipt lifecycle settled\nneedle evidence",
        }),
        "history": MappingContextProvider({
            "index.md": "# HISTORY\n- sessions/current/turn-1.md",
            "sessions/current/turn-1.md": "# old turn\nneedle history",
            "search.md": "# HISTORY SEARCH\nproject-filtered by default",
        }),
        "work": MappingContextProvider({
            "active.md": "# ACTIVE WORK\n- implement ContextFS",
            "plan.md": "# PLAN\n- tests pending",
            "dependencies.md": "# DEPENDENCIES\n- none",
            "receipts.md": "# WORK RECEIPTS\n- receipt-1",
        }),
        "memory": MappingContextProvider({
            "index.md": "# KNOWLEDGE\n- records/k-1.md",
            "status.md": "# KNOWLEDGE STATUS\nnative healthy",
            "user/index.md": "# USER KNOWLEDGE\n- concise replies",
            "project/index.md": "# PROJECT KNOWLEDGE\n- use ContextFS",
            "craft/index.md": "# CRAFT KNOWLEDGE\n- verify boundaries",
            "records/k-1.md": "# lesson\nneedle knowledge",
        }),
        "roster": MappingContextProvider({
            "index.md": "# ROSTER\n- scout/",
            "scout/profile.md": "# scout\nstanding explorer",
        }),
    }


def _status():
    return ContextStatus(
        current_project="sliceagent",
        current_workspace="/work/sliceagent",
        logical_request="build the permanent context namespace",
        regions={name: CapabilityStatus("available") for name in _providers()},
        open_active_work_count=0,
        knowledge_counts={"unique": 4, "user": 2, "project": 3, "craft": 1},
        native_index=CapabilityStatus("healthy", "FTS ready"),
        memem=CapabilityStatus("disabled", "not configured"),
        cross_project_search_policy="explicit only",
    )


@check
def canonical_manifest_is_live_truthful_and_has_direct_locators():
    live = {"project": "sliceagent", "open": 0}

    def status():
        base = _status()
        return ContextStatus(
            current_project=live["project"], current_workspace=base.current_workspace,
            logical_request=base.logical_request, regions=base.regions,
            open_active_work_count=live["open"], knowledge_counts=base.knowledge_counts,
            native_index=base.native_index, memem=base.memem,
            cross_project_search_policy=base.cross_project_search_policy,
        )

    fs = ContextFS(_providers(), status=status)
    first = fs.read_file("@sliceagent/index.md")
    assert "project: sliceagent" in first and "unresolved request roots: 0" in first
    assert "native index: healthy — FTS ready" in first
    assert "Memem: disabled — not configured" in first
    assert 'read_file("@sliceagent/work/active.md")' in first
    live.update(project="hunter", open=4)
    second = fs.read_file("@sliceagent")
    assert "project: hunter" in second and "unresolved request roots: 4" in second
    fs.set_status_provider({"current_project": "another-project", "open_work_count": 1})
    replaced = fs.read_file("@sliceagent/index.md")
    assert "project: another-project" in replaced and "unresolved request roots: 1" in replaced


@check
def manifest_names_exactly_three_layers_and_separates_adjacent_capabilities():
    manifest = ContextFS(_providers(), status=_status()).read_file("@sliceagent/index.md")
    assert "## Memory model — exactly three layers" in manifest
    assert "L0 · HISTORY AND EVIDENCE" in manifest
    assert "L1 · ACTIVE WORK" in manifest
    assert "L2 · TYPED KNOWLEDGE" in manifest
    assert "## Adjacent capabilities (not memory layers)" in manifest
    assert "roster: available" in manifest
    assert "Indexes, retrieval backends, roster, skills, and subagents are not additional" in manifest
    assert "## Self-inspection status" in manifest
    assert "## Specific content drill-down" in manifest
    assert "this is not a traversal checklist" in manifest
    assert 'read_file("@sliceagent/memory/status.md")' in manifest


@check
def canonical_memory_status_cannot_be_shadowed_and_reports_lifecycle_truth():
    # A stale provider-owned status document must not override the host's live canonical report.
    fs = ContextFS(
        {"memory": MappingContextProvider({
            "index.md": "# KNOWLEDGE\n(empty)",
            "status.md": "# STALE FOUR-LAYER MODEL\n/private/home/.sliceagent/vault",
            "diagnostics.md": "# STALE INVENTORY\n/private/home/.sliceagent/vault",
        })},
        status={
            "knowledge_counts": {"unique": 3, "user": 0, "project": 2, "craft": 1},
            "legacy_inventory": {
                "legacy_episodes": 147, "tasks": 126, "sessions": 126,
                "roster_records": 38, "subagent_records": 7,
            },
            "legacy_inventory_scope": "global compatibility store; not typed project knowledge",
            "legacy_inventory_status": {
                "state": "available", "detail": "host-counted aggregate snapshot",
            },
            "compatibility_transition": {
                "state": "retained", "detail": "legacy compatibility layout retained",
            },
            "last_consolidation": {
                "state": "completed_with_rejections", "attempted_at": "2026-07-12T12:00:00Z",
                "source_episode_count": 12, "mode": "deterministic",
                "lessons": 2, "skills": 1, "skills_rejected": 3, "errors": 0,
            },
            "native_index": {"state": "healthy", "detail": "native-fts5"},
            "memem": {"state": "disabled", "detail": "not connected"},
        },
    )
    status = fs.read_file("@sliceagent/memory/status.md")
    assert "MEMORY STATUS — GENERAL SUMMARY" in status
    assert "canonical layer-size total not reported" in status
    assert "unique active current-scope records: 3" in status
    assert "USER scope memberships: 0" in status
    assert "PROJECT scope memberships: 2" in status
    assert "scope memberships overlap; never add them" in status
    assert "compatibility layout (global): retained — legacy compatibility layout retained" in status
    assert "selective knowledge consolidation (current project): completed-with-rejections" in status
    assert "answer boundary: report the measured layer, lifecycle, and component rows" in status
    assert "do not append an `in short`" in status
    assert "episodic session files" not in status and "legacy episodes: 147" not in status
    assert '@sliceagent/memory/diagnostics.md' in status
    assert "STALE FOUR-LAYER" not in status and "/private/home" not in status
    assert "STALE FOUR-LAYER" not in fs.grep("STALE", path="@sliceagent/memory/status.md")
    diagnostics = fs.read_file("@sliceagent/memory/diagnostics.md")
    assert "MEMORY DIAGNOSTICS" in diagnostics
    assert "legacy episodes: 147" in diagnostics and "tasks: 126" in diagnostics
    assert "scope: global compatibility store; not typed project knowledge" in diagnostics
    assert "never add or compare them as layer sizes, unique memories" in diagnostics
    assert "source episodes considered: 12" in diagnostics
    assert "consolidation mode: deterministic" in diagnostics
    assert "lessons=2, skills=1, skills rejected=3, errors=0" in diagnostics
    assert "STALE INVENTORY" not in diagnostics and "/private/home" not in diagnostics
    assert "STALE INVENTORY" not in fs.grep("STALE", path="@sliceagent/memory/diagnostics.md")


@check
def absent_consolidation_telemetry_stays_unknown():
    status = ContextFS(_providers(), status={}).read_file("@sliceagent/memory/status.md")
    assert "selective knowledge consolidation (current project): unknown — not reported by host" in status
    assert "selective knowledge consolidation (current project): not-recorded" not in status


@check
def canonical_tree_reads_and_lists_injected_provider_documents():
    fs = ContextFS(_providers(), status=_status())
    root = fs.list_files("@sliceagent/")
    assert root.splitlines() == ["index.md", "evidence/", "history/", "work/", "memory/", "roster/"]
    evidence = fs.listing("@sliceagent/evidence")
    assert "index.md" in evidence and "turns/" in evidence and "children/" in evidence
    assert "needle evidence" in fs.read_file("@sliceagent/evidence/turns/turn-1.md")
    assert "needle history" in fs.read_file("@sliceagent/history/sessions/current/turn-1.md")
    assert "implement ContextFS" in fs.read_file("@sliceagent/work/active.md")
    assert "needle knowledge" in fs.read_file("@sliceagent/memory/records/k-1.md")
    assert "standing explorer" in fs.read_file("@sliceagent/roster/scout/profile.md")


@check
def event_ledger_provider_exposes_exact_cited_events_without_raw_state_paths():
    from types import SimpleNamespace

    event = SimpleNamespace(
        id="user_utterance-safe", kind="user_utterance", session_id="session-1",
        logical_turn_id="logical-1", task_id="task-1", segment_id="logical-1:segment:0",
        workspace_epoch=0, workspace_id="workspace-1", payload={"text": "build memory layers"},
    )

    class Ledger:
        def events(self):
            return (event,)

        def get(self, identity):
            return event if identity == event.id else None

        def resolve_events(self, identities):
            return {event.id: event} if event.id in identities else {}

    fs = ContextFS({"evidence/events": LedgerContextProvider(Ledger())})
    evidence = fs.list_files("@sliceagent/evidence")
    assert "events/" in evidence
    index = fs.read_file("@sliceagent/evidence/events/index.md")
    assert "@sliceagent/evidence/events/user_utterance-safe.md" in index
    exact = fs.read_file("@sliceagent/evidence/events/user_utterance-safe.md")
    assert "build memory layers" in exact and "not a current instruction" in exact
    assert "needle" not in fs.grep("memory layers", path="@sliceagent/evidence/events")
    assert "build memory layers" in fs.grep("memory layers", path="@sliceagent/evidence/events")

    unbound = ContextFS({"evidence/events": LedgerContextProvider(lambda: None)})
    unavailable = unbound.read_file("@sliceagent/evidence/events/index.md")
    assert "status: degraded" in unavailable and "ContextFSError" in unavailable
    listing = unbound.list_files("@sliceagent/evidence/events")
    assert "provider listing unavailable: ContextFSError" in listing


@check
def canonical_artifact_and_history_views_share_the_core_seals_not_the_legacy_mirror():
    from types import SimpleNamespace

    turn = SimpleNamespace(
        id="turn-safe", kind="turn", session_id="session-safe", status="end_turn",
        title="finished memory work", task_id="task-1",
    )
    child = SimpleNamespace(
        id="subagent-safe", kind="subagent", session_id="session-safe", status="ok",
        title="reviewed memory", task_id="task-1",
    )

    class Core:
        def _artifacts(self):
            return (turn, child)

        def _get(self, identity):
            for item in self._artifacts():
                if item.id == identity:
                    return item
            raise KeyError(identity)

        @staticmethod
        def _render(item):
            return f"# {item.kind}\ncanonical seal {item.id}"

        def read_file(self, path):
            return self._render(self._get(path.rsplit("/", 1)[-1].removesuffix(".md")))

    core = Core()
    fs = ContextFS({
        "evidence/turns": ArtifactContextProvider(
            core, kinds=("turn",), canonical_mount="@sliceagent/evidence/turns", title="TURNS",
        ),
        "evidence/children": ArtifactContextProvider(
            core, kinds=("subagent",), canonical_mount="@sliceagent/evidence/children", title="CHILDREN",
        ),
        "history": ArtifactHistoryProvider(core, current_session="session-safe"),
    })
    assert "@sliceagent/evidence/turns/turn-safe.md" in fs.read_file(
        "@sliceagent/evidence/turns/index.md"
    )
    assert "subagent-safe" not in fs.list_files("@sliceagent/evidence/turns")
    assert "canonical seal turn-safe" in fs.read_file("@sliceagent/history/turn-1.md")
    assert "@sliceagent/history/sessions/session-safe/turn-safe.md" in fs.read_file(
        "@sliceagent/history/sessions/session-safe/index.md"
    )
    history_hits = fs.grep("canonical seal turn-safe", path="@sliceagent/history")
    assert history_hits.count("canonical seal turn-safe") == 1
    assert "@sliceagent/history/turn-1.md" in history_hits
    # The current session's artifact-ID compatibility spelling remains directly
    # readable, but a root history walk must not inject the same sealed bytes twice.
    hits = fs.grep("canonical seal turn-safe", path="@sliceagent/history")
    assert hits.count("canonical seal turn-safe") == 1


@check
def canonical_child_view_exposes_the_same_page_backed_evidence_as_artifacts_mount():
    import hashlib
    import tempfile

    from sliceagent.persistence import Artifact, ArtifactStore
    from sliceagent.runtime_persistence import CoreArtifactFS

    store = ArtifactStore(tempfile.mkdtemp(prefix="contextfs-child-pages-"))
    view = "     1\tdef answer():\n     2\t    return 42"
    encoded = view.encode()
    store.put(Artifact(
        id="subagent-context-pages", kind="subagent", workspace_id="workspace",
        session_id="session", task_id="task", status="ok",
        structured_body={
            "report": "answer returns 42",
            "observations": [{
                "v": 1, "tool": "read_file", "args": {"path": "answer.py"},
                "status": "succeeded", "view": view,
                "raw_sha256": hashlib.sha256(encoded).hexdigest(),
                "view_sha256": hashlib.sha256(encoded).hexdigest(),
                "raw_bytes": len(encoded), "view_bytes": len(encoded),
                "redacted": False, "truncated": False,
            }],
        },
    ))
    core = CoreArtifactFS(store)
    fs = ContextFS({
        "evidence/children": ArtifactContextProvider(
            core, kinds=("subagent",), canonical_mount="@sliceagent/evidence/children",
            title="CHILDREN",
        ),
    })
    report = fs.read_file("@sliceagent/evidence/children/subagent-context-pages.md")
    assert "answer returns 42" in report and view not in report
    index = fs.read_file(
        "@sliceagent/evidence/children/subagent-context-pages/evidence/index.md"
    )
    assert "obs-001-page-001.md" in index
    page = fs.read_file(
        "@sliceagent/evidence/children/subagent-context-pages/evidence/obs-001-page-001.md"
    )
    assert view in page
    listing = fs.list_files(
        "@sliceagent/evidence/children/subagent-context-pages/evidence"
    )
    assert "index.md" in listing and "obs-001-page-001.md" in listing


@check
def exact_artifact_corruption_is_degraded_evidence_not_a_false_clean_miss():
    from types import SimpleNamespace

    item = SimpleNamespace(
        id="turn-corrupt", kind="turn", session_id="session", status="end_turn",
        title="corrupt evidence", task_id="task",
    )

    class Core:
        def _artifacts(self):
            return (item,)

        def _get(self, _identity):
            raise RuntimeError("canonical bytes failed verification")

        def read_file(self, _path):
            raise AssertionError("the canonical adapter must use exact _get")

    fs = ContextFS({
        "evidence/turns": ArtifactContextProvider(
            Core(), kinds=("turn",), canonical_mount="@sliceagent/evidence/turns", title="TURNS",
        ),
    })
    body = fs.read_file("@sliceagent/evidence/turns/turn-corrupt.md")
    assert "status: degraded" in body and "RuntimeError" in body
    assert "no retained artifact" not in body


@check
def missing_regions_stay_addressable_and_never_claim_to_be_healthy():
    # Even a contradictory caller-supplied status cannot advertise a region with no provider as available.
    fs = ContextFS(status={
        "regions": {"memory": {"state": "available", "detail": "claimed"}},
        "knowledge_counts": {"unique": 0, "user": 0, "project": 0, "craft": 0},
        "native_index_health": "unavailable",
        "memem_status": "disabled",
    })
    index = fs.read_file("@sliceagent/index.md")
    assert "memory: unavailable — no provider mounted" in index
    assert "unique records: 0" in index
    assert "USER scope: 0" in index and "PROJECT scope: 0" in index and "CRAFT scope: 0" in index
    unavailable = fs.read_file("@sliceagent/memory/index.md")
    assert "status: unavailable — no provider mounted" in unavailable
    status = fs.read_file("@sliceagent/memory/status.md")
    assert "native index: unavailable" in status and "Memem: disabled" in status
    assert "index.md" in fs.list_files("@sliceagent/memory/user")

    invalid = ContextFS(
        {"work": MappingContextProvider({"active.md": "# ACTIVE WORK\n(none)"})},
        status={"regions": {"work": {"state": "definitely-perfect"}}},
    ).read_file("@sliceagent/index.md")
    assert "work: unknown — unrecognized status: definitely-perfect" in invalid


@check
def grep_is_cross_region_scoped_canonical_and_paged():
    fs = ContextFS(_providers(), status=_status())
    all_hits = fs.grep("needle", path="@sliceagent")
    assert "@sliceagent/evidence/turns/turn-1.md:3:needle evidence" in all_hits
    assert "@sliceagent/history/sessions/current/turn-1.md:2:needle history" in all_hits
    assert "@sliceagent/memory/records/k-1.md:2:needle knowledge" in all_hits
    scoped = fs.grep("needle", path="@sliceagent/history")
    assert "needle history" in scoped and "needle evidence" not in scoped and "needle knowledge" not in scoped
    files = fs.grep("needle", path="@sliceagent", output_mode="files_with_matches")
    assert "@sliceagent/evidence/turns/turn-1.md" in files and ":3:" not in files
    counts = fs.grep("needle", path="@sliceagent", output_mode="count")
    assert "@sliceagent/memory/records/k-1.md:1" in counts
    page = fs.grep("needle", path="@sliceagent", limit=1)
    assert "[truncated; use offset=1" in page
    assert "invalid regex" in fs.grep("(unclosed", path="@sliceagent")


@check
def grep_searches_a_canonical_fallback_below_a_mounted_parent():
    # The production memory provider owns record/index documents and deliberately leaves status.md to
    # ContextFS. Its synthesized status is still a real read surface and must therefore be searchable.
    fs = ContextFS(
        {"memory": MappingContextProvider({"index.md": "# KNOWLEDGE\n(empty)"})},
        status={
            "native_index": {"state": "healthy", "detail": "sqlite-fts5"},
            "memem": {"state": "disabled", "detail": "not connected"},
        },
    )
    status = fs.read_file("@sliceagent/memory/status.md")
    assert "sqlite-fts5" in status
    matches = fs.grep("sqlite-fts5", path="@sliceagent/memory/status.md")
    assert "@sliceagent/memory/status.md" in matches and "sqlite-fts5" in matches


@check
def exact_memory_status_grep_does_not_touch_the_shadowed_provider():
    fs = ContextFS(
        {"memory": _BrokenProvider()},
        status={"native_index": {"state": "healthy", "detail": "sqlite-fts5"}},
    )
    matches = fs.grep("MEMORY STATUS", path="@sliceagent/memory/status.md")
    assert "@sliceagent/memory/status.md" in matches
    assert "incomplete" not in matches and "RuntimeError" not in matches
    diagnostics = fs.grep("MEMORY DIAGNOSTICS", path="@sliceagent/memory/diagnostics.md")
    assert "@sliceagent/memory/diagnostics.md" in diagnostics
    assert "incomplete" not in diagnostics and "RuntimeError" not in diagnostics


@check
def traversal_is_routed_to_contextfs_and_rejected_for_every_operation():
    fs = ContextFS(_providers())
    attempts = (
        "@sliceagent/../secret",
        "@sliceagent/history/../../etc/passwd",
        ".\\@sliceagent\\memory\\..\\secret",
    )
    for path in attempts:
        assert is_context_path(path), path
        for operation in (
            lambda: fs.read_file(path),
            lambda: fs.list_files(path),
            lambda: fs.grep("x", path=path),
        ):
            try:
                operation()
            except ContextPathError:
                pass
            else:
                raise AssertionError(f"traversal was accepted: {path}")
    assert normalize_context_path("./@sliceagent//history/./index.md/") == "@sliceagent/history/index.md"
    assert not is_context_path("@sliceagent-other/history")


@check
def namespace_is_explicitly_read_only():
    fs = ContextFS(_providers())
    try:
        fs.deny_write("@sliceagent/work/active.md", operation="edit")
    except ContextReadOnlyError as error:
        assert "read-only internal context" in str(error) and "edit is not allowed" in str(error)
    else:
        raise AssertionError("ContextFS allowed a write")
    assert "read-only" in fs.read_only_message("@sliceagent/memory/index.md")


class _BrokenProvider:
    def read_file(self, _path):
        raise RuntimeError("private backend detail")

    def list_files(self, _path=""):
        raise RuntimeError("private backend detail")

    def grep_matches(self, _pattern, _path=""):
        raise RuntimeError("private backend detail")


@check
def provider_and_status_failures_do_not_remove_the_permanent_namespace():
    def broken_status():
        raise LookupError("private status detail")

    fs = ContextFS({"memory": _BrokenProvider()}, status=broken_status)
    index = fs.read_file("@sliceagent/index.md")
    assert index.startswith("# SLICEAGENT INTERNAL CONTEXT")
    assert "live status: unavailable (LookupError)" in index and "private status detail" not in index
    memory = fs.read_file("@sliceagent/memory/index.md")
    assert "degraded" in memory and "RuntimeError" in memory and "private backend detail" not in memory
    listing = fs.list_files("@sliceagent/memory")
    assert "index.md" in listing and "provider listing unavailable: RuntimeError" in listing
    grep = fs.grep("anything", path="@sliceagent/memory")
    assert "search incomplete" in grep and "memory: RuntimeError" in grep


class _LegacyHistory:
    def read_file(self, path):
        if path in {"history", "history/index.md"}:
            return "# HISTORY\n- turn-2.md"
        if path == "history/turn-2.md":
            return "# turn two\nlegacy needle"
        return f"{path}: no such turn"

    def listing(self, _path="history"):
        return "index.md\nturn-2.md\n(read index.md for titles)"

    def grep(self, pattern, *, path="history", output_mode="content", context=0, offset=0, limit=50):
        del output_mode, context, offset, limit
        if "needle" in pattern and path in {"history", "history/turn-2.md"}:
            return "history/turn-2.md:2:legacy needle"
        return "grep: no matches found."


@check
def legacy_mount_adapter_removes_old_names_and_emits_canonical_locators():
    fs = ContextFS({"history": LegacyMountProvider(
        _LegacyHistory(), "history", canonical_mount="@sliceagent/history",
    )})
    assert "turn-2.md" in fs.list_files("@sliceagent/history")
    assert "legacy needle" in fs.read_file("@sliceagent/history/turn-2.md")
    result = fs.grep("needle", path="@sliceagent/history")
    assert "@sliceagent/history/turn-2.md:2:legacy needle" in result
    assert not any(line.startswith("history/") for line in result.splitlines())


@check
def unknown_paths_fail_cleanly_while_dynamic_canonical_paths_report_backend_absence():
    fs = ContextFS()
    for path in ("@sliceagent/nope/index.md", "@sliceagent/work/not-a-surface.md"):
        try:
            fs.read_file(path)
        except ContextNotFoundError as error:
            assert CONTEXTFS_SCHEMA_MARKER in str(error)
        else:
            raise AssertionError(f"unknown path resolved: {path}")
    # records/<id> is a canonical dynamic address family, so absence is a truthful backend status.
    assert "unavailable" in fs.read_file("@sliceagent/memory/records/missing.md")


@check
def schema_marker_is_detected_only_on_live_file_tools():
    absent = [{"type": "function", "function": {
        "name": "read_file", "description": "Read a workspace file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    }}]
    present = [{"type": "function", "function": {
        "name": "read_file", "description": f"Read a file; internal context starts at {CONTEXTFS_SCHEMA_MARKER}.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    }}]
    unrelated = [{"type": "function", "function": {
        "name": "send_email", "description": f"Mention {CONTEXTFS_SCHEMA_MARKER} but cannot read it.",
        "parameters": {"type": "object", "properties": {}},
    }}]
    assert not schemas_advertise_contextfs(absent)
    assert schemas_advertise_contextfs(present)
    assert not schemas_advertise_contextfs(unrelated)


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(error).__name__}: {error}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
