"""Runtime-facing adapter over the local artifact/checkpoint/journal protocol.

This is the only persistence object the agent host needs to know.  It captures a stable
task identity at turn start, journals execution, and seals an immutable artifact before
publishing the next active-state checkpoint.  Semantic memory remains an optional consumer
of the resulting artifacts rather than the durability mechanism itself.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .persistence import (
    Artifact,
    Checkpoint,
    JournalCorruptError,
    PendingTurnJournal,
    PersistenceError,
    RecoveryResult,
    SealCoordinator,
    SealResult,
    WorkspaceLease,
)
from .recovery import root_key, state_dir
from .safety import redact_text


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {redact_text(str(key)): _redact(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(child) for child in value]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return redact_text(str(value))


def _confirmed_transition_ids(snapshot) -> tuple[str, ...]:
    """Only complete outcome effect sets are publishable after a partial journal failure."""
    recorded = {
        str(event.get("payload", {}).get("transition_id"))
        for event in snapshot.events
        if event.get("type") == "semantic-transition"
        and event.get("payload", {}).get("transition_id")
    }
    confirmed = []
    for event in snapshot.events:
        if event.get("type") != "tool-outcome":
            continue
        effects = (event.get("payload", {}).get("outcome", {}).get("effects") or ())
        ids = [str(effect.get("id")) for effect in effects
               if isinstance(effect, Mapping) and effect.get("id")]
        if ids and set(ids).issubset(recorded):
            confirmed.extend(ids)
    return tuple(dict.fromkeys(confirmed))


@dataclass(frozen=True)
class ActiveTurn:
    task_id: str
    logical_id: str
    artifact_id: str
    journal: PendingTurnJournal


class LocalTurnStore:
    """Always-on local durability with one live writer per workspace store."""

    def __init__(self, workspace_root: str, session_id: str, *, store_root: str | None = None,
                 coordinator: SealCoordinator | None = None, exclusive: bool = True):
        self.workspace_root = os.path.realpath(workspace_root)
        self.workspace_id = root_key(self.workspace_root)
        self.session_id = str(session_id)
        self.store_root = store_root or state_dir("core", self.workspace_id)
        self._lease = WorkspaceLease.acquire(self.store_root) if exclusive else None
        self._closed = False
        self.coordinator = coordinator or SealCoordinator(self.store_root)
        self.active: ActiveTurn | None = None
        self._active_refs: list[str] = []

    def close(self) -> None:
        if self._closed:
            return
        if self._lease is not None:
            self._lease.close()
            self._lease = None
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("local turn store is closed")

    def begin(self, *, task_id: str, logical_id: str, user_request: str) -> ActiveTurn:
        self._ensure_open()
        if self.active is not None:
            raise RuntimeError(f"turn {self.active.logical_id!r} is already active")
        journal = self.coordinator.begin_turn(
            workspace_id=self.workspace_id,
            session_id=self.session_id,
            task_id=str(task_id),
            logical_id=str(logical_id),
            user_request=redact_text(str(user_request)),
        )
        self.active = ActiveTurn(str(task_id), str(logical_id), journal.snapshot().artifact_id, journal)
        self._active_refs = []
        return self.active

    def record_artifact_ref(self, artifact_id: str) -> None:
        """Attach one already-durable child/source artifact to the running turn checkpoint."""
        turn = self._turn()
        value = str(artifact_id)
        turn.journal.record_artifact_ref(value)
        if value not in self._active_refs:
            self._active_refs.append(value)

    def _turn(self) -> ActiveTurn:
        self._ensure_open()
        if self.active is None:
            raise RuntimeError("no active turn")
        return self.active

    def record_invocation(self, invocation_id: str, *, name: str, args: Mapping[str, Any]) -> None:
        self._turn().journal.record_invocation(
            str(invocation_id), name=str(name), args=_redact(dict(args)),
        )

    def record_outcome(self, invocation_id: str, *, status: str, text: str,
                       effects: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = ()) -> None:
        self._turn().journal.record_outcome(str(invocation_id), _redact({
            "status": str(status), "text": str(text), "effects": list(effects),
        }))

    def record_transition(self, transition_id: str, transition: Mapping[str, Any]) -> None:
        self._turn().journal.record_transition(str(transition_id), _redact(dict(transition)))

    def observe_event(self, event) -> None:
        """Journal invocation/outcome truth before authoritative reduction.

        Applied semantic transitions are deliberately recorded by :meth:`observe_reduction` only after
        the required reducer returns successfully.  Keeping those two phases separate prevents a failed
        reducer from publishing effect IDs that the active checkpoint never actually applied.
        """
        if self.active is None:
            return
        # Import lazily so the durable store itself remains independent from the UI/event layer.
        from .events import ToolResult, ToolStarted
        if isinstance(event, ToolStarted) and getattr(event, "invocation", None) is not None:
            inv = event.invocation
            self.record_invocation(inv.id, name=inv.name, args=inv.args)
            return
        if not isinstance(event, ToolResult) or getattr(event, "outcome", None) is None:
            return
        outcome = event.outcome
        # A cancelled/not-started call may have no ToolStarted event. Record its invocation here so the
        # journal still gives every provider call one logical outcome; executed calls idempotently match
        # the pre-dispatch record.
        self.record_invocation(
            outcome.invocation.id, name=outcome.invocation.name, args=outcome.invocation.args,
        )
        effects = [
            {"id": effect.id, "kind": effect.kind, "payload": dict(effect.payload)}
            for effect in outcome.effects
        ]
        self.record_outcome(
            outcome.invocation.id, status=outcome.status.value, text=outcome.text, effects=effects,
        )

    def observe_reduction(self, event) -> None:
        """Record effect IDs only after the authoritative state reducer succeeded."""
        if self.active is None:
            return
        from .events import ToolResult
        if not isinstance(event, ToolResult) or getattr(event, "outcome", None) is None:
            return
        outcome = event.outcome
        for effect in outcome.effects:
            self.record_transition(effect.id, {
                "kind": effect.kind, "payload": dict(effect.payload),
                "invocation_id": outcome.invocation.id,
            })

    def seal(
        self,
        *,
        state: Mapping[str, Any],
        record: Mapping[str, Any],
        status: str,
        title: str = "",
        summary: str = "",
        files: tuple[str, ...] | list[str] = (),
        refs: tuple[str, ...] | list[str] = (),
        uncertainty: tuple[str, ...] | list[str] = (),
        error: str = "",
        workspace_versions: Mapping[str, Any] | None = None,
        cleanup: bool = True,
    ) -> SealResult:
        turn = self._turn()
        snapshot = turn.journal.snapshot()
        transitions = _confirmed_transition_ids(snapshot)
        safe_record = _redact(dict(record))
        safe_state = _redact(dict(state))
        effective_status = str(status)
        uncertainty = tuple(str(item) for item in uncertainty)
        unresolved = snapshot.unresolved_invocations
        if unresolved:
            from .execution import reconciliation_targets

            invocation_ids = []
            targets = []
            for invocation in unresolved:
                invocation_id = str(invocation.get("invocation_id") or "unknown")
                invocation_ids.append(invocation_id)
                args = invocation.get("args")
                targets.extend(reconciliation_targets(
                    str(invocation.get("name") or ""),
                    args if isinstance(args, Mapping) else {},
                ))
            detail = (
                "invocation(s) without conclusive outcomes: " + ", ".join(invocation_ids)
                + "; re-observe affected live state before further side effects"
            )
            existing_marker = str(safe_state.get("reconciliation_required") or "")
            safe_state["reconciliation_required"] = (
                existing_marker if detail in existing_marker else
                " | ".join(item for item in (existing_marker, detail) if item)
            )
            existing_targets = safe_state.get("reconciliation_targets")
            safe_state["reconciliation_targets"] = list(dict.fromkeys((
                *(existing_targets if isinstance(existing_targets, (list, tuple)) else ()),
                *(targets or ("workspace:*",)),
            )))
            safe_state["status"] = "indeterminate"
            effective_status = "indeterminate"
            uncertainty = tuple(dict.fromkeys((*uncertainty, detail)))
            error = str(error or detail)
        refs = tuple(dict.fromkeys((
            *snapshot.artifact_refs, *self._active_refs, *tuple(str(ref) for ref in refs),
        )))
        artifact = Artifact(
            id=snapshot.artifact_id,
            kind="turn",
            workspace_id=self.workspace_id,
            session_id=self.session_id,
            task_id=turn.task_id,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            status=effective_status,
            title=redact_text(str(title)),
            brief={"request": snapshot.header.get("user_request", "")},
            summary=redact_text(str(summary)),
            structured_body=safe_record,
            files=tuple(redact_text(str(path)) for path in files),
            refs=tuple(redact_text(str(ref)) for ref in refs),
            uncertainty=tuple(redact_text(str(item)) for item in uncertainty),
            error=redact_text(str(error)),
        )
        checkpoint = Checkpoint.create(
            workspace_id=self.workspace_id,
            session_id=self.session_id,
            task_id=turn.task_id,
            generation=snapshot.base_generation + 1,
            state=safe_state,
            artifact_refs=(artifact.id, *tuple(str(ref) for ref in refs)),
            applied_transition_ids=transitions,
            workspace_versions=_redact(dict(workspace_versions or {})),
        )
        result = self.coordinator.seal(turn.journal, artifact, checkpoint, cleanup=cleanup)
        # Ownership ends only after the artifact/checkpoint protocol succeeds. A failed seal retains the
        # active turn and blocks a newer generation from overtaking its recovery journal.
        self.active = None
        self._active_refs = []
        return result

    def checkpoints(self):
        """Return startup-safe authoritative checkpoints for task hydration.

        Checkpoint bytes and every referenced artifact are revalidated on each process start. A dangling or
        corrupt live dependency is an explicit conflict; silently dropping it could resume from invented state.
        """
        checkpoints = self.coordinator.checkpoints.list_workspace(self.workspace_id)
        for checkpoint in checkpoints:
            self.coordinator.validate_checkpoint_refs(checkpoint)
        return checkpoints

    def recover_pending(self) -> tuple[RecoveryResult, ...]:
        self._ensure_open()
        results = []
        for journal in PendingTurnJournal.pending(self.store_root):
            try:
                try:
                    snapshot = journal.snapshot()
                except JournalCorruptError:
                    snapshot = journal.salvage_torn_tail()
                checkpoint = None
                if snapshot.seal_intent is None and snapshot.artifact_id.startswith("turn-"):
                    checkpoint = self._replay_unprepared_checkpoint(snapshot)
                results.append(self.coordinator.recover(
                    journal, unprepared_checkpoint=checkpoint,
                ))
            except Exception as exc:  # noqa: BLE001 - isolate one schema-incompatible journal, not BaseException
                # One malformed or schema-incompatible journal is quarantined in place and reported, but
                # cannot hide every later valid journal in lexical order.
                artifact_id = os.path.basename(journal.path).removesuffix(".jsonl")
                results.append(RecoveryResult(
                    status="conflict", artifact_id=artifact_id, detail=f"{type(exc).__name__}: {exc}",
                ))
        return tuple(results)

    def _replay_unprepared_checkpoint(self, snapshot) -> Checkpoint:
        """Rebuild state from confirmed reducer events without re-running any external tool."""
        from dataclasses import asdict
        from .events import ToolResult
        from .execution import (ToolEffect, ToolInvocation, ToolOutcome, ToolStatus,
                                coerce_tool_status, reconciliation_targets)
        from .pfc import Slice, record_user, slice_sink
        from .taskstate import (slice_to_task_state, task_state_from_checkpoint,
                                task_state_to_slice)

        header = snapshot.header
        task_id = str(header.get("task_id") or "unknown-task")
        base_generation = snapshot.base_generation
        base = self.coordinator.checkpoints.load(self.workspace_id, task_id)
        if base_generation:
            if base is None or base.generation != base_generation:
                raise PersistenceError(
                    f"cannot replay turn from base generation {base_generation}; current base is "
                    f"{getattr(base, 'generation', 0)}")
            state = task_state_to_slice(task_state_from_checkpoint(base))
        else:
            state = Slice()
            state.reset(str(header.get("user_request") or ""))

        # An unsealed turn is unresolved by definition.  A previously provisional objective becomes active
        # again before replay so crash recovery cannot publish it as mere background.
        state.task.activate_objective()
        record_user(
            state, str(header.get("user_request") or ""),
            source_artifact=snapshot.artifact_id,
        )
        invocations = {}
        outcomes = []
        applied = set()
        for event in snapshot.events:
            payload = event.get("payload", {})
            if event.get("type") == "tool-invocation":
                invocation_id = str(payload.get("invocation_id") or "")
                if invocation_id:
                    invocations[invocation_id] = payload
            elif event.get("type") == "tool-outcome":
                outcomes.append(payload)
            elif event.get("type") == "semantic-transition":
                transition_id = str(payload.get("transition_id") or "")
                if transition_id:
                    applied.add(transition_id)

        outcome_ids = {
            str(payload.get("invocation_id") or "") for payload in outcomes
            if isinstance(payload, Mapping) and payload.get("invocation_id")
        }
        unresolved_ids = []
        valid_statuses = {status.value for status in ToolStatus}

        def _valid_status(value) -> bool:
            return isinstance(value, bool) or (isinstance(value, str) and value in valid_statuses)

        for payload in outcomes:
            if not isinstance(payload, Mapping):
                continue
            record = payload.get("outcome") or {}
            raw_status = record.get("status") if isinstance(record, Mapping) else None
            status_valid = _valid_status(raw_status)
            if not status_valid or coerce_tool_status(raw_status) is ToolStatus.INDETERMINATE:
                unresolved_ids.append(str(payload.get("invocation_id") or "unknown"))
        missing_ids = sorted(set(invocations) - outcome_ids)
        uncertain_ids = tuple(dict.fromkeys((*unresolved_ids, *missing_ids)))
        uncertainty_detail = ""
        uncertainty_targets = []
        if uncertain_ids:
            details = []
            if unresolved_ids:
                details.append("indeterminate or invalid outcomes: " + ", ".join(unresolved_ids))
            if missing_ids:
                details.append("invocations without conclusive outcomes: " + ", ".join(missing_ids))
            uncertainty_detail = (
                "; ".join(details) + "; re-observe affected live state before further side effects"
            )
            for invocation_id in uncertain_ids:
                invocation = invocations.get(invocation_id) or {}
                args = invocation.get("args") if isinstance(invocation, Mapping) else {}
                uncertainty_targets.extend(reconciliation_targets(
                    str(invocation.get("name") or "") if isinstance(invocation, Mapping) else "",
                    args if isinstance(args, Mapping) else {},
                ))

        reducer = slice_sink(state)
        replayed = []
        for payload in outcomes:
            invocation_id = str(payload.get("invocation_id") or "")
            invocation_record = invocations.get(invocation_id) or {}
            outcome_record = payload.get("outcome") or {}
            raw_status = outcome_record.get("status") if isinstance(outcome_record, Mapping) else None
            if (not _valid_status(raw_status)
                    or coerce_tool_status(raw_status) is ToolStatus.INDETERMINATE):
                continue
            raw_effects = outcome_record.get("effects") or ()
            effects = tuple(ToolEffect(
                str(effect.get("id") or ""), str(effect.get("kind") or "tool_outcome"),
                dict(effect.get("payload") or {}),
            ) for effect in raw_effects if isinstance(effect, Mapping) and effect.get("id"))
            effect_ids = {effect.id for effect in effects}
            # Outcome journaling precedes reduction. Only a complete transition set proves publication;
            # a partial multi-effect append is treated as unapplied rather than replayed ambiguously.
            if not effect_ids or not effect_ids.issubset(applied):
                continue
            invocation = ToolInvocation(
                invocation_id, str(invocation_record.get("name") or ""),
                dict(invocation_record.get("args") or {}), len(replayed),
            )
            outcome = ToolOutcome(
                invocation, coerce_tool_status(outcome_record.get("status")),
                str(outcome_record.get("text") or ""), effects,
            )
            reducer(ToolResult(
                invocation.name, dict(invocation.args), outcome.text, outcome.failing,
                status=outcome.status.value, invocation_id=invocation.id, outcome=outcome,
            ))
            replayed.extend(effect.id for effect in effects)

        # Replay can legitimately clear an older reconciliation marker.  Crash uncertainty from THIS turn
        # is applied last so a later missing/invalid invocation can never be erased by an earlier committed
        # reconcile_execution transition in the same journal.
        if uncertainty_detail:
            if uncertainty_detail not in state.reconciliation_required:
                state.reconciliation_required = " | ".join(
                    item for item in (state.reconciliation_required, uncertainty_detail) if item
                )
            state.reconciliation_targets = list(dict.fromkeys((
                *state.reconciliation_targets, *(uncertainty_targets or ("workspace:*",)),
            )))

        state.seal()
        task_state = slice_to_task_state(
            state, task_id, session_id=str(header.get("session_id") or self.session_id),
            status="indeterminate" if state.reconciliation_required else "parked",
        )
        discovered_refs = tuple(
            child.id for child in self.coordinator.artifacts.list_all()
            if child.parent_id == snapshot.artifact_id
        )
        source_refs = tuple(dict.fromkeys(
            source for source in (
                state.task.goal_source,
                *(entry.source_artifact for entry in state.intent.entries),
            )
            if source and self.coordinator.artifacts.exists(source)
        ))
        return Checkpoint.create(
            workspace_id=str(header.get("workspace_id") or self.workspace_id),
            session_id=str(header.get("session_id") or self.session_id), task_id=task_id,
            generation=base_generation + 1, state=asdict(task_state),
            artifact_refs=tuple(dict.fromkeys((
                snapshot.artifact_id, *snapshot.artifact_refs, *source_refs, *discovered_refs,
            ))),
            applied_transition_ids=tuple(dict.fromkeys(replayed)), workspace_versions={},
            updated_at=str(header.get("created_at") or ""),
        )

    def recover_active_seal(self) -> RecoveryResult | None:
        """Retry a prepared failed seal; never guess state for an unprepared active turn."""
        self._ensure_open()
        if self.active is None or self.active.journal.snapshot().seal_intent is None:
            return None
        result = self.coordinator.recover(self.active.journal)
        if result.status in ("replayed", "attached", "cleaned"):
            self.active = None
            self._active_refs = []
        return result


class CoreArtifactFS:
    """Read-only virtual view over the authoritative local artifact store."""

    MOUNT = "artifacts"

    def __init__(self, artifact_store):
        self.store = artifact_store

    @staticmethod
    def _leaf(path: str) -> str:
        value = str(path or "").replace("\\", "/").strip("/")
        if value == "artifacts":
            return ""
        return value[len("artifacts/"):] if value.startswith("artifacts/") else value

    def _artifacts(self):
        return self.store.list_all()

    @staticmethod
    def _name(artifact) -> str:
        return artifact.id + ".md"

    @staticmethod
    def _render(artifact) -> str:
        body = dict(artifact.structured_body)
        markdown = body.get("markdown")
        lines = [
            f"# {artifact.kind.upper()} ARTIFACT — {artifact.title or artifact.id}",
            f"- id: {artifact.id}", f"- task: {artifact.task_id}",
            f"- status: {artifact.status}", f"- timestamp: {artifact.timestamp or '(unknown)'}",
        ]
        if artifact.summary:
            lines += ["", "## Summary", artifact.summary]
        if markdown:
            lines += ["", "## Record", str(markdown)]
        else:
            lines += ["", "## Structured record", "```json",
                      json.dumps(body, ensure_ascii=False, indent=2, default=str), "```"]
        if artifact.refs:
            lines += ["", "## References", *[f'- read_file("artifacts/{ref}.md")' for ref in artifact.refs]]
        return "\n".join(lines)

    def index(self) -> str:
        artifacts = self._artifacts()
        lines = ["# LOCAL ARTIFACTS — immutable turn and subagent records"]
        lines += [
            f'- {self._name(item)} · {item.status} · {item.title or item.task_id} '
            f'→ read_file("artifacts/{self._name(item)}")'
            for item in artifacts
        ]
        return "\n".join(lines + ([] if artifacts else ["(none yet)"]))

    def read_file(self, path: str) -> str:
        leaf = self._leaf(path)
        if leaf in ("", "index.md"):
            return self.index()
        if not leaf.endswith(".md"):
            return f"artifacts/{leaf}: not an artifact file; read artifacts/index.md"
        artifact_id = leaf[:-3]
        try:
            return self._render(self.store.get(artifact_id))
        except Exception:
            return f"artifacts/{leaf}: no such retained artifact; read artifacts/index.md"

    def listing(self, path: str = MOUNT) -> str:
        return "\n".join(["index.md", *[self._name(item) for item in self._artifacts()]])

    def _docs(self, path: str):
        leaf = self._leaf(path)
        if leaf == "":
            return [("index.md", self.index()), *[
                (self._name(item), self._render(item)) for item in self._artifacts()
            ]]
        return [(leaf, self.read_file(path))]

    def grep(self, pattern: str, *, path: str = MOUNT, output_mode: str = "content",
             context: int = 0, offset: int = 0, limit: int = 50) -> str:
        try:
            matcher = re.compile(pattern)
        except re.error as exc:
            return f"grep: invalid regex ({exc})."
        hits, counts = [], {}
        for name, text in self._docs(path):
            for line_no, line in enumerate(text.splitlines(), 1):
                if matcher.search(line):
                    hits.append(f"artifacts/{name}:{line_no}:{line}")
                    counts[name] = counts.get(name, 0) + 1
        if output_mode == "files_with_matches":
            rows = [f"artifacts/{name}" for name in counts]
        elif output_mode == "count":
            rows = [f"artifacts/{name}:{count}" for name, count in counts.items()]
        else:
            rows = hits
        rows = rows[offset:offset + limit]
        return "\n".join(rows) if rows else "grep: no matches found."


__all__ = ["ActiveTurn", "CoreArtifactFS", "LocalTurnStore"]
