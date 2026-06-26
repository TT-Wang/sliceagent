"""Skill usage sidecar (item 13) — per-skill last-used + use-count telemetry in ONE JSON
file, so consolidate.py can frequency-weight skills the way it already weights pitfalls and
procedures, and a future curator can prune stale AUTO skills.

PORTED (trimmed) from /tmp/hermes-agent/tools/skill_usage.py:
  - sidecar (`.usage.json`) keyed by skill name, NOT frontmatter — keeps telemetry out of
    user-authored SKILL.md content (frontmatter carries only provenance, which is durable
    intent; usage is mutable observability).
  - atomic write (tempfile + os.replace).
  - best-effort everywhere: a broken sidecar never breaks a skill load.
Dropped vs Hermes: lifecycle states, hub/bundled manifests, archive/restore, file locking
(memagent has no concurrent-process skill writes today — add fcntl only if that changes).

NO-TRANSCRIPT INVARIANT: the sidecar is a durable store; it feeds consolidate's frequency
weight, never the slice.

PUBLIC SIGNATURES (pinned):
    usage_path(skills_dir: str) -> str
    load_usage(skills_dir: str) -> dict[str, dict]
    bump_use(skills_dir: str, name: str) -> None         # loader calls this on skill load
    record(skills_dir: str, name: str) -> dict           # one skill's record (defaults backfilled)
    use_count(skills_dir: str, name: str) -> int         # consolidate frequency weight
    last_used_at(skills_dir: str, name: str) -> str | None
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone

_USAGE_FILE = ".usage.json"
_BUMP_LOCK = threading.Lock()   # the skill tool declares no access → concurrent skill() calls run in parallel
#                                 threads; serialize load→increment→save so an increment isn't lost.


def usage_path(skills_dir: str) -> str:
    return os.path.join(skills_dir, _USAGE_FILE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _empty_record() -> dict:
    return {"use_count": 0, "last_used_at": None}


def load_usage(skills_dir: str) -> dict:
    """Read the whole sidecar map. Returns {} on missing/corrupt (never raises)."""
    path = usage_path(skills_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def _save_usage(skills_dir: str, data: dict) -> None:
    """Atomic write. Best-effort — errors are swallowed (telemetry must never break a load)."""
    path = usage_path(skills_dir)
    try:
        os.makedirs(skills_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=skills_dir, prefix=".usage_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception:
        pass


def bump_use(skills_dir: str, name: str) -> None:
    """Increment use_count and stamp last_used_at for `name`. Called by the SkillManager
    when a skill body is loaded into the slice. Best-effort."""
    if not name:
        return
    try:
        with _BUMP_LOCK:   # atomic read-modify-write so parallel skill() threads don't lose an increment
            data = load_usage(skills_dir)
            rec = data.get(name)
            if not isinstance(rec, dict):
                rec = _empty_record()
            rec["use_count"] = int(rec.get("use_count") or 0) + 1
            rec["last_used_at"] = _now_iso()
            data[name] = rec
            _save_usage(skills_dir, data)
    except Exception:
        pass


def record(skills_dir: str, name: str) -> dict:
    """Return `name`'s record with defaults backfilled (never None)."""
    rec = dict(_empty_record())
    raw = load_usage(skills_dir).get(name)
    if isinstance(raw, dict):
        rec.update({k: raw.get(k, rec[k]) for k in rec})
        for k, v in raw.items():
            rec.setdefault(k, v)
    return rec


def use_count(skills_dir: str, name: str) -> int:
    try:
        return int(record(skills_dir, name).get("use_count") or 0)
    except (TypeError, ValueError):
        return 0


def last_used_at(skills_dir: str, name: str) -> str | None:
    return record(skills_dir, name).get("last_used_at")
