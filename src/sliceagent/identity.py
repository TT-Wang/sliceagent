"""Stable local identities for project-scoped context and knowledge.

The physical workspace path is recovery identity, not semantic project identity.  Git worktrees share the
same project through their common directory; non-Git workspaces receive a private registry UUID.  The registry
is host-private and never trusts a tracked repository file to choose another project's identity.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

from .private_state import private_dir, private_file


_LOCK = threading.RLock()


@contextmanager
def _registry_lock(path: str):
    """Serialize first-registration across processes as well as threads.

    The registry uses atomic replacement, which prevents torn JSON but cannot by itself prevent two fresh
    processes from both minting different UUIDs from the same empty snapshot. A separate stable lock inode
    closes that identity split; process death releases the OS lock automatically.
    """
    directory = private_dir(os.path.dirname(path))
    lock_path = os.path.join(directory, ".projects.lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    private_file(lock_path)
    locked = False
    try:
        if os.name == "nt":
            try:
                import msvcrt
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                    os.fsync(fd)
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                locked = True
            except (ImportError, OSError):
                pass
        else:
            try:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
                locked = True
            except (ImportError, OSError):
                pass
        yield
    finally:
        if locked:
            try:
                if os.name == "nt":
                    import msvcrt
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
        os.close(fd)


def _registry_path() -> str:
    configured = os.environ.get("SLICEAGENT_PROJECT_REGISTRY", "").strip()
    if configured:
        return os.path.realpath(os.path.expanduser(configured))
    base = os.environ.get("SLICEAGENT_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".sliceagent")
    return os.path.join(os.path.expanduser(base), "registry", "projects.json")


def _git_common_dir(root: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", root, "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return os.path.realpath(proc.stdout.strip())


def _locator(root: str) -> tuple[str, str]:
    common = _git_common_dir(root)
    target = common or os.path.realpath(root)
    try:
        stat = os.stat(target)
        inode = f"{int(stat.st_dev)}:{int(stat.st_ino)}"
    except OSError:
        inode = ""
    kind = "git" if common else "workspace"
    return f"{kind}:{inode or target}", target


def _load(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _write(path: str, value: dict) -> None:
    directory = os.path.dirname(path)
    private_dir(directory)
    fd, temp = tempfile.mkstemp(prefix=".projects-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temp, path)
        private_file(path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class ProjectIdentity:
    project_id: str
    label: str
    workspace_root: str
    locator: str
    git_common_dir: str | None = None


def resolve_project_identity(workspace_root: str) -> ProjectIdentity:
    """Return or create the private logical identity for one workspace.

    Device/inode identity keeps an ordinary moved checkout linked on the same filesystem.  The canonical path
    remains as an alias for platforms/filesystems without stable inode information.
    """
    root = os.path.realpath(workspace_root)
    locator, target = _locator(root)
    common = _git_common_dir(root)
    path = _registry_path()
    with _LOCK, _registry_lock(path):
        data = _load(path)
        projects = data.get("projects") if isinstance(data.get("projects"), dict) else {}
        aliases = data.get("aliases") if isinstance(data.get("aliases"), dict) else {}
        project_id = aliases.get(locator) or aliases.get(target) or aliases.get(root)
        if not isinstance(project_id, str) or not project_id:
            project_id = "project-" + uuid.uuid4().hex
        label = os.path.basename(os.path.dirname(common)) if common and os.path.basename(common) == ".git" \
            else os.path.basename(root)
        previous = projects.get(project_id) if isinstance(projects.get(project_id), dict) else {}
        paths = list(dict.fromkeys([*(previous.get("paths") or []), root]))
        projects[project_id] = {
            "label": label or "project",
            "locator": locator,
            "git_common_dir": common or "",
            "paths": paths,
        }
        aliases.update({locator: project_id, target: project_id, root: project_id})
        _write(path, {"version": 1, "projects": projects, "aliases": aliases})
    return ProjectIdentity(
        project_id=project_id, label=label or "project", workspace_root=root,
        locator=locator, git_common_dir=common,
    )


__all__ = ["ProjectIdentity", "resolve_project_identity"]
