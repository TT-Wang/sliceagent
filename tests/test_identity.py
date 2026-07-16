import json
import os
import subprocess
import sys

from sliceagent.identity import resolve_project_identity


def test_non_git_workspace_identity_is_stable(tmp_path, monkeypatch):
    registry = tmp_path / "state" / "projects.json"
    monkeypatch.setenv("SLICEAGENT_PROJECT_REGISTRY", str(registry))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first = resolve_project_identity(str(workspace))
    second = resolve_project_identity(str(workspace))
    assert first.project_id == second.project_id
    assert first.label == "workspace"
    assert json.loads(registry.read_text())["version"] == 1


def test_git_worktrees_share_project_identity(tmp_path, monkeypatch):
    registry = tmp_path / "state" / "projects.json"
    monkeypatch.setenv("SLICEAGENT_PROJECT_REGISTRY", str(registry))
    repo = tmp_path / "repo"
    worktree = tmp_path / "worktree"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "README.md").write_text("test\n")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q", "-b", "feature", str(worktree)], check=True)
    try:
        assert resolve_project_identity(str(repo)).project_id == resolve_project_identity(str(worktree)).project_id
    finally:
        subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(worktree)], check=False)


def test_concurrent_first_registration_cannot_split_project_identity(tmp_path):
    registry = tmp_path / "state" / "projects.json"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source_root = os.path.join(os.path.dirname(__file__), "..", "src")
    code = (
        "from sliceagent.identity import resolve_project_identity; "
        f"print(resolve_project_identity({str(workspace)!r}).project_id)"
    )
    env = {
        **os.environ,
        "SLICEAGENT_PROJECT_REGISTRY": str(registry),
        "PYTHONPATH": source_root + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env,
        )
        for _ in range(8)
    ]
    results = [process.communicate(timeout=15) for process in processes]
    assert all(process.returncode == 0 for process in processes), results
    identities = {stdout.strip() for stdout, _stderr in results}
    assert len(identities) == 1, results
