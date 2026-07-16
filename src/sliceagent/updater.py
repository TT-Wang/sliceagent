"""Safe, process-boundary updates for installed SliceAgent releases.

The canonical installers create an isolated ``uv tool`` environment.  We only
mutate that environment when its receipt positively identifies the running
interpreter as SliceAgent's tool.  Other install managers receive exact,
manager-owned guidance instead of a guessed mutation.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path


_PACKAGE = "sliceagent"
_CANONICAL_SPEC = "sliceagent[tui]"
_PYPI_SIMPLE = "https://pypi.org/simple"


@dataclass(frozen=True)
class InstallOrigin:
    kind: str
    tool_dir: str = ""
    bin_dir: str = ""


def _normalized(value: object) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _metadata_text(distribution, name: str) -> str:
    try:
        return distribution.read_text(name) or ""  # windows-footgun: ok — importlib metadata API, not pathlib
    except Exception:  # noqa: BLE001 — incomplete metadata must fail closed to guidance
        return ""


def _direct_origin(distribution) -> str:
    raw = _metadata_text(distribution, "direct_url.json")
    if not raw:
        return ""
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return "direct"
    info = value.get("dir_info") if isinstance(value, dict) else None
    return "editable" if isinstance(info, dict) and info.get("editable") is True else "direct"


def _public_pypi_options(options: dict) -> bool:
    """Recognize both legacy and current uv receipt encodings for public PyPI."""
    canonical_keys = {"index", "default-index", "default_index", "index-url", "index_url"}
    if any(key not in canonical_keys for key in options):
        return False

    legacy_urls = [
        str(options[key]).strip().rstrip("/").lower()
        for key in ("default-index", "default_index", "index-url", "index_url")
        if options.get(key)
    ]
    if any(url != _PYPI_SIMPLE for url in legacy_urls):
        return False

    indexes = options.get("index")
    if indexes is None:
        return bool(legacy_urls)
    if not isinstance(indexes, list) or len(indexes) != 1:
        return False
    index = indexes[0]
    if not isinstance(index, dict):
        return False
    url = str(index.get("url") or "").strip().rstrip("/").lower()
    return (
        url == _PYPI_SIMPLE
        and index.get("default") is True
        and index.get("explicit") in (None, False)
    )


def _uv_receipt(prefix: str) -> tuple[str, str, str] | None:
    """Return ``(kind, tool_dir, bin_dir)`` for a SliceAgent uv-tool receipt."""
    receipt = Path(prefix) / "uv-receipt.toml"
    try:
        if receipt.stat().st_size > 1_000_000:
            return None
        value = tomllib.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return None
    tool = value.get("tool") if isinstance(value, dict) else None
    if not isinstance(tool, dict):
        return None
    # uv 0.11 records the interpreter request as ``python = "3.12"`` alongside the
    # requirements and entrypoints, including for the exact canonical installer command.
    # It is install metadata, not an alternate package source or resolver override.  Keep
    # failing closed on every unknown truthy field, while accepting only the real scalar
    # shape for this one standard field.
    python = tool.get("python")
    custom_tool_fields = (
        any(
            key not in {"requirements", "entrypoints", "options", "python"} and bool(item)
            for key, item in tool.items()
        )
        or ("python" in tool and (not isinstance(python, str) or not python.strip()))
    )
    requirements = tool.get("requirements")
    entrypoints = tool.get("entrypoints")
    if not isinstance(requirements, list):
        return None
    requirement = next(
        (item for item in requirements if isinstance(item, dict)
         and _normalized(item.get("name")) == _PACKAGE),
        None,
    )
    if requirement is None:
        return None
    extras = requirement.get("extras") or []
    custom_requirements = (
        custom_tool_fields
        or len(requirements) != 1
        or any(key not in {"name", "extras", "specifier"} for key in requirement)
        or not isinstance(extras, list)
        or sorted(_normalized(extra) for extra in extras) != ["tui"]
    )
    if not isinstance(entrypoints, list):
        return None
    entrypoint = next(
        (item for item in entrypoints if isinstance(item, dict)
         and _normalized(item.get("name")) == _PACKAGE
         and (_normalized(item.get("from")) in ("", _PACKAGE))),
        None,
    )
    if entrypoint is None:
        return None
    install_path = str(entrypoint.get("install-path") or entrypoint.get("install_path") or "")
    if not install_path or not os.path.isabs(install_path):
        custom_requirements = True
        install_path = ""
    # Preserve the public shim's lexical parent. Resolving the symlink first points inside the tool venv
    # and would move UV_TOOL_BIN_DIR away from ~/.local/bin during replacement.
    bin_dir = os.path.dirname(os.path.normpath(install_path)) if install_path else ""
    options = tool.get("options") or {}
    if not isinstance(options, dict):
        options = {}
        custom_requirements = True
    canonical_source = _public_pypi_options(options)
    kind = "uv-tool" if canonical_source and not custom_requirements else "uv-tool-custom"
    return kind, os.path.dirname(os.path.realpath(prefix)), bin_dir


def detect_installation(*, prefix: str | None = None, distribution=None) -> InstallOrigin:
    """Classify the running distribution without consulting the workspace or ``PATH``."""
    prefix = os.path.realpath(prefix or sys.prefix)
    if distribution is None:
        try:
            distribution = importlib.metadata.distribution(_PACKAGE)
        except importlib.metadata.PackageNotFoundError:
            return InstallOrigin("unknown")

    direct = _direct_origin(distribution)
    if direct:
        return InstallOrigin(direct)

    receipt = _uv_receipt(prefix)
    if receipt is not None:
        return InstallOrigin(receipt[0], tool_dir=receipt[1], bin_dir=receipt[2])

    if (Path(prefix) / "pipx_metadata.json").is_file():
        return InstallOrigin("pipx")

    installer = _normalized(_metadata_text(distribution, "INSTALLER"))
    if installer == "pip":
        return InstallOrigin("pip")
    if installer == "uv":
        return InstallOrigin("uv-environment")
    return InstallOrigin("unknown")


def _find_uv(origin: InstallOrigin, which) -> str:
    """Use only known package-manager bins; never execute a repository or arbitrary PATH shim."""
    cwd = Path(os.path.realpath(os.getcwd()))
    repo_root = cwd
    for parent in (cwd, *cwd.parents):
        if (parent / ".git").exists():
            repo_root = parent
            break
    fixed_locations = (
        (os.path.expanduser("~/.local/bin"), os.path.expanduser("~/.local")),
        (os.path.expanduser("~/.cargo/bin"), os.path.expanduser("~/.cargo")),
        # A Nix profile generation links executables into sibling package paths under /nix/store; the
        # generation directory itself is not their ancestor.
        (os.path.expanduser("~/.nix-profile/bin"), "/nix/store"),
        ("/opt/homebrew/bin", "/opt/homebrew"),
        ("/opt/local/bin", "/opt/local"),
        ("/usr/local/bin", "/usr/local"),
        ("/usr/bin", "/usr"),
        ("/bin", "/usr"),
        ("/run/current-system/sw/bin", "/nix/store"),
    )
    trusted_dirs = {os.path.realpath(path) for path, _ in fixed_locations}
    trusted_symlink_roots: dict[str, set[str]] = {}
    for path, root in fixed_locations:
        trusted_symlink_roots.setdefault(os.path.realpath(path), set()).add(os.path.realpath(root))
    if origin.bin_dir:
        origin_bin = os.path.realpath(origin.bin_dir)
        try:
            origin_in_repo = os.path.commonpath((origin_bin, str(repo_root))) == str(repo_root)
        except ValueError:
            origin_in_repo = False
        if not origin_in_repo:
            trusted_dirs.add(origin_bin)

    def accepted(raw: str) -> str:
        lexical = os.path.abspath(raw)
        candidate = os.path.realpath(raw)
        lexical_parent, real_parent = os.path.dirname(lexical), os.path.dirname(candidate)
        trusted_parent = os.path.realpath(lexical_parent)
        try:
            target_in_repo = os.path.commonpath((candidate, str(repo_root))) == str(repo_root)
        except ValueError:
            target_in_repo = False
        if target_in_repo:
            return ""
        if real_parent in trusted_dirs:
            return candidate
        if trusted_parent not in trusted_dirs:
            return ""
        for root in trusted_symlink_roots.get(trusted_parent, ()):
            try:
                if os.path.commonpath((candidate, root)) == root:
                    return candidate
            except ValueError:
                continue
        return ""

    for filename in ("uv", "uv.exe"):
        sibling = os.path.join(origin.bin_dir, filename) if origin.bin_dir else ""
        if sibling and os.path.isfile(sibling) and os.access(sibling, os.X_OK):
            candidate = accepted(sibling)
            if candidate:
                return candidate
    candidate = which("uv")
    if not candidate:
        return ""
    return accepted(candidate)


def _display_command(args: list[str], os_name: str) -> str:
    return subprocess.list2cmdline(args) if os_name == "nt" else shlex.join(args)


def _manual_guidance(origin: InstallOrigin, executable: str, out, os_name: str) -> int:
    if origin.kind == "editable":
        out("  This is an editable source checkout; it was not replaced with a PyPI build.")
        out("  Update the checkout, then refresh its environment:")
        out("    git pull --ff-only")
        out("    uv sync --all-extras    # or: python -m pip install -e '.[tui]'")
    elif origin.kind == "direct":
        out("  This install came from a local, URL, or VCS artifact; its source was preserved.")
        out("  Update it with the same source command you originally used.")
    elif origin.kind == "pipx":
        out("  This install is owned by pipx. Exit SliceAgent, then run:")
        out("    pipx upgrade sliceagent")
    elif origin.kind == "pip":
        out("  This install is owned by pip. Exit SliceAgent, then run:")
        out("    " + _display_command(
            [executable, "-m", "pip", "install", "--upgrade", _CANONICAL_SPEC], os_name,
        ))
    elif origin.kind == "uv-tool-custom":
        out("  This uv-tool receipt does not prove the canonical requirements and public-PyPI source;")
        out("  the environment was preserved instead of guessing at its package source.")
        out("  Exit and update it with the same source settings you originally used.")
        out("  For an unpinned public-PyPI install:  uv tool upgrade sliceagent")
    elif origin.kind == "uv-environment":
        out("  This package is in a uv-managed Python environment, not an isolated uv tool.")
        out("  Exit SliceAgent and update that exact interpreter with the original source settings:")
        out("    " + _display_command(
            ["uv", "pip", "install", "--python", executable, "--upgrade", _CANONICAL_SPEC], os_name,
        ))
    else:
        out("  Could not identify which package manager owns this installation; nothing was changed.")
        out("  Re-run the one-line installer from the README in a clean shell.")
    return 1


def run_update(
    *,
    prefix: str | None = None,
    executable: str | None = None,
    distribution=None,
    environ: dict | None = None,
    which=shutil.which,
    runner=subprocess.run,
    out=print,
    os_name: str | None = None,
    python_version: tuple[int, int] | None = None,
    neutral_cwd: str | None = None,
) -> int:
    """Update a canonical install and return a shell-correct exit code.

    This function intentionally does not hot-reload or restart the current process.
    Its seams are injectable so the updater is fully testable without network access.
    """
    prefix = prefix or sys.prefix
    executable = executable or sys.executable
    platform_name = os_name or os.name
    try:
        from . import __version__
        out(f"  SliceAgent {__version__}")
    except Exception:  # noqa: BLE001 — version display cannot block maintenance guidance
        pass
    origin = detect_installation(prefix=prefix, distribution=distribution)
    if origin.kind != "uv-tool":
        return _manual_guidance(origin, executable, out, platform_name)

    # Windows copies console-script executables instead of symlinking them. Replacing the active shim is
    # not reliable, so keep that platform at an explicit external process boundary.
    if platform_name == "nt":
        out("  Windows keeps the running SliceAgent executable open.")
        out("  Exit, then re-run the one-line PowerShell installer from the README.")
        return 1

    uv = _find_uv(origin, which)
    if not uv:
        out("  A trusted `uv` executable was not found outside the repository; nothing was changed.")
        out("  Install uv from https://docs.astral.sh/uv/, then re-run:  sliceagent update")
        return 1

    major, minor = python_version or (sys.version_info.major, sys.version_info.minor)
    command = [
        uv, "tool", "install", "--force", "--upgrade",
        "--python", f"{major}.{minor}", "--no-config", "--default-index", _PYPI_SIMPLE,
        _CANONICAL_SPEC,
    ]
    child_env = dict(os.environ if environ is None else environ)
    for key in tuple(child_env):
        upper = key.upper()
        if (
            upper.startswith(("UV_", "PIP_"))
            or upper in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "AWS_SECRET_ACCESS_KEY"}
            or upper.endswith(("_API_KEY", "_TOKEN"))
        ):
            child_env.pop(key, None)
    child_env["UV_TOOL_DIR"] = origin.tool_dir
    if origin.bin_dir:
        child_env["UV_TOOL_BIN_DIR"] = origin.bin_dir
    cwd = neutral_cwd or tempfile.gettempdir()

    out("  Updating SliceAgent from the latest stable PyPI release…")
    try:
        result = runner(command, cwd=cwd, env=child_env, check=False)
    except OSError as exc:
        out(f"  Update could not start: {type(exc).__name__}: {exc}")
        return 1
    code = int(getattr(result, "returncode", 1) or 0)
    if code != 0:
        out(f"  Update failed (uv exited {code}). Nothing is claimed as updated.")
        out("  Fix the reported uv error, then re-run:  sliceagent update")
        return code
    out("  ✓ Latest stable release installed. Restart SliceAgent to run the new version.")
    return 0
