"""Offline tests for the provenance-aware, process-boundary updater.

No package manager or network operation is executed.
Run: PYTHONPATH=src python tests/test_updater.py
"""
import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.updater import detect_installation, run_update  # noqa: E402


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


class _Dist:
    def __init__(self, *, installer="uv", direct=None):
        self._values = {"INSTALLER": installer, "direct_url.json": direct}

    def read_text(self, name):
        return self._values.get(name)


def _uv_tool(*, specifier="==0.1.0", extra_requirement=False, requirement_fields="",
             tool_lines='python = "3.12"',
             index_url="https://pypi.org/simple/", symlink_entrypoint=False, option_lines=""):
    root = tempfile.mkdtemp(prefix="updater-uv-")
    tool_dir = os.path.join(root, "custom-tools")
    prefix = os.path.join(tool_dir, "sliceagent")
    bin_dir = os.path.join(root, "custom-bin")
    os.makedirs(prefix)
    os.makedirs(bin_dir)
    requirements = [
        f'{{ name = "sliceagent", extras = ["tui"], specifier = "{specifier}"{requirement_fields} }}',
    ]
    if extra_requirement:
        requirements.append('{ name = "custom-addon" }')
    install_path = os.path.join(bin_dir, "sliceagent.exe" if os.name == "nt" else "sliceagent")
    if symlink_entrypoint:
        target_dir = os.path.join(prefix, "bin")
        os.makedirs(target_dir)
        target = os.path.join(target_dir, os.path.basename(install_path))
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("#!/bin/sh\n")
        os.symlink(target, install_path)
    receipt = (
        "[tool]\n"
        f"requirements = [{', '.join(requirements)}]\n"
        "entrypoints = [\n"
        f'  {{ name = "sliceagent", install-path = {json.dumps(install_path)}, from = "sliceagent" }},\n'
        "]\n"
        f"{tool_lines}\n"
        "[tool.options]\n"
        f"{('index-url = ' + json.dumps(index_url) + chr(10)) if index_url is not None else ''}"
        f"{option_lines}"
    )
    with open(os.path.join(prefix, "uv-receipt.toml"), "w", encoding="utf-8") as fh:
        fh.write(receipt)
    return prefix, tool_dir, bin_dir


@check
def canonical_uv_tool_is_identified_from_the_running_prefix():
    prefix, tool_dir, bin_dir = _uv_tool()
    origin = detect_installation(prefix=prefix, distribution=_Dist())
    assert origin.kind == "uv-tool"
    assert origin.tool_dir == os.path.realpath(tool_dir)
    assert origin.bin_dir == os.path.abspath(bin_dir)


@check
def public_entrypoint_symlink_keeps_its_lexical_bin_directory():
    if os.name == "nt":
        return
    prefix, _, bin_dir = _uv_tool(symlink_entrypoint=True)
    origin = detect_installation(prefix=prefix, distribution=_Dist())
    assert origin.kind == "uv-tool"
    assert origin.bin_dir == os.path.abspath(bin_dir), origin
    assert origin.bin_dir != os.path.join(prefix, "bin"), "do not resolve the public shim into the tool venv"


@check
def updater_replaces_an_old_exact_pin_via_canonical_install_command():
    prefix, tool_dir, bin_dir = _uv_tool(specifier="==0.1.0")
    calls, messages = [], []
    uv_path = os.path.join(os.path.expanduser("~"), ".local", "bin", "uv.exe" if os.name == "nt" else "uv")

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0)

    rc = run_update(
        prefix=prefix,
        executable="/tool/python",
        distribution=_Dist(),
        environ={
            "PATH": "/bin", "LLM_API_KEY": "secret", "OPENAI_API_KEY": "secret",
            "UV_INDEX": "https://repo-controlled.invalid/simple", "PYTHONPATH": "/untrusted/repo",
            "GITHUB_TOKEN": "secret", "UV_OVERRIDE": "/repo/override.txt", "UV_PRERELEASE": "allow",
            "UV_EXCLUDE_NEWER": "2020-01-01", "PIP_CONFIG_FILE": "/repo/pip.conf",
        },
        which=lambda name: uv_path if name == "uv" else None,
        runner=runner,
        out=messages.append,
        os_name="posix",
        python_version=(3, 12),
        neutral_cwd="/neutral",
    )
    assert rc == 0 and len(calls) == 1
    argv, kwargs = calls[0]
    assert argv == [
        os.path.realpath(uv_path), "tool", "install", "--force", "--upgrade",
        "--python", "3.12", "--no-config", "--default-index", "https://pypi.org/simple",
        "sliceagent[tui]",
    ], argv
    assert kwargs["cwd"] == "/neutral" and kwargs["check"] is False
    assert kwargs.get("shell") is not True, "the updater must never invoke a shell"
    assert kwargs["env"]["UV_TOOL_DIR"] == os.path.realpath(tool_dir)
    assert kwargs["env"]["UV_TOOL_BIN_DIR"] == os.path.abspath(bin_dir)
    assert "LLM_API_KEY" not in kwargs["env"] and "OPENAI_API_KEY" not in kwargs["env"]
    assert "UV_INDEX" not in kwargs["env"], "repo/custom index env must not redirect the canonical update"
    assert "PYTHONPATH" not in kwargs["env"] and "GITHUB_TOKEN" not in kwargs["env"]
    assert not any(key.startswith(("UV_", "PIP_")) for key in kwargs["env"]
                   if key not in ("UV_TOOL_DIR", "UV_TOOL_BIN_DIR"))
    assert any("Restart SliceAgent" in line for line in messages)


@check
def manager_failure_propagates_and_never_claims_success():
    prefix, _, _ = _uv_tool()
    messages = []
    rc = run_update(
        prefix=prefix, distribution=_Dist(), which=lambda _: "/bin/uv",
        runner=lambda *a, **k: SimpleNamespace(returncode=7),
        out=messages.append, os_name="posix", neutral_cwd="/neutral",
    )
    assert rc == 7
    assert any("failed" in line.lower() for line in messages)
    assert not any("Latest stable release installed" in line for line in messages)


@check
def editable_and_direct_installs_are_never_replaced_with_pypi():
    for direct, kind in (
        (json.dumps({"url": "file:///src", "dir_info": {"editable": True}}), "editable"),
        (json.dumps({"url": "git+https://example.invalid/repo"}), "direct"),
        ("{malformed", "direct"),
    ):
        origin = detect_installation(prefix=tempfile.mkdtemp(), distribution=_Dist(direct=direct))
        assert origin.kind == kind
        calls, messages = [], []
        rc = run_update(
            prefix=tempfile.mkdtemp(), distribution=_Dist(direct=direct),
            runner=lambda *a, **k: calls.append(a), out=messages.append,
        )
        assert rc == 1 and not calls
        assert any(("checkout" in line or "source" in line) for line in messages)


@check
def alternative_managers_receive_exact_guidance_without_mutation():
    cases = []
    pipx_prefix = tempfile.mkdtemp(prefix="updater-pipx-")
    open(os.path.join(pipx_prefix, "pipx_metadata.json"), "w").close()
    cases.append((pipx_prefix, _Dist(installer="pip"), "pipx upgrade sliceagent"))
    cases.append((tempfile.mkdtemp(), _Dist(installer="pip"), "-m pip install --upgrade"))
    cases.append((tempfile.mkdtemp(), _Dist(installer="uv"), "uv pip install --python"))
    for prefix, dist, expected in cases:
        calls, messages = [], []
        rc = run_update(
            prefix=prefix, executable="/owned/python", distribution=dist,
            runner=lambda *a, **k: calls.append(a), out=messages.append,
        )
        assert rc == 1 and not calls
        assert expected in "\n".join(messages), messages


@check
def ambiguous_or_custom_uv_receipts_fail_closed():
    prefix, _, _ = _uv_tool(extra_requirement=True)
    calls, messages = [], []
    rc = run_update(
        prefix=prefix, distribution=_Dist(installer="uv"),
        runner=lambda *a, **k: calls.append(a), out=messages.append,
    )
    assert rc == 1 and not calls
    assert any("does not prove" in line for line in messages)

    prefix, _, _ = _uv_tool(requirement_fields=', directory = "/untrusted/source"')
    origin = detect_installation(prefix=prefix, distribution=_Dist())
    assert origin.kind == "uv-tool-custom", "receipt source fields must fail closed without direct_url.json"

    prefix, _, _ = _uv_tool(tool_lines='constraints = [{ name = "openai", specifier = "<3" }]')
    origin = detect_installation(prefix=prefix, distribution=_Dist())
    assert origin.kind == "uv-tool-custom", "tool-level resolver constraints must fail closed"


@check
def private_index_uv_tool_is_not_silently_switched_to_public_pypi():
    prefix, _, _ = _uv_tool(index_url="https://packages.example.invalid/simple")
    calls, messages = [], []
    rc = run_update(
        prefix=prefix, distribution=_Dist(),
        runner=lambda *a, **k: calls.append(a), out=messages.append, os_name="posix",
    )
    assert rc == 1 and not calls
    assert any("does not prove" in line for line in messages)

    prefix, _, _ = _uv_tool(option_lines='index = ["https://packages.example.invalid/simple"]\n')
    origin = detect_installation(prefix=prefix, distribution=_Dist())
    assert origin.kind == "uv-tool-custom", origin

    prefix, _, _ = _uv_tool(index_url=None)
    origin = detect_installation(prefix=prefix, distribution=_Dist())
    assert origin.kind == "uv-tool-custom", "missing provenance is not positive public-PyPI proof"


@check
def current_uv_structured_public_index_receipt_is_recognized_exactly():
    public = (
        'index = [{ url = "https://pypi.org/simple", explicit = false, default = true, '
        'format = "simple", authenticate = "auto" }]\n'
    )
    prefix, _, _ = _uv_tool(index_url=None, option_lines=public, tool_lines="")
    assert detect_installation(prefix=prefix, distribution=_Dist()).kind == "uv-tool"

    # uv 0.11.26 emits this ordinary scalar on every canonical receipt. It identifies
    # the tool interpreter and must not be mistaken for a custom package source.
    prefix, _, _ = _uv_tool(index_url=None, option_lines=public, tool_lines='python = "3.12"')
    assert detect_installation(prefix=prefix, distribution=_Dist()).kind == "uv-tool"

    prefix, _, _ = _uv_tool(index_url=None, option_lines=public, tool_lines="python = { custom = true }")
    assert detect_installation(prefix=prefix, distribution=_Dist()).kind == "uv-tool-custom"

    custom_variants = (
        'index = [{ url = "https://packages.example.invalid/simple", explicit = false, default = true }]\n',
        'index = [{ url = "https://pypi.org/simple", explicit = true, default = true }]\n',
        'index = [{ url = "https://pypi.org/simple", explicit = false, default = false }]\n',
        'index = [{ url = "https://pypi.org/simple", explicit = false, default = true }]\n'
        'constraint = ["/untrusted/constraints.txt"]\n',
        'index = [{ url = "https://pypi.org/simple", explicit = false, default = true }, '
        '{ url = "https://packages.example.invalid/simple", explicit = false, default = false }]\n',
    )
    for options in custom_variants:
        prefix, _, _ = _uv_tool(index_url=None, option_lines=options)
        origin = detect_installation(prefix=prefix, distribution=_Dist())
        assert origin.kind == "uv-tool-custom", (options, origin)


@check
def pip_guidance_quotes_an_interpreter_path_with_spaces():
    messages = []
    rc = run_update(
        prefix=tempfile.mkdtemp(), executable="/Applications/Python Tools/bin/python",
        distribution=_Dist(installer="pip"), out=messages.append, os_name="posix",
    )
    assert rc == 1
    assert "'/Applications/Python Tools/bin/python' -m pip" in "\n".join(messages)


@check
def missing_uv_and_windows_keep_the_update_external():
    for os_name, which, expected in (
        ("posix", lambda _: None, "trusted `uv` executable was not found"),
        ("nt", lambda _: "C:/uv.exe", "one-line PowerShell installer"),
    ):
        prefix, _, _ = _uv_tool()
        calls, messages = [], []
        rc = run_update(
            prefix=prefix, distribution=_Dist(), which=which,
            runner=lambda *a, **k: calls.append(a), out=messages.append, os_name=os_name,
        )
        assert rc == 1 and not calls
        assert expected in "\n".join(messages)


@check
def repo_local_uv_on_path_is_not_executed():
    prefix, _, tool_bin = _uv_tool()
    repo = tempfile.mkdtemp(prefix="updater-untrusted-repo-")
    os.makedirs(os.path.join(repo, ".git"))
    subdir = os.path.join(repo, "nested", "work")
    os.makedirs(subdir)
    local_bin = os.path.join(repo, ".venv", "bin")
    os.makedirs(local_bin)
    local_uv = os.path.join(local_bin, "uv")
    with open(local_uv, "w", encoding="utf-8") as fh:
        fh.write("#!/bin/sh\nexit 99\n")
    os.chmod(local_uv, 0o755)
    if os.name != "nt":
        os.symlink(local_uv, os.path.join(tool_bin, "uv"))  # trusted shim path, hostile repo target
    calls, messages, old_cwd = [], [], os.getcwd()
    try:
        os.chdir(subdir)
        rc = run_update(
            prefix=prefix, distribution=_Dist(), which=lambda _: local_uv,
            runner=lambda *a, **k: calls.append(a), out=messages.append, os_name="posix",
        )
    finally:
        os.chdir(old_cwd)
    assert rc == 1 and not calls
    assert any("trusted `uv` executable" in line for line in messages)

    evil_dir = tempfile.mkdtemp(prefix="updater-arbitrary-path-")
    evil_uv = os.path.join(evil_dir, "uv")
    with open(evil_uv, "w", encoding="utf-8") as fh:
        fh.write("#!/bin/sh\nexit 98\n")
    os.chmod(evil_uv, 0o755)
    calls, messages = [], []
    rc = run_update(
        prefix=prefix, distribution=_Dist(), which=lambda _: evil_uv,
        runner=lambda *a, **k: calls.append(a), out=messages.append, os_name="posix",
    )
    assert rc == 1 and not calls


@check
def installer_reruns_sanitize_resolution_environment():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "install.sh"), encoding="utf-8") as fh:
        posix = fh.read()
    with open(os.path.join(root, "install.ps1"), encoding="utf-8") as fh:
        windows = fh.read()
    assert "run_uv_clean tool install" in posix and "UV_*|PIP_*" in posix
    assert "PATH=\"$HOME/.local/bin:" in posix, "installer must not discover uv from PATH=."
    assert 'Get-Command uv -CommandType Application' in windows
    assert '$Name -like "UV_*"' in windows and '$Name -like "PIP_*"' in windows
    assert "Invoke-UvClean -UvArgs" in windows
    assert "finally" in windows and "SetEnvironmentVariable" in windows
    assert "$saved[$_.Name] = $_.Value" in windows and "$saved.GetEnumerator()" in windows
    assert "& $UvExe @UvArgs | Out-Host" in windows
    assert "& {\n$ErrorActionPreference" in windows, "iex must not leak installer variables/functions"


if __name__ == "__main__":
    ok = 0
    for fn in CHECKS:
        try:
            fn()
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
