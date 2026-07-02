"""End-to-end launch smoke test: `sliceagent` must reach the input prompt WITHOUT crashing.

Regression guard for the cli.py banner `NameError: name 'policy_mode' is not defined` (the var was
renamed to the resolved `_eff_mode`/`canonical` in the 3-modes change but the banner still referenced the
old name). No prior test exercised `main()` end-to-end, so the crash shipped. Runs main() in a subprocess
with a fake key + AGENT_TUI=off + EOF stdin, so it boots, prints the banner, and exits on EOF — no network.
"""
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _launch(extra=None):
    import tempfile
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": os.path.join(_ROOT, "src"),
        "HOME": tempfile.mkdtemp(prefix="smoke-home-"),   # hermetic: the dev machine's ~/.sliceagent
                                                          # config must not leak in (CI has none either)
        "AGENT_TUI": "off",                 # plain REPL — no prompt_toolkit app to drive
        "LLM_API_KEY": "sk-dummy-smoke", "OPENAI_API_KEY": "sk-dummy-smoke",
        "AGENT_MODEL": "dummy-model-smoke", # required since the no-default-model gate — without it the
                                            # CLI exits at the gate before the banner this test asserts on
        "AGENT_PROXY": "off",               # don't route through a local proxy that isn't there
    })
    env.update(extra or {})
    return subprocess.run(
        [sys.executable, "-c", "from sliceagent.cli import main; main()"],
        cwd=_ROOT, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=120,
    )


def main_reaches_prompt_without_crashing():
    r = _launch()
    out = r.stdout + r.stderr
    assert "Traceback" not in out and "NameError" not in out, out[-2000:]
    assert "policy=" in out, "startup banner never rendered (the line that held the NameError):\n" + out[-1500:]
    assert r.returncode == 0, f"nonzero exit {r.returncode}\n{out[-1500:]}"


def _no_key_env():
    """A truly blank first-run env: temp HOME, every key/model var stripped."""
    import tempfile
    env = {k: v for k, v in os.environ.items()
           if k not in ("LLM_API_KEY", "OPENAI_API_KEY", "MOONSHOT_API_KEY", "DEEPSEEK_API_KEY",
                        "AGENT_MODEL", "LLM_BASE_URL", "AGENT_PROVIDER")}
    env.update({"PYTHONPATH": os.path.join(_ROOT, "src"),
                "HOME": tempfile.mkdtemp(prefix="firstrun-home-"),
                "AGENT_TUI": "off", "AGENT_PROXY": "off"})
    return env


def piped_no_key_run_keeps_the_gate_no_prompt():
    """Non-interactive (stdin=pipe) + no key must print the gate and exit 1 — never start the wizard
    (a prompt into a pipe would hang CI/scripts)."""
    r = subprocess.run([sys.executable, "-c", "from sliceagent.cli import main; main()"],
                       cwd=_ROOT, env=_no_key_env(), stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=60)
    out = r.stdout + r.stderr
    assert "No API key found" in out, out[-800:]
    assert "guided setup.\n\n" not in out and "sliceagent setup" not in out, \
        "wizard must not auto-start without a tty:\n" + out[-800:]
    assert r.returncode == 1, f"expected gate exit 1, got {r.returncode}"


def interactive_first_run_auto_starts_the_wizard():
    """First-run UX: a bare interactive `sliceagent` (tty, nothing configured) drops straight into the
    init wizard — proven on a REAL pty by the wizard header + provider menu appearing unprompted.
    (The abort path is covered in-process below; macOS getpass on a detached pty is not reliably
    drivable, and the wizard's own logic has an injectable seam for exactly that reason.)"""
    import pty
    import select
    import signal
    import time
    try:
        m, s = pty.openpty()
    except OSError:
        return   # no pty on this host — skip
    p = subprocess.Popen([sys.executable, "-c", "from sliceagent.cli import main; main()"],
                         cwd=_ROOT, env=_no_key_env(), stdin=s, stdout=s, stderr=s,
                         start_new_session=True)
    os.close(s)
    buf = bytearray()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and b"Choose a provider" not in buf:
        r, _, _ = select.select([m], [], [], 0.2)
        if r:
            try:
                buf.extend(os.read(m, 4096))
            except OSError:
                break
        if p.poll() is not None:
            break
    try:
        os.killpg(p.pid, signal.SIGKILL)   # wizard reached (or not) — tear the child down either way
    except (ProcessLookupError, PermissionError):
        pass
    p.wait()
    try:
        os.close(m)
    except OSError:
        pass
    out = buf.decode(errors="replace")
    assert "starting guided setup" in out, "wizard did not auto-start on an interactive first run:\n" + out[-1200:]
    assert "sliceagent setup" in out and "Choose a provider" in out, "wizard header/menu missing:\n" + out[-1200:]


def aborted_wizard_falls_back_to_the_gate():
    """If the auto-started wizard is aborted (returns nonzero), main() must fall back to the plain
    gate message and exit 1 — never proceed keyless. In-process with the wizard mocked, tty faked."""
    import contextlib
    import io
    import tempfile
    from types import SimpleNamespace
    from unittest import mock

    from sliceagent import cli as cli_mod
    from sliceagent import onboarding as ob

    env_patch = {k: "" for k in ("LLM_API_KEY", "OPENAI_API_KEY", "MOONSHOT_API_KEY", "AGENT_MODEL")}
    env_patch["HOME"] = tempfile.mkdtemp(prefix="firstrun-abort-")
    out = io.StringIO()
    # one fake object doubling as tty-stdin (isatty only) and capturing tty-stdout: redirect_stdout
    # would swap in a StringIO whose isatty() is False and silently close the wizard path.
    fake_tty = SimpleNamespace(isatty=lambda: True, write=out.write, flush=lambda: None)
    called = {"n": 0}

    def _fake_init():
        called["n"] += 1
        return 1                                   # user aborted the wizard

    _ = contextlib  # (kept import shape stable)
    with mock.patch.dict(os.environ, env_patch), \
         mock.patch.object(ob, "run_init", _fake_init), \
         mock.patch.object(sys, "argv", ["sliceagent"]), \
         mock.patch.object(sys, "stdin", fake_tty), \
         mock.patch.object(sys, "stdout", fake_tty):
        try:
            cli_mod.main()
            raise AssertionError("main() must exit after an aborted wizard")
        except SystemExit as e:
            assert e.code == 1, f"expected exit 1, got {e.code}"
    text = out.getvalue()
    assert called["n"] == 1, "the wizard was never invoked"
    assert "No API key found" in text, "aborted wizard must fall back to the gate message:\n" + text


if __name__ == "__main__":
    main_reaches_prompt_without_crashing()
    print("PASS main_reaches_prompt_without_crashing")
    piped_no_key_run_keeps_the_gate_no_prompt()
    print("PASS piped_no_key_run_keeps_the_gate_no_prompt")
    interactive_first_run_auto_starts_the_wizard()
    print("PASS interactive_first_run_auto_starts_the_wizard")
    aborted_wizard_falls_back_to_the_gate()
    print("PASS aborted_wizard_falls_back_to_the_gate")
