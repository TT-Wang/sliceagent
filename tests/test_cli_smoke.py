"""End-to-end launch smoke test: `sliceagent` must reach the input prompt WITHOUT crashing.

No prior test exercised `main()` end-to-end, so a banner-only crash once shipped. Runs main() in a subprocess
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
    assert "model=dummy-model-smoke" in out, "startup banner never rendered:\n" + out[-1500:]
    assert "policy=" not in out
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
    try:
        import pty
    except ImportError:
        return   # Windows: no pty module — this check is POSIX-only by nature
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


def update_routes_before_env_or_api_key_startup():
    """Updating is a host subcommand: it must not load a repo .env or enter provider/session startup."""
    from unittest import mock

    from sliceagent import cli as cli_mod
    from sliceagent import onboarding as ob

    seen = []
    with mock.patch.object(sys, "argv", ["sliceagent", "update"]), \
         mock.patch.object(ob, "dispatch", side_effect=lambda argv: seen.append(argv) or 23), \
         mock.patch.object(cli_mod, "_load_env", side_effect=AssertionError("repo .env must not load")):
        try:
            cli_mod.main()
            raise AssertionError("the update subcommand must exit through onboarding dispatch")
        except SystemExit as exc:
            assert exc.code == 23
    assert seen == [["update"]]


def chitchat_consumes_stale_continuity_but_remains_exactly_adjacent():
    from sliceagent.cli import _fold_chitchat_continuity
    from sliceagent.pfc import Slice

    state = Slice(); state.reset("task")
    state.continuity.pending_proposal = {"action": {"tool": "edit_file", "args": {"path": "a.py"}}}
    state.continuity.previous_evidence_snapshot = {
        "v": 1, "source_turn_id": "turn-before", "execution_query": {
            "source": "execution_receipt", "family": "delegation", "predicate": "failure_detail",
            "scope": "task",
        },
    }
    _fold_chitchat_continuity(state, "thanks", "You're welcome.")
    assert state.continuity.pending_proposal is None
    assert state.continuity.previous_evidence_snapshot is None
    assert state.conversation[-1] == {
        "user": "thanks", "assistant": "You're welcome.", "artifact_id": "",
    }


def pending_proposal_assents_bypass_chitchat_in_both_entry_paths():
    from sliceagent.cli import _use_chitchat_fast_path
    from sliceagent.discourse import extract_pending_proposal, interpret_turn
    from sliceagent.pfc import Slice

    proposal = extract_pending_proposal(
        "The hunter workspace is `/tmp/hunter`. Could you confirm it?"
    )
    assert proposal and proposal.get("action")
    state = Slice(); state.reset("switch workspace")
    state.continuity.pending_proposal = proposal

    for answer in ("ok", "okay", "sounds good"):
        contract = interpret_turn(answer, (), pending_proposal=proposal).contract
        assert contract.effect_authority == "continuation" and contract.effect_grants
        assert not _use_chitchat_fast_path(answer, state), \
            f"{answer!r} must reach the normal admission path while the proposal is adjacent"

    # The optimization remains intact for the same social phrases when no action is awaiting an answer.
    state.continuity.pending_proposal = None
    for message in ("hi", "thanks", "ok", "okay", "sounds good"):
        assert _use_chitchat_fast_path(message, state), message
    assert not _use_chitchat_fast_path("ok do it", state)


def cost_projection_always_includes_session_token_totals():
    from sliceagent.cli import _cost_lines

    stats = {
        "model": "deepseek-reasoner", "tokens": 12_345, "fresh": 678,
        "saved_cached_tok": 9_876, "cost": 0.1234,
    }
    lines = _cost_lines(stats)
    assert any("12,345 total" in line and "678 fresh" in line
               and "9,876 cached-history saved" in line for line in lines), lines
    assert any("per-turn curve off" in line for line in lines)

    class Metrics:
        @staticmethod
        def summary():
            return {
                "per_turn_fresh": [10, 11], "avg_turn_fresh": 10.5,
                "cache_hit_rate": 0.9, "tool_calls": 3, "output": 7,
                "retries": 0, "overflows": 0,
            }

    detailed = _cost_lines(stats, Metrics())
    assert any("per_turn_fresh=[10, 11]" in line for line in detailed), detailed


def live_failure_restores_inline_bridges_and_streaming_sink():
    from sliceagent.cli import _restore_inline_after_live_failure

    class LLM:
        sink = None
        def set_delta_sink(self, sink):
            self.sink = sink

    class Rich:
        def on_delta(self, *_args):
            pass
        def subagent_notify(self, *_args):
            pass

    llm, rich = LLM(), Rich()
    live_runtime = {"active": True}
    workspace_setter = {"fn": lambda _root: None}
    ask_bridge = {"fn": lambda *_args: "stale-live-answer"}
    subagent_bridge = {"fn": None}
    inline_ask = lambda *_args: "inline-answer"
    _restore_inline_after_live_failure(
        llm=llm, rich_sink=rich, live_runtime=live_runtime,
        workspace_setter=workspace_setter, ask_bridge=ask_bridge,
        ask_user=inline_ask, subagent_bridge=subagent_bridge, interactive=True,
    )
    assert live_runtime == {"active": False}
    assert workspace_setter["fn"] is None
    assert ask_bridge["fn"] is inline_ask
    assert llm.sink == rich.on_delta
    assert subagent_bridge["fn"] == rich.subagent_notify


def durable_debug_log_is_created_and_repaired_private():
    import stat
    import tempfile
    from sliceagent import cli as cli_mod
    from sliceagent.cli import log_sink
    from sliceagent.events import AssistantText

    root = tempfile.mkdtemp(prefix="private-log-")
    path = os.path.join(root, "durable-log.jsonl")
    old_umask = os.umask(0o022)
    try:
        sink = log_sink(path=path)
        sink(AssistantText("first private answer"))
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        os.chmod(path, 0o644)
        sink(AssistantText("second private answer"))
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        os.chmod(path, 0o644)
        old_cap, cli_mod.LOG_MAX_BYTES = cli_mod.LOG_MAX_BYTES, 0
        try:
            sink(AssistantText("rotate private answer"))
        finally:
            cli_mod.LOG_MAX_BYTES = old_cap
        assert stat.S_IMODE(os.stat(path + ".1").st_mode) == 0o600
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    finally:
        os.umask(old_umask)


if __name__ == "__main__":
    main_reaches_prompt_without_crashing()
    print("PASS main_reaches_prompt_without_crashing")
    piped_no_key_run_keeps_the_gate_no_prompt()
    print("PASS piped_no_key_run_keeps_the_gate_no_prompt")
    interactive_first_run_auto_starts_the_wizard()
    print("PASS interactive_first_run_auto_starts_the_wizard")
    aborted_wizard_falls_back_to_the_gate()
    print("PASS aborted_wizard_falls_back_to_the_gate")
    update_routes_before_env_or_api_key_startup()
    print("PASS update_routes_before_env_or_api_key_startup")
    chitchat_consumes_stale_continuity_but_remains_exactly_adjacent()
    print("PASS chitchat_consumes_stale_continuity_but_remains_exactly_adjacent")
    pending_proposal_assents_bypass_chitchat_in_both_entry_paths()
    print("PASS pending_proposal_assents_bypass_chitchat_in_both_entry_paths")
    cost_projection_always_includes_session_token_totals()
    print("PASS cost_projection_always_includes_session_token_totals")
    live_failure_restores_inline_bridges_and_streaming_sink()
    print("PASS live_failure_restores_inline_bridges_and_streaming_sink")
    durable_debug_log_is_created_and_repaired_private()
    print("PASS durable_debug_log_is_created_and_repaired_private")
