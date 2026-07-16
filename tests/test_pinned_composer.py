"""The bordered, bottom-pinned composer (TuiInput._build_composer) — driven HEADLESSLY with a prompt_toolkit
pipe input + DummyOutput, so we verify the real Application end-to-end (Enter→submit, ctrl-c/ctrl-d→quit,
multi-char input) without a tty. This is the only way to test an interactive TUI offline; it catches a broken
layout / key binding that a mere "does it import" check would miss.

No model, no pytest. Run: PYTHONPATH=src python tests/test_pinned_composer.py
"""
import sys as _sys
if _sys.platform == "win32":
    print("SKIP: prompt_toolkit needs a real Windows console; CI's Git-Bash runner has none "
          "(NoConsoleScreenBufferError) — works in a real terminal")
    _sys.exit(0)
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _drive(keys: str):
    """Build the composer, feed `keys` through a pipe input, run it, return app.run()'s result."""
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from sliceagent.tui import TuiInput

    ti = TuiInput({"model": "test-model", "topic": "demo"}, root=None)
    with create_pipe_input() as pinp:
        pinp.send_text(keys)
        app, _ta = ti._build_composer(pt_input=pinp, pt_output=DummyOutput())
        return app.run()


@check
def enter_submits_typed_text():
    # type "hello world" then Enter (\r) → the composer returns exactly that line
    assert _drive("hello world\r") == "hello world", "Enter must submit the typed text"


@check
def empty_enter_returns_empty_not_none():
    # a bare Enter returns "" (an empty turn), NOT None — None is reserved for quit
    assert _drive("\r") == "", "bare Enter is an empty line, not a quit"


@check
def ctrl_c_quits():
    # ctrl-c (\x03) at the idle composer → None (quit), matching the plain-input path
    assert _drive("\x03") is None, "ctrl-c at the idle composer must quit (None)"


@check
def ctrl_d_quits():
    assert _drive("\x04") is None, "ctrl-d at the idle composer must quit (None)"


@check
def composer_layout_builds_with_frame_and_status():
    # the Application must construct with the bordered Frame + status window (catches a bad layout import)
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.input.defaults import create_pipe_input
    import inspect
    from sliceagent.tui import TuiInput, build_live_app
    ti = TuiInput({"model": "test-model"}, root=None)
    with create_pipe_input() as pinp:
        app, ta = ti._build_composer(pt_input=pinp, pt_output=DummyOutput())
        assert app is not None and ta is not None
        assert app.full_screen is False, "must stay in the normal buffer (native copy/paste)"
        assert ta.completer is ti._completer, "file/slash completion must be wired into the composer"
    assert 'Frame(ta, title="message")' not in inspect.getsource(TuiInput._build_composer)
    assert 'Frame(ta, title="message")' not in inspect.getsource(build_live_app), \
        "the input border needs no redundant title competing with the footer"


@check
def workspace_refresh_keeps_the_same_input_session_and_replaces_file_completions():
    from prompt_toolkit.document import Document
    from sliceagent.tui import TuiInput

    current = tempfile.mkdtemp(prefix="composer-current-")
    target = tempfile.mkdtemp(prefix="composer-target-")
    with open(os.path.join(current, "current_only.py"), "w", encoding="utf-8") as f:
        f.write("CURRENT = True\n")
    with open(os.path.join(target, "target_only.py"), "w", encoding="utf-8") as f:
        f.write("TARGET = True\n")

    ti = TuiInput({"model": "test-model", "workspace": "current"}, root=current)
    input_id, session_id = id(ti), id(ti.session)

    def complete(prefix):
        return {item.text for item in ti._completer.get_completions(Document(prefix), None)}

    assert "current_only.py" in complete("@current")
    ti.set_workspace(target)
    assert id(ti) == input_id and id(ti.session) == session_id, \
        "switching workspaces must refresh the existing composer, not reconnect it"
    assert "target_only.py" in complete("@target")
    assert "current_only.py" not in complete("@current"), \
        "completion must not retain files from the abandoned workspace"
    assert ti.session.completer is ti._completer


@check
def composer_is_transient_to_avoid_duplicate_echo():
    # REGRESSION: the composer box must erase on submit (erase_when_done). Otherwise the box's last frame —
    # still showing the typed text — is left on screen AND user_echo prints "▌ you …" → the message appears
    # twice. The echo is the persistent record; the input box is transient.
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.input.defaults import create_pipe_input
    from sliceagent.tui import TuiInput
    ti = TuiInput({"model": "test-model"}, root=None)
    with create_pipe_input() as pinp:
        app, _ta = ti._build_composer(pt_input=pinp, pt_output=DummyOutput())
        assert app.erase_when_done is True, "composer must erase on submit (else the message duplicates)"


@check
def prompt_falls_back_when_app_errors():
    # if the framed Application raises a non-exit error, prompt() must fall back to the plain prompt
    # (so input is NEVER broken). Force _pinned_prompt to raise; stub _simple_prompt to observe the fallback.
    from sliceagent.tui import TuiInput
    ti = TuiInput({"model": "test-model"}, root=None)
    ti._pinned_prompt = lambda: (_ for _ in ()).throw(RuntimeError("no tty"))
    ti._simple_prompt = lambda: "FELL_BACK"
    assert ti.prompt() == "FELL_BACK", "a framed-composer error must degrade to the plain prompt"


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
