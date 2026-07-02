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
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": os.path.join(_ROOT, "src"),
        "AGENT_TUI": "off",                 # plain REPL — no prompt_toolkit app to drive
        "LLM_API_KEY": "sk-dummy-smoke", "OPENAI_API_KEY": "sk-dummy-smoke",
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


if __name__ == "__main__":
    main_reaches_prompt_without_crashing()
    print("PASS main_reaches_prompt_without_crashing")
