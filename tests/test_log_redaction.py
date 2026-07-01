"""Debug-log sink: redact secrets on persist + rotate past a size cap (a rotating file sink over the
existing safety.redact_text boundary). A .env read or a token in tool output must NOT land in the
on-disk log in plaintext. No model, no pytest. Run: PYTHONPATH=src python tests/test_log_redaction.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.cli import LOG_MAX_BYTES, log_sink  # noqa: E402
from memagent.events import ToolResult  # noqa: E402
from memagent.safety import redact_text  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def secrets_are_redacted_before_hitting_the_log():
    d = tempfile.mkdtemp(prefix="log-")
    path = os.path.join(d, "log.jsonl")
    secret = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    assert redact_text(secret) != secret, "test input must be a recognized secret"
    sink = log_sink(path=path)
    sink(ToolResult("read_file", {"path": ".env"}, f"GITHUB_TOKEN={secret}\n", failing=False))
    data = open(path, encoding="utf-8").read()
    assert secret not in data, "raw secret leaked into the debug log"
    assert json.loads(data)["role"] == "tool", "line is still valid JSON after redaction"


@check
def non_secret_output_passes_through():
    d = tempfile.mkdtemp(prefix="log-")
    path = os.path.join(d, "log.jsonl")
    log_sink(path=path)(ToolResult("read_file", {"path": "a.py"}, "def f():\n    return 42\n", failing=False))
    assert "return 42" in open(path, encoding="utf-8").read()


@check
def log_rotates_past_the_size_cap():
    d = tempfile.mkdtemp(prefix="log-")
    path = os.path.join(d, "log.jsonl")
    with open(path, "w", encoding="utf-8") as f:        # pre-seed over the cap
        f.write("x" * (LOG_MAX_BYTES + 10))
    log_sink(path=path)(ToolResult("read_file", {"path": "a"}, "fresh", failing=False))
    assert os.path.exists(path + ".1"), "oversized log should rotate to .1"
    assert "fresh" in open(path, encoding="utf-8").read(), "new line goes to the fresh file"


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
