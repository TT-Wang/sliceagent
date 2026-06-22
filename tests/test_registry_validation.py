"""Registry arg-schema validation (Kimi AJV-style): a missing required arg yields a clear, model-
actionable error instead of an opaque handler KeyError. No model, no pytest.
Run: PYTHONPATH=src python tests/test_registry_validation.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.registry import ToolEntry, ToolRegistry  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _schema(name, required):
    return {"type": "function", "function": {
        "name": name,
        "parameters": {"type": "object",
                       "properties": {r: {"type": "string"} for r in required},
                       "required": required}}}


@check
def missing_required_arg_is_a_clear_error():
    reg = ToolRegistry()
    reg.register(ToolEntry(name="read_file", schema=_schema("read_file", ["path"]),
                           handler=lambda a: f"read {a['path']}"))
    out = reg.run("read_file", {})
    assert not out.ok, out
    assert "missing required argument" in out and "path" in out, out
    ok = reg.run("read_file", {"path": "a.py"})
    assert ok.ok and ok == "read a.py", ok


@check
def present_but_empty_counts_as_supplied():
    reg = ToolRegistry()
    reg.register(ToolEntry(name="edit_file", schema=_schema("edit_file", ["path", "content"]),
                           handler=lambda a: f"wrote {len(a['content'])}"))
    out = reg.run("edit_file", {"path": "a", "content": ""})   # empty is intentional, not missing
    assert out.ok and out == "wrote 0", out


@check
def no_required_list_means_no_validation():
    reg = ToolRegistry()
    reg.register(ToolEntry(name="ping",
                           schema={"type": "function", "function": {"name": "ping",
                                   "parameters": {"type": "object", "properties": {}}}},
                           handler=lambda a: "pong"))
    assert reg.run("ping", {}).ok


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
