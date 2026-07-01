"""Exec-env safety: line-ending PRESERVATION on edit. The model emits '\\n';
writing that to a CRLF file would rewrite every line ending (corruption / huge spurious diff). No model,
no pytest. Run: PYTHONPATH=src python tests/test_exec_env.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.tools import LocalToolHost  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _write_raw(d, name, data: bytes):
    p = os.path.join(d, name)
    with open(p, "wb") as f:
        f.write(data)
    return p


def _read_raw(p) -> bytes:
    with open(p, "rb") as f:
        return f.read()


@check
def str_replace_preserves_crlf():
    d = tempfile.mkdtemp(prefix="eol-")
    _write_raw(d, "win.txt", b"alpha\r\nbeta\r\ngamma\r\n")   # CRLF file
    host = LocalToolHost(d)
    out = host.run("str_replace", {"path": "win.txt", "old_string": "beta", "new_string": "BETA"})
    assert "Replaced" in out, out
    data = _read_raw(os.path.join(d, "win.txt"))
    assert data == b"alpha\r\nBETA\r\ngamma\r\n", data       # ALL endings still CRLF, edit applied
    assert b"\n" not in data.replace(b"\r\n", b""), "no lone LF leaked"


@check
def str_replace_keeps_lf_lf():
    d = tempfile.mkdtemp(prefix="eol-")
    _write_raw(d, "unix.txt", b"alpha\nbeta\ngamma\n")        # LF file
    LocalToolHost(d).run("str_replace", {"path": "unix.txt", "old_string": "beta", "new_string": "BETA"})
    data = _read_raw(os.path.join(d, "unix.txt"))
    assert data == b"alpha\nBETA\ngamma\n", data              # stays LF, no spurious CR
    assert b"\r" not in data


@check
def edit_file_preserves_existing_crlf():
    d = tempfile.mkdtemp(prefix="eol-")
    _write_raw(d, "win.py", b"x = 1\r\ny = 2\r\n")
    LocalToolHost(d).run("edit_file", {"path": "win.py", "content": "x = 1\ny = 2\nz = 3\n"})
    data = _read_raw(os.path.join(d, "win.py"))
    assert data == b"x = 1\r\ny = 2\r\nz = 3\r\n", data       # rewritten content adopts the file's CRLF


@check
def edit_file_new_file_stays_lf():
    d = tempfile.mkdtemp(prefix="eol-")
    LocalToolHost(d).run("edit_file", {"path": "new.txt", "content": "a\nb\n"})  # no existing file
    assert _read_raw(os.path.join(d, "new.txt")) == b"a\nb\n"  # default LF, no forced CRLF


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
