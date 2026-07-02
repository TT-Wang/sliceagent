"""Binary read view — read_file returns a hexdump+magic for binaries instead of refusing,
so forensics/media/archive tasks can inspect structure. str_replace still hard-refuses binaries
(you can't text-edit them). Deterministic, no model, no pytest.
Run: PYTHONPATH=src python tests/test_binary_view.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.tools import LocalToolHost  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _host():
    wd = tempfile.mkdtemp(prefix="bin-")
    return wd, LocalToolHost(root=wd)


@check
def binary_returns_hexdump_not_refusal():
    wd, h = _host()
    png = bytes.fromhex("89504e470d0a1a0a") + b"\x00\x01\x02\x03IHDR\x00\x00"
    open(os.path.join(wd, "img.png"), "wb").write(png)
    out = h.run("read_file", {"path": "img.png"})
    assert "binary file" in out and "hexdump" in out, out
    assert "89 50 4e 47" in out, f"PNG magic missing from hexdump: {out!r}"
    assert "appears to be binary; not shown" not in out, out


@check
def magic_line_present():
    wd, h = _host()
    open(os.path.join(wd, "blob.bin"), "wb").write(b"\x7fELF\x02\x01\x01\x00rest\x00\x00")
    out = h.run("read_file", {"path": "blob.bin"})
    assert "magic: 7f454c46" in out, f"ELF magic missing: {out!r}"


@check
def text_still_reads_normally():
    wd, h = _host()
    open(os.path.join(wd, "a.txt"), "w").write("hello world\n")
    from sliceagent.tools import _strip_line_numbers      # read_file now returns cat -n numbered content
    assert _strip_line_numbers(h.run("read_file", {"path": "a.txt"})).strip() == "hello world"


@check
def str_replace_still_refuses_binary():
    wd, h = _host()
    open(os.path.join(wd, "x.bin"), "wb").write(b"\x00\x01\x02hello\x00")
    out = h.run("str_replace", {"path": "x.bin", "old_string": "hello", "new_string": "bye"})
    assert "binary" in out.lower(), f"str_replace should refuse binary: {out!r}"


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
