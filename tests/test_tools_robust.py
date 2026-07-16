"""Robust file tools — atomic write (item 6), fuzzy str_replace fallback (item 7),
binary detection in read_text (item 8). No model, no pytest. Run: python tests/test_tools_robust.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.execution import ToolStatus  # noqa: E402
from sliceagent.tools import LocalToolHost   # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _host():
    """A LocalToolHost pinned to a fresh temp workspace root."""
    d = tempfile.mkdtemp(prefix="sliceagent-tools-test-")
    return LocalToolHost(root=d), d


def _write(root, rel, content, *, mode="w"):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p) or root, exist_ok=True)
    with open(p, mode, encoding="utf-8", newline="") as f:  # no \n->\r\n translation on Windows: tests assert byte-exact round-trips
        f.write(content)
    return p


def _read(root, rel):
    with open(os.path.join(root, rel), encoding="utf-8") as f:
        return f.read()


def _no_temp_files(root):
    """True if no .sliceagent-tmp-* leftover anywhere under root."""
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.startswith(".sliceagent-tmp-"):
                return False
    return True


# --- item 6: atomic write -----------------------------------------------------

@check
def edit_file_leaves_no_temp():
    h, root = _host()
    out = h._t_edit_file({"path": "a.txt", "content": "hello world"})
    assert _read(root, "a.txt") == "hello world"
    assert "Wrote" in out
    assert _no_temp_files(root), "stray .sliceagent-tmp-* left behind by edit_file"


@check
def str_replace_leaves_no_temp():
    h, root = _host()
    _write(root, "a.txt", "alpha beta gamma")
    h._t_str_replace({"path": "a.txt", "old_string": "beta", "new_string": "BETA"})
    assert _read(root, "a.txt") == "alpha BETA gamma"
    assert _no_temp_files(root)


@check
def midwrite_failure_keeps_original_and_unlinks_temp():
    # A non-string content blows up inside fdopen.write(); the ORIGINAL file must survive
    # and no temp may leak. _atomic_write writes to a temp then os.replace()s, so a failure
    # before the replace never touches the target.
    h, root = _host()
    _write(root, "keep.txt", "ORIGINAL")
    full = h._resolve("keep.txt")
    raised = False
    try:
        h._atomic_write(full, 12345)  # int -> f.write(int) raises TypeError
    except Exception:
        raised = True
    assert raised, "_atomic_write should propagate the write error"
    assert _read(root, "keep.txt") == "ORIGINAL", "mid-write failure corrupted the original"
    assert _no_temp_files(root), "mid-write failure leaked a temp file"


@check
def edit_file_preserves_mkparent():
    # edit_file must still create missing parent dirs (mkparent kept).
    h, root = _host()
    h._t_edit_file({"path": "deep/nested/dir/new.txt", "content": "x"})
    assert _read(root, "deep/nested/dir/new.txt") == "x"
    assert _no_temp_files(root)


# --- item 7: fuzzy str_replace fallback ---------------------------------------

@check
def str_replace_exact_still_primary():
    # Exact unique match: unchanged behavior, "Replaced 1 occurrence" with NO fuzzy tag.
    h, root = _host()
    _write(root, "a.py", "def f():\n    return 1\n")
    out = h._t_str_replace({"path": "a.py", "old_string": "return 1", "new_string": "return 2"})
    assert out.startswith("Replaced 1 occurrence in")
    assert "fuzzy" not in out and "normalized" not in out
    assert _read(root, "a.py") == "def f():\n    return 2\n"


@check
def str_replace_indent_only_via_fuzzy():
    # old_string has the wrong indentation -> exact byte match fails (n==0) -> the
    # indentation-flexible fuzzy span matches uniquely -> replaced, with a
    # "normalized/fuzzy match" note. The multi-line block ensures the exact bytes do
    # NOT appear as a substring (so we genuinely take the n==0 -> fuzzy branch).
    h, root = _host()
    _write(root, "a.py", "def f():\n        if x:\n            return 1\n")  # 8/12-space body
    out = h._t_str_replace({"path": "a.py",
                            "old_string": "if x:\nreturn 1",   # no indentation supplied
                            "new_string": "if x:\n            return 2"})
    assert "fuzzy" in out or "normalized" in out, f"expected fuzzy note, got: {out!r}"
    assert "return 2" in _read(root, "a.py")
    assert _no_temp_files(root)


@check
def str_replace_nonunique_still_errors():
    # >1 exact occurrences: recoverable steer, no fuzzy attempt, file untouched.
    h, root = _host()
    _write(root, "a.txt", "x\nx\n")
    out = h._t_str_replace({"path": "a.txt", "old_string": "x", "new_string": "y"})
    assert out.status is ToolStatus.STEERED and "2 times" in out
    assert _read(root, "a.txt") == "x\nx\n", "non-unique match must not modify the file"


@check
def str_replace_fuzzy_nonunique_still_errors():
    # No exact byte match (n==0), but the line-trimmed form matches TWO differently-
    # indented blocks -> fuzzy returns None (ambiguous) -> not-found error, file untouched.
    h, root = _host()
    original = "  foo\n  bar\n\tfoo\n\tbar\n"   # two trimmed 'foo\nbar' blocks, diff indent
    _write(root, "a.py", original)
    out = h._t_str_replace({"path": "a.py", "old_string": "foo\nbar", "new_string": "baz"})
    assert out.status is ToolStatus.STEERED and "not found" in out
    assert _read(root, "a.py") == original, "ambiguous fuzzy match must not modify the file"


@check
def str_replace_no_match_errors():
    h, root = _host()
    _write(root, "a.txt", "hello\n")
    out = h._t_str_replace({"path": "a.txt", "old_string": "nope", "new_string": "x"})
    assert out.status is ToolStatus.STEERED and "not found" in out


# --- item 8: binary detection in read_text ------------------------------------

@check
def read_text_nul_byte_raises():
    h, root = _host()
    _write(root, "blob.txt", "abc\x00def")     # NUL byte -> binary by content
    raised = False
    try:
        h.read_text("blob.txt")
    except ValueError as e:
        raised = True
        assert "binary" in str(e)
    assert raised, "read_text must raise ValueError on a NUL-byte file"


@check
def read_text_binary_extension_rejected():
    # .png and .so are binary by extension even if the bytes happen to decode as text.
    h, root = _host()
    _write(root, "logo.png", "PNGish text but wrong ext")
    _write(root, "lib.so", "elf-ish")
    for rel in ("logo.png", "lib.so"):
        raised = False
        try:
            h.read_text(rel)
        except ValueError as e:
            raised = True
            assert "binary" in str(e)
        assert raised, f"read_text must reject binary-extension file {rel}"


@check
def read_text_normal_utf8_unchanged():
    h, root = _host()
    body = "plain readable text\nwith newlines\n# comment with tabs:\tok\n"
    _write(root, "notes.txt", body)
    assert h.read_text("notes.txt") == body


@check
def read_file_binary_returns_hexdump_view():
    # read_file no longer REFUSES binaries — it returns a hexdump+magic view so forensics/
    # media tasks can inspect structure (str_replace's read_text still hard-refuses; see below).
    h, root = _host()
    _write(root, "x.png", "whatever")
    out = h.run("read_file", {"path": "x.png"})
    assert "binary file" in out and "hexdump" in out and not out.startswith("Error:")


@check
def str_replace_on_binary_degrades_via_registry():
    h, root = _host()
    _write(root, "blob.bin", "abc\x00def")
    out = h.run("str_replace", {"path": "blob.bin", "old_string": "abc", "new_string": "x"})
    assert out.status is ToolStatus.STEERED and "binary" in out


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
