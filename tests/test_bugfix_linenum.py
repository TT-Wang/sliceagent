"""Wave-1 line-numbered file evidence: OPEN FILES renders cat -n line numbers (for file:line citation),
and str_replace tolerates a snippet pasted back WITH those numbers (so numbering can't break edit anchors).
No model, no pytest. Run: PYTHONPATH=src python tests/test_bugfix_linenum.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.seed import _numbered  # noqa: E402
from memagent.tools import LocalToolHost, _strip_line_numbers  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def numbered_is_cat_n_style():
    assert _numbered(["alpha", "beta"]) == "     1\talpha\n     2\tbeta"
    assert _numbered(["x"], 42) == "    42\tx", "region blocks number from their absolute start line"


@check
def strip_only_when_every_line_numbered():
    assert _strip_line_numbers("     1\tdef foo():\n     2\t    return 1") == "def foo():\n    return 1"
    assert _strip_line_numbers("def foo():\n    return 1") == "def foo():\n    return 1"      # no numbers → unchanged
    assert _strip_line_numbers("     1\tA\nplain line\n     3\tC") == "     1\tA\nplain line\n     3\tC"  # not ALL → unchanged


@check
def str_replace_tolerates_pasted_line_numbers():
    wd = tempfile.mkdtemp(prefix="ln-")
    host = LocalToolHost(root=wd)
    host._t_edit_file({"path": "m.py", "content": "def foo():\n    return 1\n"})
    # snippet copied straight out of the numbered OPEN FILES render (with the "  N\t" prefix)
    pasted = _numbered(["def foo():", "    return 1"])     # "     1\tdef foo():\n     2\t    return 1"
    res = host._t_str_replace({"path": "m.py", "old_string": pasted, "new_string": "def foo():\n    return 2"})
    assert "Replaced 1 occurrence" in str(res), res
    assert host.read_text("m.py").strip() == "def foo():\n    return 2"


@check
def read_file_returns_numbered_and_editable():
    # review2 #1: read_file's RETURN now carries cat -n numbers (immediate in-turn file:line evidence),
    # and an edit using a snippet pasted WITH those numbers still matches (str_replace strips them).
    wd = tempfile.mkdtemp(prefix="ln-rf-")
    host = LocalToolHost(root=wd)
    host._t_edit_file({"path": "m.py", "content": "def f():\n    return 1\n"})
    out = host.run("read_file", {"path": "m.py"})
    assert out.splitlines()[0].startswith("     1\t"), out          # numbered
    assert _strip_line_numbers(out).startswith("def f():")          # de-numbers cleanly to the real content
    pasted = "     2\t    return 1"                                 # copied WITH the number
    assert "Replaced 1 occurrence" in str(host._t_str_replace(
        {"path": "m.py", "old_string": pasted, "new_string": "    return 2"}))


@check
def str_replace_raw_path_unchanged():
    # the PRIMARY exact path must still work with a clean (unnumbered) snippet
    wd = tempfile.mkdtemp(prefix="ln2-")
    host = LocalToolHost(root=wd)
    host._t_edit_file({"path": "m.py", "content": "a = 1\nb = 2\n"})
    res = host._t_str_replace({"path": "m.py", "old_string": "b = 2", "new_string": "b = 3"})
    assert "Replaced 1 occurrence" in str(res), res
    assert "b = 3" in host.read_text("m.py")


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
