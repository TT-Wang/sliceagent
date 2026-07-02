import os
import argparse


# The exact (correct) source fragment present in CPython 3.13's argparse.py,
# inside _get_nargs_pattern, for the ZERO_OR_MORE ('*') case.
_GOOD = "nargs_pattern = '(A*)' if option else '(-*[A-]*)'"
# The single-character behavioral bug: '*' -> '?' in the OPTIONAL-side regex,
# so an optional with nargs='*' (or '+') greedily matches AT MOST ONE token.
_BUG = "nargs_pattern = '(A?)' if option else '(-*[A-]*)'"


def setup(workdir):
    """Copy the real stdlib argparse module into the workdir as cliparse.py and
    plant exactly one subtle behavioral bug deep in the middle of the file.

    Also drop a tiny CLI program (mini_grep.py) that imports cliparse and a
    README describing the observed-but-wrong behavior, so the user prompt can
    refer to a concrete symptom without naming any internal function.
    """
    # 1) Copy the genuine, unmodified stdlib argparse source.
    src_path = argparse.__file__
    with open(src_path, "r") as f:
        source = f.read()

    # Sanity: the fragment we intend to corrupt must exist exactly once.
    if source.count(_GOOD) != 1:
        raise RuntimeError(
            "expected exactly one occurrence of the nargs-pattern fragment in "
            "argparse.py; this stdlib version is not supported by this scenario"
        )

    cli_path = os.path.join(workdir, "cliparse.py")
    with open(cli_path, "w") as f:
        f.write(source)

    # 2) Plant the bug DEEP IN THE MIDDLE (the fragment sits near line ~2459 of
    #    a ~2690-line file: below any head window, above any tail window).
    with open(cli_path, "r") as f:
        planted = f.read()
    planted = planted.replace(_GOOD, _BUG, 1)
    with open(cli_path, "w") as f:
        f.write(planted)

    # 3) A small real program that exercises a '*'/'+' multi-value option.
    app = '''\
"""mini_grep: a toy line filter built on the local cliparse module.

Usage example:
    python mini_grep.py --exclude foo bar baz -- needle file1.txt file2.txt

--exclude takes ANY number of patterns (zero or more). Use `--` to end the
list before the positional needle/files. The remaining items after the
options (or after `--`) are the search needle and the files.
"""
import sys
import cliparse


def build_parser():
    p = cliparse.ArgumentParser(prog="mini_grep")
    # multi-value option: should swallow EVERY following non-option token
    p.add_argument("--exclude", nargs="*", default=[],
                   help="patterns to skip (zero or more)")
    p.add_argument("--limit", type=int, default=0,
                   help="max matches (0 = unlimited)")
    p.add_argument("needle", help="substring to search for")
    p.add_argument("files", nargs="*", help="files to search")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    print("exclude=%r" % (args.exclude,))
    print("needle=%r" % (args.needle,))
    print("files=%r" % (args.files,))
    print("limit=%r" % (args.limit,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    with open(os.path.join(workdir, "mini_grep.py"), "w") as f:
        f.write(app)

    # 4) README documenting the symptom WITHOUT naming the buggy internal.
    readme = '''\
mini_grep
=========

`mini_grep.py` is a small command-line tool built on top of `cliparse.py`,
a vendored copy of Python's command-line argument library.

The `--exclude` option is declared with `nargs="*"`, so it is supposed to
collect *every* pattern that follows it (zero or more), up until the next
`--option` or the positional arguments.

Observed problem
----------------
When several patterns are passed, only the FIRST one is captured and the
rest leak into the positional arguments. Use `--` to mark the end of the
exclude list so the example is unambiguous:

    $ python mini_grep.py --exclude a b c -- hello f1.txt f2.txt
    exclude=['a']
    needle='b'
    files=['c', 'hello', 'f1.txt', 'f2.txt']

Expected:

    exclude=['a', 'b', 'c']
    needle='hello'
    files=['f1.txt', 'f2.txt']

It also breaks when the exclude list is followed by another option:

    $ python mini_grep.py --exclude a b c --limit 5 hello
    error: unrecognized arguments: hello        # 'b','c' got mis-consumed

The same truncation happens for any option declared with `nargs="*"` or
`nargs="+"`: it keeps at most one value. Single-value options and fixed-count
options (e.g. `nargs=2`) behave fine.

`mini_grep.py` itself looks correct; the defect is somewhere inside
`cliparse.py`.
'''
    with open(os.path.join(workdir, "README.md"), "w") as f:
        f.write(readme)
