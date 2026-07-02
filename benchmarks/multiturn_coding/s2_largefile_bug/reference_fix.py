import os


# The planted bug and its correct form (see setup.py). The reference fix simply
# reverts the single-character corruption in _get_nargs_pattern.
_BUG = "nargs_pattern = '(A?)' if option else '(-*[A-]*)'"
_GOOD = "nargs_pattern = '(A*)' if option else '(-*[A-]*)'"


def apply(workdir):
    """Revert the planted nargs-pattern bug in cliparse.py.

    Restores the ZERO_OR_MORE ('*') optional regex from '(A?)' back to '(A*)',
    so options declared with nargs='*' (and, via the shared path, nargs='+')
    greedily collect all of their following values again.
    """
    cli = os.path.join(workdir, "cliparse.py")
    with open(cli, "r") as f:
        text = f.read()

    if _GOOD in text and _BUG not in text:
        return  # already correct

    if _BUG not in text:
        raise RuntimeError(
            "expected planted bug fragment not found in cliparse.py; "
            "cannot apply reference fix"
        )

    text = text.replace(_BUG, _GOOD, 1)
    with open(cli, "w") as f:
        f.write(text)
