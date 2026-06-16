"""CodeIndex tests — the ranked repo-MAP discovery tier (and on-demand snippets). Needs `rg`.
Run: python tests/test_code_index.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent.code_index import RipgrepCodeIndex  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


def _repo():
    root = tempfile.mkdtemp()
    files = {
        # neutral-vocabulary BUG file — shares NO words with the task → term search never ranks it
        "services/billing.py": "def settle(lines, rate):\n    return sum(u * q for u, q in lines) * (1 - rate)\n",
        # loud, CORRECT red herring — matches the task vocabulary
        "reporting.py": "def checkout_summary(member, discount, amount):\n    return f'{member} {discount} {amount}'\n",
        "api/handlers.py": "def checkout(lines, member, discount):\n    return None\n",
        "utils/formatting.py": "def format_amount(x):\n    return x\n\n\ndef format_discount(p):\n    return p\n",
        "readme.txt": "not code\n",
    }
    for rel, body in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p) or root, exist_ok=True)
        with open(p, "w") as f:
            f.write(body)
    return root


def _skip_no_rg():
    return shutil.which("rg") is None


@check
def map_ranks_matches_first_but_keeps_neutral_files():
    if _skip_no_rg():
        print("  (skip: no rg)"); return
    root = _repo()
    idx = RipgrepCodeIndex(root=root)
    q = "checkout discount member amount summary"
    m = idx.scoped_map(q)
    # the loud match is annotated and appears BEFORE the neutral bug file
    assert "reporting.py" in m and "(matches:" in m
    # the KEY robustness property: the neutral-vocabulary bug file is STILL in the map
    assert "services/billing.py" in m and "def settle" in m
    assert m.index("reporting.py") < m.index("services/billing.py")   # matched files first
    # billing has no annotation (it matched nothing)
    bill_line = next(ln for ln in m.splitlines() if ln.startswith("services/billing.py"))
    assert "(matches:" not in bill_line


@check
def retrieve_returns_single_map_snippet():
    if _skip_no_rg():
        print("  (skip: no rg)"); return
    idx = RipgrepCodeIndex(root=_repo())
    out = idx.retrieve("checkout discount", k=6)
    assert len(out) == 1 and out[0].path == "(repo map)"
    assert "def settle" in out[0].text and "reporting.py" in out[0].text


@check
def map_is_bounded():
    if _skip_no_rg():
        print("  (skip: no rg)"); return
    idx = RipgrepCodeIndex(root=_repo())
    m = idx.scoped_map("checkout", max_chars=120)
    assert len(m) <= 130            # honored the cap (small slack for the final block boundary)


@check
def empty_query_still_maps_repo():
    # zero query terms → no lexical signal, but the map still orients (all files, unranked)
    if _skip_no_rg():
        print("  (skip: no rg)"); return
    idx = RipgrepCodeIndex(root=_repo())
    m = idx.scoped_map("the and for")    # all stopwords → no terms
    assert "services/billing.py" in m and "reporting.py" in m


@check
def snippets_still_available_on_demand():
    if _skip_no_rg():
        print("  (skip: no rg)"); return
    idx = RipgrepCodeIndex(root=_repo())
    sn = idx.snippets("checkout summary", k=6)
    assert sn and any("reporting.py" in s.path for s in sn)
    # snippets carry actual code lines (line-numbered), not just signatures
    assert any("checkout_summary" in s.text for s in sn)


@check
def no_code_returns_empty():
    if _skip_no_rg():
        print("  (skip: no rg)"); return
    root = tempfile.mkdtemp()
    with open(os.path.join(root, "notes.txt"), "w") as f:
        f.write("just prose, no code files\n")
    idx = RipgrepCodeIndex(root=root)
    assert idx.retrieve("anything") == [] and idx.scoped_map("anything") == ""


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
