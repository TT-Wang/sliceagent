"""CodeIndex tests — the ranked repo-MAP discovery tier. Needs `rg`.
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
    m = idx.graph_map(q)
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
    m = idx.graph_map("checkout", max_chars=120)
    assert len(m) <= 130            # honored the cap (small slack for the final block boundary)


@check
def empty_query_still_maps_repo():
    # zero query terms → no lexical signal, but the map still orients (all files, unranked)
    if _skip_no_rg():
        print("  (skip: no rg)"); return
    idx = RipgrepCodeIndex(root=_repo())
    m = idx.graph_map("the and for")    # all stopwords → no terms
    assert "services/billing.py" in m and "reporting.py" in m


@check
def no_code_returns_empty():
    if _skip_no_rg():
        print("  (skip: no rg)"); return
    root = tempfile.mkdtemp()
    with open(os.path.join(root, "notes.txt"), "w") as f:
        f.write("just prose, no code files\n")
    idx = RipgrepCodeIndex(root=root)
    assert idx.retrieve("anything") == [] and idx.graph_map("anything") == ""


def _graph_repo():
    """A matched file -> calls a NEUTRAL-vocabulary helper; plus filler that sorts before it."""
    root = tempfile.mkdtemp()
    files = {
        "api/handlers.py": "from svc.cart import amount_due\ndef checkout(member, discount):\n    return amount_due(member)\n",
        "svc/cart.py": "from svc.billing import settle\ndef amount_due(member):\n    return settle(member)\n",
        "svc/billing.py": "def settle(qty):\n    return qty * 2   # purely neutral vocabulary\n",
    }
    for i in range(40):                      # filler: alphabetically before svc/ → eats a tight budget
        files[f"aaa_filler/mod{i:02d}.py"] = f"def widget_{i:02d}(x):\n    return x\n"
    for rel, body in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p) or root, exist_ok=True)
        with open(p, "w") as f:
            f.write(body)
    return root


@check
def graph_surfaces_neutral_callee_that_lexical_truncates():
    if _skip_no_rg():
        print("  (skip: no rg)"); return
    idx = RipgrepCodeIndex(root=_graph_repo())
    q = "checkout discount member"
    # tight budget: the neutral bug file has ZERO query overlap and sorts after 40 filler files,
    # so a purely-lexical ranking would truncate it — but PageRank flows rank cart->billing and keeps it.
    graph = idx.graph_map(q, max_chars=400)
    assert "svc/billing.py" in graph, "PageRank should surface the called neutral file"
    # and it ranks ABOVE the filler (appears before any aaa_filler line)
    assert graph.index("svc/billing.py") < graph.index("aaa_filler/")


@check
def graph_map_is_bounded():
    if _skip_no_rg():
        print("  (skip: no rg)"); return
    idx = RipgrepCodeIndex(root=_graph_repo())
    assert len(idx.graph_map("checkout", max_chars=200)) <= 230


@check
def graph_degrades_to_lexical_without_edges():
    # no cross-file symbols → no edges → ranking falls back to lexical/path, still maps the repo
    if _skip_no_rg():
        print("  (skip: no rg)"); return
    idx = RipgrepCodeIndex(root=_repo())          # the earlier repo (independent files)
    g = idx.graph_map("checkout discount")
    assert "reporting.py" in g and "services/billing.py" in g


@check
def treesitter_defs_precise_when_available():
    from memagent.code_index import _ts_def_names, _ts_python
    if _ts_python() is None:
        print("  (skip: tree-sitter not installed)"); return
    src = ('# comment with def fake_fn and class FakeCls\nS = "def string_trap"\n'
           "def real_fn(a):\n    def nested_fn(x):\n        return x\n    return nested_fn\n"
           "class RealCls:\n    def method_a(self):\n        pass\n")
    names = _ts_def_names("x.py", src)
    assert names == {"real_fn", "nested_fn", "RealCls", "method_a"}, names
    assert "fake_fn" not in names and "string_trap" not in names    # no comment/string false-positives


@check
def graph_cache_reuses_until_tree_changes():
    if _skip_no_rg():
        print("  (skip: no rg)"); return
    root = _graph_repo()
    idx = RipgrepCodeIndex(root=root)
    q = "checkout discount member"
    idx.graph_map(q)
    assert idx._graph_builds == 1                 # cold build
    idx.graph_map(q); idx.graph_map("other query")
    assert idx._graph_builds == 1                 # query changes don't rebuild the graph
    # edit a file → fingerprint changes → exactly one rebuild, new defs reflected
    with open(os.path.join(root, "svc/billing.py"), "a") as f:
        f.write("\n\ndef refund(amount):\n    return amount\n")
    out = idx.graph_map(q)
    assert idx._graph_builds == 2 and "def refund" in out
    idx.graph_map(q)
    assert idx._graph_builds == 2                 # stable again after the rebuild


@check
def graph_cache_output_matches_uncached():
    # caching must not change the map — a fresh index and a warmed index agree
    if _skip_no_rg():
        print("  (skip: no rg)"); return
    root = _graph_repo()
    a = RipgrepCodeIndex(root=root).graph_map("checkout discount member")
    warm = RipgrepCodeIndex(root=root)
    warm.graph_map("checkout discount member")    # warm the cache
    b = warm.graph_map("checkout discount member")
    assert a == b


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
