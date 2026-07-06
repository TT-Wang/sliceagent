import os

# Seed project: a tiny in-memory KV store with SINGLE-LEVEL transactions.
# This is the starting repo BEFORE turn 1. It is small and working for the
# features it has. The 6 user turns extend it. The store keeps an explicit
# stack of transaction layers; the seed only ever uses one layer at a time
# (begin() refuses to nest), and rollback/commit operate on that one layer.

SEED_KVSTORE = '''\
"""A tiny in-memory key/value store with transactions.

Design notes (READ THIS before changing behavior):
  * The store holds a base dict ``_base`` of committed data.
  * Transactions are represented as a stack ``_layers``. Each layer is a dict
    mapping key -> value for writes made inside that transaction. A value of
    the sentinel ``_TOMBSTONE`` means "this key is deleted in this layer".
  * ``get`` resolves a key by scanning layers top-down, then falling back to
    ``_base``. A tombstone short-circuits to "missing".
  * Reads/writes outside a transaction go straight to ``_base``.

Public API (do not rename without updating callers):
  set(key, value), get(key, default=None), delete(key),
  begin(), commit(), rollback(), in_transaction()
"""


class KeyError_(KeyError):
    pass


# Sentinel marking a key deleted within a transaction layer.
_TOMBSTONE = object()


class KVStore:
    def __init__(self):
        self._base = {}
        self._layers = []  # stack of dicts; last is innermost

    # ----- transaction control -------------------------------------------
    def begin(self):
        """Start a transaction. The seed allows only ONE open transaction."""
        if self._layers:
            raise RuntimeError("transaction already open")
        self._layers.append({})

    def in_transaction(self):
        return bool(self._layers)

    def commit(self):
        """Commit the open transaction, folding its writes into the base."""
        if not self._layers:
            raise RuntimeError("no transaction to commit")
        layer = self._layers.pop()
        for k, v in layer.items():
            if v is _TOMBSTONE:
                self._base.pop(k, None)
            else:
                self._base[k] = v

    def rollback(self):
        """Discard the open transaction's writes."""
        if not self._layers:
            raise RuntimeError("no transaction to rollback")
        self._layers.pop()

    # ----- data operations -----------------------------------------------
    def set(self, key, value):
        if self._layers:
            self._layers[-1][key] = value
        else:
            self._base[key] = value

    def delete(self, key):
        if self._layers:
            self._layers[-1][key] = _TOMBSTONE
        else:
            self._base.pop(key, None)

    def get(self, key, default=None):
        for layer in reversed(self._layers):
            if key in layer:
                v = layer[key]
                return default if v is _TOMBSTONE else v
        if key in self._base:
            return self._base[key]
        return default
'''

SEED_README = '''\
# tinykv

A tiny in-memory key/value store with transactions, used as a teaching toy.

Current capabilities:
  * `set` / `get` / `delete`
  * a single (non-nested) transaction via `begin` / `commit` / `rollback`

See `tinykv/store.py` for the design notes. Keep the layer-stack design; new
features should build on `_layers` rather than replacing it.
'''

SEED_TEST = '''\
"""Smoke tests that ship with the seed. These exercise only the seed feature
set (single-level transactions). Keep them green."""
from tinykv.store import KVStore


def test_basic_set_get():
    s = KVStore()
    s.set("a", 1)
    assert s.get("a") == 1
    assert s.get("missing", "d") == "d"


def test_single_tx_commit():
    s = KVStore()
    s.set("a", 1)
    s.begin()
    s.set("a", 2)
    s.set("b", 3)
    assert s.get("a") == 2
    s.commit()
    assert s.get("a") == 2
    assert s.get("b") == 3


def test_single_tx_rollback():
    s = KVStore()
    s.set("a", 1)
    s.begin()
    s.set("a", 99)
    s.delete("a")
    assert s.get("a") is None
    s.rollback()
    assert s.get("a") == 1


if __name__ == "__main__":
    test_basic_set_get()
    test_single_tx_commit()
    test_single_tx_rollback()
    print("seed tests ok")
'''


def setup(workdir):
    pkg = os.path.join(workdir, "tinykv")
    os.makedirs(pkg, exist_ok=True)
    with open(os.path.join(pkg, "__init__.py"), "w") as f:
        f.write("from .store import KVStore\n")
    with open(os.path.join(pkg, "store.py"), "w") as f:
        f.write(SEED_KVSTORE)
    with open(os.path.join(workdir, "README.md"), "w") as f:
        f.write(SEED_README)
    tests = os.path.join(workdir, "tests")
    os.makedirs(tests, exist_ok=True)
    with open(os.path.join(tests, "test_seed.py"), "w") as f:
        f.write(SEED_TEST)
