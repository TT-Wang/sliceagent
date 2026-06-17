import os

# Correct, full implementation of the KV store after all six turns.
# VALIDATION ONLY -- never shown to the benchmarked agents.
#
# Core invariant that makes the turn-5 regression vanish:
#   A transaction layer is a self-contained dict (key -> value | _TOMBSTONE).
#   rollback() simply DROPS the innermost layer dict and touches nothing else,
#   so an aborted inner tx can never alter a parent layer or _base.
#   commit() folds BOTH writes and tombstones into the parent layer (or _base
#   when outermost), so a tombstone is preserved exactly one level up.

REFERENCE = '''\
"""A tiny in-memory key/value store with NESTED transactions.

Design notes:
  * ``_base`` holds committed data.
  * ``_layers`` is a stack of dicts (key -> value, or the ``_TOMBSTONE``
    sentinel meaning "deleted in this layer"). The last element is innermost.
  * A layer is fully self-contained: rollback drops the innermost layer and
    nothing else; commit folds the innermost layer (writes AND tombstones)
    into its parent layer, or into ``_base`` when it is the outermost layer.
"""


class KeyError_(KeyError):
    pass


_TOMBSTONE = object()


class KVStore:
    def __init__(self):
        self._base = {}
        self._layers = []

    # ----- transaction control -------------------------------------------
    def begin(self):
        self._layers.append({})

    def in_transaction(self):
        return bool(self._layers)

    def commit(self):
        if not self._layers:
            raise RuntimeError("no transaction to commit")
        layer = self._layers.pop()
        target = self._layers[-1] if self._layers else self._base
        if self._layers:
            # Folding into a parent layer: copy tombstones verbatim so the
            # parent also "forgets" committed-deleted keys.
            for k, v in layer.items():
                target[k] = v
        else:
            # Folding into the base: resolve tombstones into real deletions.
            for k, v in layer.items():
                if v is _TOMBSTONE:
                    self._base.pop(k, None)
                else:
                    self._base[k] = v

    def rollback(self):
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

    def _resolve(self, key):
        """Return (found, value) for the visible state of key."""
        for layer in reversed(self._layers):
            if key in layer:
                v = layer[key]
                if v is _TOMBSTONE:
                    return (False, None)
                return (True, v)
        if key in self._base:
            return (True, self._base[key])
        return (False, None)

    def get(self, key, default=None):
        found, v = self._resolve(key)
        return v if found else default

    # ----- views ----------------------------------------------------------
    def _visible_keys(self):
        seen = {}
        # Build from base up through layers so deeper layers override.
        for k, v in self._base.items():
            seen[k] = v
        for layer in self._layers:
            for k, v in layer.items():
                seen[k] = v
        return {k: v for k, v in seen.items() if v is not _TOMBSTONE}

    def keys(self):
        return sorted(self._visible_keys().keys())

    def items(self):
        vis = self._visible_keys()
        return [(k, vis[k]) for k in sorted(vis)]

    # ----- counter --------------------------------------------------------
    def numincr(self, key, by=1):
        found, cur = self._resolve(key)
        if not found:
            cur = 0
        if not isinstance(cur, int) or isinstance(cur, bool):
            raise TypeError("numincr on non-int value")
        new = cur + by
        self.set(key, new)
        return new

    # ----- snapshot of committed state -----------------------------------
    def snapshot(self):
        return dict(self._base)
'''


def apply(workdir):
    store = os.path.join(workdir, "tinykv", "store.py")
    with open(store, "w") as f:
        f.write(REFERENCE)
