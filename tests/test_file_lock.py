"""Cross-process advisory FileLock for the episode writer (#2). Serializes concurrent appenders to the same
session JSONL so their lines can't interleave into a torn record. Real on POSIX (fcntl.flock), graceful no-op
elsewhere. No model, no network. Run: PYTHONPATH=src python tests/test_file_lock.py
"""
import json
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.platform_compat import FileLock  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def filelock_is_a_working_context_manager():
    path = os.path.join(tempfile.mkdtemp(), "x.txt")
    with open(path, "a", encoding="utf-8") as f, FileLock(f):
        f.write("hello\n")
    assert open(path, encoding="utf-8").read() == "hello\n"


@check
def filelock_never_raises_on_a_weird_object():
    # a locking failure must DEGRADE to unlocked, never propagate (a cache write must not crash a session)
    class _NoFileno:
        def fileno(self): raise OSError("no fd")
    with FileLock(_NoFileno()):        # must not raise
        pass


@check
def concurrent_appends_stay_valid_lines():
    # N threads each append a distinct padded record to the SAME file under the lock; every line must come
    # back complete and parseable (no interleaving), and all N must be present.
    path = os.path.join(tempfile.mkdtemp(), "session.jsonl")
    N = 24

    def _append(i):
        rec = json.dumps({"i": i, "pad": "x" * 800})
        with open(path, "a", encoding="utf-8") as f, FileLock(f):
            f.write(rec + "\n")

    threads = [threading.Thread(target=_append, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = [ln for ln in open(path, encoding="utf-8") if ln.strip()]
    assert len(lines) == N, f"expected {N} lines, got {len(lines)}"
    parsed = [json.loads(ln) for ln in lines]                 # every line valid JSON → no torn/interleaved record
    assert sorted(p["i"] for p in parsed) == list(range(N))


def main():
    ok = 0
    for fn in CHECKS:
        try:
            fn(); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} passed")
    sys.exit(0 if ok == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
