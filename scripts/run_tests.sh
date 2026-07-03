#!/usr/bin/env bash
# Run the offline test suite. Each tests/test_*.py is a standalone script with its own main() (no pytest),
# so this wrapper runs them all, tallies pass/fail, prints the tail of any failure, and EXITS NON-ZERO if
# anything fails — giving CI (and a local `bash scripts/run_tests.sh`) a single real signal.
set -u
cd "$(dirname "$0")/.." || exit 2

PY="${PYTHON:-.venv/bin/python}"
command -v "$PY" >/dev/null 2>&1 || PY="python3"   # CI installs the package, so a plain python3 works too
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1   # Windows console defaults to cp1252; test output contains UTF-8 (no-op on POSIX)

pass=0; fail=0; failed=""
log="$(mktemp)"
for t in tests/test_*.py; do
  if "$PY" "$t" >"$log" 2>&1; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1)); failed="$failed ${t##*/}"
    echo "── FAIL: $t ─────────────────────────────"
    # the FAIL/Traceback lines first (a chatty file scrolls them out of a blind tail), then the tail
    grep -E "^FAIL |Traceback|^[A-Za-z]*Error" "$log" | head -15
    tail -12 "$log"
  fi
done
rm -f "$log"

echo "────────────────────────────────────────"
echo "suite: ${pass} passed, ${fail} failed${failed:+  (${failed} )}"
[ "$fail" -eq 0 ]
