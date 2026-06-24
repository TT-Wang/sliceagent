# tests

Dependency-free offline suite (no pytest). Each `test_*.py` is a standalone script: a tiny `@check`
collector runs every check and exits non-zero on the first failure, printing `PASS`/`FAIL` per check.

## Run

```bash
# whole suite
for f in tests/test_*.py; do PYTHONPATH=src .venv/bin/python "$f" || break; done

# one file
PYTHONPATH=src .venv/bin/python tests/test_reliability.py
```

## Rules
- **No live API calls.** Stub the LLM. Patterns:
  - fake `complete()` returning a `SimpleNamespace(content=…, tool_calls=[], finish_reason="stop", usage={})`;
  - prompt_toolkit apps driven by `create_pipe_input()` + `DummyOutput` (`test_live_composer.py`, `test_pinned_composer.py`);
  - Rich sinks rendered into `Console(file=StringIO())` to assert on output.
- **Deterministic.** No `Date.now`-style nondeterminism; no network; no real clock dependence beyond short sleeps in the scheduler-timeout test.
- A behavior change ships a test here in the same change.

Live, key-requiring checks (h2h, routing accuracy, review quality) live in `evals/` and are run manually —
they are not part of the offline suite.
