# Contributing

## Dev setup

```bash
uv sync                 # or: pip install -e ".[dev,tui,treesitter]"   # [dev] = pytest + ruff
```

Python ≥ 3.11. The core is `openai`-free — only `llm.py`/`cli.py` import the SDK — so the whole loop is
testable offline with a fake LLM.

## Running the tests

memagent uses a **dependency-free custom test runner** (no pytest): each `tests/test_*.py` is a standalone
script with a tiny `@check` harness that exits non-zero on failure. Run the whole offline suite:

```bash
bash scripts/run_tests.sh             # runs every tests/test_*.py, tallies, exits non-zero on any failure
ruff check src/memagent tests evals   # lint (real-bug rules; house style configured in pyproject)
```

or a single file:

```bash
PYTHONPATH=src .venv/bin/python tests/test_reliability.py
```

The offline suite must stay green on every change. **No live API calls in the offline suite** — stub the
LLM (see `tests/test_reliability.py` / `tests/test_live_composer.py` for the patterns: fake `complete()`,
pipe-driven prompt_toolkit apps, recording Rich consoles). Eval tooling that needs a key lives in `evals/`
and is run manually.

### Testing interactive UI offline
Drive prompt_toolkit Applications with a pipe input + `DummyOutput`, and Rich sinks with a
`Console(file=StringIO())` — see `tests/test_live_composer.py`. This catches layout/keybinding/logic
regressions without a tty.

## Architecture in one breath

The loop (`loop.py`) is the moat: one `while(true)` per turn over a **bounded slice** that is the SEED
(built once by `make_build_slice`), then working memory **accumulates** as native messages within the turn
and is sealed to the durable cache at the turn boundary. The core depends only on contracts
(`interfaces.py`): `LLMClient`, `ToolHost`, `Retriever`, `Oracle`, plus an event `dispatch` and `hooks`. It
never imports implementations. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

**Layering rule:** `loop.py`/`pfc.py`/`seed.py` are the moat — keep them stateless and contract-only. UI
(`tui.py`, `tui_app.py`, `cli.py`), tools, providers, and extensions are periphery: borrow liberally,
keep them behind seams (event sinks, the tool registry, the `LLMClient`/`Memory` interfaces).

## Conventions

- Match the surrounding code's comment density and idiom. Comments explain *why*, not *what*.
- Every behavior change ships a test in the same PR.
- Keep the agent **task-agnostic** and **LLM/provider-agnostic**: no task-type-specific heuristics in the
  core; provider quirks stay isolated in `llm.py`.
- New environment variables go in `envspec.py` (a coverage test enforces this).
