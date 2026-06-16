"""Run the eval suite against a real model and print a scorecard.

    python -m evals.run                 # all cases on $AGENT_MODEL (default gpt-5.5)
    python -m evals.run --case calc_eval
    python -m evals.run --model gpt-4o
"""
from __future__ import annotations

import argparse
import os
import sys


def _ensure_importable() -> None:
    """Make `memagent` importable even when the editable install is flaky:
    fall back to the repo's src/ on sys.path. (No-op when the package imports fine.)"""
    try:
        import memagent  # noqa: F401
    except ModuleNotFoundError:
        src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
        if os.path.isdir(src) and src not in sys.path:
            sys.path.insert(0, src)


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)  # stream progress even when piped (no | tail blind spot)
    except Exception:
        pass
    _ensure_importable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("AGENT_MODEL", "gpt-5.5"))
    ap.add_argument("--case", default=None, help="run a single case by name (searches all suites)")
    ap.add_argument("--stress", action="store_true", help="run the stress battery instead of the core suite")
    args = ap.parse_args()

    from memagent.cli import _load_env
    _load_env()
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")):
        print("Set OPENAI_API_KEY (or MOONSHOT_API_KEY).")
        sys.exit(1)

    os.environ["AGENT_MODEL"] = args.model
    from memagent.llm import OpenAILLM

    from .cases import ALL_CASES, CASES, STRESS_CASES
    from .runner import print_footer, print_header, print_result_row, run_eval

    if args.case is not None:
        cases = [c for c in ALL_CASES if c.name == args.case]
    else:
        cases = STRESS_CASES if args.stress else CASES
    if not cases:
        print(f"no case named {args.case!r}; available: {[c.name for c in ALL_CASES]}")
        sys.exit(1)

    llm = OpenAILLM(model=args.model)
    print(f"running {len(cases)} eval case(s) on {args.model} …")
    print_header()
    results = run_eval(cases, llm, on_result=print_result_row)  # stream each row as the case finishes
    print_footer(results)
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
