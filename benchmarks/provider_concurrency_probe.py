"""Low-credit real-provider probe for the off-main model path used by subagents.

Dry-run is the default. ``--live`` sends at most ``--calls`` tiny one-shot requests (default: 2, hard cap: 4)
with 24 output tokens each. App retries are disabled unless ``--retry`` is explicitly supplied, so the default
probe has a strict request/credit ceiling and can reveal provider concurrency without creating a full agent run.

    PYTHONPATH=src python benchmarks/provider_concurrency_probe.py
    PYTHONPATH=src python benchmarks/provider_concurrency_probe.py --live
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.config import load_config, load_prefs  # noqa: E402
from sliceagent.events import ApiRetry  # noqa: E402
from sliceagent.llm import OpenAILLM  # noqa: E402
from sliceagent.model_runner import complete_model_call  # noqa: E402


def _configured_provider(args) -> tuple[str, str, str]:
    cfg = load_config()
    prefs = load_prefs()
    providers = cfg.providers()
    pinned = providers.get(str(prefs.get("provider") or ""), {})
    model = args.model or os.environ.get("AGENT_MODEL") or prefs.get("model") \
        or pinned.get("model") or cfg.model
    key = os.environ.get("LLM_API_KEY") or pinned.get("api_key") or cfg.api_key
    base = os.environ.get("LLM_BASE_URL") or pinned.get("base_url") or cfg.base_url
    return str(model or ""), str(key or ""), str(base or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="send real requests (dry-run otherwise)")
    parser.add_argument("--retry", action="store_true",
                        help="enable SliceAgent's visible 3-attempt retry policy (may send up to 3x requests)")
    parser.add_argument("--calls", type=int, default=2, help="concurrent calls, 1..4 (default: 2)")
    parser.add_argument("--max-output-tokens", type=int, default=24, help="per-call output cap, 1..64")
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request timeout seconds")
    parser.add_argument("--model", default="", help="override the configured model")
    args = parser.parse_args()
    if not 1 <= args.calls <= 4:
        parser.error("--calls must be between 1 and 4")
    if not 1 <= args.max_output_tokens <= 64:
        parser.error("--max-output-tokens must be between 1 and 64")

    model, key, base = _configured_provider(args)
    plan = {
        "live": bool(args.live), "model": model, "endpoint": base or "SDK default",
        "concurrent_calls": args.calls, "max_output_tokens_each": args.max_output_tokens,
        "app_retries": 3 if args.retry else 1,
        "strict_max_physical_requests": args.calls * (3 if args.retry else 1),
    }
    if not args.live:
        print(json.dumps(plan, indent=2))
        print("dry-run only; add --live to send the bounded probe")
        return 0
    if not model or not key:
        print("No configured model/API key; run `sliceagent init` or set AGENT_MODEL + LLM_API_KEY.",
              file=sys.stderr)
        return 2

    llm = OpenAILLM(
        model=model, api_key=key, base_url=(base or None), timeout=max(1.0, args.timeout),
    )
    llm.reasoning = "fast"
    llm.max_tokens = args.max_output_tokens
    sdk_retries = getattr(llm.client, "max_retries", None)
    if sdk_retries != 0:
        print(f"Refusing probe: SDK retry invariant is {sdk_retries!r}, expected 0.", file=sys.stderr)
        return 3

    barrier = threading.Barrier(args.calls)
    lock = threading.Lock()
    rows: list[dict] = []

    def call(index: int) -> None:
        view = copy.copy(llm)  # production children use shallow per-profile views over the shared client
        attempts = 0
        retries: list[dict] = []

        def on_attempt(number, _messages, _report):
            nonlocal attempts
            attempts = number

        def dispatch(event):
            if isinstance(event, ApiRetry):
                retries.append({"attempt": event.attempt, "delay_s": event.delay_s, "error": event.error})

        barrier.wait()
        started = time.monotonic()
        row = {"call": index + 1}
        try:
            response = complete_model_call(
                view,
                [{"role": "system", "content": "Return exactly the word OK."},
                 {"role": "user", "content": "OK"}],
                [], retry=args.retry, dispatch=dispatch, on_attempt=on_attempt,
            )
            row.update(status="ok", text=str(getattr(response, "content", "") or "")[:80],
                       usage=dict(getattr(response, "usage", {}) or {}))
        except Exception as exc:  # noqa: BLE001 - diagnostic must report provider failures as data
            row.update(status="error", error=f"{type(exc).__name__}: {exc}"[:300])
        row.update(seconds=round(time.monotonic() - started, 3), attempts=attempts, retries=retries)
        with lock:
            rows.append(row)

    wall_started = time.monotonic()
    threads = [threading.Thread(target=call, args=(index,), name=f"provider-probe-{index + 1}", daemon=True)
               for index in range(args.calls)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(max(45.0, args.timeout * (3 if args.retry else 1) + 20.0))
    hung = [thread.name for thread in threads if thread.is_alive()]
    output = {
        **plan, "sdk_max_retries": sdk_retries,
        "wall_seconds": round(time.monotonic() - wall_started, 3),
        "hung_threads": hung, "results": sorted(rows, key=lambda row: row["call"]),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 1 if hung or len(rows) != args.calls or any(row["status"] != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
