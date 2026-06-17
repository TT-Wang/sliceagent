"""Reliability batch: run given scenarios N times each (memagent, v2 code) to separate a real fix/
regression from kimi-k2.7-code's run-to-run variance. Prints pass-rate + metric spread per scenario.

  PYTHONPATH=src .venv/bin/python -m evals.h2h_reliability s3_multifile_refactor:3 s2_largefile_bug:3
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    from memagent.cli import _load_env
    _load_env()
    from evals.h2h_run import load_scenario, run_memagent

    model = os.environ.get("AGENT_MODEL", "kimi-k2.7-code")
    specs = sys.argv[1:] or ["s3_multifile_refactor:3", "s2_largefile_bug:3"]
    for spec in specs:
        name, _, n = spec.partition(":")
        n = int(n or 3)
        scn = load_scenario(name)
        rows = []
        print(f"\n=== {name} × {n} ===", flush=True)
        for i in range(n):
            wd = tempfile.mkdtemp(prefix=f"rel-{name}-")
            scn["setup"](wd)
            r = run_memagent(scn, wd, model)
            cache = 100 * r["in_cached"] / max(r["in_total"], 1)
            rows.append(r)
            print(f"  run {i+1}: {'PASS' if r['passed'] else 'FAIL':4} steps={r['steps']:>3} "
                  f"clean_wall={r.get('wall_clean', r['wall_s']):>6.0f}s stalls={r.get('stalls',0)} "
                  f"out={r['out_total']:>6} cache={cache:>3.0f}%  {r['detail'][:34]}", flush=True)
        p = sum(1 for r in rows if r["passed"])
        walls = [r.get("wall_clean", r["wall_s"]) for r in rows]
        caches = [100 * r["in_cached"] / max(r["in_total"], 1) for r in rows]
        print(f"  -> PASS {p}/{n} | clean_wall {min(walls):.0f}-{max(walls):.0f}s "
              f"(med {sorted(walls)[len(walls)//2]:.0f}) | cache {min(caches):.0f}-{max(caches):.0f}%",
              flush=True)


if __name__ == "__main__":
    main()
