"""Dump a subset of SWE-bench Lite instances to JSON (runs in the swebench venv).

Decouples the heavy dataset/swebench deps from memagent's runtime: this writes a small
JSON the agent-side runner (run_agent.py, in memagent's .venv) reads. The agent gets ONLY
the problem statement + repo@base_commit — never the gold patch or test_patch.

  python dump_instances.py --repos requests,flask,marshmallow --limit 2
  python dump_instances.py --ids psf__requests-2317
"""
from __future__ import annotations

import argparse
import json

from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/tmp/swebench-instances.json")
ap.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite")
ap.add_argument("--split", default="test")
ap.add_argument("--repos", default="", help="comma-separated repo substrings to filter")
ap.add_argument("--ids", default="", help="comma-separated explicit instance_ids")
ap.add_argument("--limit", type=int, default=2)
a = ap.parse_args()

ds = load_dataset(a.dataset, split=a.split)
ids = {x for x in a.ids.split(",") if x}
repos = [r for r in a.repos.split(",") if r]

sel = []
for r in ds:
    if ids:
        if r["instance_id"] in ids:
            sel.append(r)
    elif repos:
        if any(rp in r["repo"] for rp in repos):
            sel.append(r)
    else:
        sel.append(r)
    if not ids and len(sel) >= a.limit:
        break

out = [{"instance_id": r["instance_id"], "repo": r["repo"],
        "base_commit": r["base_commit"], "problem_statement": r["problem_statement"]}
       for r in sel]
with open(a.out, "w") as f:
    json.dump(out, f, indent=2)
print(f"wrote {len(out)} instance(s) to {a.out}")
for r in out:
    print(f"  - {r['instance_id']}  ({r['repo']})")
