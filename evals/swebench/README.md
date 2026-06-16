# SWE-bench (Lite) validation harness

Runs the **full memagent** on real GitHub issues and scores it with the **official
swebench** containerized harness. Two isolated environments (so memagent's runtime stays
clean of heavy benchmark deps), with `predictions.jsonl` as the interface:

1. **swebench venv** (`/tmp/sweb-venv`, `pip install swebench`) — loads the dataset and runs
   the official Docker-based scoring. Needs the Docker daemon.
2. **memagent `.venv`** — runs the agent per instance and emits predictions.

## Flow
```bash
# 0) prerequisites: Docker running; /tmp/sweb-venv has swebench; memagent .venv ready

# 1) pick a subset (light repos build fastest)
/tmp/sweb-venv/bin/python evals/swebench/dump_instances.py --repos marshmallow,requests --limit 2

# 2) run memagent → predictions.jsonl  (uses the issue text only; never the gold/test patch)
LLM_API_KEY=... AGENT_MODEL=gpt-5.5 \
  ./.venv/bin/python evals/swebench/run_agent.py

# 3) score with the official harness (builds per-instance Docker images, applies the patch,
#    runs FAIL_TO_PASS + PASS_TO_PASS)
/tmp/sweb-venv/bin/python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Lite \
  --predictions_path /tmp/swebench-predictions.jsonl \
  --run_id memagent --max_workers 2 --cache_level env
# → report: memagent.memagent.json  (resolved / unresolved per instance)
```

The agent is given only the issue (`problem_statement`) and the repo at `base_commit`; the
gold patch and test_patch are withheld and applied only by the scorer. Benchmark hygiene:
NullMemory, no mining, tests-untouched; CodeIndex + sandbox + guard policy on.
