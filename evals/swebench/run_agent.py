"""Run memagent on SWE-bench instances → predictions.jsonl (runs in memagent's .venv).

For each instance: clone the repo at base_commit, run the FULL agent (slice loop + CodeIndex
on the real repo + sandbox + guard policy) with the issue as the task, then take `git diff`
as the model_patch. Memory/mining off and tests untouched for benchmark hygiene. The output
predictions.jsonl is scored by the official swebench harness (see run.sh).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from memagent.cli import _load_env  # noqa: E402
from memagent.code_index import make_code_index  # noqa: E402
from memagent.events import ToolResult, make_dispatcher  # noqa: E402
from memagent.hooks import BudgetHook, CompositeHooks, PermissionHook  # noqa: E402
from memagent.llm import OpenAILLM  # noqa: E402
from memagent.memory import NullMemory  # noqa: E402
from memagent.policy import make_policy  # noqa: E402
from memagent.slice import Slice, make_build_slice, slice_sink  # noqa: E402
from memagent.loop import run_turn  # noqa: E402
from memagent.tools import LocalToolHost  # noqa: E402


def _git(*args, cwd=None, timeout=600):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)


def clone_at(repo: str, base_commit: str, dest: str) -> None:
    _git("clone", "--quiet", f"https://github.com/{repo}.git", dest, timeout=900)
    r = _git("checkout", "--quiet", base_commit, cwd=dest)
    if r.returncode != 0:  # base_commit not in default fetch → fetch it explicitly
        _git("fetch", "--quiet", "origin", base_commit, cwd=dest, timeout=900)
        _git("checkout", "--quiet", base_commit, cwd=dest)


def run_instance(inst: dict, *, model: str, max_steps: int, max_tokens: int) -> dict:
    repo_dir = os.path.join("/tmp/swebench-repos", inst["instance_id"])
    if os.path.isdir(repo_dir):
        subprocess.run(["rm", "-rf", repo_dir])
    os.makedirs(os.path.dirname(repo_dir), exist_ok=True)
    t0 = time.time()
    clone_at(inst["repo"], inst["base_commit"], repo_dir)

    task = ("Resolve this GitHub issue by editing the repository's source code. Make the "
            "minimal change that fixes it. Do NOT modify or add tests.\n\n"
            f"ISSUE:\n{inst['problem_statement']}")
    state = Slice(); state.reset(task)
    tools = LocalToolHost(repo_dir)
    retriever = make_code_index(repo_dir)
    build = make_build_slice(state, tools, retriever, NullMemory(), task)
    counter = {"n": 0}
    dispatch = make_dispatcher(slice_sink(state),
                               lambda e: counter.__setitem__("n", counter["n"] + 1) if isinstance(e, ToolResult) else None)
    hooks = CompositeHooks(PermissionHook(make_policy("guard")), BudgetHook(max_tokens))

    detail, steps, tokens = "", 0, 0
    try:
        res = run_turn(build_slice=build, llm=OpenAILLM(model=model), tools=tools,
                       dispatch=dispatch, hooks=hooks, max_steps=max_steps)
        steps = res.steps
        tokens = res.usage.get("prompt_tokens", 0) + res.usage.get("completion_tokens", 0)
        detail = res.stop_reason
    except Exception as e:  # noqa: BLE001
        detail = f"run error: {e}"

    diff = _git("diff", cwd=repo_dir).stdout
    print(f"  {inst['instance_id']}: {detail} · {steps} steps · {tokens} tok · "
          f"{counter['n']} tools · {len(diff)} diff-chars · {time.time() - t0:.0f}s")
    return {"instance_id": inst["instance_id"], "model_name_or_path": "memagent", "model_patch": diff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="/tmp/swebench-instances.json")
    ap.add_argument("--out", default="/tmp/swebench-predictions.jsonl")
    ap.add_argument("--model", default=os.environ.get("AGENT_MODEL", "gpt-5.5"))
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=200000)
    a = ap.parse_args()

    _load_env()
    instances = json.load(open(a.instances))
    print(f"running memagent on {len(instances)} instance(s) (model={a.model}) …")
    preds = [run_instance(i, model=a.model, max_steps=a.max_steps, max_tokens=a.max_tokens) for i in instances]
    with open(a.out, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    empties = sum(1 for p in preds if not p["model_patch"].strip())
    print(f"\nwrote {len(preds)} predictions to {a.out}  ({empties} empty patch)")


if __name__ == "__main__":
    main()
