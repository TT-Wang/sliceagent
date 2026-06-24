"""LIVE end-to-end test of #3 — transcript → reusable SKILL via /learn (B2). Runs a small real task
(builds the episodic CACHE), then runs the /learn turn (build_learn_prompt) with the agent holding
recall_history + write_skill; asserts the agent distilled a USER-provenance SKILL.md FROM THE CACHE.
UNCOMMITTED. Run (DeepSeek, proxy off):
  set -a; source "/Users/tongtao/Desktop/agent design/.env"; set +a; unset HTTP_PROXY HTTPS_PROXY
  LLM_API_KEY=$DEEPSEEK_API_KEY LLM_BASE_URL=https://api.deepseek.com/v1 AGENT_MODEL=deepseek-chat \
    PYTHONPATH=src .venv/bin/python evals/probe_learn.py
"""
import glob
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    os.environ["MEMAGENT_VAULT"] = tempfile.mkdtemp(prefix="learn-vault-")
    sk = tempfile.mkdtemp(prefix="learn-skills-")
    os.environ["MEMAGENT_SKILLS_DIR"] = sk
    model = os.environ.get("AGENT_MODEL", "deepseek-chat")

    from memagent.memory import make_memory, make_write_skill_tool
    from memagent.history import make_history_tool
    from memagent.code_grep import make_grep_tool
    from memagent.consolidate import build_learn_prompt
    from memagent.slice import Slice, make_build_slice, slice_sink, record_user, one_line
    from memagent.episode import make_episode_sink
    from memagent.events import AssistantText, ToolResult, make_dispatcher
    from memagent.retriever import NullRetriever
    from memagent.llm import OpenAILLM
    from memagent.loop import run_turn

    sid = "learn-probe"
    wd = tempfile.mkdtemp(prefix="learn-wd-")
    mem = make_memory()
    host = make_grep_host = __import__("memagent.tools", fromlist=["LocalToolHost"]).LocalToolHost(root=wd)
    host.registry.register(make_grep_tool(host))
    host.registry.register(make_history_tool(mem, sid))
    host.registry.register(make_write_skill_tool())
    state = Slice()
    tools_seen = []
    def cap(e):
        if isinstance(e, ToolResult):
            tools_seen.append(e.name)
    episodic = make_episode_sink(mem, session_id=sid, task_id_fn=lambda: "t", title_fn=lambda: one_line(state.goal, 80))
    dispatch = make_dispatcher(slice_sink(state), episodic, cap)
    llm = OpenAILLM(model=model, timeout=120.0); llm.set_cache_key(sid)
    build = make_build_slice(state, host, NullRetriever(), mem, "", session_id=sid)

    def turn(prompt):
        state.goal = prompt
        record_user(state, prompt)
        run_turn(build_slice=build, llm=llm, tools=host, dispatch=dispatch, max_steps=12)

    print(f"# /learn live probe · model={model} · vault={'memem' if getattr(mem,'is_durable',False) else 'null'}")
    # 1) a small real workflow → fills the episodic cache
    turn("Create greet.py with a function greet(name) that returns 'Hello, '+name, then run it with python "
         "to print greet('world').")
    # 2) /learn — the agent must read THIS session from the cache and author a skill via write_skill
    tools_seen.clear()
    turn(build_learn_prompt("the workflow we just did — making and running a small Python script"))

    skills = glob.glob(os.path.join(sk, "*", "SKILL.md"))
    print(f"called write_skill: {'write_skill' in tools_seen}  |  used recall_history: {'recall_history' in tools_seen}")
    print(f"skills written: {len(skills)}")
    ok = False
    if skills:
        body = open(skills[0]).read()
        ok = "provenance: user" in body and body.lstrip().startswith("---")
        print(f"  {os.path.relpath(skills[0], sk)}: provenance-user={('provenance: user' in body)} "
              f"valid-frontmatter={body.lstrip().startswith('---')}")
        print("  --- head ---"); print("  " + "\n  ".join(body.splitlines()[:12]))
    print(f"\n==================== /learn {'PASS' if ok else 'FAIL'} ====================")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
