<!-- Thanks for contributing to sliceagent! Keep PRs focused and small where possible. -->

## What & why
<!-- One or two sentences: what does this change and why? Link any issue (#123). -->

## Changes
-

## Checklist
- [ ] `bash scripts/run_tests.sh` passes locally
- [ ] `ruff check src/sliceagent tests evals` is clean
- [ ] Added/updated a regression test for the change
- [ ] No edits to verifiers/tests purely to make them pass (reward-hacking)
- [ ] **Moat respected:** no transcript accumulation / LLM-summarization in the loop; provider quirks stay
      behind the `LLMClient` contract in `llm.py`; changes are task- and provider-agnostic
      (see [CONTRIBUTING.md](../CONTRIBUTING.md))
