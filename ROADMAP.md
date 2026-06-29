# memagent — Launch Roadmap (demo → installable product)

**Bar for "launchable":** a stranger can discover it → install in one command → guided key setup →
run a real task → trust it → reproduce the flat-cost claim; and a contributor can PR into green CI.

**State:** engine + 106 tests green; clean `pip install` verified; startup crash fixed (`4c0d783`).
The gap is release engineering + a few moat-safe borrows from Kimi-Code / Hermes — **not** features.

**Merge rule (launch waves vs borrows):** a borrow joins the launch waves only if it's a *security/trust
gate*, a *real-use correctness defect*, or *overlaps a launch task*; otherwise it's post-launch (Wave 5).
Tags: `[launch]` OSS-readiness · `[borrow ★N/#N kimi|hermes]` from the cross-check. Effort S/M/L.

## Decisions only you can make
- [ ] Make the GitHub repo **public** (QUICKSTART step 1 fails until then)
- [ ] Publish **memem → PyPI** (gates `pip install memagent` + Homebrew; git-install works without it)
- [ ] Confirm **Hermes is MIT** (needed before signing off the verbatim ports)

## The launching kit — one command to install
- **Now (v0.1, no PyPI):** `curl -fsSL https://raw.githubusercontent.com/TT-Wang/memagent/main/install.sh | sh`
  → bootstraps `uv`, runs `uv tool install git+https://github.com/TT-Wang/memagent`, puts `memagent` on PATH.
  - Power users: `uv tool install git+…/memagent` · `pipx install git+…/memagent`
  - Container: `docker run -it ghcr.io/TT-Wang/memagent`
- **Fast-follow (v0.2, after memem on PyPI):** `pipx install memagent` · `uvx memagent` · `brew install TT-Wang/tap/memagent`
- Footprint is LIGHT (openai·httpx·numpy·mcp + memem's small deps + optional rich/prompt_toolkit).
  The 96 wheels / 150 MB at repo root are CRUFT from other tools — delete them.

---

## Wave 0 — Stop the bleeding · ~½ day
- [x] Startup `NameError` (banner used renamed `policy_mode`) — `cli.py:607` → `policy_label(_eff_mode)` `[launch] (4c0d783)`
- [ ] Sync **AGENT_POLICY docs** to baby-sitter / teenager(default) / let-it-go; stop calling `guard` the safe default — README.md:57,74 · .env.example:18 · memagent.toml.example:7 `[launch] S`
- [ ] **OpenClaw → OpenHands** — README.md:7 · docs/ARCHITECTURE.md:73 `[launch] S`
- [ ] **Delete the 96 root `*.whl`** (untracked cruft) + `.gitignore += *.whl, brain/, .mindmux/, scratch/, prototype/, runs/, .coverage, .DS_Store, .pytest_cache/, .ruff_cache/` `[launch] S`
- [ ] **`git add evals/realenv_multiturn.py`** (proof artifact, currently untracked) `[launch] S`
- [ ] Rebuild local `.venv` so `PYTHONPATH=src` workaround can be dropped `[launch] S`

## Wave 1 — Launching kit + honest docs · ~1–1.5 day  → *installable in one command*
- [ ] **`install.sh`** (detect/bootstrap uv → `uv tool install git+…` → PATH + uninstall) `[launch] M`
- [ ] **`Dockerfile`** + publish `ghcr.io/TT-Wang/memagent` `[launch] M`
- [ ] Pin **`memem@<sha>`** — pyproject.toml:13 `[launch] S`
- [ ] Fix **proxy default** (don't force `127.0.0.1:7890` for non-CN users) — llm.py:51 `[launch] S`
- [ ] Missing-key → **init wizard** (helpful path, not a stack trace) — config.py/cli.py key gate → onboarding.py `[launch] S`
- [ ] **README rewrite**: pitch · the one-command install · usage example · demo gif `[launch] M`
- [ ] **Config reference** auto-generated from `envspec.py` REGISTRY `[launch] M`
- [ ] per-provider `base_url` env override `[borrow #cfg hermes] S` · project-scoped config writes `[borrow #cfg kimi] S`

## Wave 2 — Trust & legal · ~½–1 day  → *credible & safe to run*
- [ ] `license = "MIT"` + classifier in pyproject; README `## License` `[launch] S`
- [ ] **NOTICE / THIRD-PARTY-LICENSES** mapping the ~13 Kimi/Hermes verbatim ports → sources `[launch] M`
- [ ] **SECURITY.md** + disclosure path (tongtao.wang@gmail.com) `[launch] S`
- [ ] **★3 MCP spawn-security screen** (egress/persistence/IOC check before spawn) — mcp_client.py:193 `[borrow ★3 hermes] S–M`
- [ ] **Persisted allow/deny/ask permission rules** (path/command scoped) as a PolicyChain entry — policy.py · hooks.py `[borrow #8 both] M`
- [ ] **★5 live-env tier**: re-surface running bg procs/terminals in the slice each turn — slice.py + procman.py/terminal.py `[borrow ★5 kimi] S`
- [ ] Document the safety model (sandbox local/docker, 3 modes) + warn on headless→let-it-go downgrade `[launch] M`

## Wave 3 — Prove it + relieve the slice · ~1 day
- [ ] One-command **flat-cost demo + chart** (the now-tracked realenv_multiturn) in README `[launch] M`
- [ ] Real **live-LLM smoke** on a fresh machine `[launch] M`
- [ ] **★1 bound `read_file`** (route through `_page_out` / add `offset`+`limit`) + `<system>` status footer — tools.py:739,708 `[borrow ★1 kimi] M`
- [ ] **★2 `glob` tool + enriched `grep`** (output_mode, -A/-B/-C, --type, mtime-sort) — tools.py · code_grep.py · agents.py:18 `[borrow ★2 kimi] M`
- [ ] **#12 self-updating model catalog** (context-window + pricing as data) — kills stale price table — model_catalog.py · tui.py `[borrow #12 hermes] M`
- [ ] quick-wins ride here: `replace_all` on str_replace · read-after-edit staleness hints · grep sensitive-file exclusion · abort-aware backoff + honor `Retry-After` `[borrow S]`

## Wave 4 — Ship + CI · ~1 day  → *PR → green CI → merge*
- [ ] **`.github/workflows/ci.yml`**: install + `install.sh` smoke + `memagent --help` + test loop + `ruff`, on ubuntu+macOS × py3.11/3.12 (would have caught the crash) `[launch] M`
- [ ] **`scripts/run_tests.sh`** wrapper (tally + non-zero exit) `[launch] M`
- [ ] `[tool.ruff]` + `[project.optional-dependencies] dev = [pytest, ruff]` `[launch] S`
- [ ] Single-source version (`[tool.hatch.version]` ← `__init__.py`) `[launch] S`
- [ ] CODE_OF_CONDUCT.md + ISSUE/PR templates; move design notes → `docs/design/` `[launch] S`
- [ ] **Tag `v0.1.0`** + dated CHANGELOG `[launch] S`

## Wave 5 — Depth borrows (post-launch, sub-sequenced by leverage)
**5a — high-leverage UX/perf**
- [ ] **★4 mid-turn steering** (ctrl+s → steer buffer into next slice rebuild) — tui.py · loop.py `[borrow ★4 kimi] M`
- [ ] **#10 within-turn supersede-superseded-reads** + proactive micro-compaction — slice.py `[borrow #10 both] M`
- [ ] richer footer (git/cwd/context%) + bg-task badges + rotating tips — tui.py `[borrow #5 kimi] S–M`

**5b — control + robustness**
- [ ] **#6 plan-mode** (propose→approve, read-only gate; reuse the 3 permission modes) `[borrow #6 kimi] M`
- [ ] **#7 Anthropic `cache_control`** breakpoints (finish the `_cache_kwargs` stub) — llm.py `[borrow #7 kimi] M`
- [ ] **#9 multi-checkpoint undo** + `/undo [count]` selector — recovery.py/session.py + tui.py `[borrow #9 both] M`
- [ ] **#11 skill-curator lifecycle** (stale→archive→prune AUTO skills) + pinning — consume skill_usage.py telemetry `[borrow #11 hermes] M`

**5c — extensibility / orchestration**
- [ ] **#13 background / detached subagents** (return task_id, summary next turn) — subagent.py `[borrow #13 kimi] L`
- [ ] **MCP HTTP/SSE transport** + per-server toggle + connection-status state machine — mcp_client.py `[borrow #12-skills kimi] M`
- [ ] richer skill frontmatter (real YAML) + skills → dynamic slash commands — skills.py `[borrow kimi] S–M`

---

## Where memagent already LEADS (do not regress)
Flat per-turn cost (reads re-observed from ground truth, not accumulated) · lossless zero-LLM compaction
(no summarizer-failure mode) · cross-session memory + consolidation · PageRank symbol-graph repo map +
`verify_cmd` lint-into-loop · strongest **edit** primitive (fuzzy-unique match, cat-n strip, atomic write
preserving mode+CRLF, strict-decode abort, binary hexdump) · subagent = slice applied recursively +
resource-access-based safe parallelism · secret-scrub / injection-scan re-injection stack · docker
`network=none` fail-closed · WAL crash recovery + resume-by-distillation · 3 planning tiers + SelfCheckHook
done-gate · `envspec` validated env registry.

## Moat-UNSAFE — explicitly skip
- **Resume/continue a prior subagent by id** (kimi) — needs the child's transcript persisted → drifts to accumulation.
- Adopting **transcript accumulation** or **LLM-summarization compaction** into the loop.
- Any **task-specific** or **provider-locked** heuristic (provider quirks stay in llm.py behind LLMClient).
