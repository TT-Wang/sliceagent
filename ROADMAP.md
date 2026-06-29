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

## Wave 0 — Stop the bleeding · ~½ day  ✅ DONE (50648b3; repo moved to ~/code/memagent)
- [x] Startup `NameError` (banner used renamed `policy_mode`) — `cli.py:607` → `policy_label(_eff_mode)` `(4c0d783)`
- [x] Sync **AGENT_POLICY docs** to baby-sitter / teenager(default) / let-it-go; stop calling `guard` the safe default — README · .env.example · memagent.toml.example `(50648b3)`
- [x] **OpenClaw → OpenHands** — README.md:7 · docs/ARCHITECTURE.md:73 `(50648b3)`
- [x] **Deleted the 96 root `*.whl`** + `.gitignore += *.whl, brain/, .mindmux/` `(50648b3)`
- [x] **Tracked `evals/realenv_multiturn.py`** (proof artifact) `(50648b3)`
- [ ] Rebuild local `.venv` so `PYTHONPATH=src` workaround can be dropped `[launch] S` — *(optional; deferred)*
- [ ] **Make the GitHub repo public** `[you]`

## Wave 1 — Launching kit + honest docs · ~1–1.5 day  → *installable in one command*  (mostly done, 72c3ec1)
- [x] **`install.sh`** (bootstrap uv → `uv tool install "memagent[tui] @ git+…"` → PATH + `--uninstall`) `(72c3ec1)`
- [x] **`Dockerfile`** + `.dockerignore` (publish `ghcr.io/tt-wang/memagent` at release) `(72c3ec1)`
- [x] Pin **`memem@2705e7d`** — pyproject.toml `(72c3ec1)`
- [x] Fix **proxy default** — probe a local proxy; DIRECT for non-CN users — llm.py `(72c3ec1)`
- [x] Missing-key → **init wizard** — already routed (cli.py:151–152), verified `(72c3ec1)`
- [x] **README `## Install`** — the one-command install + uv/pipx/docker alternatives `(72c3ec1)`
- [ ] **README pitch polish + demo gif/asciinema** `[launch] M` — *(remaining)*
- [ ] **Config reference** auto-generated from `envspec.py` REGISTRY `[launch] M` — *(remaining)*
- [ ] per-provider `base_url` env override `[borrow hermes] S` · project-scoped config writes `[borrow kimi] S` — *(remaining)*

## Wave 2 — Trust & legal · ~½–1 day  → *credible & safe to run*  (core done, 033afa9)
- [x] `license = "MIT"` + license-files + classifiers in pyproject; README `## License` `(033afa9)`
- [x] **NOTICE** — Hermes MIT text + file map; Kimi credited as reimplemented patterns (Hermes confirmed MIT) `(033afa9)`
- [x] **SECURITY.md** + disclosure path + 4-layer threat model `(033afa9)`
- [x] **★3 MCP spawn-security screen** (egress/persistence shapes refused before spawn) — mcp_security.py + mcp_client.py `(033afa9)`
- [x] Warn on headless→let-it-go policy downgrade (no silent weakening) — cli.py `(033afa9)`
- [ ] **Persisted allow/deny/ask permission rules** (path/command scoped) as a PolicyChain entry — policy.py · hooks.py `[borrow #8 both] M` — *(remaining: bigger trust borrow)*
- [ ] **★5 live-env tier**: re-surface running bg procs/terminals in the slice each turn — slice.py + procman.py/terminal.py `[borrow ★5 kimi] S` — *(remaining)*

## Wave 3 — Prove it + relieve the slice · ~1 day  (read-surface borrows done: 31fd5ee, 77dbcde)
- [x] **★1 bound `read_file`** — default view cap + `offset`/`limit` window + `<system>` footer; blobs exempt `(31fd5ee)`
- [x] **★2 `glob` tool + enriched `grep`** (output_mode, context, --type, mtime-sort, brace globs) `(77dbcde)`
- [x] **Flat-cost demo + chart** — render_cost_chart ASCII chart in the demo + README proof chart `(c5cb89f)`
- [x] **#12 model pricing single-sourced** in model_catalog (cost meter delegates) `(f50550b)`
- [x] quick-wins: `replace_all` on str_replace · read-after-edit staleness hint · honor `Retry-After` `(dd0af8c)`
- [ ] Real **live-LLM smoke** on a fresh machine `[launch] M` — *(needs your API key; handed off)*
- [ ] *(deferred to Wave 5)* abort-aware retry sleep — needs the abort signal threaded into errors.with_retry

## Wave 4 — Ship + CI · ~1 day  → *PR → green CI → merge*  (done: 94af7c8)
- [x] **`.github/workflows/ci.yml`** — install + `memagent version` smoke + `ruff` + test loop, ubuntu+macOS × py3.11/3.12 `(94af7c8)`
- [x] **`scripts/run_tests.sh`** wrapper (tally + non-zero exit) `(94af7c8)`
- [x] `[tool.ruff]` config + `[dev]` extra; ruff-clean across src/tests/evals (~40 unused imports removed) `(94af7c8)`
- [x] Single-source version (`[tool.hatch.version]` ← `__init__.py`) `(94af7c8)`
- [x] CODE_OF_CONDUCT.md + ISSUE/PR templates `(94af7c8)`
- [ ] **Tag `v0.1.0`** — deferred to release (after repo public + merge to main); CHANGELOG entries staged under [Unreleased]
- [ ] *(optional)* move internal design notes → `docs/design/` `[launch] S`

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
