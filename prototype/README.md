# prototype — validated proof-of-concept (JavaScript)

The ~250-line experiment that validated memagent's core thesis before the Python build. Kept for reference; **not** the production codebase.

- `core.js` — shared substrate: provider-agnostic LLM client (proxy-aware), the tool definitions + executor (`read/list/edit/append/str_replace/run_command`), token/time accounting.
- `agent.js` — **classic** loop: append-only `history`, resent every turn (the baseline to beat).
- `agent-slice.js` — the **slice** loop: NO history; each turn rebuilds a bounded Active Memory Slice from deterministic tiers (task · verbatim error · counted action tally · recent · live file artifacts). One model call per turn.

## What it proved (controlled gpt-5.5 experiments, classic vs slice, identical tasks)

- Slice matches/beats classic on **wall-clock and tokens** across build, debug, and forced-sequential tasks, with **identical test pass rates**.
- The advantage **scales with turn count / accumulated content**: up to **−61% to −80% tokens and −71% wall-clock** on long iterative-debug tasks; near-parity on short or tiny-content ones.
- Key lessons baked into the architecture: typed tiers need per-type compaction (facts→dedup, attempts→count); a counted action-tally breaks repetition-blindness; live file artifacts (re-read fresh) are the working set; bounded context is only worth it for a model strong enough to re-orient each turn.

## Run

```bash
cd prototype
npm install
export OPENAI_API_KEY=sk-...        # or MOONSHOT_API_KEY
node agent-slice.js                 # the slice agent
node agent.js                       # the classic baseline
```

`AGENT_MODEL` overrides the model; `HIDE_SLICE=1` hides the per-turn slice dump.
