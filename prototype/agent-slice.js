#!/usr/bin/env node
// SLICE-NATIVE loop, v0.6 — ONE LLM call per turn (matches classic's call rate).
//
// v0.5 made a 2nd out-of-band call every turn (updateState) to maintain plan/facts/next.
// Measured cost: 34 API calls vs classic's 20, and 3.3× the wall-clock — the updater was
// the bottleneck. v0.6 DELETES it. The slice is now built entirely from DETERMINISTIC
// tiers (zero extra LLM calls), which the earlier experiments already made reliable:
//   TASK (serves as the checklist) · CURRENT ERROR (verbatim, auto-cleared) ·
//   ACTION HISTORY (counted tally, anti-loop) · RECENT ACTIONS · OPEN FILES (live artifacts).
// Progress is read straight from OPEN FILES instead of an LLM-maintained plan.
// Net: 1 call/turn, bounded context, no separate "miner" — fast AND flat.
import readline from "readline";
import fs from "fs";
import { client, MODEL, PROVIDER, useProxy, proxyUrl, TOOLS, runTool, oneLine, track, printStats } from "./core.js";

const K = Number(process.env.SLICE_WINDOW ?? 4);              // recent raw steps kept verbatim
const MAX_STEPS = Number(process.env.SLICE_MAX ?? 40);        // safety cap on loop iterations
const MAX_ARTIFACTS = Number(process.env.SLICE_ARTIFACTS ?? 4); // # of active files inlined
const MAX_ARTIFACT_CHARS = 1500;                              // per-file cap (bounds the slice)
const LOG_FILE = "scratch/durable-log.jsonl";                // the unbounded record (on disk)

const SLICE_SYSTEM_PROMPT =
  "You are a coding agent driven by an ACTIVE MEMORY SLICE (reconstructed state, not chat history). " +
  "Each turn, advance the TASK. OPEN FILES = the live file contents and your GROUND TRUTH; base edits on it, never on remembered contents. " +
  "Editing: edit_file overwrites a whole file (new files only); append_to_file adds; str_replace replaces an exact snippet copied from OPEN FILES. Test files must import what they test. " +
  "If an action is REPEATEDLY FAILING, stop repeating it — read the file, fix the root cause, then re-run. " +
  "Work in as FEW turns as possible: each turn make ALL edits you can already determine (batch many tool calls), then run once to verify. " +
  "Never write commentary, explanation, or reasoning as text while working — call tools SILENTLY with empty message content. Output text ONLY once, as a one-line final summary, and only after the TASK is fully done and tests pass (then make no tool call).";

// --- track which files are "active" (most-recently touched), bounded ---
function touchFile(ctx, p) {
  if (!p) return;
  ctx.activeFiles = ctx.activeFiles.filter((x) => x !== p);
  ctx.activeFiles.push(p);
  if (ctx.activeFiles.length > MAX_ARTIFACTS) ctx.activeFiles = ctx.activeFiles.slice(-MAX_ARTIFACTS);
}

// --- ARTIFACTS: re-read the current contents of active files FRESH every turn ---
function renderArtifacts(ctx) {
  if (!ctx.activeFiles.length) return "(no files opened yet)";
  return ctx.activeFiles
    .map((p) => {
      let body;
      try {
        body = fs.readFileSync(p, "utf8");
      } catch {
        return `### ${p}\n(not created yet)`;
      }
      const shown =
        body.length > MAX_ARTIFACT_CHARS
          ? body.slice(0, MAX_ARTIFACT_CHARS - 500) + "\n…[middle truncated]…\n" + body.slice(-500) // keep the tail (recent appends)
          : body;
      return `### ${p} (${body.length} bytes — current contents)\n\`\`\`\n${shown}\n\`\`\``;
    })
    .join("\n\n");
}

function actionSig(name, input) {
  if (name === "run_command") return `run_command \`${oneLine(input.command, 50)}\``;
  if (name === "edit_file") return `edit_file ${input.path}`;
  if (name === "append_to_file") return `append_to_file ${input.path}`;
  if (name === "str_replace") return `str_replace ${input.path}`;
  if (name === "read_file") return `read_file ${input.path}`;
  if (name === "list_files") return `list_files ${input.path ?? "."}`;
  return name;
}

function renderActionHistory(actionLog) {
  // This tier exists for anti-loop: surface only REPEATED or FAILING actions.
  // One-off successes are already visible in RECENT ACTIONS / OPEN FILES, so listing
  // them just bloats the slice (and grows unbounded for varying-arg commands).
  const entries = Object.entries(actionLog).filter(([, a]) => a.count >= 2 || a.failing);
  if (!entries.length) return "- (nothing repeated or failing)";
  return entries
    .map(([sig, a]) => {
      const warn = a.failing
        ? a.count >= 3
          ? "  ⚠ REPEATEDLY FAILING — STOP repeating; read the file & fix the root cause"
          : "  (failing)"
        : "";
      return `- ${sig} ×${a.count}${warn} → ${a.lastOutcome}`;
    })
    .join("\n");
}

// Build the VOLATILE part of the slice (the user message) — 100% deterministic, no LLM upkeep.
// The stable parts (system instructions + the task) live in the system message so they stay in
// the cacheable prefix instead of being re-rendered here every turn.
function renderSlice(recent, actionLog, artifacts, lastError) {
  const steps = recent.length
    ? recent.map((s, i) => `${i + 1}. ${s.action}\n     → ${s.observation}`).join("\n")
    : "(none yet — first move)";
  const errBlock = lastError ? `# ERROR (fix this, verbatim)\n${lastError}\n\n` : "";
  return [
    errBlock +
    `# REPEATED/FAILING ACTIONS`, renderActionHistory(actionLog), ``,
    `# RECENT (last ${K})`, steps, ``,
    `# OPEN FILES (live — edit based on this)`, artifacts, ``,
    `# NOW: do the next step(s) with tools, or a one-line summary if the task is fully done and tests pass.`,
  ].join("\n");
}

// The unbounded record — appended to disk, NOT carried in the model's context.
function logDurable(entry) {
  try {
    fs.mkdirSync("scratch", { recursive: true });
    fs.appendFileSync(LOG_FILE, JSON.stringify(entry) + "\n");
  } catch {}
}

function printSlice(rendered, ctx, logCount) {
  if (process.env.HIDE_SLICE) return;
  console.log(`\n  ┌─ WHAT THE LLM SEES THIS TURN ─────────────────────────`);
  console.log(`  │ # TASK (stable, in system prompt): ${oneLine(ctx.goal, 80)}`);
  rendered.split("\n").forEach((l) => console.log(`  │ ${l}`));
  console.log(`  ├───────────────────────────────────────────────────────`);
  console.log(
    `  │ actions:${Object.keys(ctx.actionLog).length} files:${ctx.activeFiles.length} recent:${Math.min(ctx.recent.length, K)} err:${ctx.lastError ? "yes" : "no"}  ·  slice ${rendered.length} chars  ·  durable log: ${logCount} entries`
  );
  console.log(`  └───────────────────────────────────────────────────────\n`);
}

async function agentLoopSlice(userMessage, ctx, logRef) {
  ctx.goal = userMessage;
  ctx.recent = [];
  ctx.actionLog = {};
  ctx.activeFiles = [];
  ctx.lastError = "";

  // Stable per-task system message (instructions + task) → stays in the provider's cacheable prefix
  // instead of being re-rendered in the volatile user message every turn.
  const systemContent =
    SLICE_SYSTEM_PROMPT +
    "\n\n# TASK (your checklist — do the next item that OPEN FILES shows is not done)\n" +
    ctx.goal;

  for (let step = 0; step < MAX_STEPS; step++) {
    const artifacts = renderArtifacts(ctx); // FRESH read of active files every turn
    const rendered = renderSlice(ctx.recent.slice(-K), ctx.actionLog, artifacts, ctx.lastError);
    printSlice(rendered, ctx, logRef.count);

    let choices;
    process.stdout.write("…thinking\r");
    try {
      const resp = track(await client.chat.completions.create({
        model: MODEL,
        tools: TOOLS,
        tool_choice: "auto",
        // ⭐ exactly ONE call per turn, exactly two messages — no history, no 2nd updater call.
        messages: [
          { role: "system", content: systemContent },
          { role: "user", content: rendered },
        ],
      }));
      choices = resp.choices;
    } catch (e) {
      const code = e.cause?.code || e.code || "";
      console.log(`\n[error reaching ${PROVIDER}] ${e.message}${code ? ` (${code})` : ""}`);
      if (/ETIMEDOUT|ECONNREFUSED|ENOTFOUND|UND_ERR|Connection/i.test(`${code} ${e.message}`)) {
        console.log(`Network blocked. Proxy in use: ${useProxy ? proxyUrl : "(none)"} — is ClashX running?`);
      }
      return;
    } finally {
      process.stdout.write("         \r");
    }

    const msg = choices[0].message;
    logRef.count++;
    logDurable({ role: "assistant", content: msg.content, tool_calls: msg.tool_calls });
    if (msg.content) console.log(`\nAssistant: ${msg.content}`);
    if (!msg.tool_calls?.length) return; // done — nothing was ever appended to a history array

    for (const tc of msg.tool_calls) {
      let input;
      try {
        input = JSON.parse(tc.function.arguments);
      } catch {
        ctx.recent.push({ action: `${tc.function.name}(bad args)`, observation: "Error: invalid JSON arguments" });
        continue;
      }
      console.log(`  → ${tc.function.name}(${tc.function.arguments})`);
      const out = String(runTool(tc.function.name, input));
      console.log(`  ← ${out.slice(0, 120)}`);

      if (input.path) touchFile(ctx, input.path); // mark file active → its contents enter the slice

      ctx.recent.push({ action: `${tc.function.name}(${oneLine(tc.function.arguments, 60)})`, observation: oneLine(out, 200) });

      const failing = /^(Error|Exit code)/.test(out);
      // deterministic error tier: capture verbatim on failure (head+tail), clear on a clean run
      if (failing) ctx.lastError = out.length > 800 ? out.slice(0, 120) + "\n…[trace truncated]…\n" + out.slice(-680) : out;
      else if (tc.function.name === "run_command") ctx.lastError = "";

      const sig = actionSig(tc.function.name, input);
      const prev = ctx.actionLog[sig] ?? { count: 0 };
      ctx.actionLog[sig] = { count: prev.count + 1, failing, lastOutcome: oneLine(out, 80) };

      logRef.count++;
      logDurable({ role: "tool", name: tc.function.name, args: input, full: out });
    }

    ctx.recent = ctx.recent.slice(-K); // keep the raw window bounded
  }
  console.log(`\n[stopped: hit MAX_STEPS=${MAX_STEPS}]`);
}

async function main() {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const ctx = { goal: "", recent: [], actionLog: {}, activeFiles: [], lastError: "" };
  const logRef = { count: 0 };
  const start = Date.now();
  try {
    fs.rmSync(LOG_FILE, { force: true });
  } catch {}

  const activeKey = process.env.OPENAI_API_KEY || process.env.MOONSHOT_API_KEY || "";
  console.log(`Slice-Native Agent v0.6 (${PROVIDER} · ${MODEL})  [NO history — single-call deterministic slice]`);
  console.log(`key: …${activeKey.slice(-4)}   proxy: ${useProxy ? proxyUrl : "direct (no proxy)"}   window K=${K}`);
  console.log(`type "exit" or press Ctrl+C to quit\n`);
  process.stdout.write("You: ");

  for await (const line of rl) {
    const m = line.trim();
    if (m === "exit" || m === "quit") break;
    if (m) await agentLoopSlice(m, ctx, logRef);
    process.stdout.write("You: ");
  }
  printStats(start);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
