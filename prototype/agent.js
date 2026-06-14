#!/usr/bin/env node
// CLASSIC loop: keeps an append-only `history` array, resent in full every turn.
import readline from "readline";
import { client, MODEL, PROVIDER, useProxy, proxyUrl, TOOLS, runTool, oneLine, track, printStats } from "./core.js";

// Pretty-print the whole history array so you can watch it grow. Enable with SHOW_HISTORY=1.
function printHistory(history) {
  if (!process.env.SHOW_HISTORY) return;
  console.log(`\n  ┌─ history now has ${history.length} messages ─────────────`);
  history.forEach((m, i) => {
    let preview;
    if (m.role === "assistant" && m.tool_calls?.length) {
      const calls = m.tool_calls.map((t) => `${t.function.name}(${oneLine(t.function.arguments, 45)})`).join(", ");
      preview = (m.content ? `"${oneLine(m.content, 25)}" + ` : "") + `calls → ${calls}`;
    } else if (m.role === "tool") {
      preview = `result ← ${oneLine(m.content)}`;
    } else {
      preview = oneLine(m.content);
    }
    console.log(`  │ [${i}] ${m.role.padEnd(9)} ${preview}`);
  });
  console.log(`  └────────────────────────────────────────────\n`);
}

async function agentLoop(userMessage, history) {
  history.push({ role: "user", content: userMessage });
  while (true) {
    let choices;
    process.stdout.write("…thinking\r");
    try {
      const resp = track(await client.chat.completions.create({
        model: MODEL,
        tools: TOOLS,
        tool_choice: "auto",
        messages: history,
      }));
      choices = resp.choices;
    } catch (e) {
      const code = e.cause?.code || e.code || "";
      console.log(`\n[error reaching ${PROVIDER}] ${e.message}${code ? ` (${code})` : ""}`);
      if (/ETIMEDOUT|ECONNREFUSED|ENOTFOUND|UND_ERR|Connection/i.test(`${code} ${e.message}`)) {
        console.log(`Network blocked. Proxy in use: ${useProxy ? proxyUrl : "(none)"} — is ClashX running and set to a working node?`);
      }
      return;
    } finally {
      process.stdout.write("         \r");
    }
    const msg = choices[0].message;
    history.push(msg);
    printHistory(history);
    if (msg.content) console.log(`\nAssistant: ${msg.content}`);
    if (!msg.tool_calls?.length) return;

    for (const tc of msg.tool_calls) {
      let input;
      try {
        input = JSON.parse(tc.function.arguments);
      } catch {
        history.push({ role: "tool", tool_call_id: tc.id, content: "Error: invalid JSON arguments" });
        continue;
      }
      console.log(`  → ${tc.function.name}(${tc.function.arguments})`);
      const out = runTool(tc.function.name, input);
      console.log(`  ← ${String(out).slice(0, 120)}`);
      history.push({ role: "tool", tool_call_id: tc.id, content: String(out) });
    }
    printHistory(history);
  }
}

const SYSTEM_PROMPT =
  "You are a coding assistant. Use tools to read, list, edit files, and run shell commands — never guess file contents. " +
  "Always call a tool when the task involves the filesystem. After writing code, run it with run_command to verify it works.";

async function main() {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const history = [{ role: "system", content: SYSTEM_PROMPT }];
  const start = Date.now();

  const activeKey = process.env.OPENAI_API_KEY || process.env.MOONSHOT_API_KEY || "";
  console.log(`Mini Code Agent (${PROVIDER} · ${MODEL})  [classic history loop]`);
  console.log(`key: …${activeKey.slice(-4)}   proxy: ${useProxy ? proxyUrl : "direct (no proxy)"}`);
  console.log(`type "exit" or press Ctrl+C to quit\n`);
  process.stdout.write("You: ");

  for await (const line of rl) {
    const msg = line.trim();
    if (msg === "exit" || msg === "quit") break;
    if (msg) await agentLoop(msg, history);
    process.stdout.write("You: ");
  }
  printStats(start);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
