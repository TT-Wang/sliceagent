// Shared setup for both agent loops (classic history vs. slice-native).
// Keeps the experiment focused on the LOOP, not the plumbing.
import OpenAI from "openai";
import fs from "fs";
import path from "path";
import { execSync } from "child_process";
import { HttpsProxyAgent } from "https-proxy-agent";

try {
  process.loadEnvFile(".env");
} catch {
  // no .env — fall back to shell env vars
}

export const proxyUrl =
  process.env.AGENT_PROXY ??
  process.env.HTTPS_PROXY ??
  process.env.HTTP_PROXY ??
  "http://127.0.0.1:7890";
export const useProxy = proxyUrl && proxyUrl !== "none";
const httpAgent = useProxy ? new HttpsProxyAgent(proxyUrl) : undefined;

export let client, MODEL, PROVIDER;
if (process.env.OPENAI_API_KEY) {
  client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY, httpAgent, timeout: 60_000, maxRetries: 2 });
  MODEL = process.env.AGENT_MODEL ?? "gpt-4o-mini";
  PROVIDER = "OpenAI";
} else if (process.env.MOONSHOT_API_KEY) {
  client = new OpenAI({ apiKey: process.env.MOONSHOT_API_KEY, baseURL: "https://api.moonshot.cn/v1", httpAgent, timeout: 60_000, maxRetries: 2 });
  MODEL = process.env.AGENT_MODEL ?? "moonshot-v1-8k";
  PROVIDER = "Kimi";
} else {
  console.error("Missing API key. Set OPENAI_API_KEY (or MOONSHOT_API_KEY).");
  process.exit(1);
}

const fn = (name, desc, props, req = []) => ({
  type: "function",
  function: { name, description: desc, parameters: { type: "object", properties: props, required: req } },
});

export const TOOLS = [
  fn("read_file", "Read a file and return its contents.", { path: { type: "string" } }, ["path"]),
  fn("list_files", "List files in a directory (defaults to current directory).", { path: { type: "string" } }),
  fn("edit_file", "Create or OVERWRITE an entire file (replaces ALL existing content). Use only for brand-new files.", { path: { type: "string" }, content: { type: "string" } }, ["path", "content"]),
  fn("append_to_file", "Append content to the end of a file (creates it if missing). Use to ADD to an existing file without overwriting it.", { path: { type: "string" }, content: { type: "string" } }, ["path", "content"]),
  fn("str_replace", "Replace a unique snippet in an existing file without rewriting the whole file. old_string must match exactly and occur exactly once.", { path: { type: "string" }, old_string: { type: "string" }, new_string: { type: "string" } }, ["path", "old_string", "new_string"]),
  fn("run_command", "Run a shell command and return its combined output (and exit code if it fails).", { command: { type: "string" } }, ["command"]),
];

export function runTool(name, input) {
  try {
    if (name === "read_file") return fs.readFileSync(input.path, "utf8");
    if (name === "list_files") return fs.readdirSync(input.path ?? ".").sort().join("\n") || "(empty)";
    if (name === "edit_file") {
      fs.mkdirSync(path.dirname(path.resolve(input.path)), { recursive: true });
      fs.writeFileSync(input.path, input.content);
      return `Wrote ${input.content.length} bytes to ${input.path}`;
    }
    if (name === "append_to_file") {
      fs.mkdirSync(path.dirname(path.resolve(input.path)), { recursive: true });
      fs.appendFileSync(input.path, input.content);
      return `Appended ${input.content.length} bytes to ${input.path}`;
    }
    if (name === "str_replace") {
      const cur = fs.readFileSync(input.path, "utf8");
      const n = cur.split(input.old_string).length - 1;
      if (n === 0) return `Error: old_string not found in ${input.path}`;
      if (n > 1) return `Error: old_string occurs ${n} times in ${input.path}; include more context to make it unique`;
      const i = cur.indexOf(input.old_string); // index-splice: no $-pattern interpretation
      const updated = cur.slice(0, i) + input.new_string + cur.slice(i + input.old_string.length);
      fs.writeFileSync(input.path, updated);
      return `Replaced 1 occurrence in ${input.path} (${cur.length} → ${updated.length} bytes)`;
    }
    if (name === "run_command") {
      try {
        const out = execSync(input.command, { encoding: "utf8", stdio: "pipe", timeout: 30_000 });
        return out.trim() || "(command produced no output)";
      } catch (e) {
        const out = [e.stdout, e.stderr].filter(Boolean).join("").trim();
        return `Exit code ${e.status ?? "?"}\n${out || e.message}`;
      }
    }
    return `Error: unknown tool "${name}"`;
  } catch (e) {
    return `Error: ${e.message}`;
  }
}

export const oneLine = (s, n = 80) => String(s ?? "").replace(/\s+/g, " ").trim().slice(0, n);

// --- lightweight token/time accounting for comparing loops ---
export const stats = { in: 0, out: 0, calls: 0 };
export function track(resp) {
  const u = resp?.usage;
  if (u) {
    stats.in += u.prompt_tokens || 0;
    stats.out += u.completion_tokens || 0;
    stats.calls += 1;
  }
  return resp;
}
export function printStats(startMs) {
  const sec = ((Date.now() - startMs) / 1000).toFixed(1);
  console.log(
    `\n[STATS] model:${MODEL} api_calls:${stats.calls} in_tokens:${stats.in} out_tokens:${stats.out} total_tokens:${stats.in + stats.out} wall:${sec}s`
  );
}
