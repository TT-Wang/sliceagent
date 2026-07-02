"""Active-memory-slice monitor — a web view of EXACTLY what the LLM sees each turn.

The slice core's only output is the event stream (events.py). This module adds one more sink:
it captures every SliceBuilt (the full [system, user] the model receives) plus what the model
did with it (assistant text, tool calls, tokens, stop reason), and serves it to a tiny stdlib
HTTP page. No new dependencies, no provider coupling, no touch to the loop — a sink failure is
already contained by make_dispatcher, so the monitor can never break a run.

Wire it in any host:
    from sliceagent.monitor import start_monitor
    monitor, sink, url = start_monitor(context_fn=lambda: {"goal": s.goal, "topic": s.active_id})
    dispatch = make_dispatcher(..., sink)     # add the sink alongside the others
    print(url)                                # open in a browser

The store is task- and llm-agnostic: it shows whatever the slice is, for whatever model.
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .events import (
    AssistantText,
    Event,
    SliceBuilt,
    StepEnd,
    ToolResult,
    TurnEnd,
    TurnInterrupted,
)

_MAX_OUTPUT = 6000   # cap a single tool output in the snapshot (the page stays snappy)
_MAX_ARGS = 4000     # cap a single tool-args blob

# Invariant 0 — bound the observer. The slice is bounded; its monitor must be too.
_RING_CAP = 40       # serve only the last N steps; counters stay accurate from full tallies (MON1)


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + f"\n…[+{len(text) - n} chars]"


class SliceMonitor:
    """Thread-safe capture of the per-step slice timeline. The loop (one thread) writes via the
    sink; the HTTP server (another thread) reads via snapshot — both under one lock.

    BOUNDED (Invariant 0 / MON1): the in-memory step ring is capped at `cap` (last N steps); cumulative
    turn/step/token COUNTERS are kept from full tallies so the snapshot stays truthful while memory and
    serialized size stay O(N). `i` is a monotonic step id (not a list index) so it remains stable across
    trims and the UI can still select a step uniquely."""

    def __init__(self, context_fn=None, cap: int = _RING_CAP):
        self._ctx_fn = context_fn
        self._cap = max(1, cap)
        self._lock = threading.Lock()
        self._steps: list[dict] = []
        self._turn = 0
        self._step_in_turn = 0
        self._steps_total = 0       # full tally — survives ring trimming (MON1 counter)
        self._tokens = 0            # full token tally — survives ring trimming (MON1 counter)
        self._open = False          # is a turn currently in progress?
        self._cur: dict | None = None
        self._version = 0           # bumps on ANY mutation, so the page re-renders live detail

    def _ctx(self) -> dict:
        if self._ctx_fn is None:
            return {}
        try:
            return self._ctx_fn() or {}
        except Exception:
            return {}

    def sink(self, e: Event) -> None:
        with self._lock:
            if isinstance(e, SliceBuilt):
                if not self._open:                 # no TurnBegin event → first slice opens a turn
                    self._turn += 1
                    self._open = True
                    self._step_in_turn = 0
                self._step_in_turn += 1
                self._steps_total += 1             # monotonic full tally (survives ring trim)
                msgs = e.messages or []
                system = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")
                user = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"),
                            e.rendered)
                if isinstance(user, list):     # multimodal turn (text+image parts) → use the rendered text
                    user = e.rendered           # (mirrors loop.py) so detail render doesn't choke on a list
                ctx = self._ctx()
                self._cur = {
                    "i": self._steps_total - 1, "turn": self._turn, "step": self._step_in_turn,
                    "goal": ctx.get("goal", ""), "topic": ctx.get("topic", ""),
                    "system": system, "user": user, "assistant": "", "tools": [],
                    "usage": {}, "stop_reason": "", "interrupted": "",
                }
                self._steps.append(self._cur)
                # MON1: trim past the high-water mark — keep only the last `cap` steps. The live step
                # is always the last element, so trimming the front never drops `self._cur`.
                if len(self._steps) > self._cap:
                    del self._steps[:len(self._steps) - self._cap]
            elif isinstance(e, AssistantText):
                if self._cur is not None:
                    self._cur["assistant"] += e.content
            elif isinstance(e, ToolResult):
                if self._cur is not None:
                    self._cur["tools"].append({
                        "name": e.name, "args": _clip(json.dumps(e.args, ensure_ascii=False), _MAX_ARGS),
                        "output": _clip(e.output, _MAX_OUTPUT), "failing": e.failing})
            elif isinstance(e, StepEnd):
                if self._cur is not None:
                    cu = self._cur.setdefault("usage", {})   # ACCUMULATE across steps (don't overwrite, else the
                    for _k, _v in (e.usage or {}).items():    # turn snapshot shows only the LAST step's usage)
                        if isinstance(_v, (int, float)):
                            cu[_k] = cu.get(_k, 0) + _v
                    self._cur["stop_reason"] = e.stop_reason
                    u = e.usage or {}
                    self._tokens += u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
            elif isinstance(e, TurnEnd):
                self._open = False
                self._cur = None
            elif isinstance(e, TurnInterrupted):
                if self._cur is not None:
                    self._cur["interrupted"] = e.reason
                self._open = False
                self._cur = None
            else:
                return  # event we don't track → no version bump
            self._version += 1

    def snapshot(self) -> dict:
        with self._lock:
            # MON1: tokens/turns/steps_total come from the FULL tallies, NOT the trimmed ring — so the
            # counters stay accurate even though only the last `cap` steps are served.
            # deep-copy nested MUTABLES (tools list, usage dict): json.dumps runs outside the lock,
            # and the loop thread mutates the live step's tools list in place — a shallow dict(s) would
            # share it and risk "list changed size during iteration" mid-poll.
            steps = [{**s, "tools": [dict(t) for t in s["tools"]], "usage": dict(s["usage"])}
                     for s in self._steps]
            return {"version": self._version, "turns": self._turn, "steps_total": self._steps_total,
                    "tokens": self._tokens, "steps": steps}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):       # silence the default stderr access log
        pass

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/state":
            body = json.dumps(self.server.monitor.snapshot(), ensure_ascii=False).encode("utf-8")
            self._send(body, "application/json; charset=utf-8")
        else:
            self.send_response(404)
            self.end_headers()


def serve(monitor: SliceMonitor, host: str = "127.0.0.1", port: int = 7654):
    """Start the HTTP server on the first free port at/after `port`. Returns (server, url)."""
    last = None
    for p in range(port, port + 10):
        try:
            srv = ThreadingHTTPServer((host, p), _Handler)
            srv.monitor = monitor
            srv.daemon_threads = True
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            return srv, f"http://{host}:{p}"
        except OSError as exc:           # port busy → try the next
            last = exc
            continue
    raise RuntimeError(f"no free port in {port}..{port + 9}: {last}")


def start_monitor(context_fn=None, host: str = "127.0.0.1", port: int = 7654):
    """Convenience: build a monitor, start its server, return (monitor, sink, url)."""
    monitor = SliceMonitor(context_fn=context_fn)
    _srv, url = serve(monitor, host, port)
    return monitor, monitor.sink, url


# --- persistent monitor: a STANDING server decoupled from agent sessions --------------------
# A session writes its snapshot to <dir>/<session>.json each event; the standing server reads the
# dir and serves it, going idle when nothing's written recently. So the monitor survives sessions
# (it's a separate process) and shows whichever session is active — or idles — instead of dying.
IDLE_SECONDS = 12

# Invariant 0 — bound the observer on disk too.
_DEBOUNCE_MS = 250        # MON2: write at most this often on the hot path; settle points flush now
_SESSION_TTL_SECONDS = 24 * 3600   # MON3: prune session files older than this by mtime
_MAX_SESSION_FILES = 20            # MON3: and/or keep only the most-recent M (never the active one)


def _monitor_dir(d: str | None = None) -> str:
    d = d or os.environ.get("SLICEAGENT_MONITOR_DIR") or \
        os.path.join(os.path.expanduser("~"), ".sliceagent", "monitor")
    os.makedirs(d, exist_ok=True)
    return d


class _SnapshotWriter:
    """MON2: decouple json.dump+os.replace from the loop hot path.

    A single background daemon thread owns ALL disk I/O. The sink (loop thread) only publishes the
    latest snapshot into a single-slot coalescing ref and signals an Event — it never blocks on disk.
    The writer coalesces: if 5 events arrive while one write is in flight, only the LAST snapshot is
    written (no per-event O(N) writes). Debounced to at most one write per `_DEBOUNCE_MS`, except a
    `flush` (StepEnd/TurnEnd) forces an immediate write so settle points land promptly. Atomic
    (tmp + os.replace) and fully failure-contained — a monitor write can never break the loop."""

    def __init__(self, path: str, debounce_ms: int = _DEBOUNCE_MS):
        self._path = path
        self._debounce = debounce_ms / 1000.0
        self._lock = threading.Lock()
        self._pending: dict | None = None   # single-slot: only the freshest snapshot survives
        self._flush = False                 # the pending snapshot must be written immediately
        self._busy = False                  # a write is currently in flight on the writer thread
        self._wake = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, snap: dict, *, flush: bool = False) -> None:
        with self._lock:
            self._pending = snap            # coalesce — drop any older un-written snapshot
            if flush:
                self._flush = True
        self._wake.set()

    def _run(self) -> None:
        while True:
            self._wake.wait()
            if self._stop and self._pending is None:
                return
            with self._lock:
                snap, flush = self._pending, self._flush
                self._pending, self._flush = None, False
                self._busy = snap is not None
                self._wake.clear()
            if snap is not None:
                self._write(snap)
                with self._lock:
                    self._busy = False
            if self._stop and self._pending is None:
                return   # flushed the last pending after close() → exit now; the wake edge was consumed,
                         # so looping back to wake.wait() would park the thread forever (flush-then-stop).
            # debounce: pause so a burst of hot-path events coalesces into the NEXT single write.
            # A flush skips the pause so settle points (StepEnd/TurnEnd) land without added latency.
            if not flush and not self._stop:
                time.sleep(self._debounce)

    def _write(self, snap: dict) -> None:
        try:
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False)
            os.replace(tmp, self._path)
        except Exception:
            pass

    def drain(self, timeout: float = 2.0) -> bool:
        """Block until the single pending slot has been written (best-effort, for tests/shutdown).
        Returns True if drained within `timeout`. Never used on the loop hot path."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._pending is None and not self._busy:
                    return True
            time.sleep(0.005)
        with self._lock:
            return self._pending is None and not self._busy

    def close(self) -> None:
        with self._lock:
            self._stop = True
        self._wake.set()


def make_file_monitor_sink(session_id: str, context_fn=None, dir: str | None = None):
    """A monitor sink that PERSISTS the snapshot to <dir>/<session>.json, so a standing server
    (python -m sliceagent.monitor) can show it — decoupled from this process.

    MON2: disk I/O runs on a background daemon thread, never on the loop hot path; per-event writes
    coalesce and debounce, with an immediate flush on StepEnd/TurnEnd. MON1 keeps the snapshot O(N).
    Writes are atomic and failure-contained (a monitor write must never break the loop)."""
    monitor = SliceMonitor(context_fn=context_fn)
    path = os.path.join(_monitor_dir(dir), f"{session_id}.json")
    inner = monitor.sink
    writer = _SnapshotWriter(path)
    _settle = (StepEnd, TurnEnd, TurnInterrupted)   # the points worth a guaranteed prompt write

    def sink(event: Event) -> None:
        inner(event)
        try:
            snap = monitor.snapshot()
            snap["session"] = session_id
            writer.submit(snap, flush=isinstance(event, _settle))
        except Exception:
            pass
    sink.writer = writer       # expose for explicit close()/inspection; never required for correctness
    return sink


def _session_files(d: str):
    out = []
    try:
        for fn in os.listdir(d):
            if fn.endswith(".json"):
                try:
                    out.append((fn[:-5], os.path.getmtime(os.path.join(d, fn))))
                except OSError:
                    continue   # a file vanished mid-enumeration → skip it, don't truncate the session list
    except OSError:
        pass
    return sorted(out, key=lambda x: x[1], reverse=True)   # most-recently-active first


def _prune_sessions(d: str, ttl: float = _SESSION_TTL_SECONDS, keep: int = _MAX_SESSION_FILES):
    """MON3: drop stale session files (older than `ttl` by mtime) and/or all but the most-recent
    `keep`, then return the surviving list (most-recently-active first). NEVER deletes the freshest
    file — even if it's stale and over the cap — so the active/most-recent session is always served.
    Failure-contained: a delete error just leaves the file in place."""
    files = _session_files(d)
    if not files:
        return files
    now = time.time()
    survivors = []
    for idx, (sid, mtime) in enumerate(files):
        too_old = (now - mtime) > ttl
        over_cap = idx >= keep
        if idx > 0 and (too_old or over_cap):   # idx 0 is the freshest/active — always keep it
            try:
                os.remove(os.path.join(d, f"{sid}.json"))
                continue
            except OSError:
                pass                            # couldn't delete → keep serving it
        survivors.append((sid, mtime))
    return survivors


class _PersistentHandler(_Handler):
    """Serves the page + /api/state read from the monitor dir (freshest or ?session=), with an idle
    flag and a session list — so one standing server reflects whatever session is running, or idles."""

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif u.path == "/api/state":
            sel = (parse_qs(u.query).get("session") or [None])[0]
            self._send(json.dumps(self._state(sel), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        else:
            self.send_response(404)
            self.end_headers()

    def _state(self, sel):
        d = self.server.monitor_dir
        files = _prune_sessions(d)   # MON3: drop stale/over-cap files here (never the freshest/active)
        base = {"version": -1, "turns": 0, "steps_total": 0, "tokens": 0, "steps": []}
        if not files:
            return {**base, "idle": True, "session": None, "sessions": []}
        sessions = [s for s, _ in files]
        chosen = sel if sel in sessions else sessions[0]   # default = most-recently-active
        path = os.path.join(d, f"{chosen}.json")
        try:
            with open(path, encoding="utf-8") as fh:
                snap = json.load(fh)
            if not isinstance(snap, dict):   # a valid but non-dict JSON (list/number) would crash snap[...]=
                snap = dict(base)
            age = time.time() - os.path.getmtime(path)
        except Exception:
            snap, age = dict(base), IDLE_SECONDS + 1
        snap["idle"] = age > IDLE_SECONDS
        snap["age"] = round(age, 1)
        snap["session"] = chosen
        snap["sessions"] = sessions
        return snap


def main():
    """`python -m sliceagent.monitor` — the standing, persistent monitor server. Stays up across
    agent sessions; idle when none is running; populated when a session runs with AGENT_MONITOR=1."""
    d = _monitor_dir()
    _prune_sessions(d)   # MON3: clear stale/over-cap session files on startup (keeps the freshest)
    host = "127.0.0.1"
    port = int(os.environ.get("AGENT_MONITOR_PORT", "7654"))
    last = None
    for p in range(port, port + 10):
        try:
            srv = ThreadingHTTPServer((host, p), _PersistentHandler)
            srv.monitor_dir = d
            srv.daemon_threads = True
            print(f"sliceagent monitor (persistent) → http://{host}:{p}   ·   watching {d}", flush=True)
            print("idle until a session runs with AGENT_MONITOR=1; Ctrl-C to stop.", flush=True)
            srv.serve_forever()
            return
        except OSError as exc:
            last = exc
    raise RuntimeError(f"no free port in {port}..{port + 9}: {last}")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>sliceagent · active memory slice monitor</title>
<style>
  :root{
    --bg:#0d1117; --panel:#161b22; --panel2:#10151c; --border:#30363d; --fg:#e6edf3;
    --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --amber:#d29922; --red:#f85149;
    --purple:#bc8cff; --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.5 var(--mono)}
  header{display:flex;align-items:center;gap:16px;padding:10px 16px;background:var(--panel);
    border-bottom:1px solid var(--border);position:sticky;top:0;z-index:10}
  header h1{font-size:14px;margin:0;font-weight:600;letter-spacing:.2px}
  header .dot{width:8px;height:8px;border-radius:50%;background:var(--green);
    box-shadow:0 0 6px var(--green);display:inline-block;margin-right:6px}
  header .dot.stale{background:var(--muted);box-shadow:none}
  header .dot.idle{background:var(--amber);box-shadow:0 0 6px var(--amber)}
  .stats{display:flex;gap:14px;color:var(--muted);font-size:12px;margin-left:auto;align-items:center}
  .stats b{color:var(--fg);font-weight:600}
  label.follow{display:flex;align-items:center;gap:5px;cursor:pointer;color:var(--muted)}
  main{display:flex;height:calc(100vh - 45px)}
  nav{width:280px;min-width:280px;overflow:auto;border-right:1px solid var(--border);background:var(--panel2)}
  .turn{border-bottom:1px solid var(--border)}
  .turn-h{padding:7px 12px;color:var(--muted);font-size:11px;text-transform:uppercase;
    letter-spacing:.6px;position:sticky;top:0;background:var(--panel2)}
  .turn-h .g{color:var(--fg);text-transform:none;letter-spacing:0;font-size:12px;display:block;margin-top:2px}
  .step{padding:7px 12px 7px 22px;cursor:pointer;border-left:3px solid transparent;display:flex;
    gap:8px;align-items:baseline}
  .step:hover{background:#1b222c}
  .step.sel{background:#1f6feb22;border-left-color:var(--accent)}
  .step .n{color:var(--accent);font-weight:600}
  .step .meta{color:var(--muted);font-size:11px;margin-left:auto}
  .step .pill{font-size:10px;padding:0 5px;border-radius:8px;border:1px solid var(--border);color:var(--muted)}
  .step .pill.tool{color:var(--purple);border-color:#3a3050}
  .step .pill.end{color:var(--green);border-color:#1f3a26}
  .step .pill.err{color:var(--red);border-color:#3a1f1f}
  section.detail{flex:1;overflow:auto;padding:16px 20px}
  .empty{color:var(--muted);text-align:center;margin-top:80px}
  .dmeta{color:var(--muted);margin-bottom:12px;font-size:12px}
  .dmeta b{color:var(--fg)}
  .block{margin-bottom:16px;border:1px solid var(--border);border-radius:8px;overflow:hidden;background:var(--panel)}
  .block>.bh{padding:7px 12px;background:#11161d;border-bottom:1px solid var(--border);cursor:pointer;
    display:flex;align-items:center;gap:8px;font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted)}
  .block>.bh .c{margin-left:auto;color:var(--muted);font-size:11px;text-transform:none}
  .block>.body{padding:0}
  .block.collapsed>.body{display:none}
  .tier{border-top:1px dashed var(--border)}
  .tier:first-child{border-top:none}
  .tier-h{padding:6px 12px;color:var(--accent);font-weight:600;font-size:12px;background:#0f141b}
  .tier-h.err{color:var(--red)} .tier-h.thr{color:var(--amber)} .tier-h.est{color:var(--green)}
  .tier-h.mem{color:var(--purple)} .tier-h.now{color:var(--fg)}
  pre{margin:0;padding:8px 12px;white-space:pre-wrap;word-break:break-word;font:12px/1.5 var(--mono);color:#cdd9e5}
  .sys pre{color:#9fb0c3}
  .tool{border-top:1px solid var(--border)}
  .tool-h{padding:6px 12px;display:flex;gap:8px;align-items:center;background:#0f141b}
  .tool-h .tn{color:var(--purple);font-weight:600}
  .tool-h .fail{color:var(--red);font-size:11px}
  .asst pre{color:#e6edf3}
  .hint{color:var(--muted);font-size:11px;padding:6px 12px}
</style></head>
<body>
<header>
  <h1><span class="dot" id="dot"></span>sliceagent · active memory slice</h1>
  <span id="status" style="font-size:11px;color:var(--muted)">connecting…</span>
  <div class="stats">
    <select id="sess" title="session" style="background:var(--panel2);color:var(--fg);border:1px solid var(--border);border-radius:5px;padding:2px 6px;font:11px var(--mono)"></select>
    <span>turns <b id="s-turns">0</b></span>
    <span>steps <b id="s-steps">0</b></span>
    <span>tokens <b id="s-tok">0</b></span>
    <label class="follow"><input type="checkbox" id="follow" checked> follow latest</label>
  </div>
</header>
<main>
  <nav id="nav"></nav>
  <section class="detail" id="detail"><div class="empty">waiting for the first turn…<br>run a task in the agent.</div></section>
</main>
<script>
let STATE={steps:[],version:-1}, SEL=null, LASTVER=-1, PICK="";
document.addEventListener("change",e=>{ if(e.target.id==="sess"){ PICK=e.target.value; LASTVER=-1; SEL=null; } });
const $=id=>document.getElementById(id);
const esc=s=>(s||"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

function tierClass(h){
  h=h.toUpperCase();
  if(h.includes("CURRENT ERROR"))return"err";
  if(h.includes("OTHER OPEN THREADS"))return"thr";
  if(h.includes("YOUR NOTES")||h.includes("ESTABLISHED")||h.includes("ACTIVE SKILL"))return"est";
  if(h.includes("MEMORY"))return"mem";
  if(h.startsWith("# NOW"))return"now";
  return"";
}
// split the user slice into tier cards by leading "# HEADER" lines
function renderSlice(text){
  const lines=(text||"").split("\n"); let tiers=[],cur=null;
  for(const ln of lines){
    if(/^#\s+\S/.test(ln)){ cur={h:ln,body:[]}; tiers.push(cur); }
    else{ if(!cur){cur={h:"",body:[]};tiers.push(cur);} cur.body.push(ln); }
  }
  return tiers.map(t=>{
    const head=t.h?`<div class="tier-h ${tierClass(t.h)}">${esc(t.h)}</div>`:"";
    const body=t.body.join("\n").replace(/^\n+|\n+$/g,"");
    return `<div class="tier">${head}${body?`<pre>${esc(body)}</pre>`:""}</div>`;
  }).join("");
}
function block(title,inner,extra,collapsed){
  const id="b"+Math.random().toString(36).slice(2,8);
  return `<div class="block ${collapsed?'collapsed':''}" id="${id}">
    <div class="bh" onclick="document.getElementById('${id}').classList.toggle('collapsed')">
      <span>${title}</span><span class="c">${extra||""}</span></div>
    <div class="body">${inner}</div></div>`;
}
function renderDetail(s){
  if(!s){$("detail").innerHTML='<div class="empty">select a step</div>';return;}
  const u=s.usage||{}, tok=(u.prompt_tokens||0)+(u.completion_tokens||0);
  const stop=s.interrupted?`interrupted: ${s.interrupted}`:(s.stop_reason||"…");
  let h=`<div class="dmeta">turn <b>${s.turn}</b> · step <b>${s.step}</b> ·
     topic <b>${esc(s.topic||"—")}</b> · goal <b>${esc(s.goal||"—")}</b><br>
     stop <b>${esc(stop)}</b> · tokens <b>${tok}</b>
     (prompt ${u.prompt_tokens||0} / completion ${u.completion_tokens||0})</div>`;
  h+=block("⟶ ACTIVE MEMORY SLICE (user message — what the model reads)",
           renderSlice(s.user), s.user.length+" chars", false);
  h+=block("SYSTEM PROMPT (instructions + task)",
           `<div class="sys"><pre>${esc(s.system)}</pre></div>`, s.system.length+" chars", true);
  let did="";
  if(s.assistant) did+=`<div class="asst"><div class="hint">assistant text</div><pre>${esc(s.assistant)}</pre></div>`;
  for(const t of (s.tools||[])){
    did+=`<div class="tool"><div class="tool-h"><span class="tn">${esc(t.name)}</span>
       <span class="hint">${esc(t.args)}</span>${t.failing?'<span class="fail">● failed</span>':''}</div>
       <pre>${esc(t.output)}</pre></div>`;
  }
  if(!did) did='<div class="hint">no model output captured for this step yet…</div>';
  h+=block("⟵ WHAT THE MODEL DID (this step)", did, (s.tools||[]).length+" tool call(s)", false);
  $("detail").innerHTML=h;
}
function renderNav(){
  const byTurn={};
  for(const s of STATE.steps){ (byTurn[s.turn]=byTurn[s.turn]||[]).push(s); }
  let h="";
  for(const tn of Object.keys(byTurn).sort((a,b)=>a-b)){
    const steps=byTurn[tn], g=steps[0].goal||"";
    h+=`<div class="turn"><div class="turn-h">turn ${tn}<span class="g">${esc(g.slice(0,70))}</span></div>`;
    for(const s of steps){
      const u=s.usage||{}, tok=(u.prompt_tokens||0)+(u.completion_tokens||0);
      let pill="", cls="";
      if(s.interrupted){pill=s.interrupted;cls="err";}
      else if(s.stop_reason==="tool_use"){pill=(s.tools||[]).length+"⚒";cls="tool";}
      else if(s.stop_reason==="end_turn"){pill="done";cls="end";}
      else if(s.stop_reason){pill=s.stop_reason;cls="";}
      h+=`<div class="step ${SEL===s.i?'sel':''}" data-i="${s.i}">
        <span class="n">${s.step}</span>
        <span class="pill ${cls}">${esc(pill||"…")}</span>
        <span class="meta">${tok||""}</span></div>`;
    }
    h+="</div>";
  }
  $("nav").innerHTML=h;
  for(const el of document.querySelectorAll(".step")){
    el.onclick=()=>{ SEL=+el.dataset.i; renderNav(); renderDetail(stepById(SEL)); };
  }
}
// `i` is a MONOTONIC step id (stable across the capped ring), NOT an array index — look it up.
function stepById(id){ return (STATE.steps||[]).find(s=>s.i===id) || null; }
function syncSessions(d){
  const sel=$("sess"); if(!sel) return;
  const list=d.sessions||[];
  sel.style.display = list.length>1 ? "" : "none";
  const cur=list.join("|");
  if(sel.dataset.list!==cur){ sel.dataset.list=cur; sel.innerHTML=list.map(s=>`<option value="${s}">${s}</option>`).join(""); }
  if(d.session) sel.value=d.session;
}
async function poll(){
  try{
    const url="/api/state"+(PICK?("?session="+encodeURIComponent(PICK)):"");
    const d=await(await fetch(url)).json();
    const dot=$("dot"); dot.classList.remove("stale","idle");
    if(d.session===null){ dot.classList.add("idle"); $("status").textContent="idle — no sessions yet (run with AGENT_MONITOR=1)"; }
    else if(d.idle){ dot.classList.add("idle"); $("status").textContent=`idle · ${d.session} (last activity ${d.age}s ago)`; }
    else { $("status").textContent=`live · ${d.session}`; }
    syncSessions(d);
    $("s-turns").textContent=d.turns; $("s-steps").textContent=d.steps_total;
    $("s-tok").textContent=(d.tokens||0).toLocaleString();
    if(d.version!==LASTVER){
      LASTVER=d.version; STATE=d;
      if($("follow").checked && d.steps.length) SEL=d.steps[d.steps.length-1].i;
      renderNav();
      renderDetail(SEL!=null?stepById(SEL):null);
    }
  }catch(e){ $("dot").classList.add("stale"); $("status").textContent="disconnected"; }
}
setInterval(poll,1000); poll();
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
