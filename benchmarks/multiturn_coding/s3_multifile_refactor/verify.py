"""Independent oracle for the eventbus multi-file refactor.

Checks BEHAVIOR with fresh inputs (a brand-new handler/middleware defined HERE,
not in any seed file the agent edits), plus structural invariants:
  (A) the OLD api (handle/safe_handle, old middleware/dispatcher signatures) is
      gone everywhere in the package;
  (B) the NEW context parameter is genuinely threaded and USED (not stubbed):
      correlation_id reaches handlers, middleware sees it, and context.depth is
      incremented per middleware layer;
  (C) distractor files are unchanged.

This file is NOT one the agent is asked to touch, and the behavioral portion
imports the real package and drives it with handlers defined inside verify().
"""
import ast
import os
import re
import subprocess
import sys


def _pkg(workdir):
    return os.path.join(workdir, "eventbus")


def _read(path):
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Behavioral test: run in a subprocess against the real package with FRESH
# handlers + a fresh middleware defined right here.
# ---------------------------------------------------------------------------
_BEHAVIOR = r'''
import sys, json
sys.path.insert(0, {workdir!r})

from eventbus import EventBus, Event, Handler
from eventbus.middleware import Middleware, LoggingMiddleware, CountingMiddleware
from eventbus.handlers import AuditHandler, EmailHandler, MetricsHandler, WebhookHandler

out = {{}}

# --- Fresh handler defined in the verifier (agent never saw this) ---
class ProbeHandler(Handler):
    def __init__(self):
        self.seen = []
    def process(self, event, context):
        # Must receive a context with a correlation_id and a depth.
        self.seen.append((event.topic, context.correlation_id, context.depth))
        return {{"cid": context.correlation_id, "depth": context.depth,
                "payload": dict(event.payload)}}

# --- Fresh middleware that asserts it receives + forwards context ---
class ProbeMiddleware(Middleware):
    def __init__(self):
        self.observed = []
    def dispatch(self, handler, event, context, next_call):
        self.observed.append((handler.name(), context.correlation_id, context.depth))
        return next_call(handler, event, context)

logging_mw = LoggingMiddleware()
counting_mw = CountingMiddleware()
probe_mw = ProbeMiddleware()

bus = EventBus(middlewares=[probe_mw, logging_mw, counting_mw])

probe = ProbeHandler()
audit = AuditHandler()
email = EmailHandler()
metrics = MetricsHandler()
webhook = WebhookHandler()

bus.subscribe("order.placed", probe)
bus.subscribe("order.placed", audit)
bus.subscribe("order.placed", email)
bus.subscribe("order.placed", metrics)
bus.subscribe("order.placed", webhook)
bus.subscribe("order.shipped", metrics)

r1 = bus.publish(Event("order.placed", {{"user": "ann", "to": "ann@x.io"}}), region="eu")
r2 = bus.publish(Event("order.placed", {{"user": "bob", "to": "bob@x.io"}}))
r3 = bus.publish(Event("order.shipped", {{"user": "ann"}}))

# publish must still return one result per handler
out["r1_len"] = len(r1)            # 5 handlers on order.placed
out["r3_len"] = len(r3)            # 1 handler on order.shipped

# probe received a real context each time
out["probe_seen"] = probe.seen

# correlation ids must differ across the two publishes
cid1 = probe.seen[0][1]
cid2 = probe.seen[1][1]
out["cids_distinct"] = (cid1 != cid2)

# depth must equal number of middlewares (3) at the handler/terminal
out["probe_depths"] = sorted(set(s[2] for s in probe.seen))

# middleware observed context with SAME correlation id the handler saw,
# and a SMALLER depth (it is outside the terminal).
out["probe_mw_observed"] = probe_mw.observed
mw_cids = set(o[1] for o in probe_mw.observed)
out["mw_cid_match"] = (cid1 in mw_cids and cid2 in mw_cids)

# concrete handlers must USE the context, not stub it
out["audit_trail"] = audit.trail
out["email_sent"] = email.sent
out["metrics_last_cid"] = getattr(metrics, "last_cid", "MISSING")
out["webhook_queued"] = webhook.queued

# LoggingMiddleware must now record triples incl. correlation id
out["logging_log"] = logging_mw.log
out["counting_count"] = counting_mw.count

print("JSON_START")
print(json.dumps(out, default=str))
'''


def _run_behavior(workdir):
    script = _BEHAVIOR.format(workdir=workdir)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        return None, "behavioral test raised:\n" + (proc.stderr or proc.stdout)
    out = proc.stdout
    if "JSON_START" not in out:
        return None, "behavioral test produced no JSON:\n" + out
    import json
    blob = out.split("JSON_START", 1)[1].strip()
    try:
        return json.loads(blob), ""
    except Exception as e:  # noqa: BLE001
        return None, "could not parse behavioral JSON (%r):\n%s" % (e, blob)


def _check_behavior(workdir):
    data, err = _run_behavior(workdir)
    if data is None:
        return False, err

    if data.get("r1_len") != 5:
        return False, "publish should return 5 results for order.placed, got %r" % (data.get("r1_len"),)
    if data.get("r3_len") != 1:
        return False, "publish should return 1 result for order.shipped, got %r" % (data.get("r3_len"),)

    if not data.get("cids_distinct"):
        return False, "two publishes produced the same correlation_id; Context not per-publish"

    depths = data.get("probe_depths")
    if depths != [3]:
        return False, ("handler/terminal must see context.depth == number of middlewares (3); "
                       "got probe depths %r. (child() must be used per middleware layer)" % (depths,))

    if not data.get("mw_cid_match"):
        return False, "ProbeMiddleware did not observe the same correlation_id the handler saw"

    # middleware depth must be strictly less than the terminal depth -> proves
    # context.child() is applied as the chain descends, not a constant context.
    mw_obs = data.get("probe_mw_observed") or []
    if not mw_obs:
        return False, "ProbeMiddleware.dispatch never received a context"
    mw_depths = sorted(set(o[2] for o in mw_obs))
    if any(d >= 3 for d in mw_depths):
        return False, ("middleware saw depth >= terminal depth (%r); context.child() "
                       "is not being threaded through the layers" % (mw_depths,))

    # ------------------------------------------------------------------
    # Ground-truth correlation ids come from an INDEPENDENT observer: the
    # LoggingMiddleware, which records (name, topic, correlation_id) for every
    # dispatch it wraps. The concrete handlers' recorded cids must EQUAL the
    # ids that were actually threaded for the same dispatches (not a hardcoded
    # constant). Correlation ids are allocated by a live counter at publish
    # time, so a static stub cannot reproduce them.
    log = data.get("logging_log") or []
    if not log:
        return False, "LoggingMiddleware recorded nothing"
    for entry in log:
        if not (isinstance(entry, (list, tuple)) and len(entry) == 3):
            return False, "LoggingMiddleware.log entries must be (name, topic, correlation_id) triples; got %r" % (entry,)

    # Build name -> ordered list of the correlation ids the middleware saw.
    truth = {}
    for name, _topic, cid in log:
        truth.setdefault(name, []).append(int(cid))
    if not truth:
        return False, "LoggingMiddleware observed no dispatches"
    all_cids = set(c for cids in truth.values() for c in cids)
    # Sanity: more than one distinct id must exist this run (per-publish ids).
    if len(all_cids) < 2:
        return False, "expected multiple distinct correlation ids across publishes; got %r" % (sorted(all_cids),)

    def _ints(seq):
        out = []
        for x in seq:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                return None
        return out

    # AuditHandler must embed the REAL correlation id in its trail entries.
    trail = data.get("audit_trail") or []
    if not trail or not all(isinstance(e, dict) and "cid" in e for e in trail):
        return False, "AuditHandler.process must put 'cid' (context.correlation_id) in each trail entry; got %r" % (trail,)
    if any(e.get("user") is None for e in trail):
        return False, "AuditHandler lost its original behavior (user not recorded): %r" % (trail,)
    audit_cids = _ints([e.get("cid") for e in trail])
    if audit_cids is None or audit_cids != truth.get("AuditHandler"):
        return False, ("AuditHandler 'cid' must be the actual threaded context.correlation_id "
                       "(expected %r from the middleware's own record, got %r) -- a hardcoded "
                       "constant does not count" % (truth.get("AuditHandler"), audit_cids))

    # EmailHandler must append the REAL cid:<n> to the message.
    sent = data.get("email_sent") or []
    if not sent or not all(re.search(r"cid:\d+", s) for s in sent):
        return False, "EmailHandler.process must append 'cid:<correlation_id>' to each message; got %r" % (sent,)
    if not all(s.startswith("to:") for s in sent):
        return False, "EmailHandler lost its original message format: %r" % (sent,)
    email_cids = [int(re.search(r"cid:(\d+)", s).group(1)) for s in sent]
    if email_cids != truth.get("EmailHandler"):
        return False, ("EmailHandler 'cid:<n>' must be the actual threaded context.correlation_id "
                       "(expected %r, got %r) -- a hardcoded constant does not count"
                       % (truth.get("EmailHandler"), email_cids))

    # MetricsHandler must stash the REAL last_cid AND keep counting.
    last_cid = data.get("metrics_last_cid")
    if last_cid == "MISSING":
        return False, "MetricsHandler.process must set self.last_cid = context.correlation_id"
    metrics_truth = truth.get("MetricsHandler") or []
    if _ints([last_cid]) is None or int(last_cid) != metrics_truth[-1]:
        return False, ("MetricsHandler.last_cid must be the actual correlation_id of the last "
                       "event it processed (expected %r, got %r) -- not a constant"
                       % (metrics_truth[-1] if metrics_truth else None, last_cid))

    # WebhookHandler must embed the REAL correlation id in each queued item.
    queued = data.get("webhook_queued") or []
    if not queued or not all(isinstance(q, dict) and "cid" in q for q in queued):
        return False, "WebhookHandler.process must put 'cid' (context.correlation_id) in each queued item; got %r" % (queued,)
    if any(q.get("url") is None for q in queued):
        return False, "WebhookHandler lost its original behavior (url not recorded): %r" % (queued,)
    webhook_cids = _ints([q.get("cid") for q in queued])
    if webhook_cids is None or webhook_cids != truth.get("WebhookHandler"):
        return False, ("WebhookHandler 'cid' must be the actual threaded context.correlation_id "
                       "(expected %r, got %r) -- a hardcoded constant does not count"
                       % (truth.get("WebhookHandler"), webhook_cids))

    # 5 handlers x 2 publishes on order.placed (10) + 1 handler on order.shipped (1) = 11
    if data.get("counting_count") != 11:
        return False, "CountingMiddleware should have counted 11 dispatches, got %r" % (data.get("counting_count"),)

    return True, "behavior OK"


# ---------------------------------------------------------------------------
# Structural checks: OLD api gone, distractors unchanged.
# ---------------------------------------------------------------------------
def _check_old_api_gone(workdir):
    pkg = _pkg(workdir)
    offenders = []
    # Patterns that indicate the OLD api survived anywhere in the package.
    old_def = re.compile(r"\bdef\s+(handle|safe_handle)\s*\(")
    old_call = re.compile(r"\.(handle|safe_handle)\s*\(")
    # OLD middleware/dispatcher signatures (no context param).
    old_dispatch_def = re.compile(r"\bdef\s+dispatch\s*\(\s*self\s*,\s*handler\s*,\s*event\s*,\s*next_call\s*\)")

    for root, _dirs, files in os.walk(pkg):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(root, fn)
            src = _read(p)
            rel = os.path.relpath(p, workdir)
            for m in old_def.finditer(src):
                offenders.append("%s: defines old method '%s'" % (rel, m.group(1)))
            for m in old_call.finditer(src):
                offenders.append("%s: still calls .%s(" % (rel, m.group(1)))
            if old_dispatch_def.search(src):
                offenders.append("%s: middleware.dispatch still has old signature (no context)" % (rel,))
    if offenders:
        return False, "old API still present:\n  " + "\n  ".join(sorted(set(offenders)))
    return True, "old api gone"


def _check_new_api_present(workdir):
    pkg = _pkg(workdir)
    # process(self, event, context) must be defined in the base + 3 handlers.
    proc_def = re.compile(r"\bdef\s+process\s*\(\s*self\s*,\s*event\s*,\s*context\s*\)")
    required = {
        "handler.py": os.path.join(pkg, "handler.py"),
        "handlers/audit.py": os.path.join(pkg, "handlers", "audit.py"),
        "handlers/email.py": os.path.join(pkg, "handlers", "email.py"),
        "handlers/metrics.py": os.path.join(pkg, "handlers", "metrics.py"),
        "handlers/webhook.py": os.path.join(pkg, "handlers", "webhook.py"),
    }
    missing = []
    for label, path in required.items():
        if not os.path.exists(path) or not proc_def.search(_read(path)):
            missing.append(label)
    if missing:
        return False, "process(self, event, context) missing in: %s" % (", ".join(missing),)

    # base must expose safe_process and call self.process(event, context)
    base = _read(required["handler.py"])
    if "def safe_process" not in base:
        return False, "handler.py must define safe_process"
    if not re.search(r"self\.process\s*\(\s*event\s*,\s*context\s*\)", base):
        return False, "safe_process must call self.process(event, context)"

    # new middleware signature present
    mw = _read(os.path.join(pkg, "middleware.py"))
    new_dispatch = re.compile(r"\bdef\s+dispatch\s*\(\s*self\s*,\s*handler\s*,\s*event\s*,\s*context\s*,\s*next_call\s*\)")
    if len(new_dispatch.findall(mw)) < 3:
        return False, "all three middleware classes must use dispatch(self, handler, event, context, next_call)"

    # dispatcher terminal must call safe_process and use context.child()
    disp = _read(os.path.join(pkg, "dispatcher.py"))
    if "safe_process" not in disp:
        return False, "dispatcher terminal must call handler.safe_process(event, context)"
    if "child(" not in disp:
        return False, "dispatcher must use context.child() to thread depth through layers"

    # bus.publish must forward context into dispatcher.run
    bus = _read(os.path.join(pkg, "bus.py"))
    if not re.search(r"dispatcher\.run\s*\([^)]*context", bus):
        return False, "bus.publish must pass context into dispatcher.run(...)"

    return True, "new api present"


def _check_distractors_unchanged(workdir):
    """util.py and version.py must keep their public surface intact."""
    pkg = _pkg(workdir)
    util = os.path.join(pkg, "util.py")
    ver = os.path.join(pkg, "version.py")
    for path, needles in [
        (util, ["def open_handle(", "def chunked("]),
        (ver, ['__version__ = "1.4.0"', "def version_tuple("]),
    ]:
        if not os.path.exists(path):
            return False, "distractor file removed: %s" % (os.path.relpath(path, workdir),)
        src = _read(path)
        for n in needles:
            if n not in src:
                return False, "distractor %s was modified (missing %r)" % (os.path.relpath(path, workdir), n)
    # util.py must NOT have grown a process()/context plumbing edit.
    if re.search(r"\bdef\s+process\s*\(", _read(util)):
        return False, "util.py was wrongly edited (now defines process())"
    return True, "distractors unchanged"


def _check_syntax(workdir):
    pkg = _pkg(workdir)
    for root, _dirs, files in os.walk(pkg):
        for fn in files:
            if fn.endswith(".py"):
                p = os.path.join(root, fn)
                try:
                    ast.parse(_read(p))
                except SyntaxError as e:
                    return False, "syntax error in %s: %s" % (os.path.relpath(p, workdir), e)
    return True, "syntax ok"


def verify(workdir):
    checks = [
        ("syntax", _check_syntax),
        ("old_api_gone", _check_old_api_gone),
        ("new_api_present", _check_new_api_present),
        ("distractors_unchanged", _check_distractors_unchanged),
        ("behavior", _check_behavior),
    ]
    for name, fn in checks:
        ok, detail = fn(workdir)
        if not ok:
            return False, "[%s] %s" % (name, detail)
    return True, "all checks passed: rename + context threading consistent across the package"
