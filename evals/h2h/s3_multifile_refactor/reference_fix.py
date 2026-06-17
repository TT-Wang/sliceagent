"""Reference solution (VALIDATION ONLY — never shown to benchmarked agents).

Rewrites the touched modules with the rename + context threading applied.
"""
import os


def _w(workdir, relpath, content):
    path = os.path.join(workdir, relpath)
    with open(path, "w") as f:
        f.write(content)


def apply(workdir):
    pkg = "eventbus"

    # 1) Base Handler: handle -> process(event, context), safe_handle -> safe_process
    _w(workdir, os.path.join(pkg, "handler.py"), '''\
"""Base Handler class. Core API: process(self, event, context)."""
from .errors import HandlerError


class Handler:
    """Subclasses implement process(event, context) to react to events.

    name() identifies the handler in logs and errors.
    """

    def name(self):
        return type(self).__name__

    def process(self, event, context):
        raise NotImplementedError(
            "%s must implement process(event, context)" % (self.name(),)
        )

    def safe_process(self, event, context):
        """Call process() and wrap any failure in HandlerError."""
        try:
            return self.process(event, context)
        except Exception as exc:  # noqa: BLE001
            raise HandlerError(self.name(), exc)
''')

    # 2) Concrete handlers use the context.
    _w(workdir, os.path.join(pkg, "handlers", "audit.py"), '''\
"""Concrete handler: records an audit trail of events it sees."""
from ..handler import Handler


class AuditHandler(Handler):
    def __init__(self):
        self.trail = []

    def process(self, event, context):
        entry = {
            "topic": event.topic,
            "user": event.payload.get("user"),
            "cid": context.correlation_id,
        }
        self.trail.append(entry)
        return entry
''')

    _w(workdir, os.path.join(pkg, "handlers", "email.py"), '''\
"""Concrete handler: pretends to send an email for an event."""
from ..handler import Handler


class EmailHandler(Handler):
    def __init__(self):
        self.sent = []

    def process(self, event, context):
        msg = "to:%s subject:%s cid:%s" % (
            event.payload.get("to", "?"),
            event.topic,
            context.correlation_id,
        )
        self.sent.append(msg)
        return msg
''')

    _w(workdir, os.path.join(pkg, "handlers", "metrics.py"), '''\
"""Concrete handler: increments a per-topic counter."""
from collections import Counter

from ..handler import Handler


class MetricsHandler(Handler):
    def __init__(self):
        self.counts = Counter()
        self.last_cid = None

    def process(self, event, context):
        self.last_cid = context.correlation_id
        self.counts[event.topic] += 1
        return self.counts[event.topic]
''')

    _w(workdir, os.path.join(pkg, "handlers", "webhook.py"), '''\
"""Concrete handler: queues an outbound webhook call for an event."""
from ..handler import Handler


class WebhookHandler(Handler):
    def __init__(self, url="https://hooks.example/ingest"):
        self.url = url
        self.queued = []

    def process(self, event, context):
        item = {
            "url": self.url,
            "topic": event.topic,
            "cid": context.correlation_id,
        }
        self.queued.append(item)
        return item
''')

    # 3) Middleware: dispatch gains context.
    _w(workdir, os.path.join(pkg, "middleware.py"), '''\
"""Middleware base + built-ins. dispatch(self, handler, event, context, next_call)."""


class Middleware:
    """Subclasses override dispatch() to wrap each handler call."""

    def dispatch(self, handler, event, context, next_call):
        return next_call(handler, event, context)


class LoggingMiddleware(Middleware):
    """Records each (handler_name, topic, correlation_id) triple, in order."""

    def __init__(self):
        self.log = []

    def dispatch(self, handler, event, context, next_call):
        self.log.append((handler.name(), event.topic, context.correlation_id))
        return next_call(handler, event, context)


class CountingMiddleware(Middleware):
    """Counts how many handler dispatches passed through."""

    def __init__(self):
        self.count = 0

    def dispatch(self, handler, event, context, next_call):
        self.count += 1
        return next_call(handler, event, context)
''')

    # 4) Dispatcher threads context (with child() per layer) and calls safe_process.
    _w(workdir, os.path.join(pkg, "dispatcher.py"), '''\
"""The dispatcher composes middlewares around a terminal handler call."""


class Dispatcher:
    """Builds the middleware chain and invokes a single handler through it."""

    def __init__(self, middlewares=None):
        self.middlewares = list(middlewares or [])

    def _terminal(self, handler, event, context):
        # Terminal step: actually run the handler.
        return handler.safe_process(event, context)

    def run(self, handler, event, context):
        """Invoke handler through every middleware (last added = outermost)."""
        call = self._terminal
        for mw in self.middlewares:
            call = self._wrap(mw, call)
        return call(handler, event, context)

    @staticmethod
    def _wrap(mw, next_call):
        def descend(handler, event, context):
            return next_call(handler, event, context.child())

        def wrapped(handler, event, context):
            return mw.dispatch(handler, event, context, descend)

        return wrapped
''')

    # 5) Bus passes context into dispatcher.run.
    _w(workdir, os.path.join(pkg, "bus.py"), '''\
"""EventBus: the public entry point. publish() fans an event to handlers."""
from .context import Context
from .dispatcher import Dispatcher
from .registry import Registry


class EventBus:
    def __init__(self, middlewares=None):
        self.registry = Registry()
        self.dispatcher = Dispatcher(middlewares)

    def subscribe(self, topic, handler):
        self.registry.add(topic, handler)
        return handler

    def publish(self, event, **meta):
        """Dispatch event to all subscribed handlers, return list of results.

        A Context is created per publish and threaded into the dispatch chain.
        """
        context = Context(topic=event.topic, meta=dict(meta))
        handlers = self.registry.handlers_for(event.topic)
        results = []
        for handler in handlers:
            results.append(self.dispatcher.run(handler, event, context))
        return results
''')
