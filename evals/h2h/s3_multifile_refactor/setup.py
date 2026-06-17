import os


def _w(workdir, relpath, content):
    path = os.path.join(workdir, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def setup(workdir):
    pkg = "eventbus"

    _w(workdir, os.path.join(pkg, "__init__.py"), '''\
"""eventbus: a tiny synchronous in-process event bus framework.

Public surface re-exported here for convenience.
"""
from .bus import EventBus
from .event import Event
from .handler import Handler
from .registry import Registry
from .errors import HandlerError, UnknownTopicError

__all__ = [
    "EventBus",
    "Event",
    "Handler",
    "Registry",
    "HandlerError",
    "UnknownTopicError",
]
''')

    _w(workdir, os.path.join(pkg, "event.py"), '''\
"""The Event value object carried through the bus."""
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Event:
    topic: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def with_payload(self, **extra):
        merged = dict(self.payload)
        merged.update(extra)
        return Event(self.topic, merged)
''')

    _w(workdir, os.path.join(pkg, "errors.py"), '''\
"""Exception hierarchy for the bus. (DISTRACTOR-ADJACENT: no handle() here.)"""


class BusError(Exception):
    pass


class HandlerError(BusError):
    """Raised when a handler fails while processing an event."""

    def __init__(self, handler_name, original):
        self.handler_name = handler_name
        self.original = original
        super().__init__("handler %s failed: %r" % (handler_name, original))


class UnknownTopicError(BusError):
    def __init__(self, topic):
        self.topic = topic
        super().__init__("no handlers subscribed to topic %r" % (topic,))
''')

    _w(workdir, os.path.join(pkg, "context.py"), '''\
"""Dispatch context: per-publish metadata threaded through the call chain.

This module already exists; the refactor must THREAD this object through
the handler call chain (it is currently created but not passed to handlers).
"""
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict

_counter = itertools.count(1)


@dataclass
class Context:
    """Carries metadata about a single publish() invocation."""
    correlation_id: int = field(default_factory=lambda: next(_counter))
    topic: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    # depth tracks how many middlewares have wrapped this dispatch.
    depth: int = 0

    def child(self):
        return Context(
            correlation_id=self.correlation_id,
            topic=self.topic,
            meta=dict(self.meta),
            depth=self.depth + 1,
        )
''')

    _w(workdir, os.path.join(pkg, "handler.py"), '''\
"""Base Handler class. The OLD core API is `handle(self, event)`.

REFACTOR TARGET: this base method must become `process(self, event, context)`.
"""
from .errors import HandlerError


class Handler:
    """Subclasses implement handle(event) to react to events.

    name() identifies the handler in logs and errors.
    """

    def name(self):
        return type(self).__name__

    def handle(self, event):
        raise NotImplementedError(
            "%s must implement handle(event)" % (self.name(),)
        )

    def safe_handle(self, event):
        """Call handle() and wrap any failure in HandlerError."""
        try:
            return self.handle(event)
        except Exception as exc:  # noqa: BLE001
            raise HandlerError(self.name(), exc)
''')

    _w(workdir, os.path.join(pkg, "registry.py"), '''\
"""Registry mapping topics -> ordered list of handlers."""
from collections import defaultdict

from .errors import UnknownTopicError


class Registry:
    def __init__(self):
        self._by_topic = defaultdict(list)

    def add(self, topic, handler):
        self._by_topic[topic].append(handler)

    def handlers_for(self, topic):
        if topic not in self._by_topic or not self._by_topic[topic]:
            raise UnknownTopicError(topic)
        return list(self._by_topic[topic])

    def topics(self):
        return sorted(self._by_topic.keys())

    def all_handlers(self):
        seen = []
        for topic in self.topics():
            for h in self._by_topic[topic]:
                if h not in seen:
                    seen.append(h)
        return seen
''')

    _w(workdir, os.path.join(pkg, "middleware.py"), '''\
"""Middleware base + built-ins. Middlewares wrap handler dispatch.

OLD signature: dispatch(self, handler, event, next_call).
REFACTOR TARGET: must become dispatch(self, handler, event, context, next_call)
so middlewares receive and forward the Context.
"""


class Middleware:
    """Subclasses override dispatch() to wrap each handler call."""

    def dispatch(self, handler, event, next_call):
        return next_call(handler, event)


class LoggingMiddleware(Middleware):
    """Records each (handler_name, topic) pair it sees, in order."""

    def __init__(self):
        self.log = []

    def dispatch(self, handler, event, next_call):
        self.log.append((handler.name(), event.topic))
        return next_call(handler, event)


class CountingMiddleware(Middleware):
    """Counts how many handler dispatches passed through."""

    def __init__(self):
        self.count = 0

    def dispatch(self, handler, event, next_call):
        self.count += 1
        return next_call(handler, event)
''')

    _w(workdir, os.path.join(pkg, "dispatcher.py"), '''\
"""The dispatcher composes middlewares around a terminal handler call."""


class Dispatcher:
    """Builds the middleware chain and invokes a single handler through it."""

    def __init__(self, middlewares=None):
        self.middlewares = list(middlewares or [])

    def _terminal(self, handler, event):
        # Terminal step: actually run the handler.
        return handler.safe_handle(event)

    def run(self, handler, event):
        """Invoke handler through every middleware (last added = outermost)."""
        call = self._terminal
        for mw in self.middlewares:
            call = self._wrap(mw, call)
        return call(handler, event)

    @staticmethod
    def _wrap(mw, next_call):
        def wrapped(handler, event):
            return mw.dispatch(handler, event, next_call)
        return wrapped
''')

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

        A Context is created per publish but (BUG) is not yet threaded into
        the dispatch chain / handlers.
        """
        context = Context(topic=event.topic, meta=dict(meta))
        handlers = self.registry.handlers_for(event.topic)
        results = []
        for handler in handlers:
            results.append(self.dispatcher.run(handler, event))
        return results
''')

    _w(workdir, os.path.join(pkg, "handlers", "__init__.py"), '''\
from .audit import AuditHandler
from .email import EmailHandler
from .metrics import MetricsHandler
from .webhook import WebhookHandler

__all__ = ["AuditHandler", "EmailHandler", "MetricsHandler", "WebhookHandler"]
''')

    _w(workdir, os.path.join(pkg, "handlers", "audit.py"), '''\
"""Concrete handler: records an audit trail of events it sees."""
from ..handler import Handler


class AuditHandler(Handler):
    def __init__(self):
        self.trail = []

    def handle(self, event):
        entry = {"topic": event.topic, "user": event.payload.get("user")}
        self.trail.append(entry)
        return entry
''')

    _w(workdir, os.path.join(pkg, "handlers", "email.py"), '''\
"""Concrete handler: pretends to send an email for an event."""
from ..handler import Handler


class EmailHandler(Handler):
    def __init__(self):
        self.sent = []

    def handle(self, event):
        msg = "to:%s subject:%s" % (
            event.payload.get("to", "?"),
            event.topic,
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

    def handle(self, event):
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

    def handle(self, event):
        item = {"url": self.url, "topic": event.topic}
        self.queued.append(item)
        return item
''')

    # ---- DISTRACTOR FILES: must NOT change ----
    _w(workdir, os.path.join(pkg, "util.py"), '''\
"""Generic helpers unrelated to the dispatch chain. DO NOT change for this
refactor. Note: the word 'handle' appears here only as English prose and an
unrelated file-handle helper, NOT the Handler.handle API.
"""


def open_handle(path, mode="r"):
    """Return a file handle. (Unrelated to event handlers.)"""
    return open(path, mode)


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]
''')

    _w(workdir, os.path.join(pkg, "version.py"), '''\
"""Version metadata. DO NOT change for this refactor."""

__version__ = "1.4.0"


def version_tuple():
    return tuple(int(p) for p in __version__.split("."))
''')

    _w(workdir, "README.md", '''\
# eventbus

A tiny synchronous in-process event bus.

```python
from eventbus import EventBus, Event
from eventbus.handlers import AuditHandler

bus = EventBus()
bus.subscribe("user.created", AuditHandler())
bus.publish(Event("user.created", {"user": "ann"}))
```

Handlers subclass `Handler` and react to events. Middlewares wrap each
handler dispatch. The registry maps topics to handlers.
''')
