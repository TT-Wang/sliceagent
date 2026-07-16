"""One file-mention grammar shared by the prompt completer and CLI host.

Unquoted mentions are exact, whitespace-delimited paths (``@src/app/[id].py``).
Paths containing whitespace use shell-like quotes (``@"docs/my guide.md"``).  This
module only recognizes path text; workspace confinement and existence checks happen
in :func:`workspace_mentions` before a mention can affect the active slice.
"""
from __future__ import annotations

import os


def _mention_boundary(text: str, index: int) -> bool:
    """An ``@`` inside an email/identifier is prose, not a file attachment."""
    if index == 0:
        return True
    previous = text[index - 1]
    return previous.isspace() or previous in "([{<:;,"


def parse_mentions(text: str) -> list[str]:
    """Return syntactically valid file paths mentioned in ``text``.

    Quoted mentions support ``\\\"`` and ``\\\\``.  A missing closing quote is
    deliberately ignored: guessing where a path ends could attach a different file
    than the user named.  Duplicate paths are collapsed in first-seen order.
    """
    found: list[str] = []
    seen: set[str] = set()
    i = 0
    while i < len(text):
        at = text.find("@", i)
        if at < 0:
            break
        i = at + 1
        if not _mention_boundary(text, at) or i >= len(text):
            continue

        if text[i] in ("'", '"'):
            quote = text[i]
            i += 1
            chars: list[str] = []
            closed = False
            while i < len(text):
                char = text[i]
                if char == quote:
                    closed = True
                    i += 1
                    break
                if char == "\\" and i + 1 < len(text) and text[i + 1] in (quote, "\\"):
                    chars.append(text[i + 1])
                    i += 2
                    continue
                chars.append(char)
                i += 1
            if not closed:
                continue
            path = "".join(chars)
        else:
            end = i
            while end < len(text) and not text[end].isspace():
                end += 1
            path = text[i:end]
            i = end

        # Control/invisible formatting characters in a filename can forge terminal/prompt structure. Such
        # names remain accessible through ordinary file tools, but are deliberately not mention syntax.
        if path and all(char.isprintable() for char in path) and path not in seen:
            seen.add(path)
            found.append(path)
    return found


def completion_path(path: str) -> str | None:
    """Render one repo-relative path in syntax that :func:`parse_mentions` accepts.

    The returned value excludes the leading ``@`` because prompt_toolkit preserves
    the user's trigger and replaces only the portion after it.
    """
    value = str(path or "")
    if not value or not all(char.isprintable() for char in value):
        return None
    if any(char.isspace() for char in value) or value.startswith(("'", '"')):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def workspace_mentions(text: str, root: str) -> list[str]:
    """Resolve existing mentions confined to ``root`` and return relative paths.

    ``realpath`` plus ``commonpath`` blocks both ``..`` traversal and symlink escapes,
    while still allowing legitimate filenames containing two dots.  Absolute paths
    are accepted only when they resolve inside the active workspace.
    """
    workspace = os.path.realpath(root)
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in parse_mentions(text):
        # Exact wins, so a real filename ending in punctuation is never rewritten.  If exact does not exist,
        # accept the conventional prose spelling ``@src/a.py,`` by trimming sentence punctuation only.
        candidates = [raw]
        prose_trimmed = raw.rstrip(".,;:!?)]}>")
        if prose_trimmed and prose_trimmed != raw:
            candidates.append(prose_trimmed)
        rel = None
        for candidate in candidates:
            try:
                expanded = os.path.expanduser(candidate)
                full = os.path.realpath(
                    expanded if os.path.isabs(expanded) else os.path.join(workspace, expanded)
                )
                if os.path.commonpath((workspace, full)) != workspace or not os.path.isfile(full):
                    continue
                rel = os.path.relpath(full, workspace)
                break
            except (OSError, ValueError):
                continue
        if rel is None:
            continue
        if rel not in seen:
            seen.add(rel)
            resolved.append(rel)
    return resolved
