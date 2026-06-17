"""pin / view — the active-asker memory-management syscalls (kernel Step D).

The LLM is an ACTIVE context-asker, not a passive slice-receiver: `pin` lets it deliberately grow its
working set for a multi-file task (mlock — keep these files resident + reclaim-protected), and `view`
is the /proc-style window into kernel state (headroom + what's resident) so it can self-throttle. Both
ride the existing ToolEntry/registry ABI; the bounded SwapManager owns the mechanism (it can always say
no — pins are force-compacted past PIN_CEILING), so the moat holds: growth is TASK-driven and bounded,
never history-proportional. pin mutates the live slice in-handle (the only handler that touches the slice
during dispatch — every other slice mutation is sequential in slice_sink, so there is no race); view is
read-only.
"""
from __future__ import annotations

from .access import FileAccess
from .registry import ToolEntry
from .swap import (DEP_CEILING, EDIT_CEILING, MAX_ACTIVE_SKILLS, MAX_GHOSTS, PIN_CEILING,
                   READ_BUDGET, _DEFAULT_SWAP)


def make_pin_tool(get_slice) -> ToolEntry:
    """`pin(path)` keeps a file resident + reclaim-protected; `pin(path, unpin=true)` releases it.
    get_slice() resolves the CURRENT active slice (so a topic switch retargets pins)."""

    def handler(args: dict) -> str:
        s = get_slice()
        path = (args.get("path") or "").strip()
        if not path:
            return "pin: pass a 'path' to pin (or {\"path\":..., \"unpin\":true} to release)."
        if args.get("unpin"):
            _DEFAULT_SWAP.unpin(s, path)
            return (f"Unpinned {path} — it reverts to ordinary residue (may page out to the GHOST "
                    f"INDEX as the working set moves on). Pinned now: {len(s.pinned)}/{PIN_CEILING}.")
        _DEFAULT_SWAP.pin(s, path)
        return (f"Pinned {path} — it stays resident in OPEN FILES and is protected from reclaim "
                f"({len(s.pinned)}/{PIN_CEILING} pinned). Release it with pin(\"{path}\", unpin=true) "
                "when the multi-file change is done.")

    schema = {"type": "function", "function": {
        "name": "pin",
        "description": (
            "Deliberately KEEP a file resident in OPEN FILES for a multi-file task — it is protected "
            "from the plain-read eviction that pages out exploratory reads, so the files your change "
            "must stay consistent with don't disappear from under you. Bounded (the oldest pin is "
            "dropped past the ceiling). Release with {\"path\":...,\"unpin\":true} when done. Use this "
            "when a refactor/edit spans several files; you do NOT need it for files you are editing "
            "(the change set is already protected)."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "workspace-relative file path to pin (or unpin)"},
            "unpin": {"type": "boolean", "description": "release a previously pinned file"},
        }, "required": ["path"]}}}
    return ToolEntry(name="pin", schema=schema, handler=handler,
                     accesses=lambda a: [FileAccess("read", a.get("path") or ".")], source="builtin")


def make_view_tool(get_slice) -> ToolEntry:
    """`view(kind)` — read-only /proc introspection of the bounded slice: working-set headroom
    (mem/usage) or the resident page map (slice/maps). Lets the active-asker see its own limits."""

    def handler(args: dict) -> str:
        s = get_slice()
        kind = (args.get("kind") or "mem").strip().lower()
        edited = [p for p in s.active_files if p in s.edited_files]
        pinned = [p for p in s.active_files if p in s.pinned]
        deps = [p for p in s.active_files if p in s.protected_deps and p not in s.edited_files]
        reads = [p for p in s.active_files
                 if p not in s.edited_files and p not in s.protected_deps and p not in s.pinned]
        if kind in ("maps", "slice", "files", "map"):
            out = [f"# KERNEL VIEW: slice/maps — {len(s.active_files)} resident page(s)"]
            for p in s.active_files:
                tags = []
                if p in s.edited_files:
                    tags.append("edited")
                if p in s.pinned:
                    tags.append("pinned")
                if p in s.protected_deps and p not in s.edited_files:
                    tags.append("dep")
                out.append(f"  - {p}  [{','.join(tags) if tags else 'read'}]")
            if s.ghosts:
                out.append("GHOST INDEX (paged out — recover with one read): "
                           + ", ".join(g["ref"] for g in s.ghosts))
            if s.active_skills:
                out.append("ACTIVE SKILLS: " + ", ".join(sk["name"] for sk in s.active_skills))
            return "\n".join(out)
        return ("# KERNEL VIEW: mem/usage — working-set headroom\n"
                f"reads {len(reads)}/{READ_BUDGET} · edited {len(edited)}/{EDIT_CEILING} · "
                f"deps {len(deps)}/{DEP_CEILING} · pinned {len(pinned)}/{PIN_CEILING}\n"
                f"ghosts {len(s.ghosts)}/{MAX_GHOSTS} · skills {len(s.active_skills)}/{MAX_ACTIVE_SKILLS} · "
                f"findings {len(s.findings)}\n"
                "Exploratory reads beyond the budget page out to the GHOST INDEX (one call to recover). "
                "pin a file to keep it resident for a multi-file change; view(kind=\"maps\") lists what's resident.")

    schema = {"type": "function", "function": {
        "name": "view",
        "description": (
            "Introspect your own bounded context (read-only). kind=\"mem\" (default) shows working-set "
            "headroom — how full each tier is vs its budget — so you can decide whether to pin more or "
            "let reads page out. kind=\"maps\" lists the files currently resident in OPEN FILES and "
            "what's in the GHOST INDEX. Use it to orient before a multi-file task."),
        "parameters": {"type": "object", "properties": {
            "kind": {"type": "string", "description": "\"mem\" (headroom, default) or \"maps\" (resident page map)"},
        }}}}
    return ToolEntry(name="view", schema=schema, handler=handler, source="builtin")
