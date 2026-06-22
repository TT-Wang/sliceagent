"""Skills — reusable procedure prompt-packs (Kimi + Hermes converge on this exact shape).

A skill is a SKILL.md: frontmatter (name + description required) + a markdown body of
instructions. Progressive disclosure: the `skill` tool's *description* carries the cheap
catalog (name + one-line description); calling `skill(name=...)` returns the full body,
which slice_sink folds into the slice's ACTIVE SKILL tier (slice.add_skill) where it
PERSISTS across turns — the memagent-specific adaptation (no transcript to hold a one-shot
injection). Skills are instructions the model then follows with its existing tools; they
are NOT code.
"""
from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass

from .registry import ToolEntry
from .text_utils import one_line

_MAX_SCAN_DEPTH = 8   # bound skill-root walk depth (Kimi MAX_SKILL_SCAN_DEPTH) — defensive vs deep trees


def expand_skill_args(body: str, argstr: str) -> str:
    """Substitute a skill's parameter placeholders (Kimi expandSkillParameters): `$ARGUMENTS` → the full
    arg string, `$1`/`$2`/… → positional tokens (shell-split). A skill with no placeholders is returned
    unchanged, so existing skills are unaffected."""
    if "$ARGUMENTS" not in body and not re.search(r"\$\d", body):
        return body
    argstr = argstr or ""
    try:
        parts = shlex.split(argstr)
    except ValueError:                      # unbalanced quotes → fall back to whitespace split
        parts = argstr.split()
    out = body.replace("$ARGUMENTS", argstr)
    for i, tok in enumerate(parts, 1):
        out = out.replace(f"${i}", tok)
    return out


@dataclass
class Skill:
    name: str
    description: str
    body: str
    path: str
    provenance: str = "user"   # "user" | "consolidation" (item 13); from `provenance:` frontmatter
    root: str = ""             # the discovery root this skill came from (where its .usage.json lives)
    when_to_use: str = ""      # from `when-to-use:` frontmatter — shown in the catalog to improve routing (Kimi)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Minimal '---' fenced frontmatter → (meta, body). No YAML dependency: only simple
    `key: value` lines are read (enough for name/description/when-to-use)."""
    if not text.startswith("---"):
        return {}, text
    rest = text[3:]
    end = rest.find("\n---")
    if end == -1:
        return {}, text
    fm, body = rest[:end], rest[end + 4:].lstrip("\n")
    meta: dict = {}
    for line in fm.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, _, v = line.partition(":")
        meta[k.strip().lower()] = v.strip().strip('"').strip("'")
    return meta, body


def _default_roots() -> list[str]:
    # project skills win over user skills (discovery order = first-wins)
    return [
        os.path.join(os.getcwd(), ".memagent", "skills"),
        os.path.join(os.path.expanduser("~"), ".memagent", "skills"),
    ]


class SkillManager:
    def __init__(self, roots: list[str] | None = None):
        self.roots = roots if roots is not None else _default_roots()
        self._skills: dict[str, Skill] = {}
        self.discover()

    def discover(self) -> "SkillManager":
        self._skills = {}
        for root in self.roots:
            if not os.path.isdir(root):
                continue
            for dp, _dirs, files in os.walk(root):           # followlinks=False → no symlink cycles
                if dp[len(root):].count(os.sep) > _MAX_SCAN_DEPTH:
                    _dirs[:] = []                            # bound depth (Kimi MAX_SKILL_SCAN_DEPTH)
                    continue
                for fn in files:
                    if fn == "SKILL.md" or (fn.endswith(".md") and dp == root):
                        self._load(os.path.join(dp, fn), root)
        return self

    def _load(self, path: str, root: str = "") -> None:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            return
        meta, body = parse_frontmatter(text)
        name = (meta.get("name") or "").strip().lower()
        if not name and os.path.basename(path) != "SKILL.md":
            name = os.path.splitext(os.path.basename(path))[0].lower()
        when = (meta.get("when-to-use") or meta.get("when_to_use") or "").strip()
        desc = (meta.get("description") or when or "").strip()
        if not name or not body.strip():
            return
        prov = (meta.get("provenance") or "user").strip().lower() or "user"
        self._skills.setdefault(name, Skill(name, desc, body.strip(), path, provenance=prov,
                                            root=root, when_to_use=when))  # first-wins

    def add(self, name: str, body: str, description: str = "") -> None:
        """Register an in-memory skill (e.g. contributed by a plugin). First-wins, so a
        disk skill of the same name takes precedence."""
        name = (name or "").strip().lower()
        if name and body and body.strip():
            self._skills.setdefault(name, Skill(name, description.strip(), body.strip(), "<plugin>"))

    def names(self) -> list[str]:
        return sorted(self._skills)

    def catalog(self) -> list[tuple[str, str]]:
        return [(n, self._skills[n].description) for n in self.names()]

    def load(self, name: str) -> str | None:
        s = self._skills.get((name or "").lower())
        if s is None:
            return None
        self._bump_usage(s)   # item 13: last-used sidecar feeds consolidate's frequency weight
        return s.body

    def _bump_usage(self, s: "Skill") -> None:
        """Bump the .usage.json sidecar in the skill's discovery root. Best-effort: a sidecar
        hiccup, or a plugin/in-memory skill with no root, never breaks the skill load."""
        if not s.root or not os.path.isdir(s.root):
            return
        try:
            from .skill_usage import bump_use
            bump_use(s.root, s.name)
        except Exception:
            pass


def make_skill_tool(manager: SkillManager) -> ToolEntry | None:
    """A ToolEntry for the `skill` tool — or None when no skills are discovered. The tool
    description IS the catalog (progressive disclosure); the handler returns the body, which
    slice_sink folds into the ACTIVE SKILL tier."""
    cat = manager.catalog()
    if not cat:
        return None
    names = [n for n, _ in cat]

    def _line(n: str, d: str) -> str:                       # show when-to-use to improve routing (Kimi)
        s = manager._skills.get(n)
        w = (s.when_to_use if s else "") or ""
        base = f"- {n}: {one_line(d, 140)}"
        return base + (f" (when: {one_line(w, 80)})" if w and w != d else "")

    listing = "\n".join(_line(n, d) for n, d in cat)
    desc = (
        "Load a SKILL: a reusable procedure whose detailed instructions are added to your "
        "working context and PERSIST for the rest of the task. Call it BEFORE starting work "
        "when a task matches one of these skills. Pass `arguments` to fill a parameterized skill "
        "($ARGUMENTS / $1 $2 in its body). Available skills:\n" + listing
    )
    schema = {
        "type": "function",
        "function": {
            "name": "skill", "description": desc,
            "parameters": {"type": "object",
                           "properties": {
                               "name": {"type": "string", "enum": names},
                               "arguments": {"type": "string", "description": "Optional. Substituted "
                                             "into the skill's $ARGUMENTS / $1 $2 placeholders."}},
                           "required": ["name"]},
        },
    }

    def handler(args: dict) -> str:
        body = manager.load(args.get("name", ""))
        if body is None:
            return f"Error: no skill named {args.get('name')!r}. Available: {', '.join(names)}"
        # expand $ARGUMENTS / $N placeholders (no-op for skills without them); slice_sink folds the
        # result into the ACTIVE SKILL tier (persists across turns).
        return expand_skill_args(body, args.get("arguments") or "")

    return ToolEntry(name="skill", schema=schema, handler=handler,
                     accesses=lambda _a: [], source="skill")


def make_skill_manager(roots: list[str] | None = None) -> SkillManager:
    return SkillManager(roots)
