"""CodeIndex — the RELATED CODE tier (borrowed periphery, behind the Retriever interface).

Fills the slice's discovery tier from a real repository: given the task (and the
agent's live focus, e.g. the current error), surface the most relevant *existing*
code as ranked snippets so the model doesn't have to blind-grep. Deterministic, no
embeddings, no network — just ripgrep over the working tree (respects .gitignore).

Distinct from the Memory tier: memem indexes a curated LESSON vault; this indexes
SOURCE CODE. Two tiers, two interfaces (see interfaces.py).

v1 is ripgrep symbol/term search + a regex repo-map skeleton. tree-sitter is the
precision upgrade for definition extraction — it slots in at `_defs_in()` without
touching the Retriever contract or any caller.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

from .interfaces import Snippet

# tokens too common to discriminate on (task prose is full of them)
_STOP = frozenset((
    "the and for with that this from into your you use using make build adds add create "
    "creates function functions method module modules file files code test tests should "
    "when then return returns value values given must each all any not new run runs fix "
    "fixes bug bugs def class import only also like one two get set has have its them they "
    "such only via per out off via are was were will can may want need needs implement"
).split())

# identifier-ish tokens worth searching: snake_case, camelCase, dotted names, ≥3 chars
import re as _re
_TOKEN = _re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

# language-ish definition lines for the repo-map skeleton (tree-sitter upgrades this)
_DEF_RE = _re.compile(
    r"^\s*(?:export\s+|default\s+|public\s+|private\s+|protected\s+|static\s+|abstract\s+)*"
    r"(?:async\s+)?"
    r"(?:def|class|func|function|fn|type|interface|struct|enum|impl|trait|module|const)\b"
)
_CODE_EXT = frozenset((
    ".py .js .jsx .ts .tsx .go .rs .java .rb .c .h .cc .cpp .hpp .cs .php .swift .kt "
    ".scala .sh .lua .m .mm .ex .exs .clj .hs .ml .r .jl"
).split())


def _terms(query: str, limit: int = 12) -> list[str]:
    """Extract distinct, discriminating identifiers from a natural-language query."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in _TOKEN.findall(query or ""):
        low = tok.lower()
        if low in _STOP or low in seen:
            continue
        seen.add(low)
        out.append(tok)
        if len(out) >= limit:
            break
    return out


class RipgrepCodeIndex:
    """Retriever over a working tree using ripgrep. No index to build — queries run live.

    Robust by design: any ripgrep failure (missing binary, bad path, timeout) degrades
    to an empty result, so the discovery tier simply goes quiet rather than breaking the
    loop — same contract as NullRetriever, just populated when there's code to find.
    """

    def __init__(self, root: str = ".", *, rg: str = "rg",
                 max_filesize: str = "300K", timeout: float = 6.0,
                 ctx: int = 4, max_chars: int = 1400):
        self.root = os.path.abspath(root)
        self.rg = rg
        self.max_filesize = max_filesize
        self.timeout = timeout
        self.ctx = ctx
        self.max_chars = max_chars

    # --- Retriever contract -------------------------------------------------
    def retrieve(self, query: str, k: int = 6) -> list[Snippet]:
        terms = _terms(query)
        if not terms:
            return []
        hits = self._search(terms)
        if not hits:
            return []
        # rank: more distinct terms in a file beats more raw hits of one term
        ranked = sorted(
            hits.items(),
            key=lambda kv: (len(kv[1]["terms"]), kv[1]["count"]),
            reverse=True,
        )[:k]
        out: list[Snippet] = []
        for path, info in ranked:
            text = self._context(path, info["lines"])
            if not text:
                continue
            rel = os.path.relpath(path, self.root)
            score = len(info["terms"]) + min(info["count"], 99) / 100.0
            out.append(Snippet(path=rel, text=text, score=score))
        return out

    # --- repo map (orientation skeleton; not auto-injected per turn) ---------
    def repo_map(self, max_files: int = 40, max_defs_per_file: int = 10,
                 max_chars: int = 2500) -> str:
        """A compact tree of code files → their top-level definitions.

        Cheap orientation for turn-0 / a subagent briefing. NOT folded into every
        slice (that would add constant tokens and break the bounded-context invariant);
        the host decides when to surface it.
        """
        files = self._code_files(max_files)
        if not files:
            return ""
        blocks: list[str] = []
        for rel in files:
            defs = self._defs_in(os.path.join(self.root, rel), max_defs_per_file)
            if defs:
                blocks.append(rel + "\n" + "\n".join("  " + d for d in defs))
        return "\n".join(blocks)[:max_chars]

    # --- internals ----------------------------------------------------------
    def _search(self, terms: list[str]) -> dict[str, dict]:
        """One ripgrep pass over all terms; group matches by file."""
        cmd = [self.rg, "--json", "-i", "--max-filesize", self.max_filesize,
               "--max-columns", "400"]
        for t in terms:
            cmd += ["-e", t]
        cmd.append(self.root)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=self.timeout)
        except (OSError, subprocess.SubprocessError):
            return {}
        files: dict[str, dict] = {}
        for raw in proc.stdout.splitlines():
            if not raw or '"type":"match"' not in raw:
                continue
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            d = obj.get("data", {})
            path = (d.get("path") or {}).get("text")
            ln = d.get("line_number")
            if not path or ln is None:
                continue
            f = files.setdefault(path, {"terms": set(), "lines": [], "count": 0})
            f["count"] += 1
            if len(f["lines"]) < 60:
                f["lines"].append(ln)
            for sm in d.get("submatches", []):
                mt = ((sm.get("match") or {}).get("text") or "").lower()
                if mt:
                    f["terms"].add(mt)
        return files

    def _context(self, path: str, lines: list[int]) -> str:
        """Line-numbered context windows around the matched-line clusters in one file."""
        try:
            with open(path, "r", errors="replace") as fh:
                src = fh.read().splitlines()
        except OSError:
            return ""
        if not src:
            return ""
        pts = sorted(set(lines))
        clusters: list[list[int]] = []
        for ln in pts:
            if clusters and ln - clusters[-1][-1] <= self.ctx * 2:
                clusters[-1].append(ln)
            else:
                clusters.append([ln])
        chunks: list[str] = []
        used = 0
        for cl in clusters:
            lo = max(1, cl[0] - self.ctx)
            hi = min(len(src), cl[-1] + self.ctx)
            chunk = "\n".join(f"{i:>5} {src[i - 1]}" for i in range(lo, hi + 1))
            if used + len(chunk) > self.max_chars:
                chunk = chunk[: max(0, self.max_chars - used)]
            if chunk:
                chunks.append(chunk)
                used += len(chunk)
            if used >= self.max_chars:
                break
        return "\n   …\n".join(chunks)

    def _code_files(self, max_files: int) -> list[str]:
        try:
            proc = subprocess.run([self.rg, "--files", self.root],
                                  capture_output=True, text=True, timeout=self.timeout)
        except (OSError, subprocess.SubprocessError):
            return []
        rels: list[str] = []
        for p in proc.stdout.splitlines():
            if os.path.splitext(p)[1] in _CODE_EXT:
                rels.append(os.path.relpath(p, self.root))
        rels.sort()
        return rels[:max_files]

    def _defs_in(self, path: str, limit: int) -> list[str]:
        try:
            with open(path, "r", errors="replace") as fh:
                src = fh.read().splitlines()
        except OSError:
            return []
        out: list[str] = []
        for line in src:
            if _DEF_RE.match(line):
                out.append(line.strip()[:120])
                if len(out) >= limit:
                    break
        return out


def make_code_index(root: str = ".", *, prefer_ripgrep: bool = True):
    """Factory mirroring make_memory(): a real CodeIndex if ripgrep is on PATH,
    else NullRetriever so the loop runs unchanged."""
    if prefer_ripgrep and shutil.which("rg"):
        return RipgrepCodeIndex(root=root)
    from .retriever import NullRetriever
    return NullRetriever()
