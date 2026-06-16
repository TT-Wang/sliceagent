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

# the NAME of a definition (for the symbol graph); tree-sitter upgrades this at _scan_file()
_NAME_RE = _re.compile(
    r"\b(?:def|class|func|function|fn|type|interface|struct|enum|trait|module|const)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)")


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


_TS = {"tried": False, "parser": None}


def _ts_python():
    """Lazily build a tree-sitter Python parser; None if tree-sitter isn't installed (→ regex)."""
    if not _TS["tried"]:
        _TS["tried"] = True
        try:                                          # convenience bundle (prebuilt grammars)
            from tree_sitter_languages import get_parser
            _TS["parser"] = get_parser("python")
        except Exception:                             # noqa: BLE001 — fall back to the split packages
            try:
                import tree_sitter_python as _tspy
                from tree_sitter import Language, Parser
                _TS["parser"] = Parser(Language(_tspy.language()))
            except Exception:                         # noqa: BLE001 — not installed → regex path
                _TS["parser"] = None
    return _TS["parser"]


def _ts_def_names(path: str, src: str):
    """Definition names via tree-sitter (Python only, precise: real function/class nodes, no
    comment/string false-positives). Returns None to signal 'use the regex' — non-Python file or
    tree-sitter not installed. The Retriever contract and every caller are unchanged either way."""
    if not path.endswith(".py"):
        return None
    parser = _ts_python()
    if parser is None:
        return None
    try:
        tree = parser.parse(src.encode("utf-8", "replace"))
        names, stack = set(), [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type in ("function_definition", "class_definition"):
                nm = node.child_by_field_name("name")
                if nm is not None:
                    names.add(src[nm.start_byte:nm.end_byte])
            stack.extend(node.children)
        return names
    except Exception:                                 # noqa: BLE001 — any TS hiccup → regex
        return None


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
        self._graph_cache: dict | None = None   # query-independent def/ref graph (see _graph)
        self._graph_builds = 0                   # rebuild counter (observability + tests)

    # --- Retriever contract -------------------------------------------------
    def retrieve(self, query: str, k: int = 6) -> list[Snippet]:
        """The RELATED CODE tier: a relevance-RANKED repo MAP — which files matter for this query,
        shown as definition SIGNATURES, not code excerpts. Why a map and not snippets: an A/B on a
        lexical-trap task (bug in a neutral-vocabulary file the term search never ranks) showed the
        map ties-or-beats injected snippets AND stays robust when the lexical signal points at the
        WRONG file — the model reads real code on demand instead of anchoring on a guessed excerpt.
        Ranking is STRUCTURAL (personalized PageRank over the def/ref graph, seeded by the lexical
        matches), so a relevant file surfaces even with zero query-word overlap when a matched file
        calls it — the case a purely-lexical ranking truncates on a large repo (see graph_map)."""
        text = self.graph_map(query)
        if not text:
            return []
        return [Snippet(path="(repo map)", text=text, score=float(text.count("(matches:")))]

    def scoped_map(self, query: str, max_files: int = 200, max_chars: int = 4000) -> str:
        """Definition skeleton of the repo, ranked by lexical relevance to the query. Matched files
        come first (so truncation keeps the relevant ones) and are annotated with the terms they hit;
        zero-match files follow (cheap, and they catch neutral-vocabulary targets the term search
        would miss — the lexical-trap case). Bounded by max_chars: the same discipline as every tier."""
        files = self._code_files(max_files)
        if not files:
            return ""
        matched: dict[str, set] = {}
        terms = _terms(query)
        if terms:
            for path, info in self._search(terms).items():
                matched[os.path.relpath(path, self.root)] = info["terms"]
        files.sort(key=lambda rel: len(matched.get(rel, ())), reverse=True)  # stable → ties stay sorted
        blocks: list[str] = []
        used = 0
        for rel in files:
            defs = self._defs_in(os.path.join(self.root, rel), 12)
            if not defs:
                continue
            hit = matched.get(rel)
            head = rel + (f"   (matches: {', '.join(sorted(hit))})" if hit else "")
            block = head + "\n" + "\n".join("  " + d for d in defs)
            if used + len(block) + 1 > max_chars:
                break
            blocks.append(block)
            used += len(block) + 1
        return "\n".join(blocks)

    def snippets(self, query: str, k: int = 6) -> list[Snippet]:
        """Code-context windows around term hits (the earlier default). Kept for ON-DEMAND use —
        e.g. a future 'show me the code around X' tool — but no longer the standing discovery tier."""
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

    # --- structural map: rank by personalized PageRank over the def/ref graph ---
    def graph_map(self, query: str, max_files: int = 400, max_chars: int = 4000) -> str:
        """Repo map ranked by PERSONALIZED PAGERANK over the symbol def/ref graph, seeded on the
        files that match the query lexically. Rank flows along call/import edges, so a relevant file
        surfaces even with ZERO query-word overlap when a matched file references it — exactly the
        neutral-vocabulary target a purely-lexical ranking truncates on a large repo. Degrades to
        lexical order when there is no graph signal. Bounded by max_chars, like every tier.

        The query-INDEPENDENT graph (defs/edges/skeletons) is cached on this instance and rebuilt
        only when the tree changes (see _graph), so per-turn cost is just lexical search + PageRank,
        not re-reading every file — the cost stays flat across a multi-turn session until an edit."""
        g = self._graph(max_files)
        files = list(g["files"])
        if not files:
            return ""
        terms = _terms(query)
        matched: dict[str, set] = {}
        if terms:
            for path, info in self._search(terms).items():
                matched[os.path.relpath(path, self.root)] = info["terms"]
        seeds = {rel: float(len(t)) for rel, t in matched.items() if rel in g["fileset"]}
        pr = self._pagerank(files, g["edges"], seeds)
        # rank: structural score, then lexical strength, then path (deterministic ties)
        files.sort(key=lambda rel: (pr.get(rel, 0.0), len(matched.get(rel, ()))), reverse=True)
        blocks: list[str] = []
        used = 0
        for rel in files:
            dlines = g["skeleton"].get(rel)
            if not dlines:
                continue
            hit = matched.get(rel)
            head = rel + (f"   (matches: {', '.join(sorted(hit))})" if hit else "")
            block = head + "\n" + "\n".join("  " + d for d in dlines)
            if used + len(block) + 1 > max_chars:
                break
            blocks.append(block)
            used += len(block) + 1
        return "\n".join(blocks)

    def _graph(self, max_files: int) -> dict:
        """Build (or reuse) the query-INDEPENDENT def/ref graph. Cached on this instance and
        invalidated by a fingerprint of the code files (path + mtime + size), so it rebuilds ONLY
        when the tree actually changes (e.g. the agent edits a file) — not every turn. Reads and
        parses each file ONCE per rebuild (defs + skeleton + ref tokens in one pass)."""
        files = self._code_files(max_files)
        sig = self._fingerprint(files)
        c = self._graph_cache
        if c is not None and c["sig"] == sig:
            return c
        defs: dict[str, set] = {}
        sym2file: dict[str, set] = {}
        skeleton: dict[str, list] = {}
        tokens: dict[str, set] = {}
        for rel in files:
            names, lines, toks = self._scan_file(rel)
            if lines:
                skeleton[rel] = lines
            tokens[rel] = toks
            if names:
                defs[rel] = names
                for n in names:
                    sym2file.setdefault(n, set()).add(rel)
        edges = self._edges_from_tokens(files, defs, sym2file, tokens)
        self._graph_builds += 1
        self._graph_cache = {"sig": sig, "files": files, "fileset": set(files),
                             "skeleton": skeleton, "edges": edges}
        return self._graph_cache

    def _fingerprint(self, files: list[str]) -> tuple:
        """Cheap staleness key: (rel, mtime_ns, size) per file. Stat-only — no reads — so computing
        it each turn is far cheaper than the rebuild it guards."""
        out = []
        for rel in files:
            try:
                st = os.stat(os.path.join(self.root, rel))
                out.append((rel, st.st_mtime_ns, st.st_size))
            except OSError:
                out.append((rel, 0, 0))
        return tuple(out)

    def _scan_file(self, rel: str):
        """One read per file → (def names, skeleton lines, ref tokens). Names use tree-sitter when
        available (precise), else a regex; skeleton lines and tokens are regex (display + refs)."""
        try:
            with open(os.path.join(self.root, rel), "r", errors="replace") as fh:
                src = fh.read()
        except OSError:
            return set(), [], set()
        ts = _ts_def_names(os.path.join(self.root, rel), src)
        if ts is not None:
            names = {n for n in ts if len(n) >= 4}
        else:
            names = {m.group(1) for m in _NAME_RE.finditer(src) if len(m.group(1)) >= 4}
        lines = [ln.strip()[:120] for ln in src.splitlines() if _DEF_RE.match(ln)][:12]
        tokens = set(_TOKEN.findall(src))
        return names, lines, tokens

    @staticmethod
    def _edges_from_tokens(files: list[str], defs: dict, sym2file: dict, tokens: dict) -> dict:
        """Directed edges file → file it references, from the cached ref tokens. A references B if A
        mentions a symbol DEFINED in B; symbols defined in many files are skipped (noisy names)."""
        usable = {s: fs for s, fs in sym2file.items() if len(fs) <= 4}
        edges: dict[str, dict] = {}
        for rel in files:
            own = defs.get(rel, set())
            out: dict[str, int] = {}
            for t in tokens.get(rel, ()):
                if t in own:
                    continue
                for tgt in usable.get(t, ()):
                    if tgt != rel:
                        out[tgt] = out.get(tgt, 0) + 1
            if out:
                edges[rel] = out
        return edges

    @staticmethod
    def _pagerank(nodes: list[str], edges: dict, seeds: dict, d: float = 0.85,
                  iters: int = 40) -> dict:
        """Personalized PageRank. Personalization mass sits on the seed files (the lexical matches);
        with no seeds it's uniform (→ plain centrality). Dangling nodes redistribute to the seeds."""
        n = len(nodes)
        if n == 0:
            return {}
        total = sum(seeds.values())
        p = ({x: seeds.get(x, 0.0) / total for x in nodes} if total > 0
             else {x: 1.0 / n for x in nodes})
        r = dict(p)
        outsum = {u: sum(w.values()) for u, w in edges.items()}
        nodeset = set(nodes)
        for _ in range(iters):
            nr = {x: (1 - d) * p[x] for x in nodes}
            dangling = 0.0
            for u in nodes:
                ru = r[u]
                s = outsum.get(u, 0)
                if s > 0:
                    for v, w in edges[u].items():
                        if v in nodeset:
                            nr[v] += d * ru * (w / s)
                else:
                    dangling += d * ru
            if dangling:
                for x in nodes:
                    nr[x] += dangling * p[x]
            r = nr
        return r

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
