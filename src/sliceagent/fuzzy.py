"""Indentation-tolerant unique-span finder for str_replace.

Implements ONLY the two whitespace-tolerant strategies that are safe enough to
anchor an edit on (line-trim + indentation-flexible), plus their two position
helpers. Stripped of all replacement logic: this module NEVER writes. It only
answers "where, if anywhere uniquely, does ``old`` live in ``content``?" as a
single ``(start, end)`` character span.

The single public entry point is :func:`fuzzy_find_unique`. The caller does the
actual splice (``content[:start] + new + content[end:]``); keeping find and
replace separate is what lets the edit tool fall back from an exact match to a
fuzzy one without this module ever touching file bytes.

Uniqueness gate: a strategy that finds 0 or >1 candidates yields nothing. We
try line-trim first, then indentation-flexible, and return the span only when a
strategy produces exactly ONE candidate.
"""

from __future__ import annotations

from typing import List, Tuple

__all__ = ["fuzzy_find_unique"]


def fuzzy_find_unique(content: str, old: str) -> Tuple[int, int] | None:
    """Return the sole ``(start, end)`` char span matching ``old`` in ``content``.

    Tries, in order, the line-trimmed strategy then the indentation-flexible
    strategy. Returns the span only when a strategy finds EXACTLY ONE match.
    Returns ``None`` when:

    - ``old`` is empty,
    - a strategy finds zero candidates and no later strategy matches uniquely,
    - the first strategy that matches finds more than one candidate
      (ambiguous — the caller must supply more context).

    The returned span is a byte-correct slice into ``content``: for any
    replacement ``new``, ``content[:start] + new + content[end:]`` is the
    edited content. This function never replaces text itself.
    """
    if not old or not old.strip():   # an all-whitespace old has no anchor (matches every blank line → zero-width insert)
        return None

    for strategy in (_strategy_line_trimmed, _strategy_indentation_flexible):
        matches = strategy(content, old)
        if not matches:
            continue
        if len(matches) > 1:
            # Ambiguous under this strategy. Treat >1 as a hard failure
            # (needs more context) rather than falling through to a looser
            # strategy that would only be more ambiguous.
            return None
        return matches[0]

    return None


# =============================================================================
# Matching strategies
# =============================================================================

def _strategy_line_trimmed(content: str, pattern: str) -> List[Tuple[int, int]]:
    """Match line-by-line after stripping leading/trailing whitespace per line."""
    pattern_lines = [line.strip() for line in pattern.split("\n")]
    pattern_normalized = "\n".join(pattern_lines)

    content_lines = content.split("\n")
    content_normalized_lines = [line.strip() for line in content_lines]

    return _find_normalized_matches(
        content, content_lines, content_normalized_lines,
        pattern, pattern_normalized,
    )


def _strategy_indentation_flexible(content: str, pattern: str) -> List[Tuple[int, int]]:
    """Match line-by-line ignoring leading indentation entirely.

    Strips only LEADING whitespace (``lstrip``), so trailing whitespace still
    has to line up — this is intentionally narrower than line-trim and acts as a
    second, indentation-only tolerance pass.
    """
    content_lines = content.split("\n")
    content_stripped_lines = [line.lstrip() for line in content_lines]
    pattern_lines = [line.lstrip() for line in pattern.split("\n")]

    return _find_normalized_matches(
        content, content_lines, content_stripped_lines,
        pattern, "\n".join(pattern_lines),
    )


# =============================================================================
# Position helpers
# =============================================================================

def _calculate_line_positions(content_lines: List[str], start_line: int,
                              end_line: int, content_length: int) -> Tuple[int, int]:
    """Map a [start_line, end_line) line range to a (start, end) char span.

    Each line in ``content_lines`` is stored without its trailing newline, so the
    ``+ 1`` per line re-adds the ``\\n`` consumed by ``str.split('\\n')``. The
    end position drops the final newline and is clamped to ``content_length``.
    """
    start_pos = sum(len(line) + 1 for line in content_lines[:start_line])
    end_pos = sum(len(line) + 1 for line in content_lines[:end_line]) - 1
    end_pos = min(content_length, end_pos)
    return start_pos, end_pos


def _find_normalized_matches(content: str, content_lines: List[str],
                              content_normalized_lines: List[str],
                              pattern: str, pattern_normalized: str) -> List[Tuple[int, int]]:
    """Find every block whose normalized lines equal the normalized pattern.

    Slides a window of ``len(pattern_norm_lines)`` lines across the normalized
    content; on each equal block it maps the line window back to an
    ORIGINAL-content char span via :func:`_calculate_line_positions`. Returns
    all such spans (the caller enforces uniqueness).
    """
    pattern_norm_lines = pattern_normalized.split("\n")
    num_pattern_lines = len(pattern_norm_lines)

    matches: List[Tuple[int, int]] = []

    for i in range(len(content_normalized_lines) - num_pattern_lines + 1):
        block = "\n".join(content_normalized_lines[i:i + num_pattern_lines])
        if block == pattern_normalized:
            start_pos, end_pos = _calculate_line_positions(
                content_lines, i, i + num_pattern_lines, len(content)
            )
            matches.append((start_pos, end_pos))

    return matches
