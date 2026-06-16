"""Binary-file sniffing (build task: binsniff) — keep text-only tools from choking on binaries.

A read_file / str_replace / fuzzy edit only makes sense on TEXT. Feeding a .png, a .so, or a
mojibake blob through those paths corrupts the slice (garbage in OPEN FILES) and wastes tokens.
This module is the single cheap gate the file tools call BEFORE treating bytes as text.

Two independent signals, in cost order:
    1. Extension  — a pure string check, no I/O (`has_binary_extension`). Catches the common case
       (.png/.so/.zip/...) without ever reading the file.
    2. Content    — a NUL byte, or >30% non-printable control chars (excluding \\n \\r \\t) in a
       head sample, marks bytes that decoded to text but clearly are not text.

NO-TRANSCRIPT INVARIANT: pure functions over a path string + a head sample; no state, no growing
context, no I/O of their own. The caller supplies the sample (already bounded to a file head).

Ported from Hermes `tools/binary_extensions.py` (BINARY_EXTENSIONS + has_binary_extension) and the
NUL / non-printable sniff in `tools/file_operations.py::_is_likely_binary`.

PUBLIC SIGNATURES (pinned):
    BINARY_EXTENSIONS: frozenset[str]
    has_binary_extension(path: str) -> bool
    looks_binary(path: str, sample: str) -> bool
"""

# Extensions whose contents can't be meaningfully treated as text.
# Copied verbatim from Hermes tools/binary_extensions.py (itself ported from free-code).
BINARY_EXTENSIONS = frozenset({
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".tif",
    # Videos
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv", ".m4v", ".mpeg", ".mpg",
    # Audio
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".aiff", ".opus",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz", ".z", ".tgz", ".iso",
    # Executables/binaries
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".obj", ".lib",
    ".app", ".msi", ".deb", ".rpm",
    # Documents (exclude .pdf — text-based, agents may want to inspect)
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".odp",
    # Fonts
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    # Bytecode / VM artifacts
    ".pyc", ".pyo", ".class", ".jar", ".war", ".ear", ".node", ".wasm", ".rlib",
    # Database files
    ".sqlite", ".sqlite3", ".db", ".mdb", ".idx",
    # Design / 3D
    ".psd", ".ai", ".eps", ".sketch", ".fig", ".xd", ".blend", ".3ds", ".max",
    # Flash
    ".swf", ".fla",
    # Lock/profiling data
    ".lockb", ".dat", ".data",
})

# How many leading chars of the sample to inspect, and the non-printable fraction that
# tips a decoded-but-not-text blob over into "binary" (mirrors Hermes _is_likely_binary).
_HEAD = 1000
_NON_PRINTABLE_RATIO = 0.30


def has_binary_extension(path: str) -> bool:
    """True if `path` has a known binary extension. Pure string check, no I/O."""
    dot = path.rfind(".")
    if dot == -1:
        return False
    return path[dot:].lower() in BINARY_EXTENSIONS


def looks_binary(path: str, sample: str) -> bool:
    """True if `path` is binary by extension, or `sample` looks binary by content.

    Content is binary when it contains a NUL byte, or when more than 30% of the first
    1000 chars are non-printable control chars (ord < 32, excluding \\n \\r \\t).

    `sample == ''` means we have no content signal -> non-binary unless the extension says so.
    O(head sample) only; never reads the file itself.
    """
    if has_binary_extension(path):
        return True
    if not sample:
        return False
    head = sample[:_HEAD]
    if "\x00" in head:  # a single NUL is a definitive binary marker
        return True
    non_printable = sum(1 for c in head if ord(c) < 32 and c not in "\n\r\t")
    return non_printable / len(head) > _NON_PRINTABLE_RATIO


if __name__ == "__main__":  # smoke: a .png ext and a NUL sample are binary; plain text is not
    assert looks_binary("logo.png", "") and looks_binary("x.txt", "a\x00b")
    assert not looks_binary("notes.txt", "plain readable text\nwith newlines\n")
    print("binsniff smoke OK")
