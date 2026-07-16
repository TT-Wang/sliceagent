"""High-precision catastrophic-command floor for the autonomous local kernel.

This is not a permission or risk-classification system. It examines only shell-like command
surfaces and stops a tiny set of unambiguous machine-level destructive actions. Mentions in
arguments, quoted examples, search patterns, comments, arbitrary Python bodies, and parser
uncertainty are allowed. Sandbox and workspace boundaries remain the binding protections.
"""
from __future__ import annotations

import ast
import ntpath
import os
import posixpath
import re
import shlex
from dataclasses import dataclass


SHELL_TOOLS = frozenset({"run_command", "proc_start", "terminal_open", "terminal_send"})
_POWER = frozenset({"shutdown", "reboot", "halt", "poweroff"})
_SHELL_INTERPRETERS = frozenset({"sh", "bash", "zsh"})
_BOUNDARY = frozenset({";", "&&", "||", "&", "|"})
_CONTROL_WORDS = frozenset({
    "if", "then", "elif", "else", "fi", "case", "esac", "for", "while", "until", "select", "do", "done",
    "{", "}", "function", "coproc", "[[", "]]",
})
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", re.DOTALL)
_DEVICE = re.compile(
    r"^/dev/(?:"
    r"r?disk\d+(?:s\d+)?|"
    r"(?:sd|hd|vd|xvd)[a-z]+\d*|"
    r"nvme\d+n\d+(?:p\d+)?|mmcblk\d+(?:p\d+)?|"
    r"md\d+(?:p\d+)?|dm-\d+|loop\d+(?:p\d+)?|"
    r"mapper/[^/]+|disk/by-(?:id|path|uuid|partuuid)/[^/]+|zvol/.+"
    r")$",
    re.IGNORECASE,
)
_DISKUTIL_DEVICE = re.compile(r"^(?:/dev/)?r?disk\d+(?:s\d+)?$", re.IGNORECASE)
_FORK_BOMB = re.compile(
    r"^\s*:\s*\(\s*\)\s*\{[^\n]*:\s*\|\s*:\s*&[^\n]*\}\s*;\s*:\s*$",
)
_SHELL_COMMAND_WORDS = frozenset({
    "rm", "sudo", "command", "nohup", "exec", "time", "env", "shutdown", "shutdown.exe",
    "reboot", "halt", "poweroff", "systemctl", "launchctl", "wipefs", "mkfs", "mke2fs",
    "mkfs.ext2", "mkfs.ext3", "mkfs.ext4", "mkfs.xfs", "mkfs.btrfs", "newfs_apfs", "diskutil",
    "dd", "sh", "bash", "zsh", "tee", "exit", "true", "false",
})


@dataclass(frozen=True)
class _Stage:
    tokens: tuple[str, ...]
    separator: str = ""


def _normalized_shell_token(raw: str, value: str) -> str:
    """Use the shell's unescaped value where quote provenance is not semantically relevant.

    ``shlex(posix=False)`` retains quote kind for root/home precision, but it also retains ordinary POSIX
    backslashes: ``\\rm`` and ``r\\m`` both execute ``rm`` and are common ways to bypass aliases. Whole quoted
    words stay raw so :func:`_token_value` can distinguish single from double quotes. A quoted value attached
    to an assignment-style word (``of="/dev/sda"``) is safe to normalize as one value; this covers the normal
    ``dd`` spelling without guessing across composite shell words.
    """
    # Assignment values use shell quoting on every supported command surface.  Normalize the common
    # ``dd of="/dev/sda"`` spelling before the host-specific backslash rule below; otherwise a Windows host
    # retains the quotes inside the value and misses the same literal device operation it catches on POSIX.
    match = re.fullmatch(r"[^'\"]+=([\"'])(.*)\1", raw, re.DOTALL)
    if match is not None:
        return value
    # Windows executes this surface through Git Bash, but an executable may still be spelled as a Windows
    # absolute path whose backslashes are separators. Do not corrupt that path by applying sh unescaping.
    # Git Bash also accepts alias-bypass spellings such as ``\rm`` and ``r\m``; normalize only when POSIX shlex
    # resolves the whole token to a command word this classifier understands, never arbitrary path/arg tokens.
    if os.name == "nt":
        if value.casefold() in _SHELL_COMMAND_WORDS and re.fullmatch(
                r"(?:[^\\]|\\[A-Za-z0-9_.-])+", raw, re.DOTALL):
            return value
        return raw
    if "'" not in raw and '"' not in raw:
        if raw == value:
            return raw
        # Only erase backslashes that quote ordinary executable/path characters. Backslashes before shell
        # expansion characters are semantically protective: ``\~`` and ``\$HOME`` are literal relative names,
        # not home operands, and ``/\*`` is a literal asterisk rather than a root glob.
        safe_escapes = re.fullmatch(r"(?:[^\\]|\\[A-Za-z0-9_./-])+", raw, re.DOTALL)
        return value if safe_escapes is not None else raw
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        return raw
    return raw


def _executable_basename(value: str) -> str:
    """Return a shell executable basename under the supported host's command semantics.

    POSIX treats backslashes in quoted names literally. Windows uses Git Bash for the command surface but also
    accepts Windows executable spellings, so only that host applies both lexical path grammars. This is a
    deterministic string operation and never resolves or executes the supplied path.
    """
    basename = posixpath.basename(value)
    return ntpath.basename(basename) if os.name == "nt" else basename


def _stages(body: str) -> tuple[_Stage, ...] | None:
    """Tokenize top-level shell stages while retaining simple quote boundaries.

    Quote retention matters for precision: ``rm -rf '$HOME'`` names a literal file, while
    ``rm -rf "$HOME"`` expands to the user's home. Any syntax uncertainty abstains.
    """
    result: list[_Stage] = []
    lines = body.splitlines() or [body]
    for line in lines[:-1]:
        stripped = line.rstrip()
        if (len(stripped) - len(stripped.rstrip("\\"))) % 2:
            return None
    for line in lines:
        if not line.strip():
            continue
        try:
            lexer = shlex.shlex(line, posix=False, punctuation_chars=";&|<>")
            lexer.whitespace_split = True
            lexer.commenters = "#"
            tokens = list(lexer)
            value_lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|<>")
            value_lexer.whitespace_split = True
            value_lexer.commenters = "#"
            value_tokens = list(value_lexer)
        except (TypeError, ValueError):
            return None
        if len(tokens) != len(value_tokens):
            # ``posix=False`` is needed to retain quote kind, but it splits adjacent quoted/unquoted
            # fragments of one shell word (for example ``"build dir"/*``). Treat that composite syntax
            # as parser uncertainty; otherwise the detached ``/*`` looks like a root deletion operand. The
            # one high-confidence exception is a double-quoted HOME followed by an unquoted whole-home wipe
            # suffix: merge that exact semantic word so the quote does not become a trivial safeguard bypass.
            tokens = _merge_quoted_home_wipes(tokens)
            if len(tokens) != len(value_tokens):
                return None
        tokens = [
            _normalized_shell_token(raw, value)
            for raw, value in zip(tokens, value_tokens)
        ]
        if any(token in {"<<", "<<<"} for token in tokens):
            # A following line may be inert here-document data. This lightweight classifier deliberately
            # abstains instead of pretending to implement a shell parser and blocking documentation/scripts.
            return None
        if tokens and tokens[0] in _BOUNDARY:
            return None
        current: list[str] = []
        if tokens and result and not result[-1].separator:
            # A physical newline is a command separator unless the previous line ended with an explicit
            # continuation/operator. Retain it so simple reachability such as ``exit\nshutdown`` is sound.
            result[-1] = _Stage(result[-1].tokens, ";")
        for token in tokens:
            if token in _BOUNDARY:
                if current:
                    result.append(_Stage(tuple(current), token))
                    current = []
                continue
            current.append(token)
        if current:
            result.append(_Stage(tuple(current)))
    if any(_reserved_control_at_command_start(stage.tokens) for stage in result):
        # Correct reachability through shell conditionals requires a real shell AST. Reserved words are syntax
        # only in command position; an operand named `if`/`done` must not let `rm -rf / if` evade the floor.
        return None
    return tuple(result)


def _token_value(token: str) -> tuple[str, str]:
    """Return ``(shell value, quote kind)`` for a simple lexer token.

    Composite quoting is intentionally left opaque; the classifier would rather miss an unusual spelling
    than turn parser guesswork into a user-facing refusal.
    """
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        return token[1:-1], token[0]
    if "'" in token or '"' in token:
        return token, "complex"
    return token, ""


def _merge_quoted_home_wipes(tokens: list[str]) -> list[str]:
    """Merge exact ``"$HOME"`` + unquoted wipe suffix fragments split by ``shlex(posix=False)``."""
    homes = {'"$HOME"': "$HOME", '"${HOME}"': "${HOME}"}
    suffixes = {"/", "/.", "/*", "/.*", "/.??*", "/{*,.*}"}
    merged: list[str] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and tokens[index] in homes and tokens[index + 1] in suffixes:
            merged.append(homes[tokens[index]] + tokens[index + 1])
            index += 2
            continue
        merged.append(tokens[index])
        index += 1
    return merged


def _reserved_control_at_command_start(tokens: tuple[str, ...]) -> bool:
    """Whether a real, unquoted shell reserved word begins this top-level command stage."""
    index = 0
    while index < len(tokens) and _ASSIGNMENT.fullmatch(_token_value(tokens[index])[0]):
        index += 1
    while index < len(tokens) and _token_value(tokens[index]) == ("!", ""):
        index += 1
    if index >= len(tokens):
        return False
    value, quote = _token_value(tokens[index])
    return not quote and value in _CONTROL_WORDS


def _command_index(tokens: tuple[str, ...]) -> int | None:
    """Locate a direct executable after assignments and transparent shell wrappers."""
    index = 0
    while index < len(tokens):
        while index < len(tokens) and _ASSIGNMENT.fullmatch(_token_value(tokens[index])[0]):
            index += 1
        if index >= len(tokens):
            return None
        raw_command, quote = _token_value(tokens[index])
        if quote == "complex":
            return None
        command = _executable_basename(raw_command).casefold()
        if command == "!":
            index += 1
            continue
        if command == "sudo":
            index += 1
            while index < len(tokens):
                option = _token_value(tokens[index])[0]
                if option == "--":
                    index += 1
                    break
                if not option.startswith("-"):
                    break
                if option in {
                        "--help", "-V", "--version", "-K", "--remove-timestamp",
                        "-e", "--edit", "-v", "--validate", "-l", "--list",
                } or (
                        option.startswith("-") and not option.startswith("--") and "l" in option[1:]
                ):
                    return None
                value_options = {
                    "-u", "--user", "-g", "--group", "-h", "--host", "-p", "--prompt",
                    "-r", "--role", "-t", "--type", "-C", "--close-from", "-D", "--chdir",
                    "-R", "--chroot", "-T", "--command-timeout",
                }
                no_value_options = {
                    "-A", "--askpass", "-B", "--bell", "-b", "--background",
                    "-E", "--preserve-env", "-H", "--set-home", "-i", "--login",
                    "-k", "--reset-timestamp", "-N", "--no-update", "-n", "--non-interactive",
                    "-P", "--preserve-groups", "-S", "--stdin", "-s", "--shell",
                }
                consumes_value = option in value_options
                known_attached_value = (
                    option.startswith(("-u", "-g", "-h", "-p", "-r", "-t", "-C", "-D", "-R", "-T"))
                    and len(option) > 2
                ) or (option.startswith("--") and "=" in option and option.split("=", 1)[0] in value_options)
                attached_no_value = (
                    option.startswith("--preserve-env=")
                    or (option.startswith("-") and not option.startswith("--")
                        and len(option) > 2 and all(char in "ABbEHikNnPSs" for char in option[1:]))
                )
                if (option not in value_options and option not in no_value_options
                        and not known_attached_value and not attached_no_value):
                    # An unknown sudo option may consume the following token; guessing could misidentify
                    # an option argument as a destructive executable.
                    return None
                index += 1
                if consumes_value and index < len(tokens):
                    index += 1
            continue
        if command == "command":
            index += 1
            while index < len(tokens):
                option = _token_value(tokens[index])[0]
                if option == "--":
                    index += 1
                    break
                if option in {"-v", "-V", "--help", "--version"}:
                    return None
                if option == "-p":
                    index += 1
                    continue
                if option.startswith("-"):
                    return None
                break
            continue
        if command in {"nohup", "exec", "time"}:
            index += 1
            while index < len(tokens):
                option = _token_value(tokens[index])[0]
                if option == "--":
                    index += 1
                    break
                if option in {"--help", "--version"}:
                    return None
                if not option.startswith("-"):
                    break
                if command == "exec" and option in {"-c", "-l"}:
                    index += 1
                    continue
                if command == "exec" and option == "-a" and index + 1 < len(tokens):
                    index += 2
                    continue
                if command == "nohup":
                    return None
                if command == "time" and option in {"-p", "-v", "--verbose", "-a", "--append"}:
                    index += 1
                    continue
                if command == "time" and option in {"-f", "--format", "-o", "--output"} \
                        and index + 1 < len(tokens):
                    index += 2
                    continue
                return None
            continue
        if command == "env":
            index += 1
            while index < len(tokens):
                option = _token_value(tokens[index])[0]
                if _ASSIGNMENT.fullmatch(option):
                    index += 1
                    continue
                if option == "--":
                    index += 1
                    break
                if not option.startswith("-"):
                    break
                if option in {"--help", "--version"}:
                    return None
                value_options = {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"}
                no_value_options = {"-i", "--ignore-environment", "-0", "--null", "-v", "--debug"}
                attached_value = (
                    option.startswith(("-u", "-C", "-S")) and len(option) > 2
                ) or (option.startswith("--") and "=" in option and option.split("=", 1)[0] in value_options)
                if option not in value_options and option not in no_value_options and not attached_value:
                    return None
                index += 1
                if option in value_options and index < len(tokens):
                    index += 1
            continue
        return index
    return None


def _recursive_rm_reason(tokens: tuple[str, ...], index: int) -> str | None:
    args = tokens[index + 1:]
    recursive = False
    preserve_root = False
    operands: list[tuple[str, str]] = []
    options = True
    for raw_token in args:
        value, quote = _token_value(raw_token)
        if options and value == "--":
            options = False
            continue
        if options and value.startswith("-"):
            if value in {"--help", "--version"}:
                return None
            if value == "--recursive" or (
                    not value.startswith("--") and "r" in value[1:].casefold()
            ):
                recursive = True
            if value == "--preserve-root" or value.startswith("--preserve-root="):
                preserve_root = True
            continue
        operands.append((value, quote))
    if not recursive:
        return None
    home = os.path.realpath(os.path.expanduser("~"))
    symbolic_home_globs = {
        "~/*", "~/.*", "~/.??*", "~/{*,.*}",
        "$HOME/*", "$HOME/.*", "$HOME/.??*", "$HOME/{*,.*}",
        "${HOME}/*", "${HOME}/.*", "${HOME}/.??*", "${HOME}/{*,.*}",
    }
    literal_home_globs = {
        f"{home}/*", f"{home}/.*", f"{home}/.??*", f"{home}/{{*,.*}}",
    }
    for token, quote in operands:
        if quote == "complex":
            continue
        if token in {"/*", "/{*,.*}"} and not quote:
            return "recursive deletion of root or home"
        if token in symbolic_home_globs | literal_home_globs and not quote:
            return "recursive deletion of root or home"
        # Parameter expansion occurs inside double quotes; tilde expansion does not.
        if token in {"$HOME", "${HOME}"} and quote != "'":
            return "recursive deletion of root or home"
        if token == "~" and not quote:
            return "recursive deletion of root or home"
        expanded = token
        if quote != "'":
            expanded = os.path.expandvars(expanded)
        if not quote:
            expanded = os.path.expanduser(expanded)
        # This is shell syntax, not only a path lookup on the classifier host.  Keep the realpath check (it
        # catches POSIX spellings such as /./, /tmp/.., and symlinks to root), and add a lexical POSIX check so
        # those same root operands remain visible when the classifier itself runs on Windows.
        host_root = os.path.isabs(expanded) and os.path.realpath(expanded) == "/"
        posix_root = posixpath.isabs(expanded) and posixpath.normpath(expanded) == "/"
        if host_root or posix_root:
            if preserve_root:
                continue
            return "recursive deletion of root or home"
        # Relative cleanup is intentionally context-dependent and must abstain. The execution workspace can
        # change in-process, so resolving it against the classifier process cwd would invent certainty.
        if os.path.isabs(expanded) and os.path.realpath(expanded) == home:
            return "recursive deletion of root or home"
    return None


def _option_present(values: tuple[str, ...], names: set[str]) -> bool:
    """Return whether an option occurs before ``--``; operands after it never become control flags."""
    for value in values:
        if value == "--":
            return False
        if value in names:
            return True
    return False


def _short_option_present(values: tuple[str, ...], flag: str) -> bool:
    for value in values:
        if value == "--":
            return False
        if value.startswith("-") and not value.startswith("--") and flag in value[1:]:
            return True
    return False


def _device_target(value: str) -> bool:
    return bool(_DEVICE.fullmatch(value))


def _diskutil_reason(values: tuple[str, ...]) -> str | None:
    if not values or _option_present(values, {"--help", "-h", "--version"}):
        return None
    verb = values[0].casefold()
    rest = values[1:]
    if verb in {"erasedisk", "erasevolume"}:
        # Both forms require format, name, and a concrete device. Missing/incomplete invocations print usage.
        if len(rest) >= 3 and _DISKUTIL_DEVICE.fullmatch(rest[-1]):
            return "filesystem format"
    elif verb == "partitiondisk":
        if len(rest) >= 2 and _DISKUTIL_DEVICE.fullmatch(rest[0]):
            return "filesystem format"
    elif verb in {"zerodisk", "randomdisk"}:
        if rest and _DISKUTIL_DEVICE.fullmatch(rest[-1]):
            return "raw write to a device"
    elif verb == "secureerase":
        if len(rest) >= 2 and rest[0].isdigit() and _DISKUTIL_DEVICE.fullmatch(rest[-1]):
            return "raw write to a device"
    return None


def _systemctl_reason(values: tuple[str, ...]) -> str | None:
    if _option_present(values, {"--help", "--version", "--dry-run"}):
        return None
    scoped = {"--user", "--global", "--root", "--image", "--machine", "--host"}
    if _option_present(values, scoped) or any(
            value.startswith(tuple(f"{option}=" for option in scoped)) for value in values):
        return None
    for value in values:
        if value == "--":
            continue
        if value.startswith("-"):
            continue
        return "system power control" if value.casefold() in _POWER else None
    return None


def _windows_shutdown_reason(values: tuple[str, ...]) -> str | None:
    """Classify Windows ``shutdown[.exe]`` by the action switch, not merely its executable name."""
    options = tuple(value.casefold() for value in values)
    # Microsoft documents these as help/UI/sign-out/abort/hibernate operations. Several are required to be
    # standalone and ignore or reject combinations, so none is a high-confidence machine shutdown.
    if any(option in {"/?", "/a", "/i", "/l", "/h"} for option in options):
        return None
    return "system power control" if any(
        option in {"/s", "/sg", "/r", "/g", "/p"} for option in options
    ) else None


def _stage_reason(stage: _Stage, *, depth: int = 0) -> str | None:
    index = _command_index(stage.tokens)
    if index is None:
        return None
    executable = _executable_basename(_token_value(stage.tokens[index])[0]).casefold()
    arg_values = tuple(_token_value(token)[0] for token in stage.tokens[index + 1:])
    if _option_present(arg_values, {"--help", "--version"}):
        return None
    if executable in {"shutdown", "shutdown.exe"} and (
            executable == "shutdown.exe" or any(value.startswith("/") for value in arg_values)):
        return _windows_shutdown_reason(arg_values)
    if executable == "shutdown" and _option_present(
            arg_values, {"-c", "--cancel", "-k", "--dry-run", "--show"}):
        # Cancel and warn-only are recovery/simulation operations, not power control.
        return None
    if executable in {"reboot", "halt", "poweroff"} and _option_present(
            arg_values, {"-w", "--wtmp-only"}):
        return None
    if executable == "systemctl":
        return _systemctl_reason(arg_values)
    if executable == "launchctl":
        if len(arg_values) >= 2 and arg_values[0].casefold() == "reboot" \
                and arg_values[1].casefold() == "system":
            return "system power control"
        return None
    if (executable in {"mke2fs", "mkfs.ext2", "mkfs.ext3", "mkfs.ext4"}
            and "-n" in arg_values):
        return None
    if executable == "mkfs.xfs" and "-N" in arg_values:
        return None
    if executable == "mkfs.btrfs" and "--dry-run" in arg_values:
        return None
    if executable.startswith("newfs_") and "-N" in arg_values:
        return None
    if executable in _POWER:
        return "system power control"
    if executable == "wipefs":
        if _option_present(arg_values, {"-n", "--no-act"}) or _short_option_present(arg_values, "n"):
            return None
        destructive = (
            _option_present(arg_values, {"-a", "--all", "-o", "--offset"})
            or _short_option_present(arg_values, "a")
            or _short_option_present(arg_values, "o")
            or any(value.startswith("--offset=") for value in arg_values)
        )
        if destructive and any(_device_target(value) for value in arg_values):
            return "filesystem format"
        return None
    if (executable in {"mkfs", "mke2fs"}
            or executable.startswith(("mkfs.", "newfs_"))):
        return "filesystem format" if any(_device_target(value) for value in arg_values) else None
    if executable == "diskutil":
        return _diskutil_reason(arg_values)
    if executable == "dd" and any(
        value.casefold().startswith("of=") and _device_target(value[3:])
        for value in arg_values
    ):
        return "raw write to a device"
    if executable == "rm":
        return _recursive_rm_reason(stage.tokens, index)
    if executable in _SHELL_INTERPRETERS and depth < 3:
        args = stage.tokens[index + 1:]
        for position, raw_token in enumerate(args[:-1]):
            token = _token_value(raw_token)[0]
            if token.startswith("-") and "c" in token[1:]:
                prior_options = (_token_value(item)[0] for item in args[:position + 1])
                if any(option == "--noexec" or (
                        option.startswith("-") and not option.startswith("--") and "n" in option[1:]
                ) for option in prior_options):
                    return None
                body, quote = _token_value(args[position + 1])
                return None if quote == "complex" else _body_reason(body, depth=depth + 1)
    if executable == "tee" and any(
            _device_target(_token_value(token)[0]) for token in stage.tokens[index + 1:]):
        return "raw write to a device"
    for position, token in enumerate(stage.tokens[:-1]):
        target, quote = _token_value(stage.tokens[position + 1])
        if token in {">", ">>"} and quote != "complex" and _device_target(target):
            return "raw write to a device"
    return None


def _shell_terminates_after_stage(tokens: tuple[str, ...]) -> bool:
    """Recognize only shell-builtin ``exit``/``exec`` forms that make a later stage unreachable."""
    index = 0
    while index < len(tokens) and _ASSIGNMENT.fullmatch(_token_value(tokens[index])[0]):
        index += 1
    while index < len(tokens) and _token_value(tokens[index]) == ("!", ""):
        index += 1

    def skip_command(start: int) -> int | None:
        current = start + 1
        while current < len(tokens):
            option = _token_value(tokens[current])[0]
            if option == "--":
                return current + 1
            if option == "-p":
                current += 1
                continue
            if option.startswith("-"):
                return None
            return current
        return None

    def skip_builtin(start: int) -> int | None:
        current = start + 1
        if current < len(tokens) and _token_value(tokens[current])[0] == "--":
            current += 1
        if current >= len(tokens) or _token_value(tokens[current])[0].startswith("-"):
            return None
        return current

    if index < len(tokens) and _executable_basename(_token_value(tokens[index])[0]).casefold() == "time":
        index += 1
        while index < len(tokens):
            option = _token_value(tokens[index])[0]
            if option == "--":
                index += 1
                break
            if option in {"-p", "-v", "--verbose", "-a", "--append"}:
                index += 1
                continue
            if option in {"-f", "--format", "-o", "--output"} and index + 1 < len(tokens):
                index += 2
                continue
            if option.startswith("-"):
                return False
            break

    if index < len(tokens):
        wrapper = _executable_basename(_token_value(tokens[index])[0]).casefold()
        if wrapper == "command":
            next_index = skip_command(index)
            if next_index is None:
                return False
            index = next_index
        elif wrapper == "builtin":
            next_index = skip_builtin(index)
            if next_index is None:
                return False
            index = next_index
    if index >= len(tokens):
        return False
    value, quote = _token_value(tokens[index])
    return not quote and _executable_basename(value).casefold() in {"exit", "exec"}


def _body_reason(body: str, *, depth: int = 0) -> str | None:
    if not body.strip():
        return None
    if _FORK_BOMB.fullmatch(body):
        return "fork bomb"
    stages = _stages(body)
    if stages is None:
        return None
    last_status: bool | None = None
    terminated = False
    for position, stage in enumerate(stages):
        if terminated:
            break
        if position:
            separator = stages[position - 1].separator or ";"
            if (separator == "&&" and last_status is False) or (
                    separator == "||" and last_status is True):
                # A statically unreachable branch does not change the status of the AND/OR list.
                continue
        reason = _stage_reason(stage, depth=depth)
        if reason:
            return reason
        index = _command_index(stage.tokens)
        if index is None:
            last_status = None
            continue
        executable = _executable_basename(_token_value(stage.tokens[index])[0]).casefold()
        negated = sum(_token_value(token)[0] == "!" for token in stage.tokens[:index]) % 2
        if executable in {"true", ":"}:
            last_status = not negated
        elif executable == "false":
            last_status = bool(negated)
        else:
            last_status = None
        incoming = stages[position - 1].separator if position else ""
        if (_shell_terminates_after_stage(stage.tokens)
                and stage.separator not in {"|", "&"} and incoming != "|"):
            terminated = True
    return None


def _literal_command(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        values = []
        for item in node.elts:
            if not isinstance(item, ast.Constant) or not isinstance(item.value, (str, int, float)):
                return None
            values.append(shlex.quote(str(item.value)))
        return " ".join(values)
    return None


class _StraightLineCalls(ast.NodeVisitor):
    """Collect calls in one definitely executed module statement, excluding deferred/dynamic bodies."""

    def __init__(self):
        self.calls: list[ast.Call] = []

    def visit_Call(self, node: ast.Call):  # noqa: N802 - ast visitor protocol
        self.calls.append(node)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp):  # noqa: N802
        for value in node.values:
            self.visit(value)
            truth = _literal_truth(value)
            if truth is None:
                # Later operands are conditional on runtime state. Abstain rather than presenting a stop for
                # an action that may never execute.
                break
            if isinstance(node.op, ast.And) and not truth:
                break
            if isinstance(node.op, ast.Or) and truth:
                break

    def visit_IfExp(self, node: ast.IfExp):  # noqa: N802
        self.visit(node.test)
        truth = _literal_truth(node.test)
        if truth is not None:
            self.visit(node.body if truth else node.orelse)

    def visit_Assert(self, node: ast.Assert):  # noqa: N802
        self.visit(node.test)
        truth = _literal_truth(node.test)
        if truth is False and node.msg is not None:
            self.visit(node.msg)

    def visit_Lambda(self, node: ast.Lambda):  # noqa: N802
        return None

    def visit_ListComp(self, node: ast.ListComp):  # noqa: N802
        return None

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp


def _literal_truth(node: ast.AST) -> bool | None:
    try:
        return bool(ast.literal_eval(node))
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def _call_argument(call: ast.Call) -> ast.AST | None:
    if call.args:
        return call.args[0]
    return next((keyword.value for keyword in call.keywords if keyword.arg in {"args", "command"}), None)


def _literal_shell_flag(call: ast.Call) -> bool | None:
    keyword = next((item for item in call.keywords if item.arg == "shell"), None)
    if keyword is None:
        return False
    if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, bool):
        return keyword.value.value
    return None


def _python_call_command(call: ast.Call, owner: str) -> str | None:
    node = _call_argument(call)
    if node is None:
        return None
    if owner in {"helper", "os-shell"}:
        return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None
    if owner == "subprocess-shell":
        return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None
    if owner != "subprocess":
        return None
    shell = _literal_shell_flag(call)
    if shell is None:
        return None
    if isinstance(node, (ast.List, ast.Tuple)):
        # argv execution is direct when shell=False. A sequence with shell=True has platform-specific shell
        # argument semantics and is deliberately outside this tiny classifier.
        return _literal_command(node) if not shell else None
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return None
    if shell:
        return node.value
    # subprocess with shell=False does not parse a string. A single executable name can still run directly;
    # whitespace/metacharacters mean Python will look for one oddly named file, not execute the apparent shell.
    return node.value if not re.search(r"\s|[;&|<>]", node.value) else None


def _execute_code_reason(code: str) -> str | None:
    """Check only straight-line, module-level literal shell calls in ``execute_code``.

    Function bodies, branches, loops, comprehensions, dynamic arguments, and shadowed helpers abstain. This
    catches the normal injected ``run('...')`` path without refusing code that merely contains a dangerous
    example in dead/deferred Python.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, TypeError, ValueError):
        return None
    helper_available = True
    trusted_modules = {"_os": "os", "_sp": "subprocess"}
    compound = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith,
                ast.Match, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for statement in tree.body:
        if not isinstance(statement, compound):
            visitor = _StraightLineCalls()
            visitor.visit(statement)
            for call in visitor.calls:
                owner = "helper" if (
                    isinstance(call.func, ast.Name) and call.func.id == "run" and helper_available
                ) else ""
                method = ""
                if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                    module = trusted_modules.get(call.func.value.id, "")
                    method = call.func.attr
                    if module == "os" and method in {"system", "popen"}:
                        owner = "os-shell"
                    elif module == "subprocess" and method in {"getoutput", "getstatusoutput"}:
                        owner = "subprocess-shell"
                    elif module == "subprocess" and method in {
                            "run", "Popen", "call", "check_call", "check_output"}:
                        owner = "subprocess"
                if owner:
                    command = _python_call_command(call, owner)
                    reason = _body_reason(command) if command is not None else None
                    if reason:
                        return reason

        # Update the small binding model after the statement (Python evaluates RHS/calls before assignment).
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if statement.name == "run":
                helper_available = False
            trusted_modules.pop(statement.name, None)
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                trusted_modules.pop(bound, None)
                if alias.name in {"os", "subprocess"}:
                    trusted_modules[bound] = alias.name
                if bound == "run":
                    helper_available = False
        if isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                bound = alias.asname or alias.name
                trusted_modules.pop(bound, None)
                if bound == "run":
                    helper_available = False
        stored = {
            node.id for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
        }
        if "run" in stored:
            helper_available = False
        for name in stored:
            trusted_modules.pop(name, None)
        mutated_module_objects = {
            node.value.id for node in ast.walk(statement)
            if isinstance(node, ast.Attribute)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and isinstance(node.value, ast.Name)
        }
        for name in mutated_module_objects:
            trusted_modules.pop(name, None)
    return None


def catastrophic_reason(name: str, args: dict | None) -> str | None:
    """Return a concrete catastrophic reason, otherwise ``None`` without blocking uncertainty."""
    values = args or {}
    if name == "execute_code":
        return _execute_code_reason(str(values.get("code") or ""))
    if name not in SHELL_TOOLS:
        return None
    body = str(values.get("command") or values.get("input") or "")
    return _body_reason(body)


__all__ = ["SHELL_TOOLS", "catastrophic_reason"]
