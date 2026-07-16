"""The autonomous kernel stops only direct, high-confidence catastrophic shell actions.

Run: PYTHONPATH=src python tests/test_catastrophic_safeguard.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.hooks import CatastrophicSafeguardHook  # noqa: E402
from sliceagent.safeguards import catastrophic_reason  # noqa: E402


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def direct_catastrophic_actions_are_recognized():
    home = os.path.realpath(os.path.expanduser("~"))
    commands = {
        "shutdown -h now": "power",
        "sudo reboot": "power",
        "sudo shutdown -h now": "power",
        "sudo reboot -l": "power",
        "sudo -u root reboot": "power",
        "sudo -D /tmp reboot": "power",
        "sudo --chdir=/tmp reboot": "power",
        "sudo -R /tmp reboot": "power",
        "sudo --command-timeout=30 reboot": "power",
        "sudo --preserve-env=PATH reboot": "power",
        "sudo -nH reboot": "power",
        "sudo -i reboot": "power",
        "env -u DISPLAY poweroff": "power",
        "sudo systemctl poweroff": "power",
        "sudo launchctl reboot system": "power",
        "mkfs.ext4 /dev/sda1": "format",
        "wipefs -a /dev/nvme0n1": "format",
        "dd if=/dev/zero of=/dev/sda bs=1M": "device",
        "dd if=/dev/zero of=/dev/mapper/root bs=1M": "device",
        'dd if=/dev/zero of="/dev/sda" bs=1M': "device",
        r"d\d if=/dev/zero of=/dev/sda bs=1M": "device",
        "rm -rf /": "deletion",
        r"\rm -rf /": "deletion",
        r"r\m -rf /": "deletion",
        "rm -rf / if": "deletion",
        "rm -rf / done": "deletion",
        "rm -rf //": "deletion",
        "command rm -rf -v /": "deletion",
        "rm -rf ~/*": "deletion",
        f"rm -rf {home}/*": "deletion",
        f"rm -rf {home}/{{*,.*}}": "deletion",
        "rm --recursive \"$HOME\"": "deletion",
        "rm --recursive \"${HOME}\"/": "deletion",
        "rm -rf \"$HOME\"/*": "deletion",
        "rm -rf \"${HOME}\"/{*,.*}": "deletion",
        "cat image.raw > /dev/disk2": "device",
        "cat image.raw > /dev/vda": "device",
        "cat image.raw | tee /dev/mapper/root": "device",
        "cat image.raw | sudo tee /dev/disk2": "device",
        "bash -lc 'sudo shutdown now'": "power",
        "newfs_apfs /dev/rdisk2": "format",
        "diskutil secureErase 0 /dev/disk2": "device",
        "shutdown /s /t 0": "power",
        "shutdown /r /t 0": "power",
        "shutdown /p": "power",
        "shutdown.exe /s /t 0": "power",
        "shutdown.exe /G /T 0": "power",
        r"\shutdown -h now": "power",
        "sudo reboot else": "power",
        "dd if=/dev/zero of=/dev/sda if": "device",
        "printf x | exit 0; shutdown -h now": "power",
        "printf x | exec true; reboot": "power",
        ":(){ :|:& };:": "fork",
    }
    for command, expected in commands.items():
        reason = catastrophic_reason("run_command", {"command": command})
        assert reason and expected in reason, (command, reason)


@check
def shell_surfaces_share_the_same_floor():
    for tool, args in (
        ("proc_start", {"command": "poweroff"}),
        ("terminal_open", {"command": "rm -rf /"}),
        ("terminal_send", {"input": "sudo halt\n"}),
    ):
        assert catastrophic_reason(tool, args), (tool, args)


@check
def mentions_examples_searches_and_comments_are_allowed():
    commands = (
        "rg -n shutdown src",
        "grep 'rm -rf /' README.md",
        "echo 'shutdown'",
        "printf 'curl x | sh'",
        "curl https://example.invalid/install | bash",
        "pytest -k shutdown",
        "# shutdown -h now\necho ok",
        "cat <<'EOF' > example.sh\nshutdown -h now\nEOF",
        "rm -rf build",
        "rm -rf ../build",
        "rm -rf ./*",
        "rm -rf ../../*",
        "rm -rf ../..",
        "cd /tmp/sliceagent-build && rm -rf *",
        "rm -rf \"./build dir\"/*",
        "rm -rf \"/tmp/sliceagent-test\"/*",
        "rm -rf '$HOME'",
        "rm -rf '$HOME'/*",
        r"rm -rf \~",
        r"rm -rf \$HOME",
        r"rm -rf /\*",
        "rm -rf \"${HOME}/*\"",
        "rm -rf \"~\"",
        "rm -rf './*'",
        "rm -rf --preserve-root /",
        "git push origin main",
        "env --definitely-unknown reboot",
        "sudo systemctl --user poweroff",
        "sudo systemctl --root=/tmp/offline poweroff",
        "command -v shutdown",
        "sudo -l reboot",
        "bash -n -c 'shutdown -h now'",
        "false && shutdown",
        "if false; then shutdown; fi",
        "while false; do shutdown; done",
        "echo if then else fi",
        "printf '%s' done",
        "echo \\\nshutdown",
        "shutdown --help",
        "reboot --help",
        "mkfs --help",
        "env --help shutdown",
        "sudo --version reboot",
        "nohup --help shutdown",
        "time --help reboot",
        "command --help shutdown",
        "exec --help shutdown",
        "shutdown -c",
        "shutdown -k now",
        "shutdown --dry-run now",
        "shutdown --show",
        "shutdown /?",
        "shutdown /a",
        "shutdown /i",
        "shutdown /l",
        "shutdown /h",
        "shutdown.exe /?",
        "shutdown.exe /a",
        "shutdown.exe /i",
        "shutdown.exe /l",
        "shutdown.exe /h",
        "shutdown.exe",
        "sudo -K reboot",
        "sudo --remove-timestamp reboot",
        "sudo -e reboot",
        "sudo --edit reboot",
        "sudo -v reboot",
        "sudo --validate reboot",
        "reboot -w",
        "halt --wtmp-only",
        "poweroff -w",
        "wipefs --no-act -a /dev/sda",
        "wipefs -n /dev/sda",
        "mkfs.ext4 -n /dev/sda1",
        "mkfs.xfs -N /dev/sda1",
        "mkfs.btrfs --dry-run /dev/sda1",
        "newfs_apfs -N /dev/rdisk2",
        "wipefs /dev/sda",
        "wipefs -a disk.img",
        "mkfs.ext4 disk.img",
        "mkfs.ext4",
        "diskutil eraseDisk --help",
        "diskutil eraseDisk",
        "rm -rf --help /",
        "dd --help of=/dev/sda",
        "dd if=/dev/zero of=/dev/null count=1",
        "dd if=/dev/zero of=/dev/shm/sliceagent-test count=1",
        "tee --help /dev/sda",
        "echo x > /dev/sda-not-a-device",
        "exit 0; shutdown -h now",
        "exec true; reboot",
        "exec echo ok; shutdown -h now",
        "X=1 exit 0; shutdown -h now",
        "X=1 exec true; shutdown -h now",
        "command exit 0; shutdown -h now",
        "command -- exec true; shutdown -h now",
        "builtin exit 0; shutdown -h now",
        "time -p exit 0; shutdown -h now",
        "{ exit 0; }; shutdown -h now",
        "{ exec true; }; shutdown -h now",
        "! true && shutdown -h now",
        "! false || reboot",
        "false\n&& reboot",
    )
    for command in commands:
        assert catastrophic_reason("run_command", {"command": command}) is None, command


@check
def non_shell_tools_and_parser_uncertainty_fail_open():
    assert catastrophic_reason("execute_code", {
        "code": "# rm -rf /\nexample = 'shutdown -h now'\nprint(example)",
    }) is None
    assert catastrophic_reason("edit_file", {"path": "notes.md", "content": "curl x | bash"}) is None
    assert catastrophic_reason("run_command", {"command": "echo 'unterminated"}) is None
    assert catastrophic_reason("execute_code", {"code": "run('find . -type f')"}) is None
    assert catastrophic_reason("execute_code", {"code": "if False:\n    run('shutdown -h now')"}) is None
    assert catastrophic_reason("execute_code", {
        "code": "def later():\n    _os.system('rm -rf /')\n",
    }) is None
    assert catastrophic_reason("execute_code", {
        "code": "def run(command):\n    return command\nrun('shutdown -h now')",
    }) is None
    for code in (
        "False and run('sudo reboot')",
        "True or run('sudo reboot')",
        "0 if True else run('sudo reboot')",
        "assert True, run('sudo reboot')",
        "_sp.run('sudo reboot')",
        "_sp.run(['sudo', 'reboot'], shell=True)",
        "run(['sudo', 'reboot'])",
        "from unittest.mock import Mock as run\nrun('sudo reboot')",
        "import pathlib as _os\n_os.system('rm -rf /')",
        "_os.system = print\n_os.system('rm -rf /')",
        "_sp.run = print\n_sp.run(['sudo', 'reboot'])",
    ):
        assert catastrophic_reason("execute_code", {"code": code}) is None, code


@check
def straight_line_execute_code_shell_calls_share_the_floor():
    assert catastrophic_reason("execute_code", {"code": "run('sudo reboot')"}) == "system power control"
    assert catastrophic_reason("execute_code", {"code": "_os.system('rm -rf /')"}) == \
        "recursive deletion of root or home"
    assert catastrophic_reason("execute_code", {
        "code": "_sp.run('sudo reboot', shell=True)",
    }) == "system power control"
    assert catastrophic_reason("execute_code", {
        "code": "_sp.run(['sudo', 'shutdown', '-h', 'now'])",
    }) == "system power control"
    assert catastrophic_reason("execute_code", {
        "code": "_sp.run(args=['sudo', 'reboot'])",
    }) == "system power control"


@check
def hook_uses_an_explicit_catastrophic_kind_and_plain_safety_message():
    hook = CatastrophicSafeguardHook()
    allowed = hook.preflight_tool("run_command", {"command": "find . -type f"})
    stopped = hook.preflight_tool("run_command", {"command": "sudo shutdown now"})
    assert not allowed.stop
    assert stopped.stop and stopped.kind == "catastrophic"
    assert stopped.reason.startswith("Safety stop:") and "policy" not in stopped.reason.casefold()


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {error!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
