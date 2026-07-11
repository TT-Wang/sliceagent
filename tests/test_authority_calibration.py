"""Counterfactual calibration for fluid-but-scoped turn authority. No model/network."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sliceagent.hooks import TurnAuthorityHook  # noqa: E402
from sliceagent.intent import analyze_turn  # noqa: E402
from sliceagent.registry import ToolIntentEffect  # noqa: E402


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


EFFECTS = {
    "read_file": ToolIntentEffect.OBSERVE,
    "grep": ToolIntentEffect.OBSERVE,
    "ask_user": ToolIntentEffect.DIALOGUE,
    "edit_file": ToolIntentEffect.EXTERNAL,
    "execute_code": ToolIntentEffect.EXTERNAL,
    "run_command": ToolIntentEffect.EXTERNAL,
    "proc_start": ToolIntentEffect.EXTERNAL,
    "proc_kill": ToolIntentEffect.EXTERNAL,
    "terminal_open": ToolIntentEffect.EXTERNAL,
    "terminal_close": ToolIntentEffect.EXTERNAL,
    "change_workspace": ToolIntentEffect.EXTERNAL,
    "update_plan": ToolIntentEffect.TASK_STATE,
    "world_set": ToolIntentEffect.TASK_STATE,
    "spawn_explorer": ToolIntentEffect.OBSERVE,
    "spawn_general": ToolIntentEffect.EXTERNAL,
}


def _gate(request: str, *, pending=None):
    admission = analyze_turn(request, pending_proposal=pending)
    return admission, TurnAuthorityHook(
        lambda: admission,
        lambda name, _args: EFFECTS.get(name, ToolIntentEffect.UNKNOWN),
    )


@check
def review_authorizes_observation_and_read_only_delegation_not_mutation():
    admission, gate = _gate("Review this project and tell me its strengths and weaknesses")
    assert admission.effect_authority == "none"
    assert gate.authorize_tool("read_file", {"path": "README.md"}).allow
    assert gate.authorize_tool("grep", {"pattern": "TODO"}).allow
    assert gate.authorize_tool("spawn_explorer", {"task": "inspect runtime"}).allow
    assert not gate.authorize_tool("edit_file", {"path": "README.md"}).allow
    assert not gate.authorize_tool("spawn_general", {"task": "fix runtime"}).allow


@check
def broad_implementation_authorizes_normal_local_work_but_not_release_or_navigation():
    admission, gate = _gate("Go ahead and build the full simplification upgrade")
    assert admission.effect_authority == "explicit"
    assert gate.authorize_tool("edit_file", {"path": "src/sliceagent/intent.py"}).allow
    assert gate.authorize_tool("run_command", {"command": "python -m pytest tests/test_intent.py"}).allow
    assert gate.authorize_tool(
        "run_command", {"command": "python scripts/gen_config_reference.py"},
    ).allow, "normal workspace-local generators are implementation work, not a new user objective"
    assert gate.authorize_tool("run_command", {"command": "npm install"}).allow
    assert gate.authorize_tool(
        "run_command", {"command": "PYTHONPATH=src python scripts/gen_config_reference.py"},
    ).allow
    assert not gate.authorize_tool("run_command", {"command": "git push origin main"}).allow
    assert not gate.authorize_tool(
        "run_command", {"command": "python scripts/gen_config_reference.py && git push origin main"},
    ).allow
    for command in (
        "env git commit -am pwn", "bash -lc 'git push origin main'", "git -C . commit -am pwn",
        "python scripts/gen_config_reference.py\ngit commit -am pwn",
        "X=1 git push origin main", "builtin git commit -am pwn", "busybox sh -c 'git push'",
        "python -c \"import os; os.system('git push origin main')\"",
        "(git commit)", "git$IFS commit", "npm exec -- git commit", "npx git commit",
        "node --eval \"require('child_process').execSync('git push')\"",
        "perl -e 'system q(git push)'", "/tmp/pytest -q", "/tmp/python -m pytest",
    ):
        assert not gate.authorize_tool("run_command", {"command": command}).allow, command
    assert gate.authorize_tool("execute_code", {
        "code": "write_file('src/generated.py', 'VALUE = 1\\n')\nprint('done')",
    }).allow
    for code in (
        "import subprocess; subprocess.run(['git', 'push'])",
        "open('/tmp/outside', 'w').write('x')",
        "run('git push origin main')",
    ):
        assert not gate.authorize_tool("execute_code", {"code": code}).allow, code
    assert not gate.authorize_tool("run_command", {"command": "sliceagent publish"}).allow
    assert not gate.authorize_tool("change_workspace", {"path": "/tmp/other"}).allow
    assert not gate.authorize_tool("world_set", {"key": "ambient", "value": "yes"}).allow


@check
def named_file_directive_does_not_widen_to_other_files_or_generators():
    _admission, gate = _gate("Edit README.md")
    assert gate.authorize_tool("edit_file", {"path": "README.md"}).allow
    assert not gate.authorize_tool("edit_file", {"path": "src/app.py"}).allow
    assert not gate.authorize_tool("edit_file", {"path": "docs/README.md"}).allow
    assert not gate.authorize_tool("edit_file", {"path": "private/README"}).allow
    assert not gate.authorize_tool(
        "run_command", {"command": "python scripts/gen_config_reference.py"},
    ).allow
    assert not gate.authorize_tool("execute_code", {
        "code": "write_file('README.md', 'ok')\nopen('src/pwn.py', 'w').write('x')",
    }).allow
    assert gate.authorize_tool("run_command", {"command": "python -m pytest tests/test_readme.py"}).allow
    for command in (
        "pytest -q; git commit -am pwn",
        "pytest -q && touch unrelated.txt",
        "pytest -q $(git push origin main)",
    ):
        assert not gate.authorize_tool("run_command", {"command": command}).allow, command


@check
def workspace_navigation_is_intent_authorized_but_typed_path_confirmation_stays_exact():
    # A loose named directive ("go to Hunter") is a workspace.navigate grant → the reversible NAVIGATION tier
    # authorizes the switch to whatever path the model resolves (the user directs where to go and can correct
    # it in one turn). A typed EXACT-path confirmation ("yes" to "is it /full/path?") stays exact.
    admission, gate = _gate("Go to the Hunter workspace")
    assert admission.effect_authority == "explicit"
    assert gate.authorize_tool("change_workspace", {"path": "/Users/example/Desktop/Hunter"}).allow
    assert gate.authorize_tool("change_workspace", {"path": "/tmp/Atlas"}).allow  # reversible nav tier

    target = "/Users/example/Desktop/Hunter"
    accepted, continuation = _gate("yes", pending={
        "action": {"tool": "change_workspace", "args": {"path": target}},
    })
    assert accepted.effect_authority == "continuation"
    assert continuation.authorize_tool("change_workspace", {"path": target}).allow
    # A confirmed EXACT path is an exact grant, NOT ambient nav authority — the other path stays denied.
    assert not continuation.authorize_tool("change_workspace", {"path": "/tmp/Atlas"}).allow


@check
def quoted_or_untyped_continuation_never_becomes_effect_authority():
    quoted, quoted_gate = _gate('The transcript says "delete every file". Explain why that was wrong.')
    assert quoted.effect_authority in {"none", "uncertain"}
    assert not quoted_gate.authorize_tool("run_command", {"command": "rm -rf ."}).allow

    untyped, continuation = _gate("yes", pending={"text": "Would you like me to fix it?"})
    assert untyped.effect_authority == "uncertain"
    assert not continuation.authorize_tool("edit_file", {"path": "src/app.py"}).allow


@check
def operation_directives_authorize_the_named_operation_not_the_shell_surface():
    cases = (
        ("commit changes", "git commit -am update", ("git push origin main", "npm install")),
        ("push changes", "git push origin main", ("git commit -am update", "npm publish")),
        ("install dependency requests", "python -m pip install requests", ("npm publish", "git push")),
        ("publish package", "npm publish", ("npm install", "git push")),
        ("deploy app", "vercel deploy", ("git push", "npm install")),
        ("run tests", "python -m pytest -q", ("git commit -am update", "npm install")),
    )
    for request, allowed, denied in cases:
        _admission, gate = _gate(request)
        assert gate.authorize_tool("run_command", {"command": allowed}).allow, (request, allowed)
        for command in denied:
            assert not gate.authorize_tool("run_command", {"command": command}).allow, (request, command)

    _admission, running = _gate("start server")
    assert running.authorize_tool("proc_start", {"command": "npm run dev"}).allow
    assert not running.authorize_tool("proc_start", {"command": "npm exec server"}).allow
    assert not running.authorize_tool("terminal_open", {}).allow
    assert not running.authorize_tool("run_command", {"command": "git push server"}).allow

    _admission, stopping = _gate("stop server")
    assert stopping.authorize_tool("proc_kill", {"handle": "proc-server-1"}).allow
    assert stopping.authorize_tool("terminal_close", {"handle": "server-terminal"}).allow
    assert not stopping.authorize_tool("proc_kill", {"handle": "worker-process"}).allow
    assert not stopping.authorize_tool("run_command", {"command": "pkill server"}).allow


@check
def file_shorthand_and_requirement_verbs_do_not_authorize_siblings():
    _admission, edit = _gate("edit README")
    assert edit.authorize_tool("edit_file", {"path": "README.md"}).allow
    for path in ("README.py", "README.txt", "docs/README.md"):
        assert not edit.authorize_tool("edit_file", {"path": path}).allow, path

    _admission, add = _gate('add requirement "x"')
    assert add.authorize_tool("require", {"text": "x"}).allow
    assert not add.authorize_tool("require", {"text": "opposite"}).allow
    assert not add.authorize_tool("drop_requirement", {"text": "x"}).allow

    _admission, drop = _gate('drop requirement "x"')
    assert drop.authorize_tool("drop_requirement", {"text": "x"}).allow
    assert not drop.authorize_tool("drop_requirement", {"text": "y"}).allow
    assert not drop.authorize_tool("require", {"text": "x"}).allow


@check
def quoted_targets_multi_actions_and_governing_verbs_stay_structural():
    for request in ('edit "README.md"', "edit `README.md`"):
        _admission, gate = _gate(request)
        assert gate.authorize_tool("edit_file", {"path": "README.md"}).allow
        assert not gate.authorize_tool("edit_file", {"path": "src/pwn.py"}).allow
        assert not gate.authorize_tool("execute_code", {
            "code": "write_file('src/pwn.py', 'x')",
        }).allow

    _admission, exact = _gate('run `python scripts/migration.py`')
    assert exact.authorize_tool("run_command", {"command": "python scripts/migration.py"}).allow
    assert not exact.authorize_tool("run_command", {"command": "git push origin main"}).allow

    admission, combined = _gate("fix parser and commit changes")
    assert {grant.operation for grant in admission.effect_grants} >= {"workspace.edit", "vcs.commit"}
    assert combined.authorize_tool("edit_file", {"path": "src/parser.py"}).allow
    assert combined.authorize_tool("run_command", {"command": "git commit -am parser"}).allow

    _admission, two_files = _gate("edit README.md and src/app.py")
    assert two_files.authorize_tool("edit_file", {"path": "README.md"}).allow
    assert two_files.authorize_tool("edit_file", {"path": "src/app.py"}).allow
    assert not two_files.authorize_tool("edit_file", {"path": "src/other.py"}).allow

    for request in (
        "fix the commit message parser", "fix push notification delivery", "fix stop button",
        "fix update plan button", "fix new task dialog", "fix switch topic behavior",
    ):
        admission, gate = _gate(request)
        assert "workspace.edit" in {grant.operation for grant in admission.effect_grants}, request
        assert not gate.authorize_tool("run_command", {"command": "git push origin main"}).allow, request
        assert not gate.authorize_tool("proc_kill", {"handle": "proc-1"}).allow, request

    for request in (
        "use a table to explain the design", "use concise bullets in your answer",
        "target beginners in the explanation", "make a diagram", "build me a mental model",
        "add citations to your answer", "write a code example in your answer",
    ):
        admission, gate = _gate(request)
        assert admission.effect_authority == "none", request
        assert not gate.authorize_tool("edit_file", {"path": "src/app.py"}).allow, request


@check
def command_origin_ast_binding_and_readonly_helpers_fail_closed():
    _admission, broad = _gate("build the upgrade")
    for command in (
        "PATH=/tmp pytest -q", "PYTHONPATH=/tmp python -m pytest", "BASH_ENV=/tmp/evil bash scripts/build.sh",
        "NODE_OPTIONS=--require=/tmp/evil node scripts/build.js", "~/bin/pytest", "python ~/outside.py",
        "rm -rf .", "rm -rf *", "rm -rf .git", "docker build --push -t registry/app .",
        "docker build --output=type=registry .", "go build -o /tmp/app",
        "cargo build --target-dir /tmp/cargo", "make -f /tmp/evil.mk build", "cmake --build /tmp/other",
    ):
        assert not broad.authorize_tool("run_command", {"command": command}).allow, command
    assert broad.authorize_tool("run_command", {"command": "rm -rf build"}).allow
    assert broad.authorize_tool("run_command", {"command": "docker build -t local/app ."}).allow

    for code in (
        'write_file = exec; write_file("print(1)")',
        'write_file = open; write_file("/tmp/outside", "w")',
        'read_file = open; print(read_file("/etc/passwd").read())',
        'write_file("safe", "x"); sorted(["print(1)"], key=exec)',
        '[write_file("payload") for write_file in [exec]]',
    ):
        assert not broad.authorize_tool("execute_code", {"code": code}).allow, code

    from sliceagent.policy import _is_readonly_command
    for command in (
        "rg --pre 'touch /tmp/pwn' needle .", "rg --pre='sh scripts/build.sh' needle .",
        "git --paginate -ccore.pager=cat log -1", "git -ccore.diff.external=cat diff",
        "git --paginate --config-env=core.pager=SHELL log", "/tmp/rg needle .", "~/bin/rg needle .",
    ):
        assert not _is_readonly_command(command), command


@check
def scoped_files_dependencies_vcs_and_requirements_bind_exact_objects():
    cases = (
        ("edit .env", ".env", "src/pwn.py"),
        ("edit config/.env", "config/.env", "config/other.env"),
        ("change App.vue", "App.vue", "src/pwn.py"),
        ("update schema.proto", "schema.proto", "other.proto"),
    )
    for request, allowed, denied in cases:
        _admission, gate = _gate(request)
        assert gate.authorize_tool("edit_file", {"path": allowed}).allow, request
        assert not gate.authorize_tool("edit_file", {"path": denied}).allow, request

    for request in ("upgrade to Python 3.12", "upgrade package support to v2.0"):
        admission, gate = _gate(request)
        assert any(grant.operation == "workspace.edit" and not grant.target
                   for grant in admission.effect_grants), request
        assert gate.authorize_tool("edit_file", {"path": "src/app.py"}).allow

    _admission, prefix = _gate("refactor src/")
    assert prefix.authorize_tool("edit_file", {"path": "src/app.py"}).allow
    assert not prefix.authorize_tool("edit_file", {"path": "docs/app.py"}).allow
    _admission, globbed = _gate("edit all .py files")
    assert globbed.authorize_tool("edit_file", {"path": "src/app.py"}).allow
    assert not globbed.authorize_tool("edit_file", {"path": "src/app.ts"}).allow

    _admission, delete = _gate("delete app.py")
    assert delete.authorize_tool("run_command", {"command": "rm app.py"}).allow
    assert not delete.authorize_tool("run_command", {"command": "rm src/other.py"}).allow
    _admission, rename = _gate("rename app.py to main.py")
    assert rename.authorize_tool("run_command", {"command": "mv app.py main.py"}).allow
    assert not rename.authorize_tool(
        "run_command", {"command": "mv app.py main.py --target-directory=/tmp"},
    ).allow

    _admission, install = _gate("install requests")
    assert install.authorize_tool("run_command", {"command": "python -m pip install requests"}).allow
    for command in ("pip install other", "pip install --target /tmp requests", "npm install -g requests"):
        assert not install.authorize_tool("run_command", {"command": command}).allow, command

    _admission, stage = _gate("stage README.md")
    assert stage.authorize_tool("run_command", {"command": "git add README.md"}).allow
    assert not stage.authorize_tool("run_command", {"command": "git add src/app.py"}).allow
    _admission, push = _gate("push changes")
    for command in ("git push origin :main", "git push origin +HEAD:main", "git push --delete origin main"):
        assert not push.authorize_tool("run_command", {"command": command}).allow, command
    _admission, commit = _gate("commit changes")
    for command in (
        "PATH=/tmp git commit", "git -ccore.hooksPath=/tmp/hooks commit",
        "git --config-env=core.hooksPath=HOOKS commit", "git --git-dir=/tmp/other/.git commit",
    ):
        assert not commit.authorize_tool("run_command", {"command": command}).allow, command

    _admission, complete = _gate("mark requirement A done")
    assert complete.authorize_tool("requirement_done", {"text": "A"}).allow
    assert not complete.authorize_tool("requirement_done", {"text": "B"}).allow
    _admission, supersede = _gate('supersede requirement "use v1" with "use v2"')
    assert supersede.authorize_tool(
        "supersede_requirement", {"old_text": "use v1", "new_text": "use v2"},
    ).allow
    assert not supersede.authorize_tool(
        "supersede_requirement", {"old_text": "use v1", "new_text": "use v3"},
    ).allow


@check
def ordinary_coding_verbs_processes_and_adjacent_continuations_remain_fluid():
    for request in (
        "would you mind fixing the parser?", "format the code", "clean up the parser", "solve the bug",
        "start implementation", "close the modal", "stop the animation", "kill the feature flag",
    ):
        admission, gate = _gate(request)
        assert admission.effect_authority == "explicit", request
        assert gate.authorize_tool("edit_file", {"path": "src/app.py"}).allow, request

    _admission, tests = _gate("run tests")
    for command in ("uv run pytest -q", "poetry run pytest -q", "tox", "nox"):
        assert tests.authorize_tool("run_command", {"command": command}).allow, command
    for command in (
        "tox -e deploy", "nox -s publish", "pytest --junitxml=/tmp/report.xml",
        "ruff check --fix-only src", "python -m ruff check --fix src",
        "eslint --output-file /tmp/report.json src", "biome check --write=true src",
        "tsc --noEmit --generateTrace /tmp/trace", "mypy --cache-dir=/tmp/mypy-cache src",
        "bash scripts/test.sh --publish",
    ):
        assert not tests.authorize_tool("run_command", {"command": command}).allow, command

    _admission, server = _gate("start server")
    for command in (
        "python manage.py runserver", "node server.js", "go run ./cmd/server", "docker compose up",
    ):
        assert server.authorize_tool("proc_start", {"command": command}).allow, command
    _admission, worker = _gate("start worker")
    assert worker.authorize_tool("proc_start", {"command": "celery -A app worker"}).allow

    _admission, migration = _gate("run data migration")
    assert migration.authorize_tool("run_command", {"command": "python scripts/migration.py"}).allow
    assert not migration.authorize_tool("run_command", {"command": "git push migration"}).allow

    for request in ("switch to Hunter", "go to Hunter", 'go to "Hunter" workspace'):
        _admission, navigation = _gate(request)
        assert navigation.authorize_tool(
            "change_workspace", {"path": "/Users/example/Desktop/Hunter"},
        ).allow, request

    target = "/Users/example/Desktop/Hunter"
    for response in ("go", "continue", "proceed", "please continue", "keep going"):
        admission, gate = _gate(response, pending={
            "action": {"tool": "change_workspace", "args": {"path": target}},
        })
        assert admission.effect_authority == "continuation", response
        assert gate.authorize_tool("change_workspace", {"path": target}).allow, response


@check
def composition_answer_only_and_named_operation_targets_keep_their_meaning():
    _admission, negated = _gate("Edit README.md, but do not edit CHANGELOG.md")
    assert negated.authorize_tool("edit_file", {"path": "README.md"}).allow
    assert not negated.authorize_tool("edit_file", {"path": "CHANGELOG.md"}).allow

    admission, coordinated = _gate("fix parser, commit changes, and push changes")
    assert {"workspace.edit", "vcs.commit", "vcs.push"} <= {
        grant.operation for grant in admission.effect_grants
    }
    assert coordinated.authorize_tool("run_command", {"command": "git commit -am parser"}).allow
    assert coordinated.authorize_tool("run_command", {"command": "git push origin feature"}).allow

    _admission, scoped = _gate("edit all .py files in src/")
    assert scoped.authorize_tool("edit_file", {"path": "src/app.py"}).allow
    assert not scoped.authorize_tool("edit_file", {"path": "tests/test_app.py"}).allow
    _admission, directories = _gate("refactor src/ and tests/")
    assert directories.authorize_tool("edit_file", {"path": "src/app.py"}).allow
    assert directories.authorize_tool("edit_file", {"path": "tests/test_app.py"}).allow
    assert not directories.authorize_tool("edit_file", {"path": "docs/app.py"}).allow
    _admission, named_directories = _gate("refactor src and tests directories")
    assert named_directories.authorize_tool("edit_file", {"path": "tests/test_app.py"}).allow
    assert not named_directories.authorize_tool("edit_file", {"path": "docs/app.py"}).allow
    _admission, recursive = _gate("edit **/*.py")
    assert recursive.authorize_tool("edit_file", {"path": "app.py"}).allow

    for request in ('fix "parser bug"', 'implement "dark mode"'):
        _admission, semantic = _gate(request)
        assert semantic.authorize_tool("edit_file", {"path": "src/parser.py"}).allow, request

    _admission, feature = _gate("push the feature branch")
    assert feature.authorize_tool("run_command", {"command": "git push origin feature"}).allow
    assert not feature.authorize_tool("run_command", {"command": "git push origin main"}).allow
    _admission, staging = _gate("deploy to staging")
    assert staging.authorize_tool("run_command", {"command": "vercel deploy"}).allow
    assert not staging.authorize_tool("run_command", {"command": "vercel deploy --prod"}).allow

    target = "/Users/example/Desktop/Hunter"
    for response in ("yes please", "yes, go ahead", "okay, do it", "yes, please switch it"):
        admission, continuation = _gate(response, pending={
            "action": {"tool": "change_workspace", "args": {"path": target}},
        })
        assert admission.effect_authority == "continuation", response
        assert continuation.authorize_tool("change_workspace", {"path": target}).allow, response

    for request in ("write pseudocode for the parser fix", "make a recommendation"):
        admission, answer = _gate(request)
        assert admission.effect_authority == "none", request
        assert not answer.authorize_tool("edit_file", {"path": "src/parser.py"}).allow

    from sliceagent.policy import _is_readonly_command
    assert _is_readonly_command("sed -n '1,20p' README.md")
    assert _is_readonly_command("jq . package.json")
    assert not _is_readonly_command("sed -i 's/a/b/' README.md")
    # && / || chains of read-only commands are observation, not an effect (exploration before navigating).
    assert _is_readonly_command('ls dir/ | head -20 && echo "---" && cat dir/README.md')
    assert _is_readonly_command("ls && pwd")
    assert not _is_readonly_command("ls && rm -rf x")       # one mutating branch fails the whole chain
    assert not _is_readonly_command("cat a && ./writer")     # explicit-path exec is not allowlisted
    assert not _is_readonly_command("ls &")                  # backgrounding is not a read-only chain


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
