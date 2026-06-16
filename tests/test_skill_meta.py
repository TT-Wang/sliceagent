"""Item 13 — skill provenance frontmatter + usage sidecar. No model, no pytest.
Run: python tests/test_skill_meta.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memagent import skill_usage  # noqa: E402
from memagent.consolidate import render_skill  # noqa: E402
from memagent.skill_provenance import (  # noqa: E402
    AUTO, USER, current_authoring_origin, is_auto, provenance_of,
    reset_authoring_origin, set_authoring_origin,
)
from memagent.skills import SkillManager, parse_frontmatter  # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
def render_skill_stamps_auto_provenance():
    md = render_skill({"name": "deploy-flow", "description": "deploy the app",
                       "steps": ["run_command — make build"], "files": ["Makefile"], "freq": 2})
    meta, _ = parse_frontmatter(md)
    assert meta.get("provenance") == AUTO
    assert is_auto(meta) is True


@check
def user_skill_without_field_is_not_auto():
    md = "---\nname: my-skill\ndescription: a hand-written skill\n---\n\nbody here\n"
    meta, _ = parse_frontmatter(md)
    assert provenance_of(meta) == USER
    assert is_auto(meta) is False   # never prune a user skill


@check
def authoring_origin_contextvar_defaults_user_and_resets():
    assert current_authoring_origin() == USER
    tok = set_authoring_origin(AUTO)
    try:
        assert current_authoring_origin() == AUTO
    finally:
        reset_authoring_origin(tok)
    assert current_authoring_origin() == USER


@check
def manager_loads_provenance_and_bumps_usage():
    with tempfile.TemporaryDirectory() as root:
        # an auto skill on disk
        d = os.path.join(root, "deploy-flow")
        os.makedirs(d)
        with open(os.path.join(d, "SKILL.md"), "w", encoding="utf-8") as f:
            f.write(render_skill({"name": "deploy-flow", "description": "deploy",
                                  "steps": ["s"], "files": [], "freq": 1}))
        mgr = SkillManager([root])
        sk = mgr._skills["deploy-flow"]
        assert sk.provenance == AUTO and sk.root == root
        # loading the body bumps the sidecar
        assert skill_usage.use_count(root, "deploy-flow") == 0
        body = mgr.load("deploy-flow")
        assert body and "## Process" in body
        assert skill_usage.use_count(root, "deploy-flow") == 1
        mgr.load("deploy-flow")
        assert skill_usage.use_count(root, "deploy-flow") == 2
        assert skill_usage.last_used_at(root, "deploy-flow") is not None


@check
def usage_sidecar_survives_corrupt_file():
    with tempfile.TemporaryDirectory() as root:
        with open(skill_usage.usage_path(root), "w", encoding="utf-8") as f:
            f.write("{not valid json")
        assert skill_usage.load_usage(root) == {}      # corrupt → {} not a raise
        skill_usage.bump_use(root, "x")                 # recovers by overwriting
        assert skill_usage.use_count(root, "x") == 1


@check
def missing_root_skill_load_does_not_crash():
    # an in-memory/plugin skill has no root → load must still return the body, no sidecar write
    mgr = SkillManager([])
    mgr.add("plug", "plugin body", "a plugin skill")
    assert mgr.load("plug") == "plugin body"


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
