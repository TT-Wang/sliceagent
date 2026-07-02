"""docs/CONFIGURATION.md is generated from envspec.REGISTRY; this guard fails if it drifts (someone
added/changed an env var without regenerating). Run: PYTHONPATH=src python tests/test_config_docs_current.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from gen_config_reference import render  # noqa: E402


def main() -> None:
    path = os.path.join(ROOT, "docs", "CONFIGURATION.md")
    committed = open(path, encoding="utf-8").read()
    if committed == render():
        print("PASS config_docs_match_envspec"); print("\n1/1 passed"); sys.exit(0)
    print("FAIL config_docs_match_envspec: docs/CONFIGURATION.md is stale — run "
          "`PYTHONPATH=src python scripts/gen_config_reference.py` and commit.")
    print("\n0/1 passed"); sys.exit(1)


if __name__ == "__main__":
    main()
