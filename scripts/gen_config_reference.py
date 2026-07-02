"""Generate docs/CONFIGURATION.md from envspec.REGISTRY — the single source of truth for env knobs.

Keeps the user-facing config reference in lockstep with the code: every env var sliceagent reads is registered
in envspec (a startup test enforces this), so this doc can never silently drift. Secrets are redacted.

Run: PYTHONPATH=src python scripts/gen_config_reference.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sliceagent import envspec  # noqa: E402


def main() -> None:
    by_group: dict[str, list] = {}
    for e in envspec.REGISTRY:
        by_group.setdefault(e.group, []).append(e)

    out = [
        "# Configuration reference",
        "",
        "_Auto-generated from `src/sliceagent/envspec.py` — do not edit by hand "
        "(`python scripts/gen_config_reference.py`)._",
        "",
        f"sliceagent reads **{len(envspec.REGISTRY)}** environment variables across "
        f"**{len(by_group)}** groups; every value is validated at startup (a misspelled enum warns instead "
        "of silently defaulting). Run `sliceagent config --list` to see the resolved value of each on your "
        "machine. Secrets (🔒) are read from the environment / config and never printed.",
        "",
    ]
    for group in sorted(by_group):
        out.append(f"## {group}")
        out.append("")
        out.append("| variable | default | description |")
        out.append("|---|---|---|")
        for e in sorted(by_group[group], key=lambda x: x.name):
            name = f"`{e.name}`" + (" 🔒" if getattr(e, "secret", False) else "")
            default = "—" if getattr(e, "secret", False) or not e.default else f"`{e.default}`"
            desc = (e.desc or "").replace("|", "\\|").replace("\n", " ")
            choices = getattr(e, "choices", None)
            aliases = getattr(e, "aliases", None)
            if choices:
                desc += f" _(choices: {', '.join(choices)})_"
            if aliases:
                desc += f" _(aliases: {', '.join(aliases)})_"
            out.append(f"| {name} | {default} | {desc} |")
        out.append("")

    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    path = os.path.join(ROOT, "docs", "CONFIGURATION.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"wrote {path} — {len(envspec.REGISTRY)} vars, {len(by_group)} groups")


if __name__ == "__main__":
    main()
