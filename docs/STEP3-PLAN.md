# Step ③ — Extensions: borrow plan (from Kimi Code + Hermes Agent)

memagent's build is three steps: ① core loop ✅, ② memem connection ✅, ③ **extensions** (this doc).
We borrow patterns (and some liftable Python) from Kimi Code (MIT) and Hermes Agent (MIT).

## Convergent shape (both codebases independently agree)
1. **One tool registry, many sources** — builtin + MCP + plugin + skill tools satisfy one contract, project into one list.
2. **Skills = `SKILL.md` prompt-packs** (YAML frontmatter + body) with **progressive disclosure**: a cheap name+description catalog is always present; the full body loads on demand via a `skill` tool.
3. **MCP folds into the registry**, namespaced (`mcp__server__tool`), behind a tiny `list_tools`/`call_tool` client seam; declared in an `mcpServers` config map (stdio/http/sse).
4. **Plugins = a packaging layer that feeds the existing registries** via a `register(ctx)` facade — not a privileged new surface.
5. **Two hook layers**: typed in-process (have it) + user-authored external hooks with a richer named-event vocabulary.
6. **Sandbox**: base owns `execute()`, backends implement only `_run_bash()` + `cleanup()`.

## The memagent-specific adaptation (critical)
Kimi/Hermes inject a loaded skill body as a **one-shot user message** — it persists because they keep a transcript. **memagent has no transcript; it reconstructs the slice every turn.** A one-shot injection would vanish next turn. So **skills (and any loaded context) must be a SLICE TIER** — an `ACTIVE SKILL` tier that persists loaded instructions until deactivated. This is the same lesson as the earlier stress-test fixes: *out-of-band content must report back into the slice, or it silently disappears.* (Corollary: Hermes' prized "inject into the user message, not the system prompt" trick is something memagent already does by construction — the slice **is** the user message.)

## Borrow map
| Subsystem | Borrow from | Maps onto memagent seam | Verdict |
|---|---|---|---|
| Tool registry (keystone) | Hermes `ToolRegistry` (`generation` counter, `check_fn`) + Kimi 3-source projection | `LocalToolHost` projects `schemas/run/accesses` from a registry | Lift |
| Skills | Both; Hermes frontmatter parser/validator, Kimi arg-expansion + `disableModelInvocation` | new `ACTIVE SKILL` tier + a `skill` tool; catalog can be memem-backed | Borrow + lift parser |
| MCP client | Hermes registration adapter + Kimi `mcp__server__tool` naming/collision/glob-gating | register MCP tools into the registry behind `ToolHost` | Borrow pattern (transport on official `mcp` SDK) |
| Plugins | Hermes `PluginManifest`+`PluginContext`+`PluginManager` (`register(ctx)`, 4-source discovery, `kind`) | facade feeding registry + skills + hooks + MCP; reuse sandbox+policy | Lift + adapt (scope `ctx` to the safe set) |
| Hooks (expand) | Hermes transform/`pre_llm_call`/`subagent_*` + Kimi 15-event vocab & external shell hooks | extend `Hooks`/`CompositeHooks`; add file-based user hooks | Borrow pattern |
| Permissions (upgrade) | Kimi `ask`-continuations, session-approval memo, `Tool(argPattern)` DSL | upgrade `PolicyChain` + interactive ask | Borrow pattern |
| Sandbox backends | Hermes `base.py` (session-snapshot, CWD marker) + `docker.py`/`ssh.py` | base-`execute()`/subclass-`_run_bash()`; add Docker | Borrow + lift base |
| Config | Both (layered merge) | `memagent.toml` declaring MCP servers, plugin/skill dirs, modes | Borrow pattern |

**Keep (don't regress):** memagent's `accesses()` is more principled than Hermes' pattern-matching — keep it as the permission/concurrency basis.

## Build order (each followed by an eval run)
- **③.0 Tool registry refactor** *(keystone)* — `LocalToolHost` projects from a `ToolRegistry`. Done when: existing tools work via the registry; full suite green.
- **③.1 Skills** — `ACTIVE SKILL` slice tier + `skill` tool (catalog→load); `SKILL.md` discovery. Done when: a skill loads, persists across turns, a stress case proves survival across reconstruction.
- **③.2 Config** — `memagent.toml` declaring extensions.
- **③.3 MCP client** — connect a server; tools appear namespaced and run end-to-end.
- **③.4 Plugins** — `register(ctx)` feeding tools/skills/hooks/MCP from 3 discovery sources.
- **③.5 Hooks + permissions** — richer events + `ask` continuations/session memory.
- **③.6 Sandbox backends** — base/subclass split + Docker.

**Order rationale:** ③.0 is the keystone everything plugs into; ③.1 is highest-leverage and proves the skills-as-tier insight; config unblocks MCP/plugins; plugins aggregate skills+MCP+hooks so they come after.
