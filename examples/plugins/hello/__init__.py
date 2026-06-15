"""Reference memagent plugin. `register(ctx)` is the single entry point; through `ctx` a
plugin feeds the EXISTING seams — the tool registry, skill manager, MCP servers, and hooks.
A plugin gets no privileged surface: its tools run through the same sandbox + permission
policy + scheduler as built-in tools."""


def register(ctx):
    # 1) a tool — runs through the same registry/policy/sandbox as built-ins
    ctx.register_tool(
        "reverse_text",
        "Reverse a string. args: {text}",
        lambda args: (args.get("text", ""))[::-1],
        parameters={"type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]},
    )

    # 2) a skill — enters the progressive-disclosure catalog; loads into the ACTIVE SKILL tier
    ctx.register_skill(
        "release-notes",
        "# Writing release notes\n"
        "1. Group changes into Added / Changed / Fixed.\n"
        "2. One line each, imperative mood, link the PR.\n"
        "3. Put the most user-visible change first.\n",
        description="Use when asked to write or format release notes / a changelog.",
    )

    # 3) (optional) an MCP server — ctx.register_mcp_server(name, {"command": ..., "args": [...]})
    # 4) (optional) a hook — ctx.register_hook(MyHooksSubclass())
