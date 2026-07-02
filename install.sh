#!/bin/sh
# memagent installer — one command, isolated install via uv.
#
#   curl -fsSL https://raw.githubusercontent.com/TT-Wang/memagent/main/install.sh | sh
#
# It installs `uv` (a fast Python tool manager) if missing, then installs memagent into its own
# isolated environment and puts the `memagent` command on your PATH. Re-running upgrades in place.
#   Uninstall:  sh install.sh --uninstall
#
# As with any `curl … | sh`, you are welcome to read this script first — it does exactly the above.
set -eu

REPO="git+https://github.com/TT-Wang/memagent"
PKG="memagent[tui] @ ${REPO}"          # [tui] = rich terminal UI (pure-python; the CLI degrades without it)

info() { printf '\033[36m▸ %s\033[0m\n' "$1"; }
warn() { printf '\033[33m! %s\033[0m\n' "$1" >&2; }
err()  { printf '\033[31m✗ %s\033[0m\n' "$1" >&2; }

if [ "${1:-}" = "--uninstall" ]; then
  if command -v uv >/dev/null 2>&1; then uv tool uninstall memagent 2>/dev/null || true; fi
  info "memagent uninstalled."
  exit 0
fi

# 1. ensure uv
if ! command -v uv >/dev/null 2>&1; then
  info "Installing uv (https://docs.astral.sh/uv) …"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    err "Need curl or wget to bootstrap uv. Install uv manually, then re-run."
    exit 1
  fi
  # uv lands in ~/.local/bin (or ~/.cargo/bin); make it visible for THIS shell
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  err "uv still not on PATH. Open a new terminal and re-run this installer."
  exit 1
fi

# 2. install (or upgrade) memagent as an isolated uv tool
info "Installing memagent …"
uv tool install --force "$PKG"

# 3. make sure uv's tool bin is on PATH for future shells
uv tool update-shell >/dev/null 2>&1 || warn "Could not auto-update PATH — you may need to add uv's tool bin (see 'uv tool dir') to your PATH."

# 4. soft prerequisite: ripgrep powers the code index (memagent still runs without it, just searches less well)
if ! command -v rg >/dev/null 2>&1; then
  warn "ripgrep (rg) not found — memagent works without it, but code search is much better with it."
  warn "Install it:  brew install ripgrep  |  apt install ripgrep  |  https://github.com/BurntSushi/ripgrep"
fi

cat <<'EOF'

  ✓ memagent installed.

  Next:
    memagent init     # guided setup: provider, API key, model (tests your key)
    memagent          # start the agent

  If 'memagent' isn't found, open a NEW terminal (PATH was just updated).
  Docs: https://github.com/TT-Wang/memagent
EOF
