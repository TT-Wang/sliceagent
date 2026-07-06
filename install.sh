#!/bin/sh
# sliceagent installer — one command, isolated install via uv.
#
#   curl -fsSL https://raw.githubusercontent.com/TT-Wang/sliceagent/main/install.sh | sh
#
# It installs `uv` (a fast Python tool manager) if missing, then installs sliceagent into its own
# isolated environment and puts the `sliceagent` command on your PATH. Re-running upgrades in place.
#   Uninstall:  sh install.sh --uninstall
#
# As with any `curl … | sh`, you are welcome to read this script first — it does exactly the above.
set -eu

PKG="sliceagent[tui]"          # the PUBLISHED PyPI release; [tui] = rich terminal UI. This installer
                               # tracks PyPI stable (not git main) — one canonical, reproducible path.

info() { printf '\033[36m▸ %s\033[0m\n' "$1"; }
warn() { printf '\033[33m! %s\033[0m\n' "$1" >&2; }
err()  { printf '\033[31m✗ %s\033[0m\n' "$1" >&2; }

if [ "${1:-}" = "--uninstall" ]; then
  if command -v uv >/dev/null 2>&1; then uv tool uninstall sliceagent 2>/dev/null || true; fi
  info "sliceagent uninstalled."
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

# 2. install (or upgrade) sliceagent as an isolated uv tool
# --python 3.12: don't inherit whatever python happens to be on PATH (conda base = 3.10,
# Ubuntu 22.04 = 3.10, macOS system = 3.9 — all below the >=3.11 floor). uv fetches a managed
# CPython 3.12 automatically when none is installed, so the installer has zero prerequisites.
info "Installing sliceagent …"
uv tool install --force --python 3.12 "$PKG"

# 3. make sure uv's tool bin is on PATH for future shells
uv tool update-shell >/dev/null 2>&1 || warn "Could not auto-update PATH — you may need to add uv's tool bin (see 'uv tool dir') to your PATH."

# 4. ripgrep powers the code index — install it too (brew when available, else a ~2 MB static
# binary from GitHub into uv's tool bin: no sudo, isolated, removable with the rest).
if ! command -v rg >/dev/null 2>&1; then
  info "Installing ripgrep (code search) …"
  if command -v brew >/dev/null 2>&1; then
    brew install ripgrep >/dev/null 2>&1 || true
  fi
fi
if ! command -v rg >/dev/null 2>&1; then
  RG_VER="14.1.1"
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)              RG_TARGET="aarch64-apple-darwin" ;;
    Darwin-x86_64)             RG_TARGET="x86_64-apple-darwin" ;;
    Linux-x86_64)              RG_TARGET="x86_64-unknown-linux-musl" ;;
    Linux-aarch64|Linux-arm64) RG_TARGET="aarch64-unknown-linux-gnu" ;;
    *)                         RG_TARGET="" ;;
  esac
  BIN_DIR="$(uv tool dir --bin 2>/dev/null || true)"
  [ -n "$BIN_DIR" ] || BIN_DIR="$HOME/.local/bin"
  if [ -n "$RG_TARGET" ]; then
    RG_TMP="$(mktemp -d)"
    RG_URL="https://github.com/BurntSushi/ripgrep/releases/download/${RG_VER}/ripgrep-${RG_VER}-${RG_TARGET}.tar.gz"
    if { command -v curl >/dev/null 2>&1 && curl -fsSL "$RG_URL" -o "$RG_TMP/rg.tgz"; } \
       || { command -v wget >/dev/null 2>&1 && wget -qO "$RG_TMP/rg.tgz" "$RG_URL"; }; then
      mkdir -p "$BIN_DIR" \
        && tar -xzf "$RG_TMP/rg.tgz" -C "$RG_TMP" \
        && mv "$RG_TMP/ripgrep-${RG_VER}-${RG_TARGET}/rg" "$BIN_DIR/rg" \
        && chmod +x "$BIN_DIR/rg" \
        && info "ripgrep installed to $BIN_DIR/rg" \
        || warn "Could not unpack ripgrep — sliceagent still works, code search is just weaker without it."
    else
      warn "Could not download ripgrep — sliceagent still works, code search is just weaker without it."
    fi
    rm -rf "$RG_TMP"
  else
    warn "No prebuilt ripgrep for this platform — sliceagent works without it (weaker code search)."
  fi
fi

cat <<'EOF'

  ✓ sliceagent installed.

  Next — just one command:
    sliceagent          # first run walks you through setup (provider, API key), then you're chatting

  If 'sliceagent' isn't found, open a NEW terminal (PATH was just updated).
  Docs: https://github.com/TT-Wang/sliceagent
EOF
