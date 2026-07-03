# sliceagent Windows installer — NATIVE, no WSL, no admin (Hermes-style uv recipe).
#
# Usage (PowerShell):
#   irm https://raw.githubusercontent.com/TT-Wang/sliceagent/main/install.ps1 | iex
#
# What it does: installs uv (if missing) -> installs sliceagent[tui] as an isolated uv tool with its
# own Python 3.12 -> ensures Git Bash exists (the shell that runs the agent's commands; downloads a
# pinned PortableGit into %LOCALAPPDATA%\sliceagent\git if you have no Git for Windows) -> drops a
# pinned ripgrep next to it (powers code search). Everything is user-scoped; nothing needs UAC.
$ErrorActionPreference = "Stop"
$AppDir = Join-Path $env:LOCALAPPDATA "sliceagent"
$BinDir = Join-Path $AppDir "bin"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

# Pinned artifact URLs (never releases/latest — rate limits + supply-chain pinning):
$PortableGitUrl = "https://github.com/git-for-windows/git/releases/download/v2.55.0.windows.2/PortableGit-2.55.0.2-64-bit.7z.exe"
$RipgrepUrl     = "https://github.com/BurntSushi/ripgrep/releases/download/15.1.0/ripgrep-15.1.0-x86_64-pc-windows-msvc.zip"

function Step($msg) { Write-Host "==> $msg" }

# ── 1. uv ────────────────────────────────────────────────────────────────────
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Step "Installing uv (Python tool manager)..."
    irm https://astral.sh/uv/install.ps1 | iex
    # current session PATH pickup (the uv installer updates the user PATH for NEW shells)
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv installed but not on PATH in this session — open a NEW PowerShell and re-run this installer."
    exit 1
}

# ── 2. sliceagent ────────────────────────────────────────────────────────────
Step "Installing sliceagent (isolated env, its own Python 3.12)..."
uv tool install --force --python 3.12 "sliceagent[tui]"
if ($LASTEXITCODE -ne 0) { Write-Host "uv tool install failed (see above)."; exit 1 }
uv tool update-shell 2>$null | Out-Null   # ensure the uv tools dir is on the user PATH

# ── 3. Git Bash (runs the agent's shell commands — same strategy as Claude Code) ─
$bashCandidates = @(
    (Join-Path $AppDir "git\bin\bash.exe"),
    "$env:ProgramFiles\Git\bin\bash.exe",
    "${env:ProgramFiles(x86)}\Git\bin\bash.exe"
)
$bash = $bashCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $bash) {
    $onPath = Get-Command bash -ErrorAction SilentlyContinue
    if ($onPath -and ($onPath.Source -notmatch "(?i)system32")) { $bash = $onPath.Source }  # skip WSL's bash
}
if (-not $bash) {
    Step "No Git Bash found — downloading PortableGit (~60 MB, one time)..."
    $sfx = Join-Path $env:TEMP "PortableGit.7z.exe"
    Invoke-WebRequest -Uri $PortableGitUrl -OutFile $sfx
    $gitDir = Join-Path $AppDir "git"
    & $sfx -o"$gitDir" -y | Out-Null      # self-extracting 7z archive
    if (-not (Test-Path (Join-Path $gitDir "bin\bash.exe"))) {
        Write-Host "PortableGit extraction failed — install Git for Windows manually (https://git-scm.com), then re-run."
        exit 1
    }
    Remove-Item $sfx -ErrorAction SilentlyContinue
}

# ── 4. ripgrep (optional but recommended: powers the code-search tier) ───────
if (-not (Get-Command rg -ErrorAction SilentlyContinue) -and -not (Test-Path (Join-Path $BinDir "rg.exe"))) {
    Step "Installing ripgrep..."
    try {
        $zip = Join-Path $env:TEMP "ripgrep.zip"
        Invoke-WebRequest -Uri $RipgrepUrl -OutFile $zip
        $tmp = Join-Path $env:TEMP "ripgrep-extract"
        Expand-Archive -Path $zip -DestinationPath $tmp -Force
        Get-ChildItem -Path $tmp -Recurse -Filter rg.exe | Select-Object -First 1 |
            ForEach-Object { Copy-Item $_.FullName (Join-Path $BinDir "rg.exe") }
        Remove-Item $zip, $tmp -Recurse -ErrorAction SilentlyContinue
    } catch {
        Write-Host "    (ripgrep install failed — sliceagent still works, code search just uses the slower fallback)"
    }
}

# ── 5. user PATH for our bin dir (rg.exe) ────────────────────────────────────
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$BinDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$BinDir;$userPath", "User")
}

Write-Host ""
Write-Host "Done. Open a NEW terminal (PowerShell or Windows Terminal) and run:  sliceagent"
Write-Host "First run walks you through provider setup. Docs: https://github.com/TT-Wang/sliceagent"
