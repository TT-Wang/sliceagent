# sliceagent Windows installer — NATIVE, no WSL, no admin (Hermes-style uv recipe).
#
# Usage (PowerShell):
#   irm https://raw.githubusercontent.com/TT-Wang/sliceagent/main/install.ps1 | iex
#
# What it does: installs uv (pinned version) -> installs sliceagent[tui] as an isolated uv tool with
# its own Python 3.12 -> ensures Git Bash exists (the shell that runs the agent's commands; downloads
# a pinned, SHA256-verified PortableGit into %LOCALAPPDATA%\sliceagent\git if you have no Git for
# Windows) -> drops a pinned, SHA256-verified ripgrep into uv's bin dir (already on PATH). Everything
# is user-scoped; nothing needs UAC; no PATH registry rewrites.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # WinPS 5.1: IWR is ~10x slower with the progress bar on
$AppDir = Join-Path $env:LOCALAPPDATA "sliceagent"
New-Item -ItemType Directory -Force -Path $AppDir | Out-Null

# Pinned artifacts + SHA256 (never releases/latest — rate limits + supply-chain pinning):
$UvInstall      = "https://astral.sh/uv/0.11.26/install.ps1"
$PortableGitUrl = "https://github.com/git-for-windows/git/releases/download/v2.55.0.windows.2/PortableGit-2.55.0.2-64-bit.7z.exe"
$PortableGitSha = "b20d42da3afa228e9fa6174480de820282667e799440d655e308f700dfa0d0df"
$RipgrepUrl     = "https://github.com/BurntSushi/ripgrep/releases/download/15.1.0/ripgrep-15.1.0-x86_64-pc-windows-msvc.zip"
$RipgrepSha     = "124510b94b6baa3380d051fdf4650eaa80a302c876d611e9dba0b2e18d87493a"

function Step($msg) { Write-Host "==> $msg" }

function Get-Verified($url, $sha256, $outFile) {
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $outFile
    $actual = (Get-FileHash -Algorithm SHA256 -Path $outFile).Hash.ToLowerInvariant()
    if ($actual -ne $sha256.ToLowerInvariant()) {
        Remove-Item $outFile -ErrorAction SilentlyContinue
        throw "SHA256 mismatch for $url`n  expected $sha256`n  got      $actual"
    }
}

# ── 1. uv (pinned installer) ─────────────────────────────────────────────────
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Step "Installing uv (Python tool manager, pinned 0.11.26)..."
    Invoke-Expression (Invoke-RestMethod -UseBasicParsing $UvInstall)
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"   # this session; the installer handles new shells
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv installed but not on PATH in this session — open a NEW PowerShell and re-run this installer."
    exit 1
}

# ── 2. sliceagent ────────────────────────────────────────────────────────────
Step "Installing sliceagent (isolated env, its own Python 3.12)..."
uv tool install --force --python 3.12 "sliceagent[tui]"
if ($LASTEXITCODE -ne 0) {
    # Most common Windows failure: antivirus (Defender / 360 / 电脑管家) blocks uv's freshly
    # written .exe script shims — 'Failed to update Windows PE resources ... Access denied /
    # 拒绝访问'. Usually a transient scan race: clear the cache and retry once.
    Step "Install failed — clearing uv cache and retrying once (antivirus often blocks the first attempt)..."
    try { uv cache clean | Out-Null } catch { }
    Start-Sleep -Seconds 3
    uv tool install --force --python 3.12 "sliceagent[tui]"
}
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "uv tool install failed twice. If the error above mentions 'Windows PE resources' /"
    Write-Host "'Access denied' / '拒绝访问', your antivirus is blocking uv's script shims. Fix:"
    Write-Host "  1. Temporarily pause the antivirus real-time protection (Defender / 360 / 电脑管家),"
    Write-Host "     or add exclusions for:  %LOCALAPPDATA%\uv  and  %USERPROFILE%\.local"
    Write-Host "  2. Re-run this installer. (Re-enable the antivirus afterwards.)"
    exit 1
}
try { uv tool update-shell | Out-Null } catch { }   # best-effort PATH help; WinPS 5.1-safe (no 2>$null under EAP=Stop)

# ── 3. Git Bash (runs the agent's shell commands — same strategy as Claude Code) ─
$bashCandidates = @(
    (Join-Path $AppDir "git\bin\bash.exe"),
    (Join-Path $env:ProgramFiles "Git\bin\bash.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Git\bin\bash.exe")
)
$bash = $bashCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $bash) {
    $onPath = Get-Command bash -ErrorAction SilentlyContinue
    if ($onPath -and ($onPath.Source -notmatch "(?i)system32")) { $bash = $onPath.Source }  # skip WSL's bash
}
if (-not $bash) {
    Step "No Git Bash found — downloading PortableGit (~60 MB, one time, SHA256-verified)..."
    $sfx = Join-Path $env:TEMP "PortableGit.7z.exe"
    Get-Verified $PortableGitUrl $PortableGitSha $sfx
    $gitDir = Join-Path $AppDir "git"
    & $sfx -o"$gitDir" -y | Out-Null      # self-extracting 7z archive
    if (-not (Test-Path (Join-Path $gitDir "bin\bash.exe"))) {
        Write-Host "PortableGit extraction failed — install Git for Windows manually (https://git-scm.com), then re-run."
        exit 1
    }
    Remove-Item $sfx -ErrorAction SilentlyContinue
}

# ── 4. ripgrep (recommended: powers the code-search tier) into uv's bin dir ──
$uvBin = Join-Path $env:USERPROFILE ".local\bin"     # already on PATH courtesy of uv
if (-not (Get-Command rg -ErrorAction SilentlyContinue) -and -not (Test-Path (Join-Path $uvBin "rg.exe"))) {
    Step "Installing ripgrep (SHA256-verified)..."
    try {
        $zip = Join-Path $env:TEMP "ripgrep.zip"
        Get-Verified $RipgrepUrl $RipgrepSha $zip
        $tmp = Join-Path $env:TEMP "ripgrep-extract"
        Expand-Archive -Path $zip -DestinationPath $tmp -Force
        New-Item -ItemType Directory -Force -Path $uvBin | Out-Null
        Get-ChildItem -Path $tmp -Recurse -Filter rg.exe | Select-Object -First 1 |
            ForEach-Object { Copy-Item $_.FullName (Join-Path $uvBin "rg.exe") }
        Remove-Item $zip, $tmp -Recurse -ErrorAction SilentlyContinue
    } catch {
        Write-Host "    (ripgrep install failed — sliceagent still works, code search just uses the slower fallback)"
    }
}

Write-Host ""
Write-Host "Done. Open a NEW terminal (PowerShell or Windows Terminal) and run:  sliceagent"
Write-Host "First run walks you through provider setup. Docs: https://github.com/TT-Wang/sliceagent"
