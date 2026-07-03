# sliceagent Windows installer — routes into WSL2.
#
# sliceagent needs a Unix environment (PTY / fcntl), so on Windows it runs inside WSL2.
# Usage (PowerShell):
#   irm https://raw.githubusercontent.com/TT-Wang/sliceagent/main/install.ps1 | iex
$ErrorActionPreference = "Stop"

Write-Host "sliceagent on Windows runs inside WSL2 (it needs a Unix environment)."

if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "WSL is not installed. Run this once in an ADMIN PowerShell, reboot, then re-run this installer:"
    Write-Host ""
    Write-Host "    wsl --install"
    Write-Host ""
    exit 1
}

# Check a default distro actually boots (wsl.exe exists even with no distro installed).
wsl -e true 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "WSL is present but no Linux distro is ready. Run this, then re-run this installer:"
    Write-Host ""
    Write-Host "    wsl --install -d Ubuntu"
    Write-Host ""
    exit 1
}

Write-Host "Installing sliceagent inside your default WSL distro..."
wsl -e sh -c "curl -fsSL https://raw.githubusercontent.com/TT-Wang/sliceagent/main/install.sh | sh"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Install failed inside WSL (see output above)."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Done. Open your WSL terminal (e.g. 'Ubuntu' in the Start menu) and run:  sliceagent"
