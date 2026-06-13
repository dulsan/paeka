# =============================================================================
# PAEKA - stop.ps1
# =============================================================================
# Stops all PAEKA services.
# Safe to run even if some services are already stopped.
#
# Usage:
#   .\stop.ps1              # stop everything
#   .\stop.ps1 -KeepData    # stop processes but leave Weaviate container running
# =============================================================================

param(
    [switch]$KeepData   # leave Weaviate container running
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host ""
Write-Host "  Stopping PAEKA..." -ForegroundColor Yellow

# -- Stop llama-server.exe -----------------------------------------------------
$llama = Get-Process -Name "llama-server" -ErrorAction SilentlyContinue
if ($llama) {
    Stop-Process -Name "llama-server" -Force
    Write-Host "  llama-server stopped." -ForegroundColor DarkGray
} else {
    Write-Host "  llama-server: not running." -ForegroundColor DarkGray
}

# -- Stop uvicorn / PAEKA API --------------------------------------------------
# uvicorn runs as a Python child process under uv
$uvicorn = Get-Process -Name "uvicorn" -ErrorAction SilentlyContinue
if ($uvicorn) {
    Stop-Process -Name "uvicorn" -Force
    Write-Host "  PAEKA API stopped." -ForegroundColor DarkGray
} else {
    Write-Host "  PAEKA API: not running." -ForegroundColor DarkGray
}

# -- Stop Weaviate container ---------------------------------------------------
if (-not $KeepData) {
    try {
        docker compose down 2>&1 | Out-Null
        Write-Host "  Weaviate stopped." -ForegroundColor DarkGray
    } catch {
        Write-Host "  Weaviate: Docker not running or already stopped." -ForegroundColor DarkGray
    }
} else {
    Write-Host "  Weaviate: left running (-KeepData flag)." -ForegroundColor DarkGray
}

Write-Host "  Done." -ForegroundColor Green
Write-Host ""
