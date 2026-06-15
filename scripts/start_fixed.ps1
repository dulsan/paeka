<#
.SYNOPSIS
    PAEKA Ecosystem Startup Script (start_fixed.ps1)
    Optimized for PowerShell 7.6.2+ & Docker Compose
#>

# ==============================================================================
# 1. CONFIGURATION & ENVIRONMENT SETUP
# ==============================================================================
# Gordan's Fix: Points explicitly to host port 8090 instead of 8080
$env:WEAVIATE_URL = "http://localhost:8090"

# --- Update these paths to match your local repository layout ---
$LlamaExePath     = "llama-server.exe" 
$ModelPath        = ".\models\your-model.gguf"  # Change to your actual GGUF file
$LlamaPort        = "8080"
$GpuLayers        = 35                          # GPU layers configuration

# Build the argument list as a clean array
$llamaArgList = @(
    "-m", $ModelPath,
    "-c", "2048",
    "--port", $LlamaPort,
    "-ngl", $GpuLayers.ToString()
)

# ==============================================================================
# 2 & 3. SMART WEAVIATE START & READINESS POLL
# ==============================================================================
Write-Host "`n[1/4] Checking Weaviate Vector Database status..." -ForegroundColor Cyan

$weaviateReady = $false
try {
    # Check if Weaviate is already up and running on port 8090
    $response = Invoke-WebRequest -Uri "$($env:WEAVIATE_URL)/v1/.well-known/ready" -Method Get -TimeoutSec 2 -ErrorAction Ignore
    if ($response.StatusCode -eq 200) {
        $weaviateReady = $true
    }
} catch {}

if (-not $weaviateReady) {
    Write-Host "Weaviate is not active. Performing a clean reset via Docker Compose..." -ForegroundColor Cyan
    
    # Surgical Fix: Purges stale container states and ghost network caches
    docker compose down
    
    Write-Host "Initializing fresh container instance..." -ForegroundColor Cyan
    docker compose up -d paeka-weaviate

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to execute 'docker compose up'. Is Docker Desktop running?"
        exit $LASTEXITCODE
    }

    Write-Host "Waiting for Weaviate to finish booting on port 8090 (First boot can take 60+ seconds)..." -ForegroundColor Cyan
    
    $retryCount = 0
    $maxRetries = 22 # 22 * 4 seconds = ~88 seconds maximum polling window to accommodate Raft consensus
    
    while (-not $weaviateReady -and $retryCount -lt $maxRetries) {
        try {
            $response = Invoke-WebRequest -Uri "$($env:WEAVIATE_URL)/v1/.well-known/ready" -Method Get -TimeoutSec 2 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                $weaviateReady = $true
            }
        }
        catch {
            $retryCount++
            Write-Host "Weaviate is initializing... (Attempt $retryCount/$maxRetries)" -ForegroundColor Yellow
            Start-Sleep -Seconds 4
        }
    }

    if (-not $weaviateReady) {
        Write-Error "Weaviate failed to become healthy within the time limit. Check Docker Desktop logs."
        exit 1
    }
    Write-Host "✔ Weaviate initialized and ready!" -ForegroundColor Green
} else {
    Write-Host "✔ Weaviate is already running and healthy on port 8090! Skipping container launch." -ForegroundColor Green
}

# ==============================================================================
# 4. LAUNCH LLAMA SERVER (BULLETPROOF PROCESS START)
# ==============================================================================
Write-Host "`n[3/4] Launching llama-server..." -ForegroundColor Cyan

$psi = [System.Diagnostics.ProcessStartInfo]::new()
$psi.FileName               = $LlamaExePath
$psi.UseShellExecute        = $false
$psi.RedirectStandardOutput = $false
$psi.RedirectStandardError  = $false
$psi.CreateNoWindow         = $false # Keep window visible to easily monitor model streams

# Sidestep the ArgumentList.Add() bug by compiling arguments directly into an escaped string
$psi.Arguments = ($llamaArgList | ForEach-Object {
    if ($_ -match ' ') { "`"$_`"" } else { $_ }
}) -join ' '

try {
    $LlamaProcess = [System.Diagnostics.Process]::Start($psi)
    Write-Host "✔ llama-server started successfully (PID: $($LlamaProcess.Id))" -ForegroundColor Green
}
catch {
    Write-Error "Failed to launch llama-server. Verify that '$LlamaExePath' exists."
    exit 1
}

# ==============================================================================
# 5. LAUNCH PAEKA LOCAL API (UV / UVICORN)
# ==============================================================================
Write-Host "`n[4/4] Launching PAEKA Local API via uv..." -ForegroundColor Cyan

try {
    # Fires up your FastAPI backend using your uv environment
    # Note: Swap 'main:app' with your actual entry point script name if it differs
    uv run uvicorn main:app --host 127.0.0.1 --port 8000
}
catch {
    Write-Error "PAEKA API encountered a critical crash on startup."
    
    # Graceful cleanup: Kill llama-server if the python API dies
    if ($LlamaProcess -and -not $LlamaProcess.HasExited) {
        Stop-Process -Id $LlamaProcess.Id -Force
    }
    exit 1
}