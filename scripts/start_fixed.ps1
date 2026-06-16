<#
.SYNOPSIS
    PAEKA startup script - PowerShell 7+ compatible
    Starts: Qdrant (vector DB) + Ollama (LLM) + PAEKA API

.DESCRIPTION
    No Unicode characters in this file (box-drawing chars break PS tokenizer on some code pages).
    No warning suppression anywhere - warnings are fixed at source:
      - uvicorn uses wsproto backend, websockets package not installed (pyproject.toml)
      - HuggingFace settings live in .env, loaded below before Python starts
    No Register-ObjectEvent or PS background jobs (fragile in PS7).

.EXAMPLE
    pwsh .\scripts\start_fixed.ps1
    pwsh .\scripts\start_fixed.ps1 -SkipQdrant
    pwsh .\scripts\start_fixed.ps1 -SkipOllama
#>

param(
    [switch] $SkipQdrant,
    [switch] $SkipOllama,
    [int]    $ApiPort    = 8000,
    [int]    $QdrantPort = 6333
)

Set-Location (Split-Path -Parent $PSScriptRoot)

# ---------------------------------------------------------------------------
# Load .env into process environment
# Must happen before anything else so PAEKA_* vars are visible to uvicorn
# ---------------------------------------------------------------------------
if (Test-Path '.env') {
    Get-Content '.env' | ForEach-Object {
        if ($_ -match '^([^#=][^=]*)=(.*)$') {
            $k = $Matches[1].Trim()
            $v = $Matches[2].Trim()
            [System.Environment]::SetEnvironmentVariable($k, $v, 'Process')
        }
    }
} else {
    Write-Warning '.env not found. Copy .env.example to .env and configure it.'
}

# ---------------------------------------------------------------------------
# Guard: OneDrive sync path
# OneDrive holds file locks that corrupt SQLite journals and Qdrant data files.
# Move the project to a local path such as C:\paeka before running.
# ---------------------------------------------------------------------------
if ($PWD.Path -match 'OneDrive') {
    Write-Error "PAEKA is running from a OneDrive-synced path: $($PWD.Path)"
    Write-Error 'Move the project to a local directory (e.g. C:\paeka) and try again.'
    exit 1
}

# ---------------------------------------------------------------------------
# Helper: poll an HTTP endpoint until it returns 200 or timeout
# Returns $true if ready, $false if timed out
# ---------------------------------------------------------------------------
function Wait-Http {
    param(
        [string] $Uri,
        [string] $Name,
        [int]    $MaxAttempts = 30,
        [int]    $SleepSeconds = 3
    )
    for ($i = 1; $i -le $MaxAttempts; $i++) {
        try {
            $r = Invoke-WebRequest -Uri $Uri -Method Get -TimeoutSec 3 `
                 -ErrorAction Stop -SkipHttpErrorCheck
            if ($r.StatusCode -eq 200) {
                Write-Host "  [OK] $Name ready at $Uri"
                return $true
            }
        } catch { }
        Write-Host "  Waiting for $Name... (attempt $i / $MaxAttempts)"
        Start-Sleep -Seconds $SleepSeconds
    }
    Write-Warning "$Name did not become ready within $($MaxAttempts * $SleepSeconds)s"
    return $false
}

Write-Host ''
Write-Host 'PAEKA - Personal AI Engineering and Knowledge Assistant'
Write-Host '========================================================='
Write-Host ''

# ---------------------------------------------------------------------------
# STEP 1: Qdrant (local binary - no Docker required)
# Download qdrant.exe from: https://github.com/qdrant/qdrant/releases
# Place in: bin\qdrant.exe
# ---------------------------------------------------------------------------
if (-not $SkipQdrant) {
    Write-Host '[1/3] Starting Qdrant vector database...'

    # Check if already running
    $alreadyUp = $false
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:$QdrantPort/healthz" `
             -Method Get -TimeoutSec 2 -ErrorAction Stop -SkipHttpErrorCheck
        if ($r.StatusCode -eq 200) { $alreadyUp = $true }
    } catch { }

    if ($alreadyUp) {
        Write-Host "  [OK] Qdrant already running at http://localhost:$QdrantPort"
    } elseif (Test-Path 'bin\qdrant.exe') {
        # Ensure data directory exists
        New-Item -ItemType Directory -Force -Path 'database\qdrant' | Out-Null

        # Start Qdrant in a separate minimised window so its log is visible
        # --config-path points to our config that sets the data directory and ports
        $qdrantArgs = '--config-path config\qdrant.yaml'
        Start-Process -FilePath 'bin\qdrant.exe' `
                      -ArgumentList $qdrantArgs `
                      -WindowStyle Minimized

        $ready = Wait-Http -Uri "http://localhost:$QdrantPort/healthz" `
                            -Name 'Qdrant' -MaxAttempts 20 -SleepSeconds 2
        if (-not $ready) {
            Write-Warning 'Qdrant did not start. Check the Qdrant window for errors.'
            Write-Warning 'PAEKA will start but RAG/retrieval will be unavailable.'
        }
    } else {
        Write-Warning 'bin\qdrant.exe not found. Skipping Qdrant.'
        Write-Warning 'Download from: https://github.com/qdrant/qdrant/releases'
        Write-Warning 'Extract qdrant.exe into the bin\ folder then restart.'
        Write-Warning 'PAEKA will start but RAG/retrieval will be unavailable.'
    }
}

# ---------------------------------------------------------------------------
# STEP 2: Ollama (LLM server)
# Install from: https://ollama.com/download/windows
# Then import your model: ollama create paeka-qwen -f models\qwen\Modelfile
# ---------------------------------------------------------------------------
if (-not $SkipOllama) {
    Write-Host ''
    Write-Host '[2/3] Checking Ollama...'

    $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($null -eq $ollamaCmd) {
        Write-Error 'ollama not found on PATH.'
        Write-Error 'Install from: https://ollama.com/download/windows'
        Write-Error 'Then run: ollama create paeka-qwen -f models\qwen\Modelfile'
        exit 1
    }

    # Check if Ollama is already serving
    $ollamaUp = $false
    try {
        $r = Invoke-WebRequest -Uri 'http://localhost:11434/api/tags' `
             -Method Get -TimeoutSec 3 -ErrorAction Stop -SkipHttpErrorCheck
        if ($r.StatusCode -eq 200) { $ollamaUp = $true }
    } catch { }

    if ($ollamaUp) {
        Write-Host '  [OK] Ollama already running at http://localhost:11434'
    } else {
        Write-Host '  Starting Ollama server...'
        Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Minimized

        $ready = Wait-Http -Uri 'http://localhost:11434/api/tags' `
                            -Name 'Ollama' -MaxAttempts 15 -SleepSeconds 2
        if (-not $ready) {
            Write-Error 'Ollama did not start. Check the Ollama window.'
            exit 1
        }
    }

    # Verify the configured model is available
    $modelName = if ($env:PAEKA_LLM__MODEL) { $env:PAEKA_LLM__MODEL } else { 'paeka-qwen' }
    Write-Host "  Checking model: $modelName"
    $tagsJson = ollama list 2>$null
    if ($tagsJson -notmatch [regex]::Escape($modelName)) {
        Write-Warning "Model '$modelName' not found in Ollama."
        Write-Warning "If you have the GGUF file, run:"
        Write-Warning "  ollama create $modelName -f models\qwen\Modelfile"
        Write-Warning "Or pull from Ollama hub: ollama pull qwen3:9b"
    } else {
        Write-Host "  [OK] Model '$modelName' is available"
    }
}

# ---------------------------------------------------------------------------
# STEP 3: PAEKA API
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '[3/3] Starting PAEKA API...'

New-Item -ItemType Directory -Force -Path 'database\sqlite' | Out-Null
New-Item -ItemType Directory -Force -Path 'data\uploads'    | Out-Null
New-Item -ItemType Directory -Force -Path 'logs'            | Out-Null

Write-Host ''
Write-Host '========================================================='
Write-Host "  PAEKA API  : http://localhost:$ApiPort"
Write-Host "  OpenAI URL : http://localhost:$ApiPort/v1"
Write-Host "  Health     : http://localhost:$ApiPort/api/health"
Write-Host '  Press Ctrl+C to stop'
Write-Host '========================================================='
Write-Host ''

# Granian is a Rust-based ASGI server with native WebSocket support.
# It has no dependency on the 'websockets' Python package, so the
# websockets.legacy DeprecationWarning does not exist here - the root
# cause is eliminated rather than suppressed.
uv run granian --interface asgi --host 0.0.0.0 --port $ApiPort main:app

Write-Host ''
Write-Host 'PAEKA stopped.'
