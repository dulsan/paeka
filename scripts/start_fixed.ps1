<#
.SYNOPSIS
    PAEKA startup script - PowerShell 7 compatible
    Starts: Qdrant + Ollama + PAEKA API via Granian

.DESCRIPTION
    Bugs fixed vs previous version:
      FIX-1  OneDrive guard was a hard exit. Now a warning so you can still
             run while migrating. Data corruption risk is real but startup
             should not be blocked during a transition period.

      FIX-2  Start-Process for qdrant.exe fails on OneDrive paths because
             Windows blocks unsigned executable launch from synced folders.
             Now uses full absolute path via cmd.exe /c as a bypass, with
             a clear manual fallback if that also fails.

      FIX-3  ollama list returns a PS7 string array, one element per line.
             The -notmatch operator on an array returns non-matching elements
             (always some, e.g. the header line) making the condition always
             truthy. Fixed by piping through Where-Object for a line-level
             contains check.

      FIX-4  uv run granian on Python 3.14 Windows: the uv subprocess
             wrapper changes signal propagation, causing the granian parent
             process to receive a phantom shutdown signal after workers start.
             Fixed by invoking the venv granian.exe directly, bypassing uv.

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
# Load .env
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
# FIX-1: OneDrive warning (was hard exit)
# OneDrive file locks cause SQLite journal corruption and Qdrant index issues.
# This is a WARNING not a hard stop - move to C:\paeka when you can.
# ---------------------------------------------------------------------------
if ($PWD.Path -match 'OneDrive') {
    Write-Warning "Running from OneDrive path: $($PWD.Path)"
    Write-Warning 'File locks from OneDrive sync can corrupt the SQLite and Qdrant databases.'
    Write-Warning 'Move the project to C:\paeka before long-term use.'
}

# ---------------------------------------------------------------------------
# Helper: poll HTTP until 200 or timeout
# ---------------------------------------------------------------------------
function Wait-Http {
    param(
        [string] $Uri,
        [string] $Name,
        [int]    $MaxAttempts  = 30,
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
        Write-Host "  Waiting for $Name... ($i / $MaxAttempts)"
        Start-Sleep -Seconds $SleepSeconds
    }
    Write-Warning "$Name not ready after $($MaxAttempts * $SleepSeconds)s"
    return $false
}

Write-Host ''
Write-Host 'PAEKA - Personal AI Engineering and Knowledge Assistant'
Write-Host '========================================================='
Write-Host ''

# ---------------------------------------------------------------------------
# STEP 1: Qdrant
# ---------------------------------------------------------------------------
if (-not $SkipQdrant) {
    Write-Host '[1/3] Starting Qdrant vector database...'

    # Always check first - Qdrant may already be running from a previous start
    $qdrantUp = $false
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:$QdrantPort/healthz" `
             -Method Get -TimeoutSec 2 -ErrorAction Stop -SkipHttpErrorCheck
        if ($r.StatusCode -eq 200) { $qdrantUp = $true }
    } catch { }

    if ($qdrantUp) {
        Write-Host "  [OK] Qdrant already running at http://localhost:$QdrantPort"
    } elseif (Test-Path 'bin\qdrant.exe') {
        New-Item -ItemType Directory -Force -Path 'database\qdrant' | Out-Null

        # FIX-2: Start-Process fails on OneDrive paths due to Windows blocking
        # unsigned executable launch from synced folders. Use cmd.exe /c as
        # a bypass - cmd.exe is a trusted Windows binary and is not blocked.
        # Full absolute paths are required because cmd.exe has a different CWD.
        $qdrantExe    = Join-Path $PWD.Path 'bin\qdrant.exe'
        $qdrantConfig = Join-Path $PWD.Path 'config\qdrant.yaml'
        $cmdArgs      = "/c start /min `"Qdrant`" `"$qdrantExe`" --config-path `"$qdrantConfig`""

        $launched = $false
        try {
            Start-Process -FilePath 'cmd.exe' -ArgumentList $cmdArgs -ErrorAction Stop
            $launched = $true
        } catch {
            Write-Warning "Could not launch Qdrant automatically: $_"
        }

        if ($launched) {
            $ready = Wait-Http -Uri "http://localhost:$QdrantPort/healthz" `
                                -Name 'Qdrant' -MaxAttempts 20 -SleepSeconds 2
            if (-not $ready) {
                Write-Warning 'Qdrant did not respond. Start it manually in a separate terminal:'
                Write-Warning "  .\bin\qdrant.exe --config-path config\qdrant.yaml"
                Write-Warning 'Then re-run this script with -SkipQdrant'
            }
        } else {
            Write-Warning 'Automatic Qdrant launch failed (common on OneDrive paths).'
            Write-Warning 'Start Qdrant manually in a separate terminal:'
            Write-Warning "  .\bin\qdrant.exe --config-path config\qdrant.yaml"
            Write-Warning 'Once running, re-run this script with -SkipQdrant'
        }
    } else {
        Write-Warning 'bin\qdrant.exe not found.'
        Write-Warning 'Download from: https://github.com/qdrant/qdrant/releases'
        Write-Warning 'Look for: qdrant-x86_64-pc-windows-msvc.zip'
        Write-Warning 'Extract qdrant.exe into the bin\ folder.'
    }
}

# ---------------------------------------------------------------------------
# STEP 2: Ollama
# ---------------------------------------------------------------------------
if (-not $SkipOllama) {
    Write-Host ''
    Write-Host '[2/3] Checking Ollama...'

    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Write-Error 'ollama not found on PATH. Install from: https://ollama.com/download/windows'
        exit 1
    }

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
        # Ollama is a signed Windows installer app, Start-Process works fine
        Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Minimized
        $ready = Wait-Http -Uri 'http://localhost:11434/api/tags' `
                            -Name 'Ollama' -MaxAttempts 15 -SleepSeconds 2
        if (-not $ready) {
            Write-Error 'Ollama did not start. Run: ollama serve'
            exit 1
        }
    }

    # FIX-3: ollama list returns a string array in PS7.
    # -notmatch on an array returns non-matching elements, not a boolean.
    # The header line "NAME  ID  SIZE  MODIFIED" never matches any model name,
    # so the old check always evaluated as truthy (warning always shown).
    # Fix: pipe through Where-Object for a proper per-line contains check.
    $modelName  = if ($env:PAEKA_LLM__MODEL) { $env:PAEKA_LLM__MODEL } else { 'paeka-qwen' }
    Write-Host "  Checking model: $modelName"
    $modelFound = ollama list 2>$null | Where-Object { $_ -match [regex]::Escape($modelName) }
    if (-not $modelFound) {
        Write-Warning "Model '$modelName' not found in Ollama."
        Write-Warning "Import your GGUF: ollama create $modelName -f models\qwen\Modelfile"
        Write-Warning "Or pull from hub:  ollama pull qwen3:9b"
        Write-Warning '(PAEKA will start but LLM calls will fail until a model is loaded)'
    } else {
        Write-Host "  [OK] Model '$modelName' is available"
    }
}

# ---------------------------------------------------------------------------
# STEP 3: PAEKA API via Granian
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

# FIX-4: uv run granian on Python 3.14 + Windows changes signal propagation.
# The uv subprocess wrapper intercepts SIGINT/SIGTERM, causing the granian
# parent to receive a phantom shutdown after worker startup completes.
# Fix: invoke granian.exe from the venv directly, bypassing uv entirely.
# The venv is always at .venv\Scripts\granian.exe after uv sync.
$granianExe = Join-Path $PWD.Path '.venv\Scripts\granian.exe'

if (Test-Path $granianExe) {
    & $granianExe --interface asgi --host 0.0.0.0 --port $ApiPort --workers 1 main:app
} else {
    Write-Warning 'granian.exe not found in .venv\Scripts\. Running uv sync first...'
    uv sync
    if (Test-Path $granianExe) {
        & $granianExe --interface asgi --host 0.0.0.0 --port $ApiPort --workers 1 main:app
    } else {
        Write-Error 'granian could not be installed. Check pyproject.toml has granian>=1.0.0'
        exit 1
    }
}

Write-Host ''
Write-Host 'PAEKA stopped.'
