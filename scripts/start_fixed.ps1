# PAEKA start.ps1 — Native Windows launcher (fixed)
# Fixes applied:
#   [FIX-1] --chat-template-kwargs: removed all backslash-escaping.
#            ArgumentList.Add() passes strings verbatim; let .NET handle quoting.
#            Old:  '{\\\"enable_thinking\\\":false}'  → llama-server receives {enable_thinking:false} (INVALID)
#            New:  '{"enable_thinking":false}'         → llama-server receives {"enable_thinking":false} (valid JSON)
#   [FIX-2] --flash-attn: was passing "on" as a separate arg (wrong).
#            --flash-attn is a boolean flag — no value argument.
#   [FIX-3] Weaviate: added docker pull check, increased poll timeout to 90 s,
#            improved container-state diagnostics.
#   [FIX-4] WebSocket deprecation: set PYTHONWARNINGS before uvicorn launch
#            to silence legacy websockets.legacy and WebSocketServerProtocol
#            deprecation noise. --ws wsproto is already set but uvicorn still
#            imports websockets at module level triggering the warning.
#   [FIX-5] HF warnings: set HF_HUB_VERBOSITY=error via env if not already in .env
param(
    [int]    $GpuLayers  = 99,
    [int]    $CtxSize    = 8192,
    [int]    $BatchSize  = 512,
    [int]    $Port       = 8000,
    [int]    $LlamaPort  = 8080,
    [switch] $SkipLlama,
    [switch] $SkipWeaviate,
    [switch] $Reload,
    [switch] $NoFlashAttn,
    [switch] $EnableThinking,
    [switch] $CodingMode
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Split-Path -Parent $ScriptDir)

# ── Load .env ────────────────────────────────────────────────────────────────
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*([^#=][^=]*)=(.*)$") {
            $k = $matches[1].Trim()
            $v = $matches[2].Trim().Trim('"').Trim("'")
            [System.Environment]::SetEnvironmentVariable($k, $v, "Process")
        }
    }
} else {
    Write-Host "  WARNING: .env not found — using defaults. Copy .env.example to .env." -ForegroundColor Red
}

# ── [FIX-4] Suppress WebSocket deprecation warnings ─────────────────────────
# websockets 14.x deprecated websockets.legacy; uvicorn[standard] still imports
# it at module-load time even when --ws wsproto is passed.
$env:PYTHONWARNINGS = "ignore::DeprecationWarning:websockets,ignore::DeprecationWarning:websockets.legacy,ignore::DeprecationWarning:uvicorn.protocols.websockets.websockets_impl"

# ── [FIX-5] HuggingFace — silence unauthenticated-request noise ─────────────
if (-not $env:HF_HUB_VERBOSITY)         { $env:HF_HUB_VERBOSITY = "error" }
if (-not $env:TOKENIZERS_PARALLELISM)   { $env:TOKENIZERS_PARALLELISM = "false" }
if (-not $env:TRANSFORMERS_NO_ADVISORY_WARNINGS) { $env:TRANSFORMERS_NO_ADVISORY_WARNINGS = "true" }
if (-not $env:HF_HUB_DISABLE_SYMLINKS_WARNING)   { $env:HF_HUB_DISABLE_SYMLINKS_WARNING  = "1" }

# ── Model & binary resolution ────────────────────────────────────────────────
$ModelPath = if ($env:PAEKA_LLM__MODEL_PATH) { $env:PAEKA_LLM__MODEL_PATH } `
             else { "models\qwen\Qwen3.5-9B-Q4_K_M.gguf" }
$LlamaExe  = (Resolve-Path "bin\llama-server.exe" -ErrorAction SilentlyContinue)

# ── Probe llama-server capabilities ─────────────────────────────────────────
$helpOutput = ""
if ($LlamaExe) {
    $helpOutput = & $LlamaExe.Path "--help" 2>&1 | Out-String
}

# ── [FIX-2] Flash-attn detection ─────────────────────────────────────────────
# --flash-attn is a BOOLEAN FLAG — do NOT append a value like "on" after it.
$FlashAttnFlag = $false
if (-not $NoFlashAttn -and ($helpOutput -match "--flash-attn|-fa\b")) {
    $FlashAttnFlag = $true
}

# ── Qwen3.5 9B recommended generation parameters (Unsloth docs) ─────────────
# Non-thinking (default for 9B):  temp=0.7  top_p=0.8   presence_penalty=1.5
# Thinking / coding:              temp=0.6  top_p=0.95  presence_penalty=0.0
if ($EnableThinking -or $CodingMode) {
    $Temp            = "0.6"
    $TopP            = "0.95"
    $PresencePenalty = "0.0"
} else {
    $Temp            = "0.7"
    $TopP            = "0.8"
    $PresencePenalty = "1.5"
}
$TopK = "20"
$MinP = "0.0"

Write-Host ""
Write-Host "PAEKA - Personal AI Engineering and Knowledge Assistant" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor DarkGray
Write-Host "  Model     : $ModelPath"                                  -ForegroundColor DarkGray
Write-Host "  API       : http://localhost:$Port"                       -ForegroundColor DarkGray
Write-Host "  LLM       : http://localhost:$LlamaPort"                 -ForegroundColor DarkGray
Write-Host "  GPU layers: $GpuLayers"                                   -ForegroundColor DarkGray
Write-Host "  FlashAttn : $(if ($FlashAttnFlag) { 'on' } else { 'off (use -NoFlashAttn to suppress this line)' })" -ForegroundColor DarkGray
Write-Host "  Thinking  : $(if ($EnableThinking) { 'ENABLED' } else { 'disabled (default for 9B)' })" -ForegroundColor DarkGray
Write-Host "  Temp/TopP : $Temp / $TopP  TopK: $TopK  MinP: $MinP"   -ForegroundColor DarkGray
Write-Host ""

# ── STEP 1: Weaviate ─────────────────────────────────────────────────────────
if (-not $SkipWeaviate) {
    Write-Host "[1/3] Starting Weaviate..." -ForegroundColor Yellow

    # [FIX-3a] Pull image if not cached (silent; avoids startup failure on fresh install)
    $imageExists = docker image inspect "cr.weaviate.io/semitechnologies/weaviate:1.27.0" 2>$null
    if (-not $imageExists) {
        Write-Host "      Pulling Weaviate image (first-time only)..." -ForegroundColor DarkYellow
        docker compose pull paeka-weaviate 2>&1 | Out-Null
    }

    $null = docker compose up -d paeka-weaviate 2>&1

    # [FIX-3b] Poll for up to 90 seconds (Weaviate can be slow on first raft init)
    $ready = $false
    for ($i = 0; $i -lt 45; $i++) {
        Start-Sleep 2
        try {
            $r = Invoke-WebRequest -Uri "http://localhost:8090/v1/.well-known/ready" `
                -UseBasicParsing -TimeoutSec 2 -SkipHttpErrorCheck -ErrorAction SilentlyContinue
            if ($r -and $r.StatusCode -eq 200) { $ready = $true; break }
        } catch { }
    }

    if ($ready) {
        Write-Host "      Weaviate ready at http://localhost:8090" -ForegroundColor Green
    } else {
        # [FIX-3c] Better diagnostics on failure
        $state  = docker inspect --format "{{.State.Status}}" paeka-weaviate 2>$null
        $health = docker inspect --format "{{.State.Health.Status}}" paeka-weaviate 2>$null
        if ($state -eq "running") {
            Write-Host "      Weaviate running but not yet healthy (state=$state health=$health)" -ForegroundColor DarkYellow
            Write-Host "      PAEKA will retry automatically. Check: docker logs paeka-weaviate" -ForegroundColor DarkYellow
        } else {
            Write-Host "      ERROR: Weaviate not running (state=$state). Check Docker Desktop is running." -ForegroundColor Red
            Write-Host "      Tip: docker compose down && docker compose up -d paeka-weaviate" -ForegroundColor DarkGray
        }
    }
}

# ── STEP 2: llama-server ──────────────────────────────────────────────────────
$LlamaProcess = $null

if (-not $SkipLlama) {
    Write-Host "[2/3] Starting llama-server..." -ForegroundColor Yellow

    if (-not $LlamaExe) {
        Write-Host "  ERROR: bin\llama-server.exe not found." -ForegroundColor Red
        Write-Host "  Download a CUDA build from: https://github.com/ggml-org/llama.cpp/releases" -ForegroundColor Yellow
        exit 1
    }

    $ModelFullPath = Resolve-Path $ModelPath -ErrorAction SilentlyContinue
    if (-not $ModelFullPath) {
        Write-Host "  ERROR: Model not found at $ModelPath" -ForegroundColor Red
        exit 1
    }

    Get-Process -Name "llama-server" -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep 1

    New-Item -ItemType Directory -Force -Path "logs" | Out-Null
    $logPath = "$PWD\logs\llama-server.log"
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] llama-server starting..." | Set-Content $logPath

    # Build argument list
    $llamaArgList = @(
        "--model",            $ModelFullPath.Path,
        "--host",             "0.0.0.0",
        "--port",             "$LlamaPort",
        "--n-gpu-layers",     "$GpuLayers",
        "-c",                 "$CtxSize",
        "--batch-size",       "$BatchSize",
        "--temp",             $Temp,
        "--top-p",            $TopP,
        "--top-k",            $TopK,
        "--min-p",            $MinP,
        "--presence-penalty", $PresencePenalty,
        "--jinja"
    )

    # [FIX-1] --chat-template-kwargs: use plain JSON string, no backslash-escaping.
    # ArgumentList.Add() passes the string verbatim; .NET handles command-line quoting
    # automatically. The old '{\\\"enable_thinking\\\":false}' form caused llama-server
    # to receive {enable_thinking:false} (unquoted keys = invalid JSON).
    #
    # Unsloth docs say Qwen3.5-9B has thinking DISABLED by default.
    # Pass enable_thinking:false explicitly for non-thinking mode (no stray <think> tags).
    $llamaArgList += "--chat-template-kwargs"
    if ($EnableThinking) {
        $llamaArgList += '{"enable_thinking":true}'
    } else {
        $llamaArgList += '{"enable_thinking":false}'
    }

    # Optional llama-server features
    if ($helpOutput -match "--slots") { $llamaArgList += "--slots" }

    # [FIX-2] --flash-attn is a boolean flag — no value argument follows it.
    # The old code appended "on" as a separate arg, which llama-server misread
    # as an argument name and either errored or silently discarded flash-attn.
    if ($FlashAttnFlag) {
        $llamaArgList += "--flash-attn"
    }

    # Optional: cache types (uncomment if you see gibberish output with long contexts)
    # $llamaArgList += "--cache-type-k"; $llamaArgList += "bf16"
    # $llamaArgList += "--cache-type-v"; $llamaArgList += "bf16"

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName               = $LlamaExe.Path
    $psi.UseShellExecute        = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.CreateNoWindow         = $false
    $psi.WindowStyle            = [System.Diagnostics.ProcessWindowStyle]::Normal

    foreach ($arg in $llamaArgList) {
        $psi.ArgumentList.Add($arg)
    }

    try {
        $LlamaProcess = [System.Diagnostics.Process]::Start($psi)

        $null = Register-ObjectEvent -InputObject $LlamaProcess -EventName "OutputDataReceived" `
            -SourceIdentifier "LlamaOut_$($LlamaProcess.Id)" -MessageData $logPath -Action {
                if ($EventArgs.Data) {
                    Add-Content -Path $Event.MessageData -Value "[$(Get-Date -Format 'HH:mm:ss')] $($EventArgs.Data)"
                }
            }
        $null = Register-ObjectEvent -InputObject $LlamaProcess -EventName "ErrorDataReceived" `
            -SourceIdentifier "LlamaErr_$($LlamaProcess.Id)" -MessageData $logPath -Action {
                if ($EventArgs.Data) {
                    Add-Content -Path $Event.MessageData -Value "[$(Get-Date -Format 'HH:mm:ss')][ERR] $($EventArgs.Data)"
                }
            }
        $LlamaProcess.BeginOutputReadLine()
        $LlamaProcess.BeginErrorReadLine()

        Write-Host "      PID: $($LlamaProcess.Id)  Log: logs\llama-server.log" -ForegroundColor DarkGray
        Write-Host "      Loading model (up to 180 seconds)..." -ForegroundColor DarkGray

        $llamaReady = $false
        for ($i = 0; $i -lt 60; $i++) {
            Start-Sleep 3
            if ($LlamaProcess.HasExited) {
                Write-Host "      ERROR: llama-server exited unexpectedly." -ForegroundColor Red
                if (Test-Path $logPath) {
                    Get-Content $logPath -Tail 20 | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray }
                }
                Write-Host "" -ForegroundColor DarkGray
                Write-Host "  TROUBLESHOOTING:" -ForegroundColor Yellow
                Write-Host "    1. Check logs\llama-server.log for the full error" -ForegroundColor DarkGray
                Write-Host "    2. Ensure the GGUF file is not corrupted: Get-FileHash $ModelPath" -ForegroundColor DarkGray
                Write-Host "    3. Verify CUDA build: .\bin\llama-server.exe --version" -ForegroundColor DarkGray
                break
            }
            foreach ($probe in @("http://localhost:$LlamaPort/health", "http://localhost:$LlamaPort/v1/models")) {
                try {
                    $r = Invoke-WebRequest -Uri $probe -UseBasicParsing `
                        -TimeoutSec 3 -SkipHttpErrorCheck -ErrorAction SilentlyContinue
                    if ($r -and $r.StatusCode -eq 200) { $llamaReady = $true; break }
                } catch { }
            }
            if ($llamaReady) { break }
        }

        if ($llamaReady) {
            Write-Host "      llama-server ready at http://localhost:$LlamaPort" -ForegroundColor Green
        } elseif (-not $LlamaProcess.HasExited) {
            Write-Host "      llama-server still loading — PAEKA will retry automatically" -ForegroundColor DarkYellow
        }
    } catch {
        Write-Host "  ERROR launching llama-server: $_" -ForegroundColor Red
        $LlamaProcess = $null
    }
}

# ── STEP 3: PAEKA API ─────────────────────────────────────────────────────────
Write-Host "[3/3] Starting PAEKA API..." -ForegroundColor Yellow

New-Item -ItemType Directory -Force -Path "logs"            | Out-Null
New-Item -ItemType Directory -Force -Path "database\sqlite" | Out-Null
New-Item -ItemType Directory -Force -Path "data\uploads"    | Out-Null

Write-Host ""
Write-Host "========================================================" -ForegroundColor DarkGray
Write-Host "  PAEKA API  : http://localhost:$Port"                     -ForegroundColor Cyan
Write-Host "  OpenAI URL : http://localhost:$Port/v1"                  -ForegroundColor Cyan
Write-Host "  Health     : http://localhost:$Port/api/health"          -ForegroundColor DarkGray
Write-Host "  Thinking   : $(if ($EnableThinking) { 'ENABLED' } else { 'off  (use -EnableThinking to turn on)' })" -ForegroundColor DarkGray
Write-Host "  Coding mode: $(if ($CodingMode) { 'on (temp=0.6)' } else { 'off (use -CodingMode for precise code gen)' })" -ForegroundColor DarkGray
Write-Host "  Press Ctrl+C to stop"                                    -ForegroundColor DarkGray
Write-Host "========================================================" -ForegroundColor DarkGray
Write-Host ""

# [FIX-4] wsproto backend avoids using the deprecated websockets.legacy at runtime.
# The PYTHONWARNINGS env var (set above) suppresses import-time deprecation noise.
$uvicornArgs = @(
    "run", "uvicorn", "main:app",
    "--host", "0.0.0.0",
    "--port", "$Port",
    "--ws", "wsproto",
    "--http", "httptools",
    "--timeout-keep-alive", "75",
    "--log-level", "info"
)
if ($Reload) { $uvicornArgs += "--reload" }

try {
    uv @uvicornArgs
} finally {
    if ($LlamaProcess -and -not $LlamaProcess.HasExited) {
        Write-Host "Stopping llama-server..." -ForegroundColor DarkGray
        Unregister-Event -SourceIdentifier "LlamaOut_$($LlamaProcess.Id)" -ErrorAction SilentlyContinue
        Unregister-Event -SourceIdentifier "LlamaErr_$($LlamaProcess.Id)" -ErrorAction SilentlyContinue
        $LlamaProcess.Kill()
    }
    Write-Host "PAEKA stopped." -ForegroundColor DarkGray
}
