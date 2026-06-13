# PAEKA start.ps1 — Native Windows launcher (fixed)
param(
    [int]    $GpuLayers = 99,
    [int]    $CtxSize = 8192,
    [int]    $BatchSize = 512,
    [int]    $Port = 8000,
    [int]    $LlamaPort = 8080,
    [switch] $SkipLlama,
    [switch] $SkipWeaviate,
    [switch] $Reload,
    [switch] $NoFlashAttn,
    [switch] $EnableThinking,
    [switch] $CodingMode
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Split-Path -Parent $ScriptDir)

# === Load .env ===
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*([^#=][^=]*)=(.*)$") {
            $k = $matches[1].Trim()
            $v = $matches[2].Trim().Trim('"').Trim("'")
            [System.Environment]::SetEnvironmentVariable($k, $v, "Process")
        }
    }
} else {
    Write-Host "  WARNING: .env not found using defaults." -ForegroundColor Red
}

# === warnings suppression ===
$env:PYTHONWARNINGS = "ignore::DeprecationWarning:websockets"

if (-not $env:HF_HUB_VERBOSITY) { $env:HF_HUB_VERBOSITY = "error" }
if (-not $env:TOKENIZERS_PARALLELISM) { $env:TOKENIZERS_PARALLELISM = "false" }

# === model resolution ===
$ModelPath = if ($env:PAEKA_LLM__MODEL_PATH) {
    $env:PAEKA_LLM__MODEL_PATH
} else {
    "models\qwen\Qwen3.5-9B-Q4_K_M.gguf"
}

$LlamaExe = (Resolve-Path "bin\llama-server.exe" -ErrorAction SilentlyContinue)

# === helper: safe quoting ===
function Quote-Arg {
    param([string]$arg)

    if ($null -eq $arg) { return '""' }

    if ($arg -match '[\s"`]') {
        return '"' + ($arg -replace '"','\"') + '"'
    }
    return $arg
}

# probe capabilities
$helpOutput = ""
if ($LlamaExe) {
    $helpOutput = & $LlamaExe.Path "--help" 2>&1 | Out-String
}

$FlashAttnFlag = $false
if (-not $NoFlashAttn -and ($helpOutput -match "--flash-attn")) {
    $FlashAttnFlag = $true
}

# generation params
if ($EnableThinking -or $CodingMode) {
    $Temp = "0.6"
    $TopP = "0.95"
    $PresencePenalty = "0.0"
} else {
    $Temp = "0.7"
    $TopP = "0.8"
    $PresencePenalty = "1.5"
}

$TopK = "20"
$MinP = "0.0"

Write-Host ""
Write-Host "PAEKA - Personal AI Engineering and Knowledge Assistant" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor DarkGray

# === STEP 1: Weaviate ===
if (-not $SkipWeaviate) {
    Write-Host "[1/3] Starting Weaviate..." -ForegroundColor Yellow

    $null = docker compose up -d paeka-weaviate 2>&1
}

# === STEP 2: llama-server ===
$LlamaProcess = $null

if (-not $SkipLlama) {
    Write-Host "[2/3] Starting llama-server..." -ForegroundColor Yellow

    if (-not $LlamaExe) {
        Write-Host "ERROR: llama-server.exe missing" -ForegroundColor Red
        exit 1
    }

    $ModelFullPath = Resolve-Path $ModelPath -ErrorAction SilentlyContinue
    if (-not $ModelFullPath) {
        Write-Host "ERROR: model not found" -ForegroundColor Red
        exit 1
    }

    $llamaArgList = @(
        "--model", $ModelFullPath.Path,
        "--host", "0.0.0.0",
        "--port", "$LlamaPort",
        "--n-gpu-layers", "$GpuLayers",
        "-c", "$CtxSize",
        "--batch-size", "$BatchSize",
        "--temp", $Temp,
        "--top-p", $TopP,
        "--top-k", $TopK,
        "--min-p", $MinP,
        "--presence-penalty", $PresencePenalty,
        "--jinja"
    )

    # chat template kwargs
    if ($EnableThinking) {
        $llamaArgList += "--chat-template-kwargs"
        $llamaArgList += '{"enable_thinking":true}'
    } else {
        $llamaArgList += "--chat-template-kwargs"
        $llamaArgList += '{"enable_thinking":false}'
    }

    if ($FlashAttnFlag) {
        $llamaArgList += "--flash-attn"
    }

    # build safe argument string (FIXED PART)
    $argString = ($llamaArgList | ForEach-Object { Quote-Arg $_ }) -join " "

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $LlamaExe.Path
    $psi.Arguments = $argString
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true

    New-Item -ItemType Directory -Force -Path "logs" | Out-Null
    $logPath = "$PWD\logs\llama-server.log"

    $LlamaProcess = [System.Diagnostics.Process]::Start($psi)

    Register-ObjectEvent -InputObject $LlamaProcess -EventName OutputDataReceived `
        -SourceIdentifier "LlamaOut_$($LlamaProcess.Id)" -MessageData $logPath -Action {
        if ($EventArgs.Data) {
            Add-Content $Event.MessageData "[$(Get-Date -Format 'HH:mm:ss')] $($EventArgs.Data)"
        }
    }

    Register-ObjectEvent -InputObject $LlamaProcess -EventName ErrorDataReceived `
        -SourceIdentifier "LlamaErr_$($LlamaProcess.Id)" -MessageData $logPath -Action {
        if ($EventArgs.Data) {
            Add-Content $Event.MessageData "[$(Get-Date -Format 'HH:mm:ss')][ERR] $($EventArgs.Data)"
        }
    }

    $LlamaProcess.BeginOutputReadLine()
    $LlamaProcess.BeginErrorReadLine()

    Write-Host "PID: $($LlamaProcess.Id)"
}

# === STEP 3: API ===
Write-Host "[3/3] Starting PAEKA API..." -ForegroundColor Yellow

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
}
finally {
    if ($LlamaProcess -and -not $LlamaProcess.HasExited) {
        $LlamaProcess.Kill()
    }
    Write-Host "PAEKA stopped."
}