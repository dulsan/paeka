# =============================================================================
# PAEKA - setup.ps1
# =============================================================================
# First-time setup. Run once after cloning the repo.
#
# What it does:
#   1. Creates required directories
#   2. Copies .env.example -> .env (if .env doesn't exist)
#   3. Runs uv sync to install all Python dependencies
#   4. Downloads the latest llama-server.exe (CUDA 12.x build)
#   5. Prints next steps
#
# Usage:
#   .\setup.ps1
# =============================================================================

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Split-Path -Parent $ScriptDir)   # go to repo root

Write-Host ""
Write-Host "  PAEKA Setup" -ForegroundColor Cyan
Write-Host "  -----------------------------------------" -ForegroundColor DarkGray

# -- 1. Directories ------------------------------------------------------------
Write-Host "[1/5] Creating directories..." -ForegroundColor Yellow
$dirs = @(
    "bin",
    "logs",
    "models\qwen",
    "data\uploads",
    "data\documents",
    "data\hf_cache",
    "database\sqlite",
    "database\weaviate"
)
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
Write-Host "      Done." -ForegroundColor Green

# -- 2. .env -------------------------------------------------------------------
Write-Host "[2/5] Environment file..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "      .env created from .env.example - edit it before starting." -ForegroundColor Green
    } else {
        Write-Host "      .env.example not found - skipping." -ForegroundColor DarkYellow
    }
} else {
    Write-Host "      .env already exists - skipping." -ForegroundColor DarkGray
}

# -- 3. Python dependencies ----------------------------------------------------
Write-Host "[3/5] Installing Python dependencies (uv sync)..." -ForegroundColor Yellow
Write-Host "      This takes 3-8 minutes on first run." -ForegroundColor DarkGray
uv sync --no-dev
Write-Host "      Dependencies installed." -ForegroundColor Green

# -- 4. llama-server.exe -------------------------------------------------------
Write-Host "[4/5] Checking llama-server.exe..." -ForegroundColor Yellow
if (Test-Path "bin\llama-server.exe") {
    Write-Host "      Already present - skipping download." -ForegroundColor DarkGray
} else {
    Write-Host "      Fetching latest llama.cpp release..." -ForegroundColor DarkGray
    try {
        $release = Invoke-RestMethod `
            -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest" `
            -Headers @{ "User-Agent" = "PAEKA-setup" }

        # Find the win-cuda asset (prefer cu12, fall back to any cuda build)
        $asset = $release.assets | Where-Object {
            $_.name -match "win" -and $_.name -match "cuda" -and $_.name -match "cu12"
        } | Select-Object -First 1

        if (-not $asset) {
            $asset = $release.assets | Where-Object {
                $_.name -match "win" -and $_.name -match "cuda"
            } | Select-Object -First 1
        }

        if ($asset) {
            Write-Host "      Downloading $($asset.name)..." -ForegroundColor DarkGray
            $zipPath = "bin\llama-cpp-win.zip"
            Invoke-WebRequest -Uri $asset.browser_download_url `
                -OutFile $zipPath -UseBasicParsing
            Expand-Archive -Path $zipPath -DestinationPath "bin\llama-tmp" -Force
            # Find and copy llama-server.exe from the extracted folder
            $exe = Get-ChildItem -Path "bin\llama-tmp" -Recurse -Filter "llama-server.exe" |
                   Select-Object -First 1
            if ($exe) {
                Copy-Item $exe.FullName "bin\llama-server.exe"
                Write-Host "      llama-server.exe installed." -ForegroundColor Green
            } else {
                Write-Host "      llama-server.exe not found in archive - install manually." -ForegroundColor Red
            }
            Remove-Item $zipPath -Force
            Remove-Item "bin\llama-tmp" -Recurse -Force
        } else {
            Write-Host "      No matching CUDA release found." -ForegroundColor DarkYellow
            Write-Host "      Download manually from:" -ForegroundColor Yellow
            Write-Host "      https://github.com/ggml-org/llama.cpp/releases" -ForegroundColor Yellow
            Write-Host "      Get the win-cuda-cu12.x build, place llama-server.exe in .\bin\" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "      Auto-download failed: $_" -ForegroundColor DarkYellow
        Write-Host "      Download manually from:" -ForegroundColor Yellow
        Write-Host "      https://github.com/ggml-org/llama.cpp/releases" -ForegroundColor Yellow
    }
}

# -- 5. Model check ------------------------------------------------------------
Write-Host "[5/5] Model check..." -ForegroundColor Yellow
$modelPath = "models\qwen\Qwen3.5-9B-Q4_K_M.gguf"
if (Test-Path $modelPath) {
    $size = [math]::Round((Get-Item $modelPath).Length / 1GB, 2)
    Write-Host "      Model found: $modelPath ($size GB)" -ForegroundColor Green
} else {
    Write-Host "      Model not found." -ForegroundColor DarkYellow
    Write-Host "      Download Qwen3.5-9B-Q4_K_M.gguf (~5.7 GB) from:" -ForegroundColor Yellow
    Write-Host "      https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF" -ForegroundColor Yellow
    Write-Host "      Place at: .\models\qwen\Qwen3.5-9B-Q4_K_M.gguf" -ForegroundColor Yellow
}

# -- Summary -------------------------------------------------------------------
Write-Host ""
Write-Host "  -----------------------------------------" -ForegroundColor DarkGray
Write-Host "  Setup complete. Next steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Edit .env with your settings (optional)" -ForegroundColor White
Write-Host "  2. Make sure Docker Desktop is running" -ForegroundColor White
Write-Host "  3. Start PAEKA:" -ForegroundColor White
Write-Host "       .\scripts\start.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Terax connection:" -ForegroundColor White
Write-Host "    Base URL : http://localhost:8000/v1" -ForegroundColor Cyan
Write-Host "    API Key  : paeka-local" -ForegroundColor Cyan
Write-Host ""
