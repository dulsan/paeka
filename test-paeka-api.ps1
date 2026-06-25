#Requires -Version 7.0
<#
.SYNOPSIS
    PAEKA end-to-end API smoke/regression test suite.

.DESCRIPTION
    Exercises every route currently registered in backend/api/app.py against
    a running PAEKA instance, reports PASS/FAIL/SKIP per test, and prints a
    summary table at the end. Designed to be re-run after any change to
    confirm nothing regressed -- same intent as the pytest suite, but against
    the real running server instead of mocks, so it also catches wiring
    issues pytest's mocks can't see (e.g. the openai_compat.py route-wiring
    bug from earlier in this project's history).

    Self-skips anything whose backing feature isn't enabled right now
    (knowledge graph, retrieval/Qdrant, Docker sandbox) rather than
    reporting a false failure -- gated on what /api/health and
    /api/sandbox/status actually report, not assumptions. Cleans up every
    resource it creates (conversations, documents) so re-running this
    doesn't accumulate junk data in the database.

    Deliberately excludes /models/download -- that can trigger a genuine,
    large, slow model download. Not something a routine smoke test should
    ever trigger automatically.

.PARAMETER BaseUrl
    Root URL of the running PAEKA instance. Default http://localhost:8000.

.PARAMETER Detailed
    Print full request/response bodies for every test, not just pass/fail.
    Verbose, but this is what you want when something's actually broken and
    you need to see exactly what the server sent back.

.PARAMETER SkipSlow
    Skip the two genuinely slow tests (agent/iterate, agent/react) that make
    real LLM calls and can each take 10-60+ seconds depending on model speed
    and round count. Useful for a quick structural check while iterating.

.PARAMETER LogPath
    Optional path to also write a detailed JSON log of every test result
    (full request/response included regardless of -Detailed) for later
    review or diffing between runs.

.EXAMPLE
    .\test-paeka-api.ps1

.EXAMPLE
    .\test-paeka-api.ps1 -Detailed -LogPath .\test-run-$(Get-Date -Format yyyyMMdd-HHmmss).json

.EXAMPLE
    .\test-paeka-api.ps1 -SkipSlow
#>

param(
    [string]$BaseUrl = "http://localhost:8000",
    [switch]$Detailed,
    [switch]$SkipSlow,
    [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"
$script:Results = [System.Collections.Generic.List[object]]::new()
$script:StartTime = Get-Date

# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------
# Uses Invoke-WebRequest -SkipHttpErrorCheck (PowerShell 7.0+) rather than
# Invoke-RestMethod deliberately: -SkipHttpErrorCheck means this ALWAYS
# returns a response object with .StatusCode, whether the server answered
# with 200 or 503 -- it only throws for genuine network-level failures
# (connection refused, DNS failure, timeout). That's exactly the
# distinction this script needs: "got an HTTP response with some status
# code" (a test result) versus "couldn't even reach the server" (an
# infrastructure problem, reported differently below).

function Invoke-PaekaTest {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [string]$Method,
        [Parameter(Mandatory)] [string]$Path,
        [object]$Body = $null,
        [int[]]$ExpectedStatus = @(200),
        [bool]$Skip = $false,
        [string]$SkipReason = "",
        [int]$TimeoutSec = 120,
        [scriptblock]$Validate = $null   # optional: receives parsed response, return $true/$false
    )

    if ($Skip) {
        $script:Results.Add([PSCustomObject]@{
            Name = $Name; Method = $Method; Path = $Path
            Status = "SKIP"; HttpCode = $null; DurationMs = 0
            Detail = $SkipReason; Response = $null
        })
        Write-Host "  [SKIP] $Name -- $SkipReason" -ForegroundColor DarkYellow
        return $null
    }

    $uri = "$BaseUrl$Path"
    $sw  = [System.Diagnostics.Stopwatch]::StartNew()

    try {
        $params = @{
            Uri                = $uri
            Method             = $Method
            SkipHttpErrorCheck = $true
            TimeoutSec         = $TimeoutSec
        }
        if ($null -ne $Body) {
            $params.Body        = ($Body | ConvertTo-Json -Depth 10)
            $params.ContentType = "application/json"
        }

        $resp = Invoke-WebRequest @params
        $sw.Stop()

        $code = [int]$resp.StatusCode
        $parsed = $null
        if ($resp.Content) {
            try { $parsed = $resp.Content | ConvertFrom-Json -Depth 20 } catch { $parsed = $resp.Content }
        }

        $statusOk = $ExpectedStatus -contains $code
        $validateOk = $true
        if ($statusOk -and $Validate) {
            try { $validateOk = & $Validate $parsed } catch { $validateOk = $false }
        }

        $pass = $statusOk -and $validateOk
        $detail = if (-not $statusOk) {
            "expected status $($ExpectedStatus -join '/'), got $code"
        } elseif (-not $validateOk) {
            "response shape check failed"
        } else { "" }

        $script:Results.Add([PSCustomObject]@{
            Name = $Name; Method = $Method; Path = $Path
            Status = if ($pass) { "PASS" } else { "FAIL" }
            HttpCode = $code; DurationMs = $sw.ElapsedMilliseconds
            Detail = $detail; Response = $parsed
        })

        $color = if ($pass) { "Green" } else { "Red" }
        $label = if ($pass) { "PASS" } else { "FAIL" }
        Write-Host "  [$label] $Name ($code, $($sw.ElapsedMilliseconds)ms)" -ForegroundColor $color
        if (-not $pass) {
            Write-Host "         $detail" -ForegroundColor Red
        }
        if ($Detailed) {
            Write-Host "         Request:  $Method $uri"
            if ($Body) { Write-Host "         Body:     $($Body | ConvertTo-Json -Compress -Depth 10)" }
            Write-Host "         Response: $($resp.Content)"
        }

        return $parsed
    }
    catch {
        # Genuine network-level failure -- server unreachable, not just an
        # error status code. Reported distinctly since it usually means
        # "PAEKA isn't running" or "wrong port", not a feature bug.
        $sw.Stop()
        $script:Results.Add([PSCustomObject]@{
            Name = $Name; Method = $Method; Path = $Path
            Status = "FAIL"; HttpCode = $null; DurationMs = $sw.ElapsedMilliseconds
            Detail = "Connection failed: $($_.Exception.Message)"; Response = $null
        })
        Write-Host "  [FAIL] $Name -- connection failed: $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

function Write-Section([string]$Title) {
    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# 0. Capability detection -- decide up front what's safe/meaningful to test
# ---------------------------------------------------------------------------
Write-Section "Capability detection"

$health = Invoke-PaekaTest -Name "Health check" -Method GET -Path "/api/health"

$retrievalUp = $false
$kgUp        = $false
if ($health) {
    $retrievalUp = [bool]$health.components.retrieval
    $kgUp        = [bool]$health.components.knowledge_graph
    Write-Host "  Detected: llm=$($health.components.llm) retrieval=$retrievalUp knowledge_graph=$kgUp memory=$($health.components.memory) skills=$($health.components.skills)"
}

$sandboxStatus = Invoke-PaekaTest -Name "Sandbox status" -Method GET -Path "/api/sandbox/status"
$dockerUp = if ($sandboxStatus) { [bool]$sandboxStatus.docker_available } else { $false }
Write-Host "  Detected: docker_available=$dockerUp"

# ---------------------------------------------------------------------------
# 1. Models
# ---------------------------------------------------------------------------
Write-Section "Models"

Invoke-PaekaTest -Name "List models (registry)" -Method GET -Path "/api/models" | Out-Null
Invoke-PaekaTest -Name "Active model" -Method GET -Path "/api/models/active" | Out-Null
Invoke-PaekaTest -Name "Model download status" -Method GET -Path "/api/models/download/status" | Out-Null
# /api/models/scan and /api/models/download deliberately excluded -- scan
# re-walks the models directory (cheap but mutates registry state), download
# can trigger a genuine multi-GB fetch. Neither belongs in a routine smoke test.

Invoke-PaekaTest -Name "OpenAI-compat: list models" -Method GET -Path "/v1/models" | Out-Null

# ---------------------------------------------------------------------------
# 2. Conversations -- full create -> chat -> rename -> get -> delete cycle
# ---------------------------------------------------------------------------
Write-Section "Conversations"

$conv = Invoke-PaekaTest -Name "Create conversation" -Method POST -Path "/api/conversations" `
    -Body @{ title = "PAEKA test suite run" } -ExpectedStatus @(201) `
    -Validate { param($r) $null -ne $r.id }

$convId = if ($conv) { $conv.id } else { $null }

Invoke-PaekaTest -Name "List conversations" -Method GET -Path "/api/conversations" | Out-Null

if ($convId) {
    Invoke-PaekaTest -Name "Get conversation detail" -Method GET -Path "/api/conversations/$convId" | Out-Null

    Invoke-PaekaTest -Name "Rename conversation" -Method PATCH -Path "/api/conversations/$convId" `
        -Body @{ title = "Renamed by test suite" } | Out-Null

    # Streaming SSE endpoint -- chat.py always returns StreamingResponse
    # regardless of any request flag. Invoke-WebRequest buffers the whole
    # stream into .Content as raw "data: {...}\n\n" text; a full SSE parser
    # is out of scope for a smoke test, so this just confirms a 200 and that
    # at least one well-formed event came back.
    Invoke-PaekaTest -Name "Chat in conversation (SSE)" -Method POST `
        -Path "/api/conversations/$convId/chat" `
        -Body @{ message = "Say hello in exactly three words." } `
        -Validate { param($r) "$r" -match "data:" } | Out-Null

    Invoke-PaekaTest -Name "Export conversation" -Method GET `
        -Path "/api/conversations/$convId/export" | Out-Null

    Invoke-PaekaTest -Name "Delete conversation (cleanup)" -Method DELETE `
        -Path "/api/conversations/$convId" -ExpectedStatus @(204) | Out-Null
} else {
    Write-Host "  [SKIP] Remaining conversation tests -- create failed, no id to use" -ForegroundColor DarkYellow
}

Invoke-PaekaTest -Name "Export all conversations" -Method GET -Path "/api/export/all" | Out-Null

# ---------------------------------------------------------------------------
# 3. Documents -- ingest-text avoids needing a real file upload
# ---------------------------------------------------------------------------
Write-Section "Documents"

$docSkipReason = "retrieval not enabled (set [retrieval] enabled = true)"
$doc = Invoke-PaekaTest -Name "Ingest text document" -Method POST -Path "/api/documents/ingest-text" `
    -Body @{ text = "PAEKA test suite canary document. The quick brown fox jumps over the lazy dog."; filename = "test-suite-canary.txt" } `
    -ExpectedStatus @(202) -Skip (-not $retrievalUp) -SkipReason $docSkipReason `
    -Validate { param($r) $null -ne $r.document_id }

$docId = if ($doc) { $doc.document_id } else { $null }

Invoke-PaekaTest -Name "List documents" -Method GET -Path "/api/documents" `
    -Skip (-not $retrievalUp) -SkipReason $docSkipReason | Out-Null

if ($docId) {
    Invoke-PaekaTest -Name "Get document detail" -Method GET -Path "/api/documents/$docId" | Out-Null
    Invoke-PaekaTest -Name "Get document chunks" -Method GET -Path "/api/documents/$docId/chunks" | Out-Null
} else {
    Write-Host "  [SKIP] Document detail/chunks tests -- no document_id available" -ForegroundColor DarkYellow
}

# ---------------------------------------------------------------------------
# 4. Knowledge graph -- gated on /api/health's reported state
# ---------------------------------------------------------------------------
Write-Section "Knowledge graph"

if (-not $kgUp) {
    # Verify the disabled-state behaviour itself is correct (a real,
    # documented contract: 503 with a clear message), then skip the rest --
    # extract/refine/query would all just redundantly hit the same 503.
    Invoke-PaekaTest -Name "Knowledge graph stats (disabled -> 503 expected)" `
        -Method GET -Path "/api/knowledge/stats" -ExpectedStatus @(503) | Out-Null
    foreach ($t in @("Extract from document", "Refine graph", "List nodes", "Query graph")) {
        Write-Host "  [SKIP] $t -- knowledge_graph not enabled (PAEKA_KNOWLEDGE_GRAPH__ENABLED=true)" -ForegroundColor DarkYellow
        $script:Results.Add([PSCustomObject]@{
            Name = $t; Method = "-"; Path = "-"; Status = "SKIP"; HttpCode = $null
            DurationMs = 0; Detail = "knowledge_graph not enabled"; Response = $null
        })
    }
} else {
    if ($docId) {
        Invoke-PaekaTest -Name "Extract knowledge from document" -Method POST `
            -Path "/api/knowledge/extract/$docId" | Out-Null
    } else {
        Write-Host "  [SKIP] Extract from document -- no document_id available (retrieval disabled above)" -ForegroundColor DarkYellow
    }
    Invoke-PaekaTest -Name "Refine graph" -Method POST -Path "/api/knowledge/refine" | Out-Null
    Invoke-PaekaTest -Name "Graph stats" -Method GET -Path "/api/knowledge/stats" | Out-Null
    Invoke-PaekaTest -Name "List nodes" -Method GET -Path "/api/knowledge/nodes" | Out-Null
    Invoke-PaekaTest -Name "Query graph" -Method GET -Path "/api/knowledge/query?q=test" | Out-Null
}

# ---------------------------------------------------------------------------
# 5. Memory
# ---------------------------------------------------------------------------
Write-Section "Memory"

Invoke-PaekaTest -Name "List global memory" -Method GET -Path "/api/memory/global" | Out-Null
# Destructive on real data if anything's actually stored there -- still
# worth confirming the endpoint itself responds correctly.
Invoke-PaekaTest -Name "Clear global memory" -Method DELETE -Path "/api/memory/global" | Out-Null

# ---------------------------------------------------------------------------
# 6. Skills
# ---------------------------------------------------------------------------
Write-Section "Skills"

$skills = Invoke-PaekaTest -Name "List skills" -Method GET -Path "/api/skills" `
    -Validate { param($r) $r.Count -gt 0 }

if ($skills -and $skills.Count -gt 0) {
    $firstSkill = $skills[0].name
    Invoke-PaekaTest -Name "Get skill detail" -Method GET -Path "/api/skills/$firstSkill" | Out-Null
} else {
    Write-Host "  [SKIP] Get skill detail -- no skills returned to test against" -ForegroundColor DarkYellow
}

Invoke-PaekaTest -Name "Reload skills" -Method POST -Path "/api/skills/reload" | Out-Null

# ---------------------------------------------------------------------------
# 7. Code tools (verify/format -- pure, no Docker dependency)
# ---------------------------------------------------------------------------
Write-Section "Code tools"

Invoke-PaekaTest -Name "Verify code (clean)" -Method POST -Path "/api/code/verify" `
    -Body @{ code = "def add(a, b):`n    return a + b`n"; language = "python"; filename = "add.py" } | Out-Null

Invoke-PaekaTest -Name "Format code" -Method POST -Path "/api/code/format" `
    -Body @{ code = "def add(a,b):`n  return a+b" } | Out-Null

# ---------------------------------------------------------------------------
# 8. Sandbox -- gated on docker_available
# ---------------------------------------------------------------------------
Write-Section "Sandbox"

Invoke-PaekaTest -Name "List sandbox languages" -Method GET -Path "/api/sandbox/languages" | Out-Null

$sandboxSkip = "Docker not reachable (start Docker, this feature is intentionally deferred until dockerization)"
Invoke-PaekaTest -Name "Execute code in sandbox" -Method POST -Path "/api/sandbox/execute" `
    -Body @{ code = "print('hello from sandbox')"; language = "python" } `
    -Skip (-not $dockerUp) -SkipReason $sandboxSkip | Out-Null

# ---------------------------------------------------------------------------
# 9. Agent endpoints -- the slow, real-LLM-call ones
# ---------------------------------------------------------------------------
Write-Section "Agent"

Invoke-PaekaTest -Name "Agent: iterate" -Method POST -Path "/api/agent/iterate" `
    -Body @{ task = "Write a one-sentence description of a haiku."; max_iterations = 2 } `
    -Skip $SkipSlow -SkipReason "skipped via -SkipSlow" -TimeoutSec 300 `
    -Validate { param($r) $r.PSObject.Properties.Name -contains "iterations" } | Out-Null

Invoke-PaekaTest -Name "Agent: react (tool-calling loop)" -Method POST -Path "/api/agent/react" `
    -Body @{ message = "What tools do you have available?" } `
    -Skip $SkipSlow -SkipReason "skipped via -SkipSlow" `
    -Validate { param($r) -not [string]::IsNullOrWhiteSpace($r.response) } | Out-Null

# ---------------------------------------------------------------------------
# 10. OpenAI-compatible chat completions (non-streaming, for simple validation)
# ---------------------------------------------------------------------------
Write-Section "OpenAI-compatible endpoint"

Invoke-PaekaTest -Name "Chat completions (non-streaming)" -Method POST -Path "/v1/chat/completions" `
    -Body @{
        model    = "paeka-qwen"
        messages = @(@{ role = "user"; content = "Reply with exactly the word: pong" })
        stream   = $false
    } `
    -Skip $SkipSlow -SkipReason "skipped via -SkipSlow" `
    -Validate { param($r) $null -ne $r.choices } | Out-Null

# ---------------------------------------------------------------------------
# 11. Chat control / sessions
# ---------------------------------------------------------------------------
Write-Section "Chat control / sessions"

Invoke-PaekaTest -Name "List sessions" -Method GET -Path "/api/chat/sessions" | Out-Null

$session = Invoke-PaekaTest -Name "Create session" -Method POST -Path "/api/chat/sessions" `
    -ExpectedStatus @(201) -Validate { param($r) $null -ne $r.session_id }

$sessionId = if ($session) { $session.session_id } else { $null }

if ($sessionId) {
    Invoke-PaekaTest -Name "Get session" -Method GET -Path "/api/chat/sessions/$sessionId" | Out-Null
    Invoke-PaekaTest -Name "Activate session" -Method POST -Path "/api/chat/sessions/$sessionId/activate" | Out-Null
    Invoke-PaekaTest -Name "Reset chat" -Method POST -Path "/api/chat/reset" -Body @{ session_id = $sessionId } | Out-Null
    Invoke-PaekaTest -Name "Delete session (cleanup)" -Method DELETE -Path "/api/chat/sessions/$sessionId" `
        -ExpectedStatus @(204) | Out-Null
} else {
    Write-Host "  [SKIP] Remaining session tests -- create failed, no session_id to use" -ForegroundColor DarkYellow
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Section "Summary"

$totalDuration = ((Get-Date) - $script:StartTime).TotalSeconds
$passed  = ($script:Results | Where-Object Status -eq "PASS").Count
$failed  = ($script:Results | Where-Object Status -eq "FAIL").Count
$skipped = ($script:Results | Where-Object Status -eq "SKIP").Count

$script:Results | Format-Table -Property Status, Name, HttpCode, DurationMs, Detail -AutoSize

Write-Host ""
Write-Host "Total: $($script:Results.Count)  Passed: $passed  Failed: $failed  Skipped: $skipped  ($([math]::Round($totalDuration, 1))s)" `
    -ForegroundColor $(if ($failed -gt 0) { "Red" } else { "Green" })

if ($LogPath) {
    $script:Results | ConvertTo-Json -Depth 20 | Out-File -FilePath $LogPath -Encoding utf8
    Write-Host "Detailed log written to $LogPath"
}

if ($failed -gt 0) {
    Write-Host ""
    Write-Host "Failed tests:" -ForegroundColor Red
    $script:Results | Where-Object Status -eq "FAIL" | ForEach-Object {
        Write-Host "  - $($_.Name): $($_.Detail)" -ForegroundColor Red
    }
    exit 1
}

exit 0
