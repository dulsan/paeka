# ingest_folder.ps1
# Bulk-ingests all documents from a folder into PAEKA.
#
# Usage:
#   .\ingest_folder.ps1 -Folder "D:\PhD_Work\PhD_Research\PhD Library\collected_pdfs"
#
# Optional flags:
#   -Recurse       also scan subdirectories
#   -Endpoint      override the PAEKA API URL (default: http://localhost:8000)
#   -Extensions    comma-separated list of extensions to include
#                  (default: pdf,docx,txt,md,html,xlsx)
#
# Examples:
#   .\ingest_folder.ps1 -Folder "D:\papers" -Recurse
#   .\ingest_folder.ps1 -Folder "D:\papers" -Extensions "pdf,docx"

param(
    [Parameter(Mandatory = $true)]
    [string]$Folder,

    [switch]$Recurse,

    [string]$Endpoint = "http://localhost:8000",

    [string]$Extensions = "pdf,docx,txt,md,html,xlsx,tex"
)

$UploadUrl = "$Endpoint/api/documents/upload"
$extList   = $Extensions.Split(",") | ForEach-Object { "." + $_.Trim().TrimStart(".").ToLower() }

# Collect files
$getParams = @{ Path = $Folder; File = $true; ErrorAction = "Stop" }
if ($Recurse) { $getParams.Recurse = $true }

try {
    $files = Get-ChildItem @getParams | Where-Object {
        $extList -contains $_.Extension.ToLower()
    }
} catch {
    Write-Error "Cannot read folder: $Folder"
    exit 1
}

if ($files.Count -eq 0) {
    Write-Host "No matching files found in $Folder" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "PAEKA Bulk Ingest" -ForegroundColor Cyan
Write-Host "Folder   : $Folder" -ForegroundColor Cyan
Write-Host "Endpoint : $UploadUrl" -ForegroundColor Cyan
Write-Host "Files    : $($files.Count)" -ForegroundColor Cyan
Write-Host ""

$success  = 0
$skipped  = 0
$failed   = 0
$errors   = @()

foreach ($file in $files) {
    $label = $file.Name.PadRight(60).Substring(0, 60)
    Write-Host -NoNewline "  $label "

    try {
        # Build multipart form using .NET directly — avoids curl quoting issues on Windows
        $form    = [System.Net.Http.MultipartFormDataContent]::new()
        $bytes   = [System.IO.File]::ReadAllBytes($file.FullName)
        $content = [System.Net.Http.ByteArrayContent]::new($bytes)

        # Set Content-Type based on extension
        $mime = switch ($file.Extension.ToLower()) {
            ".pdf"  { "application/pdf" }
            ".docx" { "application/vnd.openxmlformats-officedocument.wordprocessingml.document" }
            ".xlsx" { "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" }
            ".html" { "text/html" }
            ".md"   { "text/markdown" }
            ".tex"  { "text/x-latex" }
            default { "text/plain" }
        }
        $content.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::new($mime)
        $form.Add($content, "file", $file.Name)

        $client         = [System.Net.Http.HttpClient]::new()
        $client.Timeout = [System.TimeSpan]::FromMinutes(10)
        $response = $client.PostAsync($UploadUrl, $form).GetAwaiter().GetResult()
        $body     = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        $client.Dispose()

        if ($response.IsSuccessStatusCode) {
            $json = $body | ConvertFrom-Json
            Write-Host "OK  [$($json.status)]" -ForegroundColor Green
            $success++
        } elseif ($response.StatusCode -eq 409) {
            Write-Host "SKIP [already ingested]" -ForegroundColor DarkYellow
            $skipped++
        } else {
            Write-Host "FAIL [$($response.StatusCode)]" -ForegroundColor Red
            $errors += "$($file.Name): HTTP $($response.StatusCode) — $body"
            $failed++
        }
    } catch {
        Write-Host "ERR  [$($_.Exception.Message)]" -ForegroundColor Red
        $errors += "$($file.Name): $($_.Exception.Message)"
        $failed++
    }
}

Write-Host ""
Write-Host "─────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host "Done.  Success: $success  Skipped: $skipped  Failed: $failed" -ForegroundColor Cyan

if ($errors.Count -gt 0) {
    Write-Host ""
    Write-Host "Errors:" -ForegroundColor Red
    $errors | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkRed }
}
