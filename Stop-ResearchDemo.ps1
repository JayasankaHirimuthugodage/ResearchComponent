[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$runtimePath = Join-Path $PSScriptRoot ".demo-runtime"
$statePath = Join-Path $runtimePath "state.json"

if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
    Write-Host "No launcher-managed research demo is recorded as running."
    exit 0
}

$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
$records = @($state.frontend_process, $state.backend_process) | Where-Object { $null -ne $_ }

foreach ($record in $records) {
    $process = Get-Process -Id $record.id -ErrorAction SilentlyContinue
    if ($null -eq $process) { continue }

    $actualStartedAt = $process.StartTime.ToUniversalTime().ToString("o")
    if ($process.ProcessName -ne $record.name -or $actualStartedAt -ne $record.started_at) {
        Write-Warning "PID $($record.id) no longer matches the launcher record and was not stopped."
        continue
    }

    Stop-Process -Id $process.Id
    Write-Host "Stopped launcher-managed process $($process.ProcessName) (PID $($process.Id))."
}

Remove-Item -LiteralPath $statePath
Write-Host "Research demo stopped. Unrelated processes were not changed." -ForegroundColor Green
