[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$PreferredBackendPort = 8000,
    [ValidateRange(1024, 65535)]
    [int]$PreferredFrontendPort = 5173
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$frontendPath = Join-Path $projectRoot "frontend"
$nodePath = (Get-Command node.exe -ErrorAction SilentlyContinue).Source
$vitePath = Join-Path $frontendPath "node_modules\vite\bin\vite.js"
$runtimePath = Join-Path $projectRoot ".demo-runtime"
$statePath = Join-Path $runtimePath "state.json"

function Test-PortAvailable([int]$Port) {
    $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return $null -eq $listener
}

function Find-AvailablePort([int]$PreferredPort, [int]$MaximumPort) {
    for ($port = $PreferredPort; $port -le $MaximumPort; $port++) {
        if (Test-PortAvailable $port) { return $port }
    }
    throw "No available local demo port was found between $PreferredPort and $MaximumPort."
}

function Test-Endpoint([string]$Uri) {
    try {
        $response = Invoke-RestMethod -Uri $Uri -TimeoutSec 2
        return $null -ne $response
    } catch {
        return $false
    }
}

function Wait-Endpoint([string]$Uri, [int]$Seconds, [System.Diagnostics.Process]$Process) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if ($null -ne $Process -and $Process.HasExited) { return $false }
        if (Test-Endpoint $Uri) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function New-ProcessRecord([System.Diagnostics.Process]$Process) {
    return [ordered]@{
        id = $Process.Id
        name = $Process.ProcessName
        started_at = $Process.StartTime.ToUniversalTime().ToString("o")
    }
}

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "The project virtual environment was not found at .venv\Scripts\python.exe."
}
if (-not (Test-Path -LiteralPath $frontendPath -PathType Container)) {
    throw "The frontend directory was not found."
}
if (-not $nodePath) {
    throw "node.exe was not found. Install Node.js or add it to PATH before the demo."
}
if (-not (Test-Path -LiteralPath $vitePath -PathType Leaf)) {
    throw "Frontend dependencies were not found. Run npm install in the frontend directory before the demo."
}

if (Test-Path -LiteralPath $statePath -PathType Leaf) {
    $previous = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    $trackedProcesses = @($previous.backend_process, $previous.frontend_process) | Where-Object { $null -ne $_ }
    $stillRunning = $false
    foreach ($record in $trackedProcesses) {
        if (Get-Process -Id $record.id -ErrorAction SilentlyContinue) { $stillRunning = $true }
    }
    if ($stillRunning) {
        throw "A launcher-managed demo may already be running. Run .\Stop-ResearchDemo.ps1 first."
    }
}

New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null

$frontendPort = Find-AvailablePort $PreferredFrontendPort ($PreferredFrontendPort + 99)
if ($frontendPort -ne $PreferredFrontendPort) {
    Write-Host "Port $PreferredFrontendPort is occupied; using frontend port $frontendPort safely."
}
$frontendUri = "http://127.0.0.1:$frontendPort"

Write-Host "Checking MongoDB connectivity..."
$pingCode = "from backend.config import Settings; from pymongo import MongoClient; s=Settings.from_environment(); c=MongoClient(s.mongodb_uri, serverSelectionTimeoutMS=5000); c.admin.command('ping'); c.close()"
& $pythonPath -c $pingCode
if ($LASTEXITCODE -ne 0) {
    throw "MongoDB is not reachable. The demo was not started. Review the local .env configuration and network connection."
}

$backendProcess = $null
$preferredReadyUri = "http://127.0.0.1:$PreferredBackendPort/ready"
$canReuseBackend = $frontendPort -eq $PreferredFrontendPort -and -not (Test-PortAvailable $PreferredBackendPort) -and (Test-Endpoint $preferredReadyUri)
if ($canReuseBackend) {
    $backendPort = $PreferredBackendPort
    Write-Host "Reusing the healthy backend already running on port $backendPort."
} else {
    $backendPort = Find-AvailablePort $PreferredBackendPort ($PreferredBackendPort + 99)
    if ($backendPort -ne $PreferredBackendPort) {
        Write-Host "Port $PreferredBackendPort belongs to another process; using backend port $backendPort safely."
    }
    $backendOut = Join-Path $runtimePath "backend-output.log"
    $backendErr = Join-Path $runtimePath "backend-error.log"
    $backendArguments = @("-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", $backendPort)
    $previousAllowedOrigins = $env:API_ALLOWED_ORIGINS
    try {
        $env:API_ALLOWED_ORIGINS = $frontendUri
        $backendProcess = Start-Process -FilePath $pythonPath -ArgumentList $backendArguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $backendOut -RedirectStandardError $backendErr -PassThru
    } finally {
        $env:API_ALLOWED_ORIGINS = $previousAllowedOrigins
    }
    $readyUri = "http://127.0.0.1:$backendPort/ready"
    if (-not (Wait-Endpoint $readyUri 45 $backendProcess)) {
        if (-not $backendProcess.HasExited) { Stop-Process -Id $backendProcess.Id }
        throw "The backend did not become ready. Check .demo-runtime\backend-error.log."
    }
}

$frontendOut = Join-Path $runtimePath "frontend-output.log"
$frontendErr = Join-Path $runtimePath "frontend-error.log"
$previousApiBaseUrl = $env:VITE_API_BASE_URL
try {
    $env:VITE_API_BASE_URL = "http://127.0.0.1:$backendPort"
    $frontendArguments = @($vitePath, "--host", "127.0.0.1", "--port", $frontendPort, "--strictPort")
    $frontendProcess = Start-Process -FilePath $nodePath -ArgumentList $frontendArguments -WorkingDirectory $frontendPath -WindowStyle Hidden -RedirectStandardOutput $frontendOut -RedirectStandardError $frontendErr -PassThru
} finally {
    $env:VITE_API_BASE_URL = $previousApiBaseUrl
}

if (-not (Wait-Endpoint $frontendUri 45 $frontendProcess)) {
    if (-not $frontendProcess.HasExited) { Stop-Process -Id $frontendProcess.Id }
    if ($null -ne $backendProcess -and -not $backendProcess.HasExited) { Stop-Process -Id $backendProcess.Id }
    throw "The frontend did not become ready. Check .demo-runtime\frontend-error.log."
}

$state = [ordered]@{
    schema_version = 1
    backend_url = "http://127.0.0.1:$backendPort"
    frontend_url = $frontendUri
    backend_process = if ($null -eq $backendProcess) { $null } else { New-ProcessRecord $backendProcess }
    frontend_process = New-ProcessRecord $frontendProcess
}
$state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statePath -Encoding UTF8

Write-Host ""
Write-Host "Research demo is ready." -ForegroundColor Green
Write-Host "Open: $frontendUri" -ForegroundColor Cyan
Write-Host "Stop safely with: .\Stop-ResearchDemo.ps1"
