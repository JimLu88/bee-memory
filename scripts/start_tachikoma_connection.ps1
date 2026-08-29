$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot; $backendRoot = Join-Path $projectRoot "backend"; $python = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = (Get-Command python.exe -ErrorAction Stop).Source }
$logRoot = Join-Path $projectRoot "logs"; $null = New-Item -ItemType Directory -Path $logRoot -Force
Set-Location -LiteralPath $backendRoot
$ErrorActionPreference = "Continue"
& $python -m uvicorn tachikoma_connection_app:app --host 127.0.0.1 --port 8004 1>> (Join-Path $logRoot "tachikoma-connection.out.log") 2>> (Join-Path $logRoot "tachikoma-connection.err.log"); exit $LASTEXITCODE
