$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not $env:YESTIGER_HOST) {
  $env:YESTIGER_HOST = "127.0.0.1"
}
if (-not $env:PORT) {
  $env:PORT = "8765"
}
if (-not $env:YESTIGER_CORS_ORIGIN) {
  $env:YESTIGER_CORS_ORIGIN = "*"
}
if (-not $env:YESTIGER_RUN_DIR) {
  $env:YESTIGER_RUN_DIR = Join-Path $Root "webapp_runs"
}
if (-not $env:HF_HOME) {
  $env:HF_HOME = Join-Path $Root ".hf"
}
if (-not $env:TRANSFORMERS_CACHE) {
  $env:TRANSFORMERS_CACHE = $env:HF_HOME
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
  throw "Virtual environment not found. Run .\setup_windows.ps1 first."
}

New-Item -ItemType Directory -Force -Path `
  (Join-Path $env:YESTIGER_RUN_DIR "uploads"), `
  (Join-Path $env:YESTIGER_RUN_DIR "jobs"), `
  (Join-Path $env:YESTIGER_RUN_DIR "feature_cache"), `
  (Join-Path $env:YESTIGER_RUN_DIR "custom_actions"), `
  $env:HF_HOME | Out-Null

Write-Host "Starting YesTiger at http://$($env:YESTIGER_HOST):$($env:PORT)"
& $VenvPython webapp\server.py --host $env:YESTIGER_HOST --port $env:PORT
