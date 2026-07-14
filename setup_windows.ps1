$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Find-Python {
  $py = Get-Command py -ErrorAction SilentlyContinue
  if ($py) {
    & py -3.10 --version *> $null
    if ($LASTEXITCODE -eq 0) {
      return @{ Command = "py"; Args = @("-3.10") }
    }
    & py -3.11 --version *> $null
    if ($LASTEXITCODE -eq 0) {
      return @{ Command = "py"; Args = @("-3.11") }
    }
  }
  $python = Get-Command python -ErrorAction SilentlyContinue
  if ($python) {
    return @{ Command = "python"; Args = @() }
  }
  throw "Python was not found. Install Python 3.10 or 3.11 first."
}

$Python = Find-Python

if (-not (Test-Path ".venv")) {
  & $Python.Command @($Python.Args + @("-m", "venv", ".venv"))
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
  throw "Virtual environment Python not found: $VenvPython"
}

& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 torchaudio==2.3.1
& $VenvPython -m pip install -r requirements-webapp.txt

New-Item -ItemType Directory -Force -Path "webapp_runs\uploads", "webapp_runs\jobs", "webapp_runs\feature_cache", "webapp_runs\custom_actions" | Out-Null

Write-Host ""
Write-Host "YesTiger setup complete."
Write-Host "Run: .\start_windows.ps1"
