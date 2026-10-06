$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonExecutable = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    throw 'Create .venv and install web_app/requirements-local.txt first (see web_app/README.md).'
}
& $pythonExecutable -m uvicorn web_app.backend.main:app --host 127.0.0.1 --port 8000
