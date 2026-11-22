# 让本项目的所有进程把缓存和临时文件写在项目目录内。用法：. .\env.ps1
$root = $PSScriptRoot
$cache = Join-Path $root ".cache"
foreach ($d in @("tmp", "pycache")) { New-Item -ItemType Directory -Force -Path (Join-Path $cache $d) | Out-Null }
$env:TEMP = Join-Path $cache "tmp"
$env:TMP = $env:TEMP
$env:PYTHONPYCACHEPREFIX = Join-Path $cache "pycache"
$env:PYTHONNOUSERSITE = "1"
$env:PIP_NO_INSTALL = "1"
$env:SENSORLENS_DATA = Join-Path $root "data"
# 本机专用的设置（例如指定解释器 $env:SENSORLENS_PYTHON）写在 env.local.ps1 里，它不会提交到仓库
$local = Join-Path $root "env.local.ps1"
if (Test-Path $local) { . $local }
if (-not $env:SENSORLENS_PYTHON) {
  $venv = Join-Path $root ".venv\Scripts\python.exe"
  $env:SENSORLENS_PYTHON = if (Test-Path $venv) { $venv } else { "python" }
}
Set-Location $root
