# 让本项目的所有进程把缓存和临时文件写在项目目录内。用法：. .\env.ps1
$root = $PSScriptRoot
$cache = Join-Path $root ".cache"
foreach ($d in @("tmp", "pycache")) { New-Item -ItemType Directory -Force -Path (Join-Path $cache $d) | Out-Null }
$env:TEMP = Join-Path $cache "tmp"
$env:TMP = $env:TEMP
$env:PYTHONPYCACHEPREFIX = Join-Path $cache "pycache"
$env:PYTHONNOUSERSITE = "1"
$env:PIP_NO_INSTALL = "1"
$env:ADVIZ_DATA = Join-Path $root "data"
if (-not $env:ADVIZ_PYTHON) { $env:ADVIZ_PYTHON = "E:\Must_System\Anaconda\python.exe" }
Set-Location $root
