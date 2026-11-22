# 启动 Web 服务。用法：.\run.ps1 [-Port 8765]
param([int]$Port = 8765)
. "$PSScriptRoot\env.ps1"
Write-Host "ad-viz: http://127.0.0.1:$Port  （Ctrl+C 停止）"
& $env:ADVIZ_PYTHON -m uvicorn adviz.server:app --host 127.0.0.1 --port $Port
