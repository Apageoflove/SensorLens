# 运行单元测试。
. "$PSScriptRoot\env.ps1"
& $env:SENSORLENS_PYTHON -m pytest @args
exit $LASTEXITCODE
