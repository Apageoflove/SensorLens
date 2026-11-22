# 运行单元测试。
. "$PSScriptRoot\env.ps1"
& $env:ADVIZ_PYTHON -m pytest @args
exit $LASTEXITCODE
