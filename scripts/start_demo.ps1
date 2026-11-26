# 一键开演：启动服务 + 自检 + 暖机
# 用法： .\scripts\start_demo.ps1            （默认 8078）
#        .\scripts\start_demo.ps1 -Port 8079
param([int]$Port = 8078, [switch]$E2E)

$ErrorActionPreference = 'Stop'
$root = 'C:\vlmdisaster'
Set-Location $root
$py = Join-Path $root 'venv\Scripts\python.exe'

if (-not (Test-Path $py)) { Write-Host "找不到 venv：$py" -ForegroundColor Red; exit 1 }

# 端口占用检查
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
  Write-Host "端口 $Port 已被占用（进程 $($busy.OwningProcess)）。换端口：-Port 8079" -ForegroundColor Yellow
  exit 1
}

Write-Host "启动服务（端口 $Port）…" -ForegroundColor Cyan
$stamp = Get-Date -Format 'HHmmss'
$log = Join-Path $root ("outputs\serve_demo_{0}.log" -f $stamp)
$elog = Join-Path $root ("outputs\serve_demo_{0}.err.log" -f $stamp)
$proc = Start-Process -FilePath $py -ArgumentList @('-m','uvicorn','scripts.serve:app','--host','127.0.0.1','--port',"$Port") `
  -WorkingDirectory $root -RedirectStandardOutput $log -RedirectStandardError $elog -PassThru -WindowStyle Hidden

# 等服务起来
$ok = $false
for ($i = 0; $i -lt 30; $i++) {
  Start-Sleep -Seconds 1
  try {
    $null = Invoke-WebRequest "http://127.0.0.1:$Port/api/status" -TimeoutSec 3
    $ok = $true; break
  } catch { }
}
if (-not $ok) {
  Write-Host "服务没起来，日志：$log" -ForegroundColor Red
  if (Test-Path $log) { Get-Content $log -Tail 15 }
  exit 1
}

Write-Host "服务已在 http://127.0.0.1:$Port 运行（PID $($proc.Id)）" -ForegroundColor Green
Write-Host "开始自检 + 暖机（首次约 1–2 分钟）…" -ForegroundColor Cyan
& $py scripts\demo_check.py --warm --py "http://127.0.0.1:$Port"
if ($E2E) {
  Write-Host "全元素端到端自测（38 个，约 7 分钟）…" -ForegroundColor Cyan
  & $py scripts\e2e_all.py --py "http://127.0.0.1:$Port"
}
$code = $LASTEXITCODE

Write-Host ""
Write-Host "浏览器打开：http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "停止服务：Stop-Process -Id $($proc.Id)" -ForegroundColor DarkGray
exit $code
