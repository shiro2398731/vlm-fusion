# 一键建立运行环境（Windows）
# 说明：本仓库脚本默认以 C:\vlmdisaster 为根目录（作者机器路径）。
# 克隆后执行本脚本，会在 C:\vlmdisaster 建立指向本仓库的目录联接（junction），
# 这样所有脚本无需改路径即可直接运行。
#
# 用法（普通 PowerShell，不需要管理员）：
#   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
#   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -Root "D:\vlmdisaster"

param([string]$Root = "C:\vlmdisaster")

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path

if (Test-Path $Root) {
  $item = Get-Item $Root -Force
  if ($item.LinkType -eq "Junction") {
    Write-Host "已存在联接 $Root -> $($item.Target)，无需重复创建。" -ForegroundColor Yellow
    exit 0
  }
  Write-Host "$Root 已存在且不是联接。请先移走该目录，或用 -Root 指定别的路径。" -ForegroundColor Red
  exit 1
}

New-Item -ItemType Junction -Path $Root -Target $repo | Out-Null
Write-Host "已创建联接：$Root -> $repo" -ForegroundColor Green

Write-Host ""
Write-Host "下一步：" -ForegroundColor Cyan
Write-Host "  1) 准备 Python 3.11 虚拟环境： py -3.11 -m venv venv"
Write-Host "  2) 安装依赖： .\venv\Scripts\pip install -r requirements.txt"
Write-Host "  3) 下载模型与 LoRA（见 README 第三、四节）"
Write-Host "  4) 启动演示： .\venv\Scripts\python -m uvicorn scripts.serve:app --port 8078"
