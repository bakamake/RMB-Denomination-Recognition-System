# RMB-Dataset 克隆和预处理脚本
# 用于克隆数据集并清理不需要的文件

param(
    [string]$TargetDir = ".",
    [switch]$Force
)

Write-Host "=== RMB-Dataset 设置脚本 ===" -ForegroundColor Cyan
$datasetPath = Join-Path $TargetDir "RMB-Dataset"

# 检查目标目录
if (Test-Path $datasetPath) {
    if ($Force) {
        Write-Host "删除现有目录..." -ForegroundColor Yellow
        Remove-Item -Path $datasetPath -Recurse -Force
    } else {
        Write-Host "目录已存在，使用 -Force 强制删除" -ForegroundColor Red
        exit 1
    }
}

# 克隆数据集
Write-Host "克隆 RMB-Dataset..." -ForegroundColor Blue
Write-Host "从 GitHub 下载: https://github.com/bat67/RMB-Dataset.git" -ForegroundColor Gray
git clone "https://github.com/bat67/RMB-Dataset.git" $datasetPath

# 清理不需要的文件
Write-Host "清理数据集..." -ForegroundColor Blue
$rmdbPath = Join-Path $datasetPath "RMBDataset"

# 保留的目录
$keepDirs = @("1", "5", "10", "20", "50", "100")

# 删除不需要的目录
Get-ChildItem -Path $rmdbPath -Directory | Where-Object { $keepDirs -notcontains $_.Name } | ForEach-Object {
    Write-Host "删除: $($_.Name)" -ForegroundColor Yellow
    Remove-Item -Path $_.FullName -Recurse -Force
}

# 删除Git文件
Remove-Item -Path (Join-Path $datasetPath ".git") -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path (Join-Path $datasetPath ".gitattributes") -Force -ErrorAction SilentlyContinue
Remove-Item -Path (Join-Path $datasetPath ".gitignore") -Force -ErrorAction SilentlyContinue
Remove-Item -Path (Join-Path $datasetPath "README.md") -Force -ErrorAction SilentlyContinue
Remove-Item -Path (Join-Path $rmdbPath "README.md") -Force -ErrorAction SilentlyContinue

Write-Host "完成！数据集已准备在: $datasetPath" -ForegroundColor Green

# 显示结果
Get-ChildItem -Path $rmdbPath -Directory | Sort-Object Name | ForEach-Object {
    $count = (Get-ChildItem -Path $_.FullName -File).Count
    Write-Host "$($_.Name): $count 个文件" -ForegroundColor White
}