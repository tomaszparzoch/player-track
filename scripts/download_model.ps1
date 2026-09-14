# Downloads YOLOv8 ONNX models for player_track auto mode.
# Usage:
#   .\scripts\download_model.ps1
#   .\scripts\download_model.ps1 -Model yolov8s
#   .\scripts\download_model.ps1 -Model yolov8n

param(
    [ValidateSet("yolov8n", "yolov8s")]
    [string]$Model = "yolov8s"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$modelsDir = Join-Path $root "models"
$outFile = Join-Path $modelsDir "$Model.onnx"

New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null

if (Test-Path $outFile) {
    Write-Host "Model already exists: $outFile"
    exit 0
}

$urls = @(
    "https://huggingface.co/Kalray/yolov8/resolve/main/$Model.onnx"
)

foreach ($url in $urls) {
    Write-Host "Downloading $Model from $url ..."
    try {
        Invoke-WebRequest -Uri $url -OutFile $outFile -UseBasicParsing
        if ((Get-Item $outFile).Length -gt 1MB) {
            Write-Host "Saved: $outFile"
            exit 0
        }
        Remove-Item $outFile -Force -ErrorAction SilentlyContinue
    } catch {
        Write-Warning "Failed: $_"
    }
}

Write-Error "Could not download $Model.onnx. Download manually and place at $outFile"
