# Downloads ONNX Runtime with DirectML (GPU on Windows, no CUDA toolkit required).
# Usage: .\scripts\download_onnxruntime.ps1

param(
    [string]$Version = "1.22.1"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ortRoot = Join-Path $root "third_party\onnxruntime"
$tmpDir = Join-Path $env:TEMP "ort-directml-$Version"
$nugetUrl = "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/$Version"
$nugetFile = Join-Path $tmpDir "ort-directml.nupkg"

if (Test-Path (Join-Path $ortRoot "include\onnxruntime_cxx_api.h")) {
    Write-Host "ONNX Runtime already present: $ortRoot"
    exit 0
}

Write-Host "Downloading ONNX Runtime DirectML $Version ..."
New-Item -ItemType Directory -Force -Path $tmpDir, $ortRoot | Out-Null
Invoke-WebRequest -Uri $nugetUrl -OutFile $nugetFile -UseBasicParsing

$extractDir = Join-Path $tmpDir "extract"
if (Test-Path $extractDir) { Remove-Item $extractDir -Recurse -Force }
$zipFile = Join-Path $tmpDir "ort-directml.zip"
Copy-Item $nugetFile $zipFile -Force
Expand-Archive -Path $zipFile -DestinationPath $extractDir -Force

$includeSrc = Join-Path $extractDir "build\native\include"
$nativeSrc = Join-Path $extractDir "runtimes\win-x64\native"
if (!(Test-Path $includeSrc) -or !(Test-Path $nativeSrc)) {
    Write-Error "Unexpected ONNX Runtime package layout"
}

$includeDst = Join-Path $ortRoot "include"
$libDst = Join-Path $ortRoot "lib"
$binDst = Join-Path $ortRoot "bin"
New-Item -ItemType Directory -Force -Path $includeDst, $libDst, $binDst | Out-Null

Copy-Item -Path (Join-Path $includeSrc "*") -Destination $includeDst -Recurse -Force
Copy-Item -Path (Join-Path $nativeSrc "*") -Destination $libDst -Force
Copy-Item -Path (Join-Path $nativeSrc "*.dll") -Destination $binDst -Force

Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "Installed ONNX Runtime DirectML to $ortRoot"
