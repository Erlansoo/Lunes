# Builds kinect\bin\kinect_bridge.exe with the C# compiler that ships with Windows (.NET Framework 4)
# against the Kinect for Windows SDK 1.8. Run from any folder:  powershell -File kinect\build.ps1
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$csc = "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
$kinect = "C:\Program Files\Microsoft SDKs\Kinect\v1.8\Assemblies\Microsoft.Kinect.dll"
if (-not (Test-Path $kinect)) { throw "Kinect for Windows SDK 1.8 not found at $kinect" }
$bin = Join-Path $here "bin"
New-Item -ItemType Directory -Force $bin | Out-Null
& $csc /nologo /optimize /platform:x64 /target:exe "/reference:$kinect" /reference:System.Drawing.dll "/out:$bin\kinect_bridge.exe" "$here\KinectBridge.cs"
if ($LASTEXITCODE -ne 0) { throw "build failed" }
Copy-Item $kinect $bin -Force  # the bridge loads it from its own folder
Write-Output "built $bin\kinect_bridge.exe"
