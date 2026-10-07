@echo off
rem world_shift.dll derlemesi (VS 2022 x64). WORLD kaynağı: ..\..\github\World (değiştirilmez).
setlocal
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VS=%%i
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set WORLD=%~dp0..\..\github\World\src
if not exist "%~dp0bin" mkdir "%~dp0bin"
if not exist "%~dp0obj" mkdir "%~dp0obj"
cl /nologo /O2 /EHsc /MT /LD /utf-8 /I "%WORLD%" "%~dp0world_shift.cpp" "%WORLD%\cheaptrick.cpp" "%WORLD%\common.cpp" "%WORLD%\d4c.cpp" "%WORLD%\dio.cpp" "%WORLD%\fft.cpp" "%WORLD%\harvest.cpp" "%WORLD%\matlabfunctions.cpp" "%WORLD%\stonemask.cpp" "%WORLD%\synthesis.cpp" /Fo"%~dp0obj\\" /Fe"%~dp0bin\world_shift.dll"
