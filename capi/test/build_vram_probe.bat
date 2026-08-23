@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cd /d F:\M1AO_Projects\audio.cpp\capi\test
cl /nologo /O2 /Fe:vram_probe.exe vram_probe.c /I..\include /link /LIBPATH:G:\iStation\Sound audiocpp.lib
if errorlevel 1 exit /b 1
echo BUILD_OK
