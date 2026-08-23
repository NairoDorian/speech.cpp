@echo off
rem Build + run the fork-regression ctest suite after an upstream merge.
rem Needs a configure with ENGINE_BUILD_TESTS=ON first, e.g.:
rem   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_windows.ps1 ^
rem     -Preset windows-cpu-release -Target audiocpp -BuildTests ON -Jobs 2
rem This script only sets up the MSVC env (vcvars) and builds the labeled
rem test targets, then runs ctest -L fork_regression.
setlocal

rem VS Community vcvars (adjust if the edition moves); cmake/ctest from mingw64.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=D:\dev-tools\mingw64\bin;%PATH%

cd /d "%~dp0..\.."

cmake --build build/windows-cpu-release -j 2 --target ^
  asr_vad_model_path_test capi_enum_sync_test capi_shared_lib_surface_test ^
  tensor_source_memory_backed_test progress_callback_test ^
  backend_weight_store_commit_test capi_option_number_test capi_session_options_test
if errorlevel 1 exit /b 1

ctest --test-dir build/windows-cpu-release -L fork_regression --output-on-failure
