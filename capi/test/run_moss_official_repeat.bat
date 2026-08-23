@echo off
set EXE=D:\audiocpp_test\audiocpp_cli.exe
set "TEXT=Hello, this is a short test of the audio text to speech system."
set OUT=D:\audiocpp_test\out
%EXE% --task tts --family moss_tts_local --model D:\iStation\Models\sound\audiocpp\moss-tts-local-v1.5-q8_0.gguf --backend cuda --threads 8 --text "%TEXT%" --out %OUT%\moss_tts_local.official2.wav --metrics > %OUT%\moss_tts_local.official2.log 2>&1
%EXE% --task tts --family moss_tts_local --model D:\iStation\Models\sound\audiocpp\moss-tts-local-v1.5-q8_0.gguf --backend cuda --threads 8 --text "%TEXT%" --out %OUT%\moss_tts_local.official3.wav --metrics > %OUT%\moss_tts_local.official3.log 2>&1
%EXE% --task tts --family moss_tts_nano --model D:\iStation\Models\sound\audiocpp\moss-tts-nano-100m-q8_0.gguf --backend cuda --threads 8 --text "%TEXT%" --out %OUT%\moss_tts_nano.official2.wav --metrics > %OUT%\moss_tts_nano.official2.log 2>&1
%EXE% --task tts --family moss_tts_nano --model D:\iStation\Models\sound\audiocpp\moss-tts-nano-100m-q8_0.gguf --backend cuda --threads 8 --text "%TEXT%" --out %OUT%\moss_tts_nano.official3.wav --metrics > %OUT%\moss_tts_nano.official3.log 2>&1
echo ALL_DONE
