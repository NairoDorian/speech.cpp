@echo off
set EXE=D:\audiocpp_test\capi_tts_test.exe
set "TEXT=Hello, this is a short test of the audio text to speech system."
set OUT=D:\audiocpp_test\out
%EXE% --model D:\iStation\Models\sound\audiocpp\irodori-tts-v4-small-q8_0.gguf --family irodori_tts --task 5 --backend 1 --threads 8 --text "%TEXT%" --out %OUT%\irodori_v4_small.capi2.wav > %OUT%\irodori_v4_small.capi2.log 2>&1
%EXE% --model D:\iStation\Models\sound\audiocpp\irodori-tts-v4-small-q8_0.gguf --family irodori_tts --task 5 --backend 1 --threads 8 --text "%TEXT%" --out %OUT%\irodori_v4_small.capi3.wav > %OUT%\irodori_v4_small.capi3.log 2>&1
echo ALL_DONE
