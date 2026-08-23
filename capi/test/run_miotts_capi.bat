@echo off
set EXE=D:\audiocpp_test\capi_tts_test.exe
set "TEXT=Hello, this is a short test of the audio text to speech system."
%EXE% --model D:\iStation\Models\sound\audiocpp\miotts-1.7b-q8_0.gguf --family miotts --task 5 --backend 1 --threads 8 --text "%TEXT%" --out D:\audiocpp_test\out\miotts.capi.wav --options-file D:\audiocpp_test\voice_ref.json > D:\audiocpp_test\out\miotts.capi.log 2>&1
echo ALL_DONE
