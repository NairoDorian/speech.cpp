@echo off
set EXE=D:\audiocpp_test\audiocpp_cli.exe
set CAPI=D:\audiocpp_test\capi_tts_test.exe
set "TEXT=Hello, this is a short test of the audio text to speech system."
set REF=D:\audiocpp_test\ref_voice.wav
set OUT=D:\audiocpp_test\out
%EXE% --task tts --family miotts --model D:\iStation\Models\sound\audiocpp\miotts-1.7b-q8_0.gguf --backend cuda --threads 8 --text "%TEXT%" --out %OUT%\miotts.official2.wav --metrics --voice-ref %REF% > %OUT%\miotts.official2.log 2>&1
%EXE% --task tts --family miotts --model D:\iStation\Models\sound\audiocpp\miotts-1.7b-q8_0.gguf --backend cuda --threads 8 --text "%TEXT%" --out %OUT%\miotts.official3.wav --metrics --voice-ref %REF% > %OUT%\miotts.official3.log 2>&1
%CAPI% --model D:\iStation\Models\sound\audiocpp\miotts-1.7b-q8_0.gguf --family miotts --task 5 --backend 1 --threads 8 --text "%TEXT%" --out %OUT%\miotts.capi2.wav --options-file D:\audiocpp_test\voice_ref.json > %OUT%\miotts.capi2.log 2>&1
echo ALL_DONE
