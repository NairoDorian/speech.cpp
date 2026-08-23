@echo off
set EXE=D:\audiocpp_test\capi_tts_test.exe
set "TEXT=Hello, this is a short test of the audio text to speech system."
set OUT=D:\audiocpp_test\out
%EXE% --model D:\iStation\Models\sound\audiocpp\qwen3-tts-12hz-1.7b-voicedesign-q8_0.gguf --family qwen3_tts --task 10 --backend 1 --threads 8 --text "%TEXT%" --out %OUT%\qwen3_voicedesign.capi2.wav --options-file D:\audiocpp_test\instruct.json > %OUT%\qwen3_voicedesign.capi2.log 2>&1
%EXE% --model D:\iStation\Models\sound\audiocpp\moss-tts-local-v1.5-q8_0.gguf --family moss_tts_local --task 5 --backend 1 --threads 8 --text "%TEXT%" --out %OUT%\moss_tts_local.capi2.wav > %OUT%\moss_tts_local.capi2.log 2>&1
echo ALL_DONE
