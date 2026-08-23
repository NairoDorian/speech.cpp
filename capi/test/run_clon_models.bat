@echo off
REM Voice-clone models via CAPI (--options-file, QUOTED text; qwen3_customvoice uses task tts)
set EXE=D:\audiocpp_test\capi_tts_test.exe
set "TEXT=Hello, this is a short test of the audio text to speech system."
set OUT=D:\audiocpp_test\out

%EXE% --model D:\iStation\Models\sound\audiocpp\chatterbox-q8_0.gguf --family chatterbox --task 8 --backend 1 --threads 8 --text "%TEXT%" --out %OUT%\chatterbox.capi.wav --options-file D:\audiocpp_test\voice_ref.json > %OUT%\chatterbox.capi.log 2>&1
%EXE% --model D:\iStation\Models\sound\audiocpp\confucius4-tts-orig.gguf --family confucius4_tts --task 8 --backend 1 --threads 8 --text "%TEXT%" --out %OUT%\confucius4_tts.capi.wav --options-file D:\audiocpp_test\voice_ref.json > %OUT%\confucius4_tts.capi.log 2>&1
%EXE% --model D:\iStation\Models\sound\audiocpp\qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf --family qwen3_tts --task 5 --backend 1 --threads 8 --text "%TEXT%" --out %OUT%\qwen3_customvoice.capi.wav --options-file D:\audiocpp_test\voice_ref_text.json > %OUT%\qwen3_customvoice.capi.log 2>&1
%EXE% --model D:\iStation\Models\sound\audiocpp\qwen3-tts-12hz-1.7b-voicedesign-q8_0.gguf --family qwen3_tts --task 10 --backend 1 --threads 8 --text "%TEXT%" --out %OUT%\qwen3_voicedesign.capi.wav --options-file D:\audiocpp_test\instruct.json > %OUT%\qwen3_voicedesign.capi.log 2>&1
echo ALL_DONE
