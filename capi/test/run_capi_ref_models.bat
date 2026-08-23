@echo off
REM CAPI voice-ref model retries with --options-file and QUOTED text
set EXE=D:\audiocpp_test\capi_tts_test.exe
set "TEXT=Hello, this is a short test of the audio text to speech system."
set OUT=D:\audiocpp_test\out

%EXE% --model D:\iStation\Models\sound\audiocpp\pocket-tts-english-q8_0.gguf --family pocket_tts --task 5 --backend 1 --threads 8 --text "%TEXT%" --out %OUT%\pocket_tts.capi.wav --options-file D:\audiocpp_test\voice_ref.json > %OUT%\pocket_tts.capi.log 2>&1
%EXE% --model D:\iStation\Models\sound\audiocpp\Qwen3-TTS-12Hz-0.6B-Base --family qwen3_tts --task 5 --backend 1 --threads 8 --text "%TEXT%" --out %OUT%\qwen3_06b.capi.wav --options-file D:\audiocpp_test\voice_ref_text.json > %OUT%\qwen3_06b.capi.log 2>&1
%EXE% --model D:\iStation\Models\sound\audiocpp\glm-tts-q8_0.gguf --family glm_tts --task 5 --backend 1 --threads 8 --text "%TEXT%" --out %OUT%\glm_tts.capi.wav --options-file D:\audiocpp_test\voice_ref_text.json > %OUT%\glm_tts.capi.log 2>&1
%EXE% --model D:\iStation\Models\sound\audiocpp\index-tts2-q8_0.gguf --family index_tts2 --task 5 --backend 1 --threads 8 --text "%TEXT%" --out %OUT%\index_tts2.capi.wav --options-file D:\audiocpp_test\voice_ref.json > %OUT%\index_tts2.capi.log 2>&1
%EXE% --model D:\iStation\Models\sound\audiocpp\vietneu-tts-v3-turbo-q8_0.gguf --family vietneu_tts --task 5 --backend 1 --threads 8 --text "%TEXT%" --out %OUT%\vietneu_tts.capi.wav --options-file D:\audiocpp_test\voice_ref_text.json > %OUT%\vietneu_tts.capi.log 2>&1
echo ALL_DONE
