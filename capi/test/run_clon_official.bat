@echo off
REM Voice-clone models via OFFICIAL CLI (voice-ref / instruct)
set EXE=D:\audiocpp_test\audiocpp_cli.exe
set "TEXT=Hello, this is a short test of the audio text to speech system."
set REF=D:\audiocpp_test\ref_voice.wav
set OUT=D:\audiocpp_test\out

%EXE% --task clon --family chatterbox --model D:\iStation\Models\sound\audiocpp\chatterbox-q8_0.gguf --backend cuda --threads 8 --text "%TEXT%" --out %OUT%\chatterbox.official.wav --metrics --voice-ref %REF% > %OUT%\chatterbox.official.log 2>&1
%EXE% --task clon --family confucius4_tts --model D:\iStation\Models\sound\audiocpp\confucius4-tts-orig.gguf --backend cuda --threads 8 --text "%TEXT%" --out %OUT%\confucius4_tts.official.wav --metrics --voice-ref %REF% > %OUT%\confucius4_tts.official.log 2>&1
%EXE% --task tts --family qwen3_tts --model D:\iStation\Models\sound\audiocpp\qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf --backend cuda --threads 8 --text "%TEXT%" --out %OUT%\qwen3_customvoice.official.wav --metrics --voice-ref %REF% --reference-text "%TEXT%" > %OUT%\qwen3_customvoice.official.log 2>&1
%EXE% --task vdes --family qwen3_tts --model D:\iStation\Models\sound\audiocpp\qwen3-tts-12hz-1.7b-voicedesign-q8_0.gguf --backend cuda --threads 8 --text "%TEXT%" --out %OUT%\qwen3_voicedesign.official.wav --metrics --instruct "Speak in a calm, clear, and natural tone." > %OUT%\qwen3_voicedesign.official.log 2>&1
echo ALL_DONE
