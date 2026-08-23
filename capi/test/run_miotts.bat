@echo off
set EXE=D:\audiocpp_test\audiocpp_cli.exe
set "TEXT=Hello, this is a short test of the audio text to speech system."
set REF=D:\audiocpp_test\ref_voice.wav
%EXE% --task tts --family miotts --model D:\iStation\Models\sound\audiocpp\miotts-1.7b-q8_0.gguf --backend cuda --threads 8 --text "%TEXT%" --out D:\audiocpp_test\out\miotts.official.wav --metrics --voice-ref %REF% > D:\audiocpp_test\out\miotts.official.log 2>&1
%EXE% --task tts --family pocket_tts --model D:\iStation\Models\sound\audiocpp\pocket-tts-english-q8_0.gguf --backend cuda --threads 8 --text "%TEXT%" --out D:\audiocpp_test\out\pocket_tts.official3.wav --metrics --voice-ref %REF% > D:\audiocpp_test\out\pocket_tts.official3.log 2>&1
echo ALL_DONE
