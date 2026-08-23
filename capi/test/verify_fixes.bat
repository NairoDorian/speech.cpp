@echo off
REM Verify the CAPI fixes on 4080: session options (load_model_ex) + channels.
set EXE=D:\audiocpp_test\capi_tts_test_new.exe
set "TEXT=Hello, this is a short test of the audio text to speech system."
set OUT=D:\audiocpp_test\out
echo === 1. miotts with load_model_ex codec path (should work) ===
%EXE% --model D:\iStation\Models\sound\audiocpp\miotts-1.7b-q8_0.gguf --family miotts --task 5 --backend 1 --threads 8 --text "%TEXT%" --out %OUT%\miotts_verify.wav --options-file D:\audiocpp_test\voice_ref.json --load-options-file D:\audiocpp_test\load_codec.json > %OUT%\miotts_verify.log 2>&1
echo === 2. miotts with WRONG codec path (error must mention our path) ===
%EXE% --model D:\iStation\Models\sound\audiocpp\miotts-1.7b-q8_0.gguf --family miotts --task 5 --backend 1 --threads 8 --text "%TEXT%" --out %OUT%\miotts_verify2.wav --options-file D:\audiocpp_test\voice_ref.json --load-options-file D:\audiocpp_test\load_codec_bad.json > %OUT%\miotts_verify2.log 2>&1
echo === 3. malformed session options JSON (hard error) ===
%EXE% --model D:\iStation\Models\sound\audiocpp\pocket-tts-english-q8_0.gguf --family pocket_tts --task 5 --backend 1 --threads 8 --text "%TEXT%" --out %OUT%\pocket_verify.wav --load-options-file D:\audiocpp_test\load_bad.json > %OUT%\pocket_verify.log 2>&1
echo === 4. pocket_tts channels field (should be 1) ===
%EXE% --model D:\iStation\Models\sound\audiocpp\pocket-tts-english-q8_0.gguf --family pocket_tts --task 5 --backend 1 --threads 8 --text "%TEXT%" --out %OUT%\pocket_verify2.wav --options-file D:\audiocpp_test\voice_ref.json > %OUT%\pocket_verify2.log 2>&1
echo ALL_DONE
