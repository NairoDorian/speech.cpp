# run_tts_retry.ps1 — phase-2 special-case TTS comparisons (runs on 4080)
# Models that need voice refs / extra options, run with identical conditions
# through both binaries.
param(
    [string]$Text = "Hello, this is a short test of the audio text to speech system.",
    [string]$OutDir = "D:\audiocpp_test\out"
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OfficialCli = Join-Path $Root "audiocpp_cli.exe"
$CapiTest    = Join-Path $Root "capi_tts_test.exe"
$ModelBase   = "D:\iStation\Models\sound\audiocpp"
$RefWav      = Join-Path $Root "ref_voice.wav"

# name, path, family, task, tasknum, extra official args, extra capi options json
$cases = @(
    @{ name = "pocket_tts";        path = "pocket-tts-english-q8_0.gguf";             family = "pocket_tts";     task = "tts"; num = 5;
       oextra = @("--voice-ref", $RefWav); cextra = '{"voice_ref":"' + $RefWav.Replace('\','/') + '"}' },
    @{ name = "qwen3_06b";         path = "Qwen3-TTS-12Hz-0.6B-Base";                 family = "qwen3_tts";      task = "tts"; num = 5;
       oextra = @("--voice-ref", $RefWav, "--reference-text", $Text);
       cextra = '{"voice_ref":"' + $RefWav.Replace('\','/') + '","reference_text":"' + $Text + '"}' },
    @{ name = "glm_tts";           path = "glm-tts-q8_0.gguf";                        family = "glm_tts";        task = "tts"; num = 5;
       oextra = @("--voice-ref", $RefWav, "--reference-text", $Text);
       cextra = '{"voice_ref":"' + $RefWav.Replace('\','/') + '","reference_text":"' + $Text + '"}' },
    @{ name = "index_tts2";        path = "index-tts2-q8_0.gguf";                     family = "index_tts2";     task = "tts"; num = 5;
       oextra = @("--voice-ref", $RefWav); cextra = '{"voice_ref":"' + $RefWav.Replace('\','/') + '"}' },
    @{ name = "vietneu_tts";       path = "vietneu-tts-v3-turbo-q8_0.gguf";           family = "vietneu_tts";    task = "tts"; num = 5;
       oextra = @("--voice-ref", $RefWav, "--reference-text", $Text);
       cextra = '{"voice_ref":"' + $RefWav.Replace('\','/') + '","reference_text":"' + $Text + '"}' },
    @{ name = "vevo2";             path = "vevo2-q8_0.gguf";                          family = "vevo2";          task = "tts"; num = 5;
       oextra = @("--target-voice", $RefWav); cextra = '{"target_voice":"' + $RefWav.Replace('\','/') + '"}' },
    @{ name = "miotts";            path = "miotts-1.7b-q8_0.gguf";                    family = "miotts";         task = "tts"; num = 5;
       oextra = @("--request-option", "miotts.codec_model_path=$ModelBase\miocodec-25hz-44khz-v2-q8_0.gguf");
       cextra = '{"miotts.codec_model_path":"' + ($ModelBase + "\miocodec-25hz-44khz-v2-q8_0.gguf").Replace('\','/') + '"}' },
    @{ name = "vibevoice";         path = "vibevoice-1.5b-q8_0.gguf";                 family = "vibevoice";      task = "tts"; num = 5;
       oextra = @(); cextra = "{}"; textOverride = "Speaker 0: Hello, this is a short test of the audio text to speech system." }
)

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$summary = @()
foreach ($c in $cases) {
    $modelPath = Join-Path $ModelBase $c.path
    $base = Join-Path $OutDir $c.name
    $officialLog = "$base.retry.official.log"
    $capiLog     = "$base.retry.capi.log"
    $officialWav = "$base.retry.official.wav"
    $capiWav     = "$base.retry.capi.wav"
    $text = if ($c.textOverride) { $c.textOverride } else { $Text }

    $oargs = @("--task",$c.task,"--family",$c.family,"--model",$modelPath,"--backend","cuda",
               "--threads","8","--text",$text,"--out",$officialWav,"--metrics") + $c.oextra
    Write-Host "--- $($c.name) official (retry) ---"
    & $OfficialCli @oargs 2>&1 | Out-File -Encoding utf8 $officialLog
    $oExit = $LASTEXITCODE

    $cargs = @("--model",$modelPath,"--family",$c.family,"--task",$c.num,"--backend","1","--threads","8",
               "--text",$text,"--out",$capiWav,"--options",$c.cextra)
    Write-Host "--- $($c.name) capi (retry) ---"
    & $CapiTest @cargs 2>&1 | Out-File -Encoding utf8 $capiLog
    $cExit = $LASTEXITCODE

    $oWall = (Select-String -Path $officialLog -Pattern '^metrics.wall_ms=([\d.]+)' | ForEach-Object { $_.Matches[0].Groups[1].Value })
    $oRtf  = (Select-String -Path $officialLog -Pattern '^metrics.rtf=([\d.]+)' | ForEach-Object { $_.Matches[0].Groups[1].Value })
    $oSr   = (Select-String -Path $officialLog -Pattern '^metrics.sample_rate=(\d+)' | ForEach-Object { $_.Matches[0].Groups[1].Value })
    $oErr  = (Select-String -Path $officialLog -Pattern '^audiocpp_cli failed: (.*)' | ForEach-Object { $_.Matches[0].Groups[1].Value })
    $cLoad = (Select-String -Path $capiLog -Pattern '^load_ms=([\d.]+)' | ForEach-Object { $_.Matches[0].Groups[1].Value })
    $cSyn  = (Select-String -Path $capiLog -Pattern '^synth_ms=([\d.]+)' | ForEach-Object { $_.Matches[0].Groups[1].Value })
    $cSr   = (Select-String -Path $capiLog -Pattern '^sample_rate=(\d+)' | ForEach-Object { $_.Matches[0].Groups[1].Value })
    $cNs   = (Select-String -Path $capiLog -Pattern '^n_samples=(\d+)' | ForEach-Object { $_.Matches[0].Groups[1].Value })
    $cErr  = (Select-String -Path $capiLog -Pattern '^error_message=(.*)' | ForEach-Object { $_.Matches[0].Groups[1].Value })
    $cNan  = (Select-String -Path $capiLog -Pattern '^pcm_nan=(\d+)' | ForEach-Object { $_.Matches[0].Groups[1].Value })
    $cClip = (Select-String -Path $capiLog -Pattern '^pcm_clip=(\d+)' | ForEach-Object { $_.Matches[0].Groups[1].Value })

    $row = "{0}|exit_o={1}|o_wall_ms={2}|o_rtf={3}|o_sr={4}|o_err={5}|exit_c={6}|c_load_ms={7}|c_synth_ms={8}|c_sr={9}|c_n={10}|c_nan={11}|c_clip={12}|c_err={13}" -f `
        $c.name, $oExit, $oWall, $oRtf, $oSr, $oErr, $cExit, $cLoad, $cSyn, $cSr, $cNs, $cNan, $cClip, $cErr
    Write-Host $row
    $summary += $row
}
Write-Host "=== SUMMARY ==="
$summary | ForEach-Object { Write-Host $_ }
$summary | Out-File -Encoding utf8 (Join-Path $OutDir "summary_retry.txt")
