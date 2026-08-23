# run_tts_compare.ps1 — CAPI vs official CLI TTS comparison harness (runs on 4080)
#
# For each model: run the official CLI (--metrics) and the CAPI test CLI with
# identical conditions (backend cuda, 8 threads, same text), save logs, then
# print a one-line summary. Uses whichever binaries sit next to this script.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File run_tts_compare.ps1 [-Phase main|clon] [-Text <text>] [-OutDir <dir>]

param(
    [string]$Phase = "main",
    [string]$Text = "Hello, this is a short test of the audio text to speech system.",
    [string]$OutDir = "D:\audiocpp_test\out"
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OfficialCli = Join-Path $Root "audiocpp_cli.exe"
$CapiTest    = Join-Path $Root "capi_tts_test.exe"
$ModelBase   = "D:\iStation\Models\sound\audiocpp"

$models = @(
    @{ name = "pocket_tts";        path = "pocket-tts-english-q8_0.gguf";              family = "pocket_tts";    task = "tts"; num = 5 },
    @{ name = "miotts";            path = "miotts-1.7b-q8_0.gguf";                     family = "miotts";        task = "tts"; num = 5 },
    @{ name = "moss_tts_nano";     path = "moss-tts-nano-100m-q8_0.gguf";              family = "moss_tts_nano"; task = "tts"; num = 5 },
    @{ name = "moss_tts_local";    path = "moss-tts-local-v1.5-q8_0.gguf";             family = "moss_tts_local";task = "tts"; num = 5 },
    @{ name = "qwen3_tts_base";    path = "qwen3-tts-12hz-1.7b-base-q8_0.gguf";        family = "qwen3_tts";     task = "tts"; num = 5 },
    @{ name = "qwen3_tts_06b";     path = "Qwen3-TTS-12Hz-0.6B-Base";                  family = "qwen3_tts";     task = "tts"; num = 5 },
    @{ name = "glm_tts";           path = "glm-tts-q8_0.gguf";                         family = "glm_tts";       task = "tts"; num = 5 },
    @{ name = "index_tts2";        path = "index-tts2-q8_0.gguf";                      family = "index_tts2";    task = "tts"; num = 5 },
    @{ name = "dramabox";          path = "dramabox-q8_0.gguf";                        family = "dramabox";      task = "tts"; num = 5 },
    @{ name = "fish_audio";        path = "fish-audio-s2-pro-q8_0.gguf";               family = "fish_audio";    task = "tts"; num = 5 },
    @{ name = "higgs_audio_tts";   path = "higgs-audio-v3-tts-4b-q8_0.gguf";           family = "higgs_audio_tts";task = "tts"; num = 5 },
    @{ name = "irodori_v4_small";  path = "irodori-tts-v4-small-q8_0.gguf";            family = "irodori_tts";   task = "tts"; num = 5 },
    @{ name = "irodori_v3_600m";   path = "irodori-tts-600m-v3-voicedesign-q8_0.gguf"; family = "irodori_tts";   task = "tts"; num = 5 },
    @{ name = "irodori_500m_v3";   path = "Irodori-TTS-500M-v3";                       family = "irodori_tts";   task = "tts"; num = 5 },
    @{ name = "outetts";           path = "outetts-1.0-1b-q8_0.gguf";                  family = "outetts";       task = "tts"; num = 5 },
    @{ name = "vietneu_tts";       path = "vietneu-tts-v3-turbo-q8_0.gguf";            family = "vietneu_tts";   task = "tts"; num = 5 },
    @{ name = "vibevoice";         path = "vibevoice-1.5b-q8_0.gguf";                  family = "vibevoice";     task = "tts"; num = 5 },
    @{ name = "voxcpm2";           path = "voxcpm2-q8_0.gguf";                         family = "voxcpm2";       task = "tts"; num = 5 },
    @{ name = "vevo2";             path = "vevo2-q8_0.gguf";                           family = "vevo2";         task = "tts"; num = 5 },
    @{ name = "omnivoice";         path = "OmniVoice";                                 family = "omnivoice";     task = "tts"; num = 5 }
)

$clonModels = @(
    @{ name = "chatterbox";        path = "chatterbox-q8_0.gguf";                      family = "chatterbox";    task = "clon"; num = 8 },
    @{ name = "confucius4_tts";    path = "confucius4-tts-orig.gguf";                  family = "confucius4_tts";task = "clon"; num = 8 },
    @{ name = "qwen3_customvoice"; path = "qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf";  family = "qwen3_tts";     task = "clon"; num = 8 },
    @{ name = "qwen3_voicedesign"; path = "qwen3-tts-12hz-1.7b-voicedesign-q8_0.gguf";  family = "qwen3_tts";     task = "vdes"; num = 10 }
)

$selected = if ($Phase -eq "clon") { $clonModels } else { $models }
$RefWav = Join-Path $Root "ref_voice.wav"
$RefText = $Text

if (-not (Test-Path $OfficialCli)) { Write-Host "MISSING $OfficialCli"; exit 1 }
if (-not (Test-Path $CapiTest))    { Write-Host "MISSING $CapiTest"; exit 1 }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "=== phase=$Phase text='$Text' out=$OutDir ==="
$summary = @()

foreach ($m in $selected) {
    $modelPath = Join-Path $ModelBase $m.path
    $base = Join-Path $OutDir $m.name
    $officialLog = "$base.official.log"
    $capiLog     = "$base.capi.log"
    $officialWav = "$base.official.wav"
    $capiWav     = "$base.capi.wav"

    $common = @("--backend","cuda","--threads","8","--family",$m.family,"--model",$modelPath,"--text",$Text)
    $extra = @()
    if ($Phase -eq "clon") {
        $extra = @("--voice-ref",$RefWav)
        if ($m.family -eq "qwen3_tts") { $extra += @("--reference-text",$RefText) }
    }
    $oargs = @("--task",$m.task,"--out",$officialWav,"--metrics") + $common + $extra
    Write-Host "--- $($m.name) official ---"
    & $OfficialCli @oargs 2>&1 | Out-File -Encoding utf8 $officialLog
    $oExit = $LASTEXITCODE

    $cargs = @("--model",$modelPath,"--family",$m.family,"--task",$m.num,"--backend","1","--threads","8","--text",$Text,"--out",$capiWav)
    if ($Phase -eq "clon") {
        $cjson = '{"voice_ref":"' + $RefWav.Replace('\','/') + '"}'
        if ($m.family -eq "qwen3_tts") { $cjson = '{"voice_ref":"' + $RefWav.Replace('\','/') + '","reference_text":"' + $RefText + '"}' }
        $cargs += @("--options",$cjson)
    }
    Write-Host "--- $($m.name) capi ---"
    & $CapiTest @cargs 2>&1 | Out-File -Encoding utf8 $capiLog
    $cExit = $LASTEXITCODE

    # Parse key fields
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
        $m.name, $oExit, $oWall, $oRtf, $oSr, $oErr, $cExit, $cLoad, $cSyn, $cSr, $cNs, $cNan, $cClip, $cErr
    Write-Host $row
    $summary += $row
}

Write-Host "=== SUMMARY ==="
$summary | ForEach-Object { Write-Host $_ }
$summary | Out-File -Encoding utf8 (Join-Path $OutDir "summary_$Phase.txt")
