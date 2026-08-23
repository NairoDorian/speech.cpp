# measure_cli.ps1 — full-process wall time of the official CLI for one TTS run
param(
    [string]$Family,
    [string]$Model,
    [string]$Text = "Hello, this is a short test of the audio text to speech system.",
    [string]$Out = "D:\audiocpp_test\measure.wav",
    [string]$ExtraArgs = ""
)
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$args = @("--task","tts","--family",$Family,"--model",$Model,"--backend","cuda","--threads","8",
          "--text",$Text,"--out",$Out,"--metrics") + ($ExtraArgs -split " ")
& "D:\audiocpp_test\audiocpp_cli.exe" @args > "$Out.process.log" 2>&1
$sw.Stop()
Write-Output ("process_ms=" + [int]$sw.Elapsed.TotalMilliseconds)
Write-Output ("exit=" + $LASTEXITCODE)
