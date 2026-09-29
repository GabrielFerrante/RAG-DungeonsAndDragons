<#
Pipeline completo de fine-tuning do retriever (ColQwen3 + LoRA), retomavel em cada etapa:

  1) Ingestao            PDF -> PNG + texto por pagina                     ~5 min, CPU
  2) Perguntas sinteticas  Qwen3-VL-4B local gera perguntas pt-BR por pagina  horas, GPU (retoma de onde parou)
  3) Treino LoRA         retriever adaptado aos livros                     horas, GPU (retoma do ultimo checkpoint)

Rode na raiz do repositorio, em um PowerShell:

  .\scripts\run_finetune.ps1                    # pipeline completo
  .\scripts\run_finetune.ps1 -Smoke             # teste rapido ponta a ponta (dados em data_smoke\, saidas em outputs\retriever-smoke)
  .\scripts\run_finetune.ps1 -SkipSynthetic     # so o treino (data\synth\queries.jsonl ja existe)
  .\scripts\run_finetune.ps1 -Epochs 2 -MaxHours 10

Se interromper (Ctrl+C, queda de energia), rode o mesmo comando de novo: cada etapa retoma.
Acompanhar:  Get-Content outputs\retriever-lora\log.jsonl -Wait -Tail 5
Resultado:   outputs\retriever-lora\summary.json  (baseline x melhor x final) e o adapter em best\
#>
param(
    [int]$QuestionsPerPage = 5,
    [int]$Epochs = 1,
    [double]$MaxHours = 0,
    [switch]$SkipIngest,
    [switch]$SkipSynthetic,
    [switch]$Smoke
)

Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"

$logDir = "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir ("finetune-{0:yyyyMMdd-HHmmss}.log" -f (Get-Date))

# Impede o Windows de suspender o PC enquanto o pipeline roda (todas as etapas, nao so o treino)
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"

function Invoke-Step([string]$Name, [string]$Module, [string]$StepArgs) {
    Write-Host "`n=== $Name ===  (log: $log)" -ForegroundColor Cyan
    # cmd /c evita que o PowerShell 5.1 trate as linhas de stderr do Python como erros
    cmd /c "conda run -n ragded --no-capture-output python -u -m $Module $StepArgs 2>&1" | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Falha em '$Name' (codigo $LASTEXITCODE). Veja $log"
    }
}

if ($Smoke) {
    $data, $out = "data_smoke", "outputs\retriever-smoke"
    $ingestArgs = "--out $data --max-pages 40"
    $synthArgs = "--data $data --questions-per-page 5 --limit 30"
    $trainArgs = "--data $data --out $out --max-steps 2 --grad-accum 4 --eval-every 1 --save-every 1 " +
                 "--val-fraction 0.25 --val-block 5 --val-queries 6 --distractors 6 --no-resume"
} else {
    $data, $out = "data", "outputs\retriever-lora"
    $ingestArgs = ""
    $synthArgs = "--questions-per-page $QuestionsPerPage"
    $trainArgs = "--epochs $Epochs --out $out"
    if ($MaxHours -gt 0) { $trainArgs += " --max-hours $MaxHours" }
}

[void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
try {
    if (-not $SkipIngest)    { Invoke-Step "1/3 Ingestao (PDF -> paginas)" "src.ingestion.loader" $ingestArgs }
    if (-not $SkipSynthetic) { Invoke-Step "2/3 Perguntas sinteticas (Qwen3-VL local)" "src.train.make_synthetic" $synthArgs }
    Invoke-Step "3/3 Treino LoRA do retriever" "src.train.train_retriever" $trainArgs
    Write-Host "`nConcluido. Resumo em $out\summary.json" -ForegroundColor Green
}
finally {
    [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)  # devolve o controle de energia ao Windows
}
