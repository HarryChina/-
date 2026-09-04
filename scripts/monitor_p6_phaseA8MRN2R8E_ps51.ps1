param(
    [int]$ProcessId = 0,
    [string]$ExpectedCommandLine = '',
    [ValidateRange(2, 300)]
    [int]$RefreshSeconds = 10
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$resultDir = Join-Path $projectRoot 'results\airsas_bunny20k\p6_complex_psf_field_deconv'
$statusPath = Join-Path $resultDir 'p6_phaseA8MRN2R8E_status.json'
$stderrPath = Join-Path $resultDir 'p6_phaseA8MRN2R8E_stderr.log'

function Read-StatusSafely {
    for ($attempt = 0; $attempt -lt 5; $attempt++) {
        if (Test-Path -LiteralPath $statusPath -PathType Leaf) {
            try { return Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json }
            catch { }
        }
        Start-Sleep -Milliseconds 100
    }
    return $null
}

function Normalize-CommandLine {
    param([Parameter(Mandatory = $true)][string]$CommandLine)
    $tokens = [regex]::Matches($CommandLine, '"[^"]*"|\S+') | ForEach-Object {
        $_.Value.Trim().Trim('"').Replace('/', '\').ToLowerInvariant()
    }
    return (($tokens -join ' ').Replace('\\', '\'))
}

function Value-OrDash {
    param($Value)
    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) { return '-' }
    return [string]$Value
}

function Get-ValidatedWorker {
    param([int]$Id, [string]$Expected, $FrozenStatus)
    $process = Get-Process -Id $Id -ErrorAction SilentlyContinue
    if ($null -eq $process) { return $null }
    $mode = 'CIM_COMMAND_LINE'
    try {
        $record = Get-CimInstance Win32_Process -Filter ("ProcessId = {0}" -f $Id) -ErrorAction Stop
        if ($null -eq $record -or [string]::IsNullOrWhiteSpace($record.CommandLine)) {
            throw [Microsoft.Management.Infrastructure.CimException]::new('Command line unavailable')
        }
        if ((Normalize-CommandLine $record.CommandLine) -ne (Normalize-CommandLine $Expected)) {
            throw ("PID {0} command mismatch. Expected: {1} Actual: {2}" -f $Id, $Expected, $record.CommandLine)
        }
    }
    catch [Microsoft.Management.Infrastructure.CimException] {
        $mode = 'PID_PATH_AND_FROZEN_STATUS_FALLBACK'
        if ($null -eq $FrozenStatus) { throw 'Status unavailable during CIM fallback.' }
        if ([int]$FrozenStatus.PID -ne $Id) { throw 'Frozen status PID mismatch.' }
        if ((Normalize-CommandLine ([string]$FrozenStatus.CommandLine)) -ne (Normalize-CommandLine $Expected)) {
            throw 'Frozen status command line mismatch.'
        }
        $expectedExe = ([regex]::Matches($Expected, '"[^"]*"|\S+')[0].Value).Trim('"')
        if (-not [string]::IsNullOrWhiteSpace($process.Path)) {
            if ([IO.Path]::GetFullPath($process.Path) -ne [IO.Path]::GetFullPath($expectedExe)) {
                throw 'Executable path mismatch.'
            }
        }
    }
    return [pscustomobject]@{ Process = $process; VerificationMode = $mode }
}

$lastStatus = Read-StatusSafely
if ($ProcessId -le 0) {
    if ($null -eq $lastStatus) { throw 'Status JSON unavailable; provide ProcessId and ExpectedCommandLine.' }
    $ProcessId = [int]$lastStatus.PID
}
if ([string]::IsNullOrWhiteSpace($ExpectedCommandLine)) {
    if ($null -eq $lastStatus) { throw 'Status JSON unavailable; provide ExpectedCommandLine.' }
    $ExpectedCommandLine = [string]$lastStatus.CommandLine
}

Write-Host ("Monitoring frozen A8MRN2R8E H4 worker PID {0}" -f $ProcessId)
Write-Host ("Expected command: {0}" -f (Normalize-CommandLine $ExpectedCommandLine))
Write-Host 'Ctrl+C stops only this monitor; it does not stop the worker.'

while ($true) {
    $freshStatus = Read-StatusSafely
    if ($null -ne $freshStatus) { $lastStatus = $freshStatus }
    $status = $lastStatus
    $worker = Get-ValidatedWorker -Id $ProcessId -Expected $ExpectedCommandLine -FrozenStatus $status
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    if ($null -eq $worker) {
        if ($null -ne $status) {
            Write-Host ("[{0}] Worker exited. status={1}; phase={2}; case={3}; classification={4}" -f `
                $stamp, $status.status, $status.phase, (Value-OrDash $status.outcome_case), (Value-OrDash $status.classification))
        }
        else { Write-Host ("[{0}] Worker exited; status JSON unavailable." -f $stamp) }
        break
    }

    $ramGiB = [math]::Round($worker.Process.WorkingSet64 / 1GB, 2)
    Write-Host ("[{0}] verified ({1}); status={2}; phase={3}; update={4}/{5}; RAM={6} GiB" -f `
        $stamp, $worker.VerificationMode, $status.status, $status.phase, `
        (Value-OrDash $status.current_update), (Value-OrDash $status.total_updates), $ramGiB)
    Write-Host ("             H4={0}; parent={1}; params/input={2}/{3}; lambda={4}; field/diag pairs={5}/{6}" -f `
        (Value-OrDash $status.candidate), (Value-OrDash $status.parent), `
        (Value-OrDash $status.params), (Value-OrDash $status.input), `
        (Value-OrDash $status.lambda_diag), (Value-OrDash $status.field_pairs_per_update), `
        (Value-OrDash $status.diag_pairs_per_update))
    Write-Host ("             loss field/diag/weighted/total={0}/{1}/{2}/{3}" -f `
        (Value-OrDash $status.latest_field_loss), (Value-OrDash $status.latest_diag_loss), `
        (Value-OrDash $status.latest_weighted_diag_loss), (Value-OrDash $status.latest_total_loss))
    Write-Host ("             VAL latest pooled/max={0}/{1}; best update={2}, pooled/max={3}/{4}" -f `
        (Value-OrDash $status.latest_VAL_pooled), (Value-OrDash $status.latest_VAL_max), `
        (Value-OrDash $status.best_update), (Value-OrDash $status.best_VAL_pooled), `
        (Value-OrDash $status.best_VAL_max))
    Write-Host ("             streams field={0}; diag={1}; blacklist hits={2}/{3}" -f `
        (Value-OrDash $status.field_stream_status), (Value-OrDash $status.diag_stream_status), `
        (Value-OrDash $status.field_blacklist_hits), (Value-OrDash $status.diag_blacklist_hits))
    Write-Host ("             TEST={0} {1}/{2}; CANCEL4={3}; burned={4} {5}/{6}; READY4={7}; case={8}" -f `
        (Value-OrDash $status.INTERNAL_TEST_opened), (Value-OrDash $status.INTERNAL_TEST_complete), `
        (Value-OrDash $status.INTERNAL_TEST_total), (Value-OrDash $status.CANCEL4), `
        (Value-OrDash $status.burned_opened), (Value-OrDash $status.burned_complete), `
        (Value-OrDash $status.burned_total), (Value-OrDash $status.READY4), `
        (Value-OrDash $status.outcome_case))
    Write-Host ("             fresh30={0}/{1}; operator={2}; stderr={3} bytes" -f `
        (Value-OrDash $status.fresh30_selected), (Value-OrDash $status.fresh30_accessed), `
        (Value-OrDash $status.operator_cases), (Value-OrDash $status.stderr_bytes))

    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -ne $nvidiaSmi) {
        try {
            $gpuProcess = & $nvidiaSmi.Source --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>$null |
                Where-Object { $_ -match ('^\s*{0}\s*,' -f $ProcessId) }
            $gpuDevice = & $nvidiaSmi.Source --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
            if ($gpuProcess) { Write-Host ("             worker VRAM: {0} MiB" -f (($gpuProcess -split ',')[1].Trim())) }
            else { Write-Host '             worker VRAM: not listed yet' }
            if ($gpuDevice) {
                $fields = $gpuDevice -split ',' | ForEach-Object { $_.Trim() }
                Write-Host ("             GPU {0}: utilization={1}%; VRAM={2}/{3} MiB" -f $fields[0], $fields[1], $fields[2], $fields[3])
            }
        }
        catch { Write-Host ("             GPU telemetry unavailable: {0}" -f $_.Exception.Message) }
    }
    if ((Test-Path -LiteralPath $stderrPath) -and (Get-Item -LiteralPath $stderrPath).Length -gt 0) {
        Write-Host '             stderr tail:'
        Get-Content -LiteralPath $stderrPath -Tail 3 -Encoding UTF8 | ForEach-Object { Write-Host ("               {0}" -f $_) }
    }
    Start-Sleep -Seconds $RefreshSeconds
}
