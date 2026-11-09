param([Parameter(Mandatory=$true)][string]$OutputDirectory, [Parameter(Mandatory=$true)][string]$TaskName)
$ErrorActionPreference = 'Stop'
$metadataPath = Join-Path $OutputDirectory 'metadata.json'
if (-not (Test-Path -LiteralPath $metadataPath)) {
    $launch = Get-Content -Raw -LiteralPath (Join-Path $OutputDirectory 'launch.json') | ConvertFrom-Json
    $process = Get-Process -Id $launch.pid -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        [ordered]@{checked_at=(Get-Date).ToString('o'); status='failed'; attention='Evaluation exited before initializing metadata; inspect stderr.log.'} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDirectory 'monitor.json') -Encoding utf8
        Disable-ScheduledTask -TaskName $TaskName | Out-Null
    }
    exit 0
}
$metadata = Get-Content -Raw -LiteralPath $metadataPath | ConvertFrom-Json
$process = Get-Process -Id $metadata.pid -ErrorAction SilentlyContinue
$terminal = $metadata.status -in @('completed','failed')
$snapshot = [ordered]@{checked_at=(Get-Date).ToString('o'); status=$metadata.status; process_alive=($null -ne $process); slots_completed=$metadata.slots_completed; slots_planned=$metadata.slots_planned; elapsed_s=$metadata.elapsed_s; estimated_seconds_remaining=$metadata.estimated_seconds_remaining}
if (-not $terminal -and $null -eq $process) {
    $snapshot.status = 'failed'
    $snapshot.attention = 'Evaluation process exited; inspect stderr.log.'
    $terminal = $true
}
$snapshot | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDirectory 'monitor.json') -Encoding utf8
if ($terminal) { Disable-ScheduledTask -TaskName $TaskName | Out-Null }
