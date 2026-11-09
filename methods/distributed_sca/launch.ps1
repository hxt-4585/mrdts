param(
    [string]$Config = "config/methods/distributed_sca.toml",
    [int]$MonitorMinutes = 5
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pythonWindowless = Join-Path $projectRoot ".venv\Scripts\pythonw.exe"
$stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmss_fffffffZ")
$output = Join-Path $projectRoot "results\distributed_sca\$stamp"
$taskName = "MRDTS_DistributedSCA_$stamp"
$configPath = (Resolve-Path (Join-Path $projectRoot $Config)).Path
New-Item -ItemType Directory -Path $output | Out-Null

$action = New-ScheduledTaskAction -Execute $pythonWindowless `
    -Argument "-B -m methods.distributed_sca.monitor --output `"$output`"" `
    -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes($MonitorMinutes) `
    -RepetitionInterval (New-TimeSpan -Minutes $MonitorMinutes) -RepetitionDuration (New-TimeSpan -Days 1)
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 2) `
    -StartWhenAvailable -Hidden
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal `
    -Settings $settings | Out-Null

$arguments = @("-B", "-u", "-m", "methods.distributed_sca.run", "--config", "`"$configPath`"", "--output", "`"$output`"")
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $projectRoot `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $output "stdout.log") `
    -RedirectStandardError (Join-Path $output "stderr.log") -PassThru
$launch = [ordered]@{
    pid = $process.Id
    task_name = $taskName
    output = $output
    monitor_minutes = $MonitorMinutes
    launched_utc = [DateTime]::UtcNow.ToString("o")
    expected_minutes = "10-15 based on the 100-slot paired pilot"
}
$launch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output "launch.json") -Encoding utf8
$launch | ConvertTo-Json
