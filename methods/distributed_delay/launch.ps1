param([string]$Config = '')

$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if ($Config -eq '') {
    $Config = Join-Path $projectDirectory 'config\methods\distributed_delay.toml'
}
$pythonExecutable = Join-Path $projectDirectory '.venv\Scripts\python.exe'
$hiddenPython = Join-Path $projectDirectory '.venv\Scripts\pythonw.exe'
$identifier = Get-Date -Format 'yyyyMMddTHHmmss_fffffff'
$launchDirectory = Join-Path $projectDirectory "results\distributed_delay\launches\$identifier"
New-Item -ItemType Directory -Path $launchDirectory -Force | Out-Null
$runDirectory = Join-Path $launchDirectory 'run'
$taskName = "MRDTS_DistributedDelay_$identifier"
$monitorCommand = '-B -m methods.distributed_delay.monitor "{0}" --task-name "{1}"' -f $runDirectory, $taskName
$taskAction = New-ScheduledTaskAction -Execute $hiddenPython -Argument $monitorCommand -WorkingDirectory $projectDirectory
$taskTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(15) -RepetitionInterval (New-TimeSpan -Minutes 15)
$taskSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 3)
$taskPrincipal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $taskAction -Trigger $taskTrigger -Settings $taskSettings -Principal $taskPrincipal | Out-Null
$stdout = Join-Path $launchDirectory 'stdout.log'
$stderr = Join-Path $launchDirectory 'stderr.log'
$trainingCommand = '-B -u -m methods.distributed_delay.run --config "{0}" --output-dir "{1}"' -f $Config, $runDirectory
try {
    $process = Start-Process -FilePath $pythonExecutable -ArgumentList $trainingCommand -WorkingDirectory $projectDirectory -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
} catch {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    throw
}
$manifest = [ordered]@{run=$runDirectory; task_name=$taskName; pid=$process.Id; config=$Config; stdout=$stdout; stderr=$stderr; monitoring_interval_minutes=15; created_at=(Get-Date).ToString('o')}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $launchDirectory 'launch.json') -Encoding utf8
$manifest | ConvertTo-Json
