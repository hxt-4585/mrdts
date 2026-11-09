param(
    [string]$Config = ""
)

$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if ($Config -eq '') {
    $Config = Join-Path $workspacePath 'config\methods\centralized_ppo.toml'
}
$pythonPath = Join-Path $workspacePath '.venv\Scripts\python.exe'
$pythonWindowlessPath = Join-Path $workspacePath '.venv\Scripts\pythonw.exe'
$stamp = Get-Date -Format 'yyyyMMddTHHmmss_fffffff'
$launchRoot = Join-Path $workspacePath 'results\centralized_ppo\launches'
New-Item -ItemType Directory -Path $launchRoot -Force | Out-Null
$runPath = Join-Path $launchRoot "$stamp\run"
$launchDirectory = Split-Path $runPath -Parent
New-Item -ItemType Directory -Path $launchDirectory | Out-Null
$taskName = "MRDTS_CentralizedPPO_$stamp"
$monitorArguments = '-B -m methods.centralized_ppo.monitor "{0}" --task-name "{1}"' -f $runPath, $taskName
$action = New-ScheduledTaskAction -Execute $pythonWindowlessPath -Argument $monitorArguments -WorkingDirectory $workspacePath
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(30) -RepetitionInterval (New-TimeSpan -Minutes 30)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null
$stdoutPath = Join-Path $launchDirectory 'stdout.log'
$stderrPath = Join-Path $launchDirectory 'stderr.log'
$trainingArguments = '-B -u -m methods.centralized_ppo.run --config "{0}" --output-dir "{1}"' -f $Config, $runPath
try {
    $process = Start-Process -FilePath $pythonPath -ArgumentList $trainingArguments -WorkingDirectory $workspacePath -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
} catch {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    throw
}
$manifest = [ordered]@{
    run = $runPath
    task_name = $taskName
    launcher_pid = $process.Id
    config = $Config
    stdout = $stdoutPath
    stderr = $stderrPath
    created_at = (Get-Date).ToString('o')
    monitoring_interval_minutes = 30
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $launchDirectory 'launch.json') -Encoding utf8
$manifest | ConvertTo-Json
