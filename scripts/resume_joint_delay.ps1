param([Parameter(Mandatory=$true)][string]$SourceLaunch, [string]$Checkpoint = 'candidate.pt')

$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$pythonExecutable = Join-Path $projectDirectory '.venv\Scripts\python.exe'
$hiddenPython = Join-Path $projectDirectory '.venv\Scripts\pythonw.exe'
$sourceManifestPath = (Resolve-Path -LiteralPath $SourceLaunch).Path
$sourceManifest = Get-Content -Raw -LiteralPath $sourceManifestPath | ConvertFrom-Json
$checkpointPath = (Resolve-Path -LiteralPath (Join-Path $sourceManifest.runs.joint_delay_acac.output $Checkpoint)).Path
$identifier = Get-Date -Format 'yyyyMMddTHHmmss_fffffff'
$launchDirectory = Join-Path $projectDirectory "results\joint_delay_comparison\resume_$identifier"
New-Item -ItemType Directory -Path $launchDirectory -Force | Out-Null
$manifestPath = Join-Path $launchDirectory 'launch.json'
$outputDirectory = Join-Path $launchDirectory 'joint_delay_acac'
$stdout = Join-Path $launchDirectory 'joint_delay_acac.stdout.log'
$stderr = Join-Path $launchDirectory 'joint_delay_acac.stderr.log'
$taskName = "MRDTS_JointDelayResume_$identifier"
$monitorArguments = '-B -m scripts.monitor_joint_delay "{0}"' -f $manifestPath
$action = New-ScheduledTaskAction -Execute $hiddenPython -Argument $monitorArguments -WorkingDirectory $projectDirectory
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(15) -RepetitionInterval (New-TimeSpan -Minutes 15)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 3)
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null
$process = $null
try {
    $trainingArguments = '-B -u -m methods.joint_delay_acac.run --resume "{0}" --output-dir "{1}"' -f $checkpointPath, $outputDirectory
    $process = Start-Process -FilePath $pythonExecutable -ArgumentList $trainingArguments -WorkingDirectory $projectDirectory -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $runs = [ordered]@{
        joint_delay_ppo=$sourceManifest.runs.joint_delay_ppo
        joint_delay_acac=[ordered]@{pid=$process.Id; output=$outputDirectory; stdout=$stdout; stderr=$stderr}
    }
    $manifest = [ordered]@{created_at=(Get-Date).ToString('o'); task_name=$taskName; monitoring_interval_minutes=15; source_launch=$sourceManifestPath; resumed_from=$checkpointPath; runs=$runs}
    $manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding utf8
    $manifest | ConvertTo-Json -Depth 6
} catch {
    if ($null -ne $process -and -not $process.HasExited) { Stop-Process -Id $process.Id }
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    throw
}
