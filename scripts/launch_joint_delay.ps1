param([int]$Episodes = 125, [int]$Workers = 2, [string]$SourceLaunch = '')

$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$pythonExecutable = Join-Path $projectDirectory '.venv\Scripts\python.exe'
$hiddenPython = Join-Path $projectDirectory '.venv\Scripts\pythonw.exe'
$sourceManifest = $null
if ($SourceLaunch -ne '') {
    $SourceLaunch = (Resolve-Path -LiteralPath $SourceLaunch).Path
    $sourceManifest = Get-Content -Raw -LiteralPath $SourceLaunch | ConvertFrom-Json
}
$identifier = Get-Date -Format 'yyyyMMddTHHmmss_fffffff'
$launchDirectory = Join-Path $projectDirectory "results\joint_delay_comparison\$identifier"
New-Item -ItemType Directory -Path $launchDirectory -Force | Out-Null
$manifestPath = Join-Path $launchDirectory 'launch.json'
$taskName = "MRDTS_JointDelay_$identifier"
$monitorArguments = '-B -m scripts.monitor_joint_delay "{0}"' -f $manifestPath
$action = New-ScheduledTaskAction -Execute $hiddenPython -Argument $monitorArguments -WorkingDirectory $projectDirectory
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(15) -RepetitionInterval (New-TimeSpan -Minutes 15)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 3)
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null
$runs = [ordered]@{}
$processes = @()
try {
    foreach ($method in @('joint_delay_ppo', 'joint_delay_acac')) {
        $outputDirectory = Join-Path $launchDirectory $method
        $stdout = Join-Path $launchDirectory "$method.stdout.log"
        $stderr = Join-Path $launchDirectory "$method.stderr.log"
        $trainingArguments = '-B -u -m methods.{0}.run --output-dir "{1}" --episodes {2} --workers {3}' -f $method, $outputDirectory, $Episodes, $Workers
        $checkpointPath = $null
        if ($null -ne $sourceManifest) {
            $checkpointPath = (Resolve-Path -LiteralPath (Join-Path $sourceManifest.runs.$method.output 'latest.pt')).Path
            $trainingArguments += ' --resume "{0}"' -f $checkpointPath
        }
        $process = Start-Process -FilePath $pythonExecutable -ArgumentList $trainingArguments -WorkingDirectory $projectDirectory -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        $processes += $process
        $runs[$method] = [ordered]@{pid=$process.Id; output=$outputDirectory; stdout=$stdout; stderr=$stderr; resumed_from=$checkpointPath}
    }
    $manifest = [ordered]@{created_at=(Get-Date).ToString('o'); task_name=$taskName; monitoring_interval_minutes=15; episodes=$Episodes; workers_per_baseline=$Workers; source_launch=$SourceLaunch; runs=$runs}
    $manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding utf8
    $manifest | ConvertTo-Json -Depth 6
} catch {
    foreach ($process in $processes) {
        if (-not $process.HasExited) { Stop-Process -Id $process.Id }
    }
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    throw
}
