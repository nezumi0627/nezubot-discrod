param(
    [string]$Distro = 'Ubuntu',
    [ValidatePattern('^[a-z][a-z0-9_-]*$')][string]$BotUser = 'nezubot',
    [switch]$AtStartup,
    [switch]$DockerEngine,
    [string]$TaskName = 'DiscordSandboxAI'
)
$ErrorActionPreference = 'Stop'
if ($AtStartup -and -not $DockerEngine) { throw 'Boot mode requires the WSL Docker engine, not Docker Desktop.' }
$account = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$extra = if ($DockerEngine) { ' -DockerEngine' } else { '' }
$arguments = '-NoProfile -File "{0}\run.ps1" -Distro "{1}" -BotUser "{2}"{3}' -f $PSScriptRoot, $Distro, $BotUser, $extra
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument $arguments
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -StartWhenAvailable -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
if ($AtStartup) {
    $credential = Get-Credential -UserName $account -Message 'Password for the Windows account owning this WSL distro (stored by Task Scheduler)'
    if ($credential.UserName -ne $account) { throw 'Use the current Windows account that owns the WSL distro.' }
    $trigger = New-ScheduledTaskTrigger -AtStartup
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -User $account -Password $credential.GetNetworkCredential().Password -Force
} else {
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $account
    $principal = New-ScheduledTaskPrincipal -UserId $account -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force
}
Write-Host 'Registered but NOT started. Stop the old bot before Start-ScheduledTask.'
