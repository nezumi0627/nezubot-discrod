param(
    [string]$Distro = 'Ubuntu',
    [ValidatePattern('^[a-z][a-z0-9_-]*$')][string]$BotUser = 'nezubot',
    [switch]$DockerEngine,
    [string]$Repository = 'https://github.com/your-account/nezubot-discrod.git',
    [string]$Branch = 'main'
)
$ErrorActionPreference = 'Stop'
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw 'Install WSL2 first (elevated PowerShell: wsl --install -d Ubuntu), reboot, then rerun.'
}
$distros = (& wsl.exe --list --quiet) -replace "`0", ''
if ($distros.Trim() -notcontains $Distro) {
    throw "Install the distro first: wsl --install -d $Distro ; reboot if requested."
}
$kernel = & wsl.exe -d $Distro -u root -- uname -r
if ($LASTEXITCODE) { throw 'Start the Ubuntu distro and finish its first-run setup, then rerun.' }
if ($kernel -notmatch 'WSL2') {
    & wsl.exe --set-version $Distro 2
    if ($LASTEXITCODE) { throw 'WSL2 is required. Enable virtualization/Virtual Machine Platform and reboot.' }
}
$mode = if ($DockerEngine) { 'engine' } else { 'desktop' }
$script = (Get-Content -Raw "$PSScriptRoot\install-wsl.sh") -replace "`r", ''
$oldWslEnv = $env:WSLENV
try {
    if ($env:GITHUB_TOKEN) { $env:WSLENV = "${oldWslEnv}:GITHUB_TOKEN".Trim(':') }
    $script | & wsl.exe -d $Distro -u root -- bash -s -- $BotUser $mode $Repository $Branch
} finally {
    $env:WSLENV = $oldWslEnv
}
if ($LASTEXITCODE) { throw 'WSL installation failed; see output above. It is safe to rerun.' }
Write-Host "Installed. Configure /home/$BotUser/nezubot-discrod/.env in WSL; do not start a second live bot."
