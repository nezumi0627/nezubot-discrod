param(
    [string]$Distro = 'Ubuntu',
    [ValidatePattern('^[a-z][a-z0-9_-]*$')][string]$BotUser = 'nezubot',
    [switch]$DockerEngine
)
$ErrorActionPreference = 'Stop'
if ($DockerEngine) {
    & wsl.exe -d $Distro -u root -- systemctl start docker
    if ($LASTEXITCODE) { throw 'Could not start the WSL Docker engine.' }
}
& wsl.exe -d $Distro -u $BotUser -- bash "/home/$BotUser/nezubot-discrod/scripts/windows/run-wsl.sh"
if ($LASTEXITCODE) { throw "WSL launcher exited: $LASTEXITCODE" }
