# tuningfork inside WSL, from a checkout on Windows, from PowerShell:
#
#   .\scripts\tuningfork <command> [args]         any tuningfork command, as on Linux
#   .\scripts\tuningfork --run <program> [args]   anything else, in tuningfork's venv
#
# The checkout stays where Windows editors see it; tuningfork's venv and its recordings stay
# in the distro's Linux file system, where SQLite's locking works. scripts/tuningfork is its
# twin for Git Bash: change both together.
#
# It runs nothing until TUNINGFORK_WSL_DISTRO names the distro: with more than one installed,
# the default one may not be the one meant. Each argument reaches tuningfork as it was typed.

$ErrorActionPreference = 'Stop'
$env:WSL_UTF8 = '1'  # wsl.exe's own messages in UTF-8, not UTF-16

$distro = $env:TUNINGFORK_WSL_DISTRO
if (-not $distro) {
    Write-Host 'Name the WSL distro tuningfork runs in first, for example:  $env:TUNINGFORK_WSL_DISTRO = "Ubuntu"'
    Write-Host 'The distros installed here:'
    & wsl.exe --list --verbose
    exit 2
}

# Run in the distro: one fixed script, the arguments after it, never read back as shell.
$script = 'export UV_PROJECT_ENVIRONMENT="$HOME/.local/share/tuningfork/venv"; ' +
    'command -v uv >/dev/null || { echo "uv is not installed in the Linux tuningfork runs in: install it there with  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 127; }; ' +
    'if [ "${1:-}" = --run ]; then shift; exec uv run "$@"; fi; exec uv run tuningfork "$@"'
$passed = @($script, 'tuningfork') + @($args | ForEach-Object { "$_" })
if ($PSVersionTable.PSVersion -lt [version]'7.3') {
    # Windows PowerShell hands a program the double quotes inside an argument unescaped.
    $passed = @($passed | ForEach-Object { $_ -replace '"', '\"' })
}
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
& wsl.exe --distribution $distro --cd $repo --exec bash -lc @passed
exit $LASTEXITCODE
