@echo off
REM install_requirements.bat - installs packages listed in requirements.txt
cd /d "%~dp0"
setlocal EnableDelayedExpansion

set "CMD="

python --version >NUL 2>&1
if !errorlevel! == 0 (
    set "CMD=python"
) else (
    py --version >NUL 2>&1
    if !errorlevel! == 0 set "CMD=py"
)

if not defined CMD (
    echo Neither python nor py was found in PATH.
    exit /b 1
)

for /F "usebackq tokens=* eol=#" %%P in ("requirements.txt") do (
    if not "%%P"=="" (
        echo Installing %%P ...
        !CMD! -m pip install %%P
    )
)

endlocal
pause
