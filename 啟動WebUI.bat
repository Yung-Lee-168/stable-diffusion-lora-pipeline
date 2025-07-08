@echo off
chcp 65001 >nul
title Start Stable Diffusion WebUI
echo ============================================================
echo            Start Stable Diffusion WebUI
echo ============================================================
echo.
echo Starting Stable Diffusion WebUI...
echo Please be patient, first startup may take 5-10 minutes
echo.
echo Please watch for these important messages:
echo   - "Running on local URL: http://127.0.0.1:7860"
echo   - Any error messages
echo.

REM Switch to correct directory
cd /d "e:\Yung_Folder\Project\stable-diffusion-webui"

REM Check configuration file
echo Current configuration:
type webui-user.bat
echo.
echo ============================================================
echo Starting WebUI...
echo ============================================================

REM Start WebUI
call webui.bat

echo.
echo ============================================================
echo WebUI has stopped running
echo ============================================================
pause
