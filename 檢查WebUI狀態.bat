@echo off
chcp 65001 >nul
title Stable Diffusion WebUI Status Check
echo ============================================================
echo           Stable Diffusion WebUI Status Checker
echo ============================================================
echo.
echo This tool will automatically check if WebUI is running
echo If WebUI hasn't started yet, please double-click webui-user.bat first
echo.
echo Checking WebUI status...
echo.

REM Switch to correct directory
cd /d "e:\Yung_Folder\Project\stable-diffusion-webui"

REM Run status check
python wait_for_webui.py

echo.
echo Check completed.
pause
