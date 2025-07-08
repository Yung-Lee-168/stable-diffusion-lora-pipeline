@echo off
chcp 65001 >nul
title Quick WebUI Check
echo ============================================================
echo               Quick WebUI Status Check
echo ============================================================
echo.

cd /d "e:\Yung_Folder\Project\stable-diffusion-webui"

echo Performing quick check...
echo.

python wait_for_webui.py

pause
