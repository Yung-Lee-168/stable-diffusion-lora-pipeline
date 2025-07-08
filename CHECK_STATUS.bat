@echo off
chcp 65001 >nul
title Detailed WebUI Status Check
echo ============================================================
echo           Detailed WebUI Status Checker
echo ============================================================
echo.
echo This tool will perform 8 comprehensive checks:
echo   1. Basic Network Connection
echo   2. API Endpoints Availability  
echo   3. Available Models
echo   4. Memory Status
echo   5. Image Generation Test
echo   6. Available Samplers
echo   7. Progress API
echo   8. Configuration Info
echo.
echo Starting detailed check...
echo.

cd /d "e:\Yung_Folder\Project\stable-diffusion-webui"

python detailed_webui_checker.py

pause
