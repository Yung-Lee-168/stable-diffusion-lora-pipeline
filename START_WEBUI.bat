@echo off
chcp 65001 >nul
title Start Stable Diffusion WebUI
echo ============================================================
echo              Start Stable Diffusion WebUI
echo ============================================================
echo.
echo Step 1: Starting WebUI (This may take 5-10 minutes)
echo ============================================================
echo.
echo IMPORTANT: Watch for this message:
echo   "Running on local URL: http://127.0.0.1:7860"
echo.
echo When you see that message:
echo   1. WebUI is ready!
echo   2. Keep this window open
echo   3. Open a new file explorer window
echo   4. Double-click CHECK_STATUS.bat to verify
echo.
echo Starting WebUI now...
echo.

cd /d "e:\Yung_Folder\Project\stable-diffusion-webui"

call webui.bat

echo.
echo ============================================================
echo WebUI has stopped running
echo ============================================================
echo If this happened unexpectedly, check for error messages above.
pause
