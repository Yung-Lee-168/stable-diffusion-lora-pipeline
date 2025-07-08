@echo off
chcp 65001 >nul
echo ============================================================
echo        Stable Diffusion WebUI Quick Start Guide
echo ============================================================
echo.

echo Step 1: Check Configuration
echo --------------
type webui-user.bat
echo.

echo Step 2: Start WebUI (This will take a few minutes)
echo --------------
echo Starting Stable Diffusion WebUI...
echo Please be patient, first startup may require downloading models...
echo.

echo Please watch for these important messages:
echo   - "Running on local URL: http://127.0.0.1:7860"
echo   - "Running on public URL: ..." (if available)
echo   - Any error messages
echo.

echo Step 3: Verify After Startup
echo --------------
echo After WebUI starts successfully, you will see:
echo   - Web interface accessible in browser
echo   - API endpoints working normally
echo.

echo Common Issues:
echo   - If you see CUDA errors, may be GPU memory insufficient
echo   - If startup is slow, this is normal (especially first time)
echo   - If port is occupied, you can modify port settings
echo.

echo Ready? Press any key to start WebUI...
pause >nul

REM Start WebUI
echo.
echo Starting Stable Diffusion WebUI...
echo ============================================================
call webui.bat

echo.
echo ============================================================
echo WebUI has stopped or been interrupted
echo ============================================================
echo.
echo If startup was successful, please:
echo   1. Keep this window open
echo   2. Open new command prompt
echo   3. Run: python webui_diagnostic.py
echo   4. After API confirmed working, run: python day1_basic_test.py
echo.
pause
