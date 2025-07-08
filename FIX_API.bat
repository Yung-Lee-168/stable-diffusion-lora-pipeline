@echo off
chcp 65001 > nul
echo ===============================================
echo        WebUI API 診斷與修復工具
echo ===============================================
echo.

echo 🔍 安裝診斷依賴...
pip install psutil requests > nul 2>&1

echo 🚀 開始診斷...
python diagnose_webui_api.py

echo.
echo 如果修復成功，請執行：
echo   DEBUG_CLIP.bat
echo 或
echo   python debug_clip_test.py
echo.
pause
