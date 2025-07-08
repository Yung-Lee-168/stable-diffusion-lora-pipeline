@echo off
chcp 65001 > nul
echo ===============================================
echo          快速 WebUI 狀態檢查
echo ===============================================
echo.

echo 🔍 檢查 WebUI 狀態...
python quick_webui_check.py

echo.
echo 如果 WebUI 正常，接下來執行：
echo   python debug_clip_test.py
echo.
pause
