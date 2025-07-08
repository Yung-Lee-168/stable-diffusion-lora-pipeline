@echo off
chcp 65001 > nul
echo ===============================================
echo        CLIP 測試調試工具
echo ===============================================
echo.

echo 🔍 檢查 WebUI 是否正在運行...
netstat -an | find ":7860" > nul
if %errorlevel% equ 0 (
    echo ✅ WebUI 正在運行 ^(端口 7860^)
) else (
    echo ❌ WebUI 未運行，正在啟動...
    start "WebUI" cmd /c "webui.bat"
    echo ⏳ 等待 WebUI 啟動 ^(30秒^)...
    timeout /t 30 /nobreak > nul
)

echo.
echo 🚀 開始調試測試...
python debug_clip_test.py

echo.
echo 如果調試測試成功，你可以執行：
echo   python day2_enhanced_test.py
echo.
pause
