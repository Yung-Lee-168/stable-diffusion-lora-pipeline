@echo off
chcp 65001 > nul
echo ===============================================
echo        確保 WebUI 正確啟動
echo ===============================================
echo.

echo 🔍 檢查當前 WebUI 狀態...
python detailed_webui_status.py

echo.
echo 🔄 如果上述檢查失敗，將重新啟動 WebUI...
echo.

set /p restart="是否要重新啟動 WebUI？(y/n): "
if /i "%restart%"=="y" (
    echo 🛑 正在終止現有的 WebUI 進程...
    taskkill /f /im python.exe 2>nul
    taskkill /f /im cmd.exe /fi "WINDOWTITLE eq *webui*" 2>nul
    timeout /t 3 /nobreak > nul
    
    echo 🚀 啟動 WebUI（帶 API 支持）...
    echo 請等待看到 "Running on local URL: http://127.0.0.1:7860"
    echo.
    start "Stable Diffusion WebUI" webui.bat
    
    echo ⏳ 等待 WebUI 啟動（最多 2 分鐘）...
    timeout /t 10 /nobreak > nul
    
    echo 🔍 檢查啟動狀態...
    python detailed_webui_status.py
)

echo.
echo 如果 WebUI 正常運行，接下來執行：
echo   python debug_clip_test.py
echo   python day2_enhanced_test.py
echo.
pause
