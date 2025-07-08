@echo off
echo 🚀 啟動 Stable Diffusion WebUI (API 模式)
echo ============================================================
echo 為 day2_enhanced_test.py 提供圖片生成服務
echo ============================================================
echo.

cd /d "e:\Yung_Folder\Project\stable-diffusion-webui"

echo 📥 檢查 WebUI 環境...
if not exist "webui.py" (
    echo ❌ 找不到 webui.py，請確認目錄正確
    pause
    exit /b 1
)

echo ✅ WebUI 環境檢查通過
echo.

echo 🎮 啟動 WebUI (API 模式)...
echo 💡 請等待看到 "Running on local URL: http://127.0.0.1:7860"
echo 💡 然後在另一個終端執行 day2_enhanced_test.py
echo.

REM 啟動 WebUI 並開啟 API
python webui.py --api --listen --enable-insecure-extension-access

pause
