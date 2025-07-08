@echo off
chcp 65001 > nul
echo ===============================================
echo        Stable Diffusion WebUI 完整修復
echo ===============================================
echo.

echo 第 1 步：安裝 OpenAI CLIP 套件
echo ----------------------------------------
pip install git+https://github.com/openai/CLIP.git
if %errorlevel% neq 0 (
    echo ✗ CLIP 安裝失敗，請檢查網路連線
    pause
    exit /b 1
)
echo ✓ CLIP 安裝成功
echo.

echo 第 2 步：檢查安裝狀態
echo ----------------------------------------
python check_clip_status.py
echo.

echo 第 3 步：啟動 WebUI（背景執行）
echo ----------------------------------------
echo 正在啟動 WebUI，請稍候...
start "WebUI" cmd /c "webui.bat"

echo 等待 WebUI 啟動...
timeout /t 30 /nobreak > nul

echo 第 4 步：測試 API 功能
echo ----------------------------------------
python check_webui_for_clip.py
echo.

echo 第 5 步：執行完整測試（可選）
echo ----------------------------------------
set /p choice="是否執行完整的 CLIP vs FashionCLIP 測試？(y/n): "
if /i "%choice%"=="y" (
    python day2_enhanced_test.py
)

echo.
echo ===============================================
echo              修復完成！
echo ===============================================
echo.
echo 你現在可以：
echo 1. 使用 WebUI 網頁界面生成圖像
echo 2. 執行 API 測試腳本
echo 3. 運行完整的 3 天測試計劃
echo.
pause
