@echo off
chcp 65001 > nul
echo ================================================================
echo                 Stable Diffusion API 測試自動化腳本
echo ================================================================
echo.

echo 📋 開始自動化測試流程...
echo.

echo 🔍 第1步：檢查 WebUI 是否已在運行...
python quick_webui_status.py
if %errorlevel% equ 0 (
    echo ✅ WebUI 已在運行，API 可用
    goto run_test
) else (
    echo ⚠️ WebUI 未運行或 API 不可用
)

echo.
echo 🚀 第2步：啟動 WebUI...
echo 正在啟動 WebUI，請稍候...
echo 這可能需要1-3分鐘，請耐心等待
echo.

start "WebUI" cmd /c "webui.bat --api"

echo ⏳ 等待 WebUI 載入...
timeout /t 30 /nobreak > nul

echo.
echo 🔍 第3步：檢查 WebUI 載入狀態...
:check_loop
python quick_webui_status.py
if %errorlevel% equ 0 (
    echo ✅ WebUI 載入完成，API 可用
    goto run_test
) else (
    echo ⏳ WebUI 仍在載入中，再等待10秒...
    timeout /t 10 /nobreak > nul
    goto check_loop
)

:run_test
echo.
echo 🧪 第4步：開始執行增強版第2天測試...
echo ================================================================
echo                    開始 CLIP vs FashionCLIP 比較測試
echo ================================================================
echo.

python day2_enhanced_test.py

echo.
echo ================================================================
echo                         測試完成
echo ================================================================
echo.
echo 📊 測試結果已保存在 day2_enhanced_results 資料夾
echo 📄 詳細報告：day2_enhanced_results\day2_enhanced_report.json
echo.
echo 按任意鍵退出...
pause > nul
