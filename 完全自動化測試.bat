@echo off
chcp 65001 > nul
cls
echo ================================================================
echo           完全自動化 WebUI 測試腳本 - 解決終端機衝突
echo ================================================================
echo.

echo 🚀 第1步：在新視窗啟動 WebUI
echo.
echo 正在新視窗中啟動 WebUI...
start "Stable Diffusion WebUI" cmd /c "webui.bat --api & pause"

echo ✅ WebUI 已在新視窗中啟動
echo 📋 請注意：
echo    - WebUI 在獨立視窗中運行
echo    - 瀏覽器會自動打開 http://localhost:7860
echo    - 不要關閉 WebUI 視窗
echo.

echo ⏳ 等待 WebUI 載入 (30秒)...
timeout /t 30 /nobreak > nul

echo.
echo 🔍 第2步：循環檢查 WebUI 狀態
echo.

:check_loop
echo 正在檢查 WebUI 狀態...
python 檢查狀態.py > temp_check.txt 2>&1

findstr /C:"可以開始測試了" temp_check.txt > nul
if %errorlevel% equ 0 (
    echo ✅ WebUI 完全就緒！
    type temp_check.txt
    del temp_check.txt
    goto run_test
)

findstr /C:"WebUI 未啟動" temp_check.txt > nul
if %errorlevel% equ 0 (
    echo ⏳ WebUI 仍在啟動中，繼續等待 15 秒...
    del temp_check.txt
    timeout /t 15 /nobreak > nul
    goto check_loop
)

findstr /C:"API 不可用" temp_check.txt > nul
if %errorlevel% equ 0 (
    echo ⚠️ WebUI 已啟動但 API 未開啟
    echo 這可能需要更多時間，繼續等待...
    del temp_check.txt
    timeout /t 20 /nobreak > nul
    goto check_loop
)

echo 🔄 繼續檢查...
del temp_check.txt
timeout /t 10 /nobreak > nul
goto check_loop

:run_test
echo.
echo ================================================================
echo 🧪 第3步：執行 CLIP vs FashionCLIP 比較測試
echo ================================================================
echo.

python day2_enhanced_test.py

echo.
echo ================================================================
echo ✅ 測試完成！
echo ================================================================
echo.
echo 📊 結果保存在：day2_enhanced_results 資料夾
echo 📄 詳細報告：day2_enhanced_results\day2_enhanced_report.json
echo.
echo 💡 WebUI 仍在另一個視窗中運行，可以繼續使用
echo.
pause
