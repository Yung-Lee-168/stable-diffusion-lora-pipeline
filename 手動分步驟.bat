@echo off
chcp 65001 > nul
cls
echo ================================================================
echo                    手動分步驟操作指南
echo ================================================================
echo.
echo 這個腳本會分步驟引導您完成所有操作
echo.

echo 🚀 第1步：啟動 WebUI
echo.
echo 選擇啟動方式：
echo [1] 自動在新視窗啟動 (推薦)
echo [2] 手動啟動說明
echo.
set /p choice="請選擇 (1 或 2): "

if "%choice%"=="1" (
    echo.
    echo 正在新視窗中啟動 WebUI...
    start "Stable Diffusion WebUI" cmd /c "webui.bat --api & echo WebUI 已啟動，請保持此視窗開啟 & pause"
    echo ✅ WebUI 已在新視窗中啟動
) else (
    echo.
    echo 請手動執行以下步驟：
    echo 1. 開啟新的命令提示字元視窗
    echo 2. 切換到目錄：cd /d "%~dp0"
    echo 3. 執行：webui.bat --api
    echo 4. 等待瀏覽器自動打開
)

echo.
echo 📋 重要提醒：
echo    - 必須等待瀏覽器自動打開 http://localhost:7860
echo    - 確認可以看到 WebUI 界面
echo    - 不要關閉 WebUI 視窗
echo.
echo 完成後按任意鍵繼續...
pause > nul

cls
echo.
echo ================================================================
echo 🔍 第2步：檢查 WebUI 狀態
echo ================================================================
echo.

python 檢查狀態.py

echo.
echo 如果上面顯示 "可以開始測試了"，請按 Y 繼續
echo 如果顯示錯誤，請按 N 重新啟動 WebUI
echo.
set /p test_choice="繼續測試? (Y/N): "

if /i "%test_choice%"=="Y" goto run_test
if /i "%test_choice%"=="y" goto run_test

echo.
echo 請重新啟動 WebUI 後再運行此腳本
pause
exit

:run_test
cls
echo.
echo ================================================================
echo 🧪 第3步：執行測試
echo ================================================================
echo.

python day2_enhanced_test.py

echo.
echo ================================================================
echo ✅ 完成！
echo ================================================================
pause
