@echo off
chcp 65001 > nul
cls
echo ================================================================
echo               STEP BY STEP 操作指南 - 絕對清楚版本
echo ================================================================
echo.
echo 請按照以下步驟依序執行，每個步驟完成後再進行下一步：
echo.
echo ================================================================
echo 第 1 步：自動啟動 WebUI
echo ================================================================
echo.
echo 正在新視窗中啟動 WebUI...
echo.

start "Stable Diffusion WebUI" cmd /c "webui.bat --api"

echo ✅ WebUI 已在新視窗中啟動
echo.
echo 重要提醒：
echo - 請等待瀏覽器自動打開 http://localhost:7860
echo - 看到 WebUI 界面後才算啟動完成
echo - 不要關閉 WebUI 視窗
echo.
echo 等待 WebUI 載入中...
timeout /t 30 /nobreak > nul
echo.
echo ================================================================
echo 第 2 步：驗證 WebUI 是否正確啟動
echo ================================================================
echo.
echo 現在檢查 WebUI 狀態...
echo.

python 快速檢查.py

echo.
echo ================================================================
echo 第 3 步：根據檢查結果決定下一步
echo ================================================================
echo.
echo 如果上面顯示 "一切就緒"，請按 Y 繼續測試
echo 如果顯示其他錯誤，請按 N 退出並重新啟動 WebUI
echo.
set /p choice="請輸入 Y 繼續或 N 退出: "
if /i "%choice%"=="Y" goto run_test
if /i "%choice%"=="y" goto run_test
echo.
echo 請重新啟動 WebUI 後再運行此腳本
pause
exit

:run_test
cls
echo.
echo ================================================================
echo 第 4 步：開始執行測試
echo ================================================================
echo.
echo 正在啟動 CLIP vs FashionCLIP 比較測試...
echo 這個過程可能需要 10-15 分鐘
echo.

python day2_enhanced_test.py

echo.
echo ================================================================
echo 測試完成！
echo ================================================================
echo.
echo 結果已保存在 day2_enhanced_results 資料夾
echo.
pause
