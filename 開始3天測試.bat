@echo off
title 開始3天可行性測試
echo ============================================================
echo              3天 Stable Diffusion 可行性測試
echo ============================================================
echo.
echo 這個工具會自動運行3天的測試流程
echo.
echo 前置要求：
echo ✅ Stable Diffusion WebUI 必須正在運行
echo ✅ API 端點 http://localhost:7860 可以訪問
echo.

REM 切換到正確的目錄  
cd /d "e:\Yung_Folder\Project\stable-diffusion-webui"

echo 🔍 首先檢查 WebUI 是否運行...
python -c "import requests; r=requests.get('http://localhost:7860/sdapi/v1/memory',timeout=5); print('✅ WebUI API 正常')" 2>nul
if errorlevel 1 (
    echo ❌ WebUI API 無法連接
    echo.
    echo 請先：
    echo   1. 雙擊執行 "啟動WebUI.bat"
    echo   2. 等待 WebUI 完全啟動
    echo   3. 再次執行此測試
    echo.
    pause
    exit /b 1
)

echo ✅ WebUI API 連接正常
echo.

:menu
echo ============================================================
echo                     測試選項
echo ============================================================
echo 1. 第1天測試 - 基礎功能測試
echo 2. 第2天測試 - 進階功能測試  
echo 3. 第3天測試 - 結果評估
echo 4. 完整3天測試（自動執行全部）
echo 5. 查看測試結果
echo 0. 退出
echo ============================================================

set /p choice="請選擇 (0-5): "

if "%choice%"=="1" goto day1
if "%choice%"=="2" goto day2  
if "%choice%"=="3" goto day3
if "%choice%"=="4" goto full_test
if "%choice%"=="5" goto view_results
if "%choice%"=="0" goto exit

echo 無效選擇，請重新輸入
goto menu

:day1
echo.
echo 📅 執行第1天測試：基礎功能測試
echo ============================================================
python day1_basic_test.py
echo.
pause
goto menu

:day2
echo.
echo 📅 執行第2天測試：進階功能測試
echo ============================================================
python day2_advanced_test.py
echo.
pause
goto menu

:day3
echo.
echo 📅 執行第3天測試：結果評估
echo ============================================================
python day3_evaluation.py
echo.
pause
goto menu

:full_test
echo.
echo 🚀 執行完整3天測試流程
echo ============================================================
echo.
echo 📅 第1天：基礎功能測試
python day1_basic_test.py
if errorlevel 1 (
    echo ❌ 第1天測試失敗
    pause
    goto menu
)
echo.

echo 📅 第2天：進階功能測試
python day2_advanced_test.py
if errorlevel 1 (
    echo ❌ 第2天測試失敗
    pause
    goto menu
)
echo.

echo 📅 第3天：結果評估
python day3_evaluation.py
echo.
echo 🎉 完整3天測試完成！
pause
goto menu

:view_results
echo.
echo 📊 測試結果概覽
echo ============================================================
if exist "day1_results\day1_report.json" (
    echo ✅ 第1天測試結果存在
) else (
    echo ❌ 第1天測試結果不存在
)

if exist "day2_results\day2_report.json" (
    echo ✅ 第2天測試結果存在
) else (
    echo ❌ 第2天測試結果不存在
)

if exist "day3_evaluation\final_feasibility_report.json" (
    echo ✅ 最終評估報告存在
    echo.
    echo 🔍 顯示評估摘要...
    python -c "import json; data=json.load(open('day3_evaluation/final_feasibility_report.json', encoding='utf-8')); print(f'可行性評估: {data[\"feasibility_assessment\"][\"feasibility_level\"]}'); print(f'整體成功率: {data[\"technical_analysis\"][\"overall_success_rate\"]*100:.1f}%')" 2>nul
) else (
    echo ❌ 最終評估報告不存在
)
echo.
pause
goto menu

:exit
echo.
echo 👋 謝謝使用！
echo 如需查看詳細結果，請檢查：
echo   📁 day1_results/
echo   📁 day2_results/  
echo   📁 day3_evaluation/
pause
exit /b 0
