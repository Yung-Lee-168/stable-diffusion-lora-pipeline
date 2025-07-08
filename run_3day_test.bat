@echo off
REM 3天 Stable Diffusion 時尚圖片生成可行性測試 - 自動化腳本
echo ============================================================
echo        3天 Stable Diffusion 時尚圖片生成可行性測試
echo ============================================================
echo.

REM 檢查 Python 是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 未安裝或不在 PATH 中
    echo 請安裝 Python 3.7+ 並添加到系統 PATH
    pause
    exit /b 1
)

echo ✅ Python 環境檢查通過

REM 安裝必要套件
echo.
echo 📦 安裝必要的 Python 套件...
python -m pip install --upgrade pip
python -m pip install requests pillow torch transformers matplotlib pandas numpy

if errorlevel 1 (
    echo ⚠️ 套件安裝可能有問題，但繼續執行測試...
) else (
    echo ✅ 套件安裝完成
)

echo.
echo 🔍 檢查 Stable Diffusion WebUI 是否運行...
timeout /t 2 >nul

REM 測試 API 連接
python -c "import requests; response = requests.get('http://localhost:7860/sdapi/v1/memory', timeout=5); print('✅ WebUI API 運行正常' if response.status_code == 200 else '❌ WebUI API 無法連接')" 2>nul
if errorlevel 1 (
    echo.
    echo ⚠️ 無法連接到 WebUI API
    echo 請確保:
    echo   1. 已啟動 webui-user.bat
    echo   2. WebUI 包含 --api --listen 參數
    echo   3. API 在 http://localhost:7860 可用
    echo.
    echo 是否要繼續測試? (Y/N)
    set /p continue="請輸入 Y 或 N: "
    if /i "%continue%" neq "Y" (
        echo 測試取消
        pause
        exit /b 1
    )
)

echo.
echo 🚀 開始3天可行性測試...
echo.

:menu
echo ============================================================
echo                     測試選項菜單
echo ============================================================
echo 1. 運行第1天測試 (基礎功能測試)
echo 2. 運行第2天測試 (進階功能測試)  
echo 3. 運行第3天測試 (結果評估)
echo 4. 運行完整3天測試流程
echo 5. 查看測試結果
echo 6. 清理測試結果
echo 7. 顯示幫助信息
echo 0. 退出
echo ============================================================

set /p choice="請選擇操作 (0-7): "

if "%choice%"=="1" goto day1
if "%choice%"=="2" goto day2
if "%choice%"=="3" goto day3
if "%choice%"=="4" goto full_test
if "%choice%"=="5" goto view_results
if "%choice%"=="6" goto cleanup
if "%choice%"=="7" goto help
if "%choice%"=="0" goto exit
echo 無效選擇，請重新輸入
goto menu

:day1
echo.
echo 📅 開始第1天測試：基礎功能測試
echo ============================================================
python day1_basic_test.py
if errorlevel 1 (
    echo ❌ 第1天測試失敗
) else (
    echo ✅ 第1天測試完成
)
echo.
pause
goto menu

:day2
echo.
echo 📅 開始第2天測試：進階功能測試
echo ============================================================
python day2_advanced_test.py
if errorlevel 1 (
    echo ❌ 第2天測試失敗
) else (
    echo ✅ 第2天測試完成
)
echo.
pause
goto menu

:day3
echo.
echo 📅 開始第3天測試：結果評估
echo ============================================================
python day3_evaluation.py
if errorlevel 1 (
    echo ❌ 第3天評估失敗
) else (
    echo ✅ 第3天評估完成
)
echo.
pause
goto menu

:full_test
echo.
echo 🚀 開始完整3天測試流程
echo ============================================================
echo.
echo 📅 第1天：基礎功能測試
python day1_basic_test.py
if errorlevel 1 (
    echo ❌ 第1天測試失敗，停止後續測試
    pause
    goto menu
)
echo ✅ 第1天測試完成
echo.

echo 📅 第2天：進階功能測試
python day2_advanced_test.py
if errorlevel 1 (
    echo ❌ 第2天測試失敗，跳過第3天測試
    pause
    goto menu
)
echo ✅ 第2天測試完成
echo.

echo 📅 第3天：結果評估
python day3_evaluation.py
if errorlevel 1 (
    echo ❌ 第3天評估失敗
) else (
    echo ✅ 完整3天測試流程完成
)
echo.
pause
goto menu

:view_results
echo.
echo 📊 測試結果查看
echo ============================================================
if exist "day1_results\day1_report.json" (
    echo ✅ 第1天測試結果: day1_results\day1_report.json
) else (
    echo ❌ 第1天測試結果不存在
)

if exist "day2_results\day2_report.json" (
    echo ✅ 第2天測試結果: day2_results\day2_report.json
) else (
    echo ❌ 第2天測試結果不存在
)

if exist "day3_evaluation\final_feasibility_report.json" (
    echo ✅ 最終評估報告: day3_evaluation\final_feasibility_report.json
    echo.
    echo 🔍 顯示評估摘要...
    python -c "import json; data=json.load(open('day3_evaluation/final_feasibility_report.json', encoding='utf-8')); print(f'整體成功率: {data[\"technical_analysis\"][\"overall_success_rate\"]*100:.1f}%'); print(f'可行性評估: {data[\"feasibility_assessment\"][\"feasibility_level\"]}'); print(f'建議: {data[\"feasibility_assessment\"][\"recommendation\"]}')"
) else (
    echo ❌ 最終評估報告不存在
)
echo.
pause
goto menu

:cleanup
echo.
echo 🧹 清理測試結果
echo ============================================================
echo ⚠️ 此操作將刪除所有測試結果，確定要繼續嗎? (Y/N)
set /p confirm="請輸入 Y 或 N: "
if /i "%confirm%"=="Y" (
    if exist "day1_results" rmdir /s /q "day1_results"
    if exist "day2_results" rmdir /s /q "day2_results"
    if exist "day3_evaluation" rmdir /s /q "day3_evaluation"
    echo ✅ 測試結果已清理
) else (
    echo 清理操作已取消
)
echo.
pause
goto menu

:help
echo.
echo 📖 幫助信息
echo ============================================================
echo 這是一個3天 Stable Diffusion 時尚圖片生成可行性測試工具
echo.
echo 測試流程:
echo   第1天: 測試基礎 API 功能和時尚提示詞效果
echo   第2天: 測試圖片分析和自動提示詞生成
echo   第3天: 評估測試結果並生成可行性報告
echo.
echo 前置要求:
echo   • Stable Diffusion WebUI 已安裝並運行
echo   • webui-user.bat 包含 --api --listen 參數
echo   • Python 3.7+ 和必要套件已安裝
echo.
echo 輸出文件:
echo   day1_results/     - 第1天測試結果和生成圖片
echo   day2_results/     - 第2天測試結果和分析圖片  
echo   day3_evaluation/  - 第3天評估報告和圖表
echo.
echo 詳細說明請參考: README_3DAY_TEST.md
echo.
pause
goto menu

:exit
echo.
echo 👋 感謝使用 Stable Diffusion 可行性測試工具
echo 如需查看詳細報告，請檢查生成的結果文件夾
pause
exit /b 0
