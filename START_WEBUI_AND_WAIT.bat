@echo off
chcp 65001 > nul
echo ===============================================
echo        啟動 WebUI 並等待 API 準備就緒
echo ===============================================
echo.

echo 🔍 檢查是否有 WebUI 進程正在運行...
tasklist /fi "imagename eq python.exe" | find "python.exe" > nul
if %errorlevel% equ 0 (
    echo ⚠️ 發現 Python 進程，可能 WebUI 已在運行
    set /p kill="是否要終止現有進程？(y/n): "
    if /i "!kill!"=="y" (
        echo 🛑 正在終止現有進程...
        taskkill /f /im python.exe 2>nul
        timeout /t 3 /nobreak > nul
    )
)

echo.
echo 🚀 啟動 WebUI (帶 API 支持)...
echo 請等待看到 "Running on local URL: http://127.0.0.1:7860"
echo 不要關閉這個視窗！
echo.

REM 啟動 WebUI
start "Stable Diffusion WebUI" cmd /k "webui.bat"

echo ⏳ 等待 WebUI 啟動...
echo 正在檢查 API 可用性...

REM 等待並檢查 API
set /a counter=0
:check_loop
timeout /t 5 /nobreak > nul
set /a counter+=1

REM 檢查主頁是否可訪問
curl -s http://localhost:7860 > nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ WebUI 主頁已可訪問！
    goto :check_api
)

if %counter% geq 24 (
    echo ❌ WebUI 啟動超時（2分鐘）
    echo 請檢查：
    echo 1. 是否有錯誤訊息
    echo 2. GPU 記憶體是否足夠
    echo 3. 模型是否正確安裝
    goto :end
)

echo   等待中... (%counter%/24)
goto :check_loop

:check_api
echo.
echo 🔍 檢查 API 端點...

REM 檢查不同的 API 端點
curl -s http://localhost:7860/sdapi/v1/options > nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ 標準 SD API 可用 (/sdapi/v1/options)
    set API_TYPE=standard
    goto :success
)

curl -s http://localhost:7860/docs > nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Gradio API 文檔可用 (/docs)
    set API_TYPE=gradio
    goto :success
)

curl -s http://localhost:7860/info > nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Gradio 信息端點可用 (/info)
    set API_TYPE=gradio
    goto :success
)

echo ⚠️ 主頁可訪問但 API 端點不明確
set API_TYPE=unknown
goto :success

:success
echo.
echo 🎉 WebUI 已成功啟動！
echo    類型: %API_TYPE%
echo    URL: http://localhost:7860
echo.
echo 下一步：
if "%API_TYPE%"=="standard" (
    echo 1. 執行 python day2_enhanced_test.py
    echo 2. 或執行 python debug_clip_test.py
) else if "%API_TYPE%"=="gradio" (
    echo 1. 執行 python test_gradio_api.py
    echo 2. 然後根據結果修改測試腳本
) else (
    echo 1. 在瀏覽器中打開 http://localhost:7860
    echo 2. 檢查是否能手動生成圖像
    echo 3. 執行 python detailed_webui_status.py
)
echo.

:end
echo 按任意鍵結束...
pause > nul
