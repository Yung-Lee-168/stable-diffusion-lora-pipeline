@echo off
echo ========================================
echo   Stable Diffusion WebUI API 使用指南
echo ========================================
echo.

echo 第一步: 啟動 Stable Diffusion WebUI 服務器
echo ----------------------------------------
echo 請在另一個命令列視窗中執行: webui-user.bat
echo 等待看到 "Running on local URL: http://127.0.0.1:7860" 訊息
echo.

echo 第二步: 確認服務器狀態
echo ----------------------
echo 在瀏覽器中開啟: http://localhost:7860/docs
echo 如果可以看到 API 文檔頁面，表示服務器正常運行
echo.

echo 第三步: 選擇使用方式
echo --------------------
echo 1. 簡單圖像生成器 (推薦新手)
echo 2. 完整 API 客戶端 (進階使用)
echo 3. 自定義 Python 腳本
echo.

:menu
set /p choice="請選擇 (1/2/3) 或按 Enter 查看說明: "

if "%choice%"=="1" (
    echo.
    echo 啟動簡單圖像生成器...
    python simple_generator.py
    goto end
)

if "%choice%"=="2" (
    echo.
    echo 啟動完整 API 客戶端...
    python api_client.py
    goto end
)

if "%choice%"=="3" (
    echo.
    echo 自定義腳本範例:
    echo.
    echo import requests
    echo import base64
    echo.
    echo # 發送請求
    echo response = requests.post("http://localhost:7860/sdapi/v1/txt2img", json={
    echo     "prompt": "您的圖像描述",
    echo     "width": 512,
    echo     "height": 512,
    echo     "steps": 20
    echo })
    echo.
    echo # 保存圖像
    echo if response.status_code == 200:
    echo     result = response.json()
    echo     with open("output.png", "wb") as f:
    echo         f.write(base64.b64decode(result['images'][0]))
    echo.
    goto end
)

echo.
echo 使用說明:
echo =========
echo.
echo 方式 1: 簡單圖像生成器
echo - 互動式介面，適合初學者
echo - 輸入文字描述即可生成圖像
echo - 自動保存到 outputs 資料夾
echo.
echo 方式 2: 完整 API 客戶端  
echo - 展示完整 API 功能
echo - 包含模型和採樣器列表
echo - 多個範例圖像生成
echo.
echo 方式 3: 自定義腳本
echo - 適合程式開發者
echo - 可整合到其他應用程式
echo - 完全自定義參數
echo.
goto menu

:end
echo.
pause
