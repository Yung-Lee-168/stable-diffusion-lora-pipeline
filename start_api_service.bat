@echo off
echo ================================================================
echo           Stable Diffusion WebUI API 完整解決方案
echo ================================================================
echo.

:menu
echo 請選擇操作:
echo 1. 啟動 Stable Diffusion WebUI 服務器
echo 2. 測試 API 連接
echo 3. 執行範例程式 (文字轉圖片)
echo 4. 啟動 Web API 服務器 (HTTP 接口)
echo 5. 安裝必要套件
echo 6. 查看使用說明
echo 0. 退出
echo.

set /p choice="請選擇 (0-6): "

if "%choice%"=="1" goto start_webui
if "%choice%"=="2" goto test_api
if "%choice%"=="3" goto run_examples  
if "%choice%"=="4" goto start_web_api
if "%choice%"=="5" goto install_packages
if "%choice%"=="6" goto show_help
if "%choice%"=="0" goto exit
goto menu

:start_webui
echo.
echo 🚀 啟動 Stable Diffusion WebUI 服務器...
echo ================================================================
echo 注意: 此服務器需要保持運行狀態
echo 首次啟動可能需要下載模型，請耐心等待
echo 看到 "Running on local URL" 訊息表示啟動成功
echo ================================================================
echo.
pause
start cmd /k "webui-user.bat"
echo.
echo ✅ 已在新視窗中啟動 WebUI 服務器
echo 請等待啟動完成後再使用其他功能
echo.
pause
goto menu

:test_api
echo.
echo 🔍 測試 API 連接...
echo.
python -c "
from text_to_image_service import StableDiffusionAPI
api = StableDiffusionAPI()
if api.is_server_ready():
    print('✅ API 服務器連接正常')
    models = api.get_models()
    if models:
        print(f'📋 當前模型: {models[0].get(\"model_name\", \"Unknown\")}')
else:
    print('❌ API 服務器未就緒')
    print('請先啟動 Stable Diffusion WebUI')
"
echo.
pause
goto menu

:run_examples
echo.
echo 🎨 執行範例程式...
echo.
python api_usage_examples.py
echo.
pause
goto menu

:start_web_api
echo.
echo 🌐 啟動 Web API 服務器...
echo.
echo 檢查必要套件...
python -c "import flask" 2>nul || (
    echo ❌ Flask 未安裝，正在安裝...
    pip install flask
)
echo.
echo 🚀 啟動中...
echo 服務地址: http://localhost:8000
echo API 文檔: http://localhost:8000
echo.
python web_api_server.py
echo.
pause
goto menu

:install_packages
echo.
echo 📦 安裝必要的 Python 套件...
echo.
echo 正在安裝 requests...
pip install requests
echo.
echo 正在安裝 Pillow...
pip install Pillow
echo.
echo 正在安裝 Flask (用於 Web API)...
pip install flask
echo.
echo ✅ 套件安裝完成!
echo.
pause
goto menu

:show_help
echo.
echo ================================================================
echo                        使用說明
echo ================================================================
echo.
echo 🎯 核心功能: 輸入文字描述 → 生成對應圖片
echo.
echo 📋 使用流程:
echo    1. 先執行選項 1 啟動 Stable Diffusion WebUI 服務器
echo    2. 等待服務器完全啟動 (看到 Running on local URL 訊息)
echo    3. 執行選項 3 運行範例程式，測試文字轉圖片功能
echo.
echo 🔧 API 使用方式:
echo.
echo    方式 1: 直接呼叫函數
echo    from text_to_image_service import text_to_image_service
echo    result = text_to_image_service("a beautiful sunset")
echo.
echo    方式 2: 使用 Web API (HTTP)
echo    POST http://localhost:8000/generate
echo    Content-Type: application/json
echo    {"prompt": "a beautiful sunset"}
echo.
echo 📁 輸出位置:
echo    - 生成的圖片保存在 'generated_images' 資料夾
echo    - 每張圖片都有時間戳記檔名
echo.
echo 🎛️ 可調整參數:
echo    - prompt: 圖片描述文字 (必需)
echo    - negative_prompt: 不想要的元素描述
echo    - width/height: 圖片尺寸 (預設 512x512)
echo    - steps: 生成步數 (預設 20，越高品質越好但越慢)
echo    - cfg_scale: 提示詞遵循度 (預設 7)
echo.
echo 💡 提示詞建議:
echo    - 使用英文描述
echo    - 描述越詳細越好
echo    - 可以加入風格關鍵字如 "photorealistic", "anime style"
echo    - 使用負向提示排除不想要的元素
echo.
echo 範例提示詞:
echo    "a serene mountain landscape at sunset, highly detailed, 4k"
echo    "a cute robot cat, cyberpunk style, neon lights"
echo    "portrait of a wise wizard, fantasy art, detailed"
echo.
echo ================================================================
pause
goto menu

:exit
echo.
echo 👋 感謝使用 Stable Diffusion API 服務!
echo.
exit

:error
echo.
echo ❌ 發生錯誤，請檢查:
echo    1. Python 是否已安裝
echo    2. 必要套件是否已安裝 (選項 5)
echo    3. Stable Diffusion WebUI 是否已啟動 (選項 1)
echo.
pause
goto menu
