@echo off
chcp 65001 >nul
echo 🔧 安裝 Stable Diffusion WebUI 依賴
echo ============================================================
echo 解決 pytorch_lightning 缺失問題
echo ============================================================
echo.

cd /d "e:\Yung_Folder\Project\stable-diffusion-webui"

echo 📦 檢查 Python 環境...
python --version
if errorlevel 1 (
    echo ❌ Python 未找到，請確認 Python 已安裝
    pause
    exit /b 1
)

echo.
echo 📦 安裝缺失的套件...
echo 正在安裝 pytorch_lightning...
pip install pytorch_lightning

echo.
echo 📦 安裝其他可能缺失的依賴...
pip install gradio==3.41.2
pip install fastapi>=0.90.1
pip install transformers==4.30.2
pip install accelerate
pip install safetensors

echo.
echo 📦 嘗試安裝完整依賴列表...
pip install -r requirements.txt

echo.
echo ✅ 依賴安裝完成！
echo.
echo 🚀 現在嘗試啟動 WebUI...
echo 💡 如果仍有錯誤，請檢查錯誤訊息
echo.

python webui.py --api --listen

pause
