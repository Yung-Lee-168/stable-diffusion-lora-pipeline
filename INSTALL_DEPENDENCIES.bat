@echo off
title 安裝 Fashion CLIP 測試依賴套件
echo.
echo ====================================================
echo   安裝 Fashion CLIP 測試所需的 Python 套件
echo ====================================================
echo.

REM 檢查 Python 是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 錯誤: 找不到 Python
    echo 請確保已安裝 Python 並添加到 PATH
    pause
    exit /b 1
)

echo 🔍 檢查當前 Python 版本...
python --version

echo.
echo 🚀 開始安裝必要套件...
echo.

REM 升級 pip
echo 📦 升級 pip...
python -m pip install --upgrade pip

echo.
echo 📦 安裝 PyTorch...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo 📦 安裝其他依賴套件...
python -m pip install transformers pillow numpy requests

echo.
echo 🔍 驗證安裝...
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import PIL; print(f'Pillow: {PIL.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

if errorlevel 1 (
    echo.
    echo ❌ 安裝驗證失敗
    echo 請檢查錯誤訊息並手動安裝套件
    pause
    exit /b 1
)

echo.
echo ✅ 所有套件安裝完成！
echo.
echo 🎯 現在您可以運行測試:
echo    - 雙擊: RUN_FASHION_CLIP_TEST.bat
echo    - 或運行: python day2_advanced_fashion_test.py
echo.

pause
