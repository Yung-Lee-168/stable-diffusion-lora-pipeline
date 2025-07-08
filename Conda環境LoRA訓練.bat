@echo off
echo ===============================================
echo Conda環境LoRA訓練啟動器
echo ===============================================

echo 🔍 檢查Conda安裝...
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未找到Conda，請確保已安裝Anaconda或Miniconda
    pause
    exit /b 1
)

echo ✅ 找到Conda

echo.
echo 📋 當前Conda環境列表:
conda env list

echo.
echo 🤔 請選擇要使用的環境:
echo 1. 使用當前環境 (默認)
echo 2. 激活指定環境
echo 3. 創建新環境

set /p choice="請輸入選擇 (1-3, 默認1): "

if "%choice%"=="" set choice=1
if "%choice%"=="1" goto run_training
if "%choice%"=="2" goto activate_env
if "%choice%"=="3" goto create_env

:activate_env
set /p env_name="請輸入環境名稱: "
echo 🔄 激活環境 %env_name%...
call conda activate %env_name%
if %errorlevel% neq 0 (
    echo ❌ 無法激活環境 %env_name%
    pause
    exit /b 1
)
goto run_training

:create_env
set /p new_env_name="請輸入新環境名稱: "
echo 🆕 創建新環境 %new_env_name%...
call conda create -n %new_env_name% python=3.10 -y
if %errorlevel% neq 0 (
    echo ❌ 無法創建環境 %new_env_name%
    pause
    exit /b 1
)
call conda activate %new_env_name%
echo 📦 安裝基本依賴...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate
goto run_training

:run_training
echo.
echo ===============================================
echo 🚀 在Conda環境中啟動LoRA訓練
echo ===============================================

echo 🔍 當前環境信息:
echo 環境名稱: %CONDA_DEFAULT_ENV%
echo Python路徑: %CONDA_PYTHON_EXE%

echo.
echo 🎯 選擇訓練模式:
echo 1. 新訓練
echo 2. 繼續訓練
echo 3. 運行環境檢查

set /p train_choice="請輸入選擇 (1-3): "

if "%train_choice%"=="1" (
    echo 🆕 開始新訓練...
    python auto_test_pipeline\train_lora.py --new
) else if "%train_choice%"=="2" (
    echo 🔄 繼續訓練...
    python auto_test_pipeline\train_lora.py --continue
) else if "%train_choice%"=="3" (
    echo 🔍 運行環境檢查...
    python -c "import sys; print('Python:', sys.executable); import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
) else (
    echo ❌ 無效選擇
)

echo.
echo ===============================================
echo 訓練完成或已退出
echo ===============================================
pause
