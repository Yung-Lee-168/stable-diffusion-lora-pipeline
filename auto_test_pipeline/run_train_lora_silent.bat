@echo off
echo 🚀 啟動 LoRA 訓練 (完全抑制警告版本)
echo.

REM 設定最強力的環境變數來抑制所有警告
set DISABLE_XFORMERS=1
set XFORMERS_MORE_DETAILS=0
set PYTHONWARNINGS=ignore
set PYTHONIOENCODING=utf-8
set CUDA_LAUNCH_BLOCKING=0
set TRANSFORMERS_VERBOSITY=error
set DIFFUSERS_VERBOSITY=error
set TOKENIZERS_PARALLELISM=false

REM 切換到腳本目錄
cd /d "%~dp0"

REM 執行訓練並重定向錯誤輸出
echo 開始執行訓練...
python train_lora.py %* 2>nul

echo.
echo 🏁 訓練完成
pause
