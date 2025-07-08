@echo off
echo 🚀 啟動 LoRA 訓練 (完整監控版本 - 抑制警告)
echo.

REM 設定環境變數來抑制警告
set DISABLE_XFORMERS=1
set XFORMERS_MORE_DETAILS=0
set PYTHONWARNINGS=ignore
set PYTHONIOENCODING=utf-8

REM 切換到腳本目錄
cd /d "%~dp0"

REM 執行訓練
python train_lora_monitored_new.py %*

echo.
echo 🏁 訓練完成
pause
