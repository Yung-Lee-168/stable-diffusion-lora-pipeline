@echo off
echo ====================================
echo        LoRA 訓練狀態檢查
echo ====================================
echo.

cd /d "%~dp0"

python check_status.py

echo.
echo 按任意鍵繼續...
pause >nul
