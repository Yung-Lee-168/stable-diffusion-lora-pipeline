@echo off
echo ====================================
echo        測試 LoRA 訓練流程
echo ====================================
echo.

cd /d "%~dp0"

echo 🧪 開始測試...
python test_training_flow.py simple

echo.
echo 測試完成！
echo 按任意鍵結束...
pause >nul
