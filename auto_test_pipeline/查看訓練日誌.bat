@echo off
chcp 65001 >nul
echo 📊 LoRA 訓練日誌查看器
echo ================================

cd /d "%~dp0"

echo 🔍 檢查訓練日誌...
python check_training_logs.py

echo.
echo 💡 如要啟動 TensorBoard 查看詳細圖表，請選擇：
echo    [1] 啟動 TensorBoard
echo    [2] 僅查看日誌摘要
echo    [3] 退出
echo.

set /p choice="請選擇 (1-3): "

if "%choice%"=="1" (
    echo 🚀 正在啟動 TensorBoard...
    echo 📊 請在瀏覽器中打開: http://localhost:6006
    echo 🛑 按 Ctrl+C 停止 TensorBoard
    cd lora_output\logs
    tensorboard --logdir .
) else if "%choice%"=="2" (
    echo 📋 顯示詳細日誌摘要...
    python check_training_logs.py
) else (
    echo 👋 退出
)

pause
