@echo off
REM 確保使用正確版本的 train_lora.py 腳本
REM 這個腳本會直接調用當前目錄的 train_lora.py

echo 🚀 啟動 LoRA 訓練（調試版本）
echo 📁 工作目錄: %CD%
echo.

REM 設定工作目錄
cd /d "%~dp0"

REM 確認文件存在
if not exist "train_lora.py" (
    echo ❌ 錯誤：找不到 train_lora.py 文件
    pause
    exit /b 1
)

REM 清理 Python 快取
if exist "__pycache__" rmdir /s /q "__pycache__"

REM 顯示即將執行的命令
echo 📋 即將執行: python "%CD%\train_lora.py" %*
echo.

REM 執行腳本
python "%CD%\train_lora.py" %*

REM 檢查執行結果
if errorlevel 1 (
    echo.
    echo ❌ 訓練過程中出現錯誤
    echo 💡 請檢查上面的錯誤訊息
) else (
    echo.
    echo ✅ 訓練完成
)

echo.
pause
