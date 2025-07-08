@echo off
REM 調試 train_lora.py 腳本
REM 此腳本將幫助識別和解決 logging_interval 參數錯誤

echo 🔍 調試 train_lora.py 問題...
echo.

REM 設定工作目錄
cd /d "%~dp0"
echo 📁 當前目錄: %CD%

REM 清理 Python 快取
echo 🧹 清理 Python 快取...
if exist "__pycache__" (
    rmdir /s /q "__pycache__"
    echo    ✅ 已清理 __pycache__
) else (
    echo    ℹ️  沒有找到 __pycache__
)

REM 檢查文件是否存在
echo.
echo 📋 檢查關鍵文件...
if exist "train_lora.py" (
    echo    ✅ train_lora.py 存在
) else (
    echo    ❌ train_lora.py 不存在
    pause
    exit /b 1
)

if exist "train_network.py" (
    echo    ✅ train_network.py 存在
) else (
    echo    ❌ train_network.py 不存在
    pause
    exit /b 1
)

REM 檢查 logging_interval 參數
echo.
echo 🔍 檢查 train_lora.py 中是否有 logging_interval...
findstr /C:"logging_interval" train_lora.py >nul 2>&1
if errorlevel 1 (
    echo    ✅ train_lora.py 中沒有發現 logging_interval
) else (
    echo    ⚠️  train_lora.py 中發現 logging_interval，這可能是問題源頭
    findstr /N /C:"logging_interval" train_lora.py
)

REM 檢查其他可能的腳本
echo.
echo 🔍 檢查其他腳本中的 logging_interval...
for %%f in (*.py) do (
    findstr /C:"logging_interval" "%%f" >nul 2>&1
    if not errorlevel 1 (
        echo    ⚠️  在 %%f 中發現 logging_interval
        findstr /N /C:"logging_interval" "%%f"
    )
)

REM 測試 train_lora.py 幫助訊息
echo.
echo 🧪 測試 train_lora.py --help...
python train_lora.py --help
if errorlevel 1 (
    echo    ❌ train_lora.py --help 失敗
) else (
    echo    ✅ train_lora.py --help 成功
)

REM 生成乾淨的訓練命令
echo.
echo 🔧 生成乾淨的訓練命令...
echo python train_lora.py --new > test_command.bat
echo    ✅ 已生成 test_command.bat

echo.
echo 🎯 調試完成！
echo.
echo 📋 下一步建議：
echo    1. 檢查上面的輸出是否有任何 ⚠️ 警告
echo    2. 運行 test_command.bat 來測試
echo    3. 如果仍有問題，請提供完整的錯誤訊息
echo.
pause
