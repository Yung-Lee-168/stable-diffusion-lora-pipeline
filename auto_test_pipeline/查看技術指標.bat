@echo off
chcp 65001
echo.
echo ============================================
echo 🎯 LoRA 調教技術指標快速查看器
echo ============================================
echo.

:menu
echo 請選擇操作：
echo.
echo 1. 查看最新技術指標
echo 2. 查看歷史趨勢
echo 3. 比較多輪結果
echo 4. 開始即時監控
echo 5. 產生完整分析報告
echo 6. 查看指標追蹤指南
echo 7. 退出
echo.

set /p choice=請輸入選項 (1-7): 

if "%choice%"=="1" goto latest
if "%choice%"=="2" goto history
if "%choice%"=="3" goto compare
if "%choice%"=="4" goto monitor
if "%choice%"=="5" goto analyze
if "%choice%"=="6" goto guide
if "%choice%"=="7" goto exit

echo 無效選項，請重新選擇
pause
goto menu

:latest
echo.
echo 🔍 查看最新技術指標...
python quick_metrics_viewer.py --latest
pause
goto menu

:history
echo.
echo 📈 查看歷史趨勢...
python quick_metrics_viewer.py --history
pause
goto menu

:compare
echo.
echo 🔍 比較多輪結果...
python quick_metrics_viewer.py --compare
pause
goto menu

:monitor
echo.
echo 🔍 開始即時監控...
echo 按 Ctrl+C 可停止監控
python quick_metrics_viewer.py --monitor
pause
goto menu

:analyze
echo.
echo 📊 產生完整分析報告...
python analyze_results.py
pause
goto menu

:guide
echo.
echo 📖 開啟指標追蹤指南...
start notepad "技術指標追蹤指南.md"
pause
goto menu

:exit
echo.
echo 👋 感謝使用！
pause
exit
