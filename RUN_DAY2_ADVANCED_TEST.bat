@echo off
title Day 2 進階 CLIP 測試 (修復版)
echo.
echo ====================================================
echo   Day 2 進階 CLIP 比較測試
echo   分析現有圖片，比較標準 CLIP 和 FashionCLIP
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

echo 🔍 檢查環境和依賴...

REM 檢查關鍵依賴
python -c "import torch, transformers, PIL, numpy" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  發現缺少必要套件，正在安裝...
    python -m pip install torch transformers pillow numpy --quiet
    echo ✅ 依賴安裝完成
)

REM 檢查是否有可分析的圖片
set HAS_IMAGES=0
if exist "day1_results\*.png" set HAS_IMAGES=1
if exist "day1_results\*.jpg" set HAS_IMAGES=1
if exist "outputs\*.png" set HAS_IMAGES=1
if exist "outputs\*.jpg" set HAS_IMAGES=1
if exist "day2_enhanced_results\*.png" set HAS_IMAGES=1

if "%HAS_IMAGES%"=="0" (
    echo ⚠️  沒有找到現有圖片文件
    echo 正在創建測試圖片...
    python create_test_images.py
    if errorlevel 1 (
        echo ❌ 創建測試圖片失敗
        pause
        exit /b 1
    )
    echo ✅ 測試圖片創建完成
) else (
    echo ✅ 找到現有圖片文件
)

echo.
echo 🚀 開始運行 Day 2 進階測試...
echo.

REM 運行測試
python day2_advanced_test.py

if errorlevel 1 (
    echo.
    echo ❌ 測試運行失敗
    echo.
    echo 🔧 可能的解決方案:
    echo 1. 檢查網路連接 (需要下載模型)
    echo 2. 確認 GPU 記憶體充足 (或使用 CPU)
    echo 3. 重新安裝依賴: pip install torch transformers pillow numpy
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ 測試完成！
echo.
echo 📊 查看結果:
if exist "day2_advanced_results" (
    echo 🗂️  開啟結果資料夾...
    start "" "day2_advanced_results"
    echo.
    echo 📄 建議查看:
    echo 1. HTML 報告 - 最佳視覺體驗
    echo 2. Markdown 報告 - 純文字版本
    echo 3. JSON 數據 - 完整原始數據
)

echo.
echo 按任意鍵關閉...
pause >nul
