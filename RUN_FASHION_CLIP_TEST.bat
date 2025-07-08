@echo off
title Fashion CLIP 進階比較測試
echo.
echo ====================================================
echo   Fashion CLIP 模型比較測試
echo   比較標準 CLIP 與 FashionCLIP 在時尚圖片分析上的差異
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

REM 快速檢查關鍵依賴
echo 🔍 檢查依賴套件...
python -c "import torch, transformers, PIL, numpy" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  發現缺少必要套件
    echo.
    echo 🔧 自動安裝依賴套件...
    echo.
    
    REM 安裝基本套件
    python -m pip install torch transformers pillow numpy requests --quiet
    
    REM 再次檢查
    python -c "import torch, transformers, PIL, numpy" >nul 2>&1
    if errorlevel 1 (
        echo ❌ 自動安裝失敗
        echo.
        echo 請手動運行: INSTALL_DEPENDENCIES.bat
        echo 或手動安裝: pip install torch transformers pillow numpy
        pause
        exit /b 1
    )
    
    echo ✅ 依賴套件安裝完成
    echo.
)

REM 檢查是否有 day1 的圖片結果
if not exist "day1_results" (
    echo ⚠️  警告: 找不到 day1_results 資料夾
    echo 將嘗試使用其他可用的圖片進行測試
    echo.
)

echo 🚀 開始運行 Fashion CLIP 比較測試...
echo.

REM 運行測試
python day2_advanced_fashion_test.py

if errorlevel 1 (
    echo.
    echo ❌ 測試運行失敗
    echo 可能的原因:
    echo 1. 缺少必要的 Python 套件
    echo 2. GPU 記憶體不足
    echo 3. 找不到測試圖片
    echo.
    echo 請檢查錯誤訊息並嘗試以下解決方案:
    echo.
    echo 安裝必要套件:
    echo pip install torch transformers pillow numpy
    echo.
    echo 如果 GPU 記憶體不足，測試會自動使用 CPU
    pause
    exit /b 1
)

echo.
echo ✅ 測試完成！
echo.
echo 📊 查看結果:
echo 1. HTML 報告: 開啟 day2_fashion_results 資料夾中的 .html 文件
echo 2. Markdown 報告: 開啟 .md 文件查看純文字版本  
echo 3. JSON 數據: .json 文件包含完整的原始數據
echo.

REM 嘗試開啟結果資料夾
if exist "day2_fashion_results" (
    echo 🗂️  開啟結果資料夾...
    start "" "day2_fashion_results"
)

echo.
echo 按任意鍵關閉...
pause >nul
