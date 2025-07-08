@echo off
title 深度時尚分析：CLIP vs FashionCLIP 詳細比較
echo.
echo ====================================================
echo   👗 深度時尚分析測試
echo   比較 Standard CLIP 和 FashionCLIP 在服裝細節識別上的差異
echo   包含：款式、材質、剪裁、風格等詳細特徵分析
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

echo 🔍 檢查深度分析環境...

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
if exist "test_images\*.png" set HAS_IMAGES=1

if "%HAS_IMAGES%"=="0" (
    echo ⚠️  沒有找到現有圖片文件
    echo 正在創建時尚測試圖片...
    python create_test_images.py
    if errorlevel 1 (
        echo ❌ 創建測試圖片失敗
        pause
        exit /b 1
    )
    echo ✅ 時尚測試圖片創建完成
) else (
    echo ✅ 找到現有圖片文件進行深度分析
)

echo.
echo 🚀 開始深度時尚分析...
echo.
echo 📋 分析內容包括:
echo    ✓ 基本分類：性別、年齡、季節、場合
echo    ✓ 服裝款式：連衣裙類型、襯衫特徵、外套種類
echo    ✓ 詳細特徵：材質、剪裁、圖案、色彩、風格
echo    ✓ 兩模型專業比較和優勢分析
echo.

REM 運行深度分析
python day2_advanced_test.py

if errorlevel 1 (
    echo.
    echo ❌ 深度分析失敗
    echo.
    echo 🔧 可能的解決方案:
    echo 1. 確認網路連接正常 (需要下載 FashionCLIP 模型)
    echo 2. 檢查 GPU 記憶體 (約需 4GB+ VRAM，或自動使用 CPU)
    echo 3. 重新安裝依賴: pip install torch transformers pillow numpy
    echo 4. 確認有足夠硬碟空間下載模型
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ 深度時尚分析完成！
echo.
echo 📊 查看詳細結果:
if exist "day2_advanced_results" (
    echo 🗂️  開啟結果資料夾...
    start "" "day2_advanced_results"
    echo.
    echo 📄 推薦查看順序:
    echo 1. 📱 HTML 報告 - 最佳視覺體驗，包含詳細特徵對比
    echo 2. 📝 Markdown 報告 - 純文字版本，適合分享
    echo 3. 💾 JSON 數據 - 完整原始分析數據
    echo.
    echo 💡 HTML 報告特點:
    echo    ✓ 基本分類 vs 詳細特徵分區顯示
    echo    ✓ 兩模型並排比較
    echo    ✓ 置信度色彩標示
    echo    ✓ 提示詞生成比較
    echo    ✓ 模型優勢分析總結
)

echo.
echo 🎯 分析重點:
echo - 觀察 FashionCLIP 在具體服裝款式識別上的優勢
echo - 比較兩模型在材質、風格判斷上的差異
echo - 參考置信度評估模型可靠性
echo - 使用分析結果優化服裝相關應用
echo.
echo 按任意鍵關閉...
pause >nul
