@echo off
:: 設定 UTF-8 編碼以正確顯示 emoji 和中文
chcp 65001 >nul
title 深度時尚分析：CLIP vs FashionCLIP 詳細比較

echo.
echo ============================================================
echo 🔍 深度時尚分析：Standard CLIP vs FashionCLIP 比較測試
echo 自動分析 day1 產出的所有時尚圖片，並進行多層次服裝特徵分析
echo ============================================================
echo.

echo 📋 檢查環境需求...
echo   1. Python 3.7+
echo   2. 建議使用 GPU (需要 4GB+ VRAM，或者使用 CPU)
echo   3. 必要套件: pip install torch transformers pillow numpy
echo   4. 確保有現有圖片可分析
echo.

echo 🔍 開始深度分析...
python day2_advanced_test.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ 分析完成！
    echo 📊 查看結果:
    echo    - HTML 報告: day2_advanced_results\ 資料夾
    echo    - JSON 數據: day2_advanced_results\ 資料夾  
    echo    - Markdown: day2_advanced_results\ 資料夾
    echo.
    echo 🎯 主要發現:
    echo    - 詳細服裝特徵識別能力比較
    echo    - 兩模型在各類別的置信度分析
    echo    - 專業時尚領域的模型優勢評估
    echo.
    echo 🚀 開啟報告資料夾...
    start explorer "day2_advanced_results"
) else (
    echo.
    echo ❌ 分析過程中發生錯誤
    echo 🔧 請檢查:
    echo    1. 是否已安裝必要套件
    echo    2. 是否有 day1_results 資料夾及圖片
    echo    3. GPU 記憶體是否足夠 (或改用 CPU)
    echo.
)

echo.
echo 按任意鍵結束...
pause >nul
