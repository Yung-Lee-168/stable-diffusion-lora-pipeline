@echo off
echo 🚀 快速 GitHub 設置和推送腳本
echo ================================

echo 📁 當前目錄: %CD%
echo.

echo 🔍 檢查 Git 狀態...
git status
echo.

echo 📋 添加所有檔案到 Git...
git add .
echo.

echo 📊 檢查將要提交的檔案...
git status
echo.

echo ⏸️  按任意鍵繼續提交，或 Ctrl+C 取消...
pause >nul

echo 💾 提交變更...
git commit -m "完整 LoRA 訓練 Pipeline: 統一性能指標實現和圖片尺寸處理"
echo.

echo 🌐 推送到 GitHub...
echo 💡 如果是首次推送，請先在 GitHub 建立倉庫
echo 💡 然後執行: git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
echo.

set /p choice="是否現在推送到 GitHub? (y/n): "
if /i "%choice%"=="y" (
    git push
    echo ✅ 推送完成!
) else (
    echo ℹ️  跳過推送，您可以稍後手動執行: git push
)

echo.
echo 🎯 完成! 您的代碼已準備好版本控制
echo 💡 下次修改後，可以執行此腳本快速提交
pause
