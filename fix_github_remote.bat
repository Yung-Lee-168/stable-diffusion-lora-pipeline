@echo off
echo 🔧 修復 GitHub 遠端倉庫設置
echo ================================

echo 📋 當前問題：您嘗試推送到 AUTOMATIC1111 的原始倉庫
echo 💡 解決方案：建立您自己的 GitHub 倉庫

echo.
echo 🔍 檢查當前遠端設置...
git remote -v
echo.

echo 📋 步驟 1: 移除現有的遠端連接
echo 💡 這會移除到 AUTOMATIC1111 倉庫的連接
set /p confirm1="確認移除現有遠端連接? (y/n): "
if /i "%confirm1%"=="y" (
    git remote remove origin
    echo ✅ 已移除原有遠端連接
) else (
    echo ❌ 已取消操作
    goto :end
)

echo.
echo 📋 步驟 2: 請在 GitHub 建立您的新倉庫
echo 🌐 請按照以下步驟操作：
echo    1. 開啟瀏覽器，前往 https://github.com
echo    2. 登入您的 GitHub 帳號
echo    3. 點擊右上角的 '+' 按鈕
echo    4. 選擇 'New repository'
echo    5. 倉庫名稱建議: stable-diffusion-lora-pipeline
echo    6. 描述: My LoRA training pipeline for Stable Diffusion
echo    7. 選擇 Public 或 Private (建議 Private 保護隱私)
echo    8. 不要勾選 'Initialize with README' (我們已有檔案)
echo    9. 點擊 'Create repository'
echo.

echo ⏸️  請完成 GitHub 倉庫建立後按任意鍵繼續...
pause >nul

echo.
echo 📋 步驟 3: 輸入您的 GitHub 倉庫信息
set /p username="輸入您的 GitHub 用戶名: "
set /p reponame="輸入倉庫名稱 (預設: stable-diffusion-lora-pipeline): "

if "%reponame%"=="" set reponame=stable-diffusion-lora-pipeline

echo.
echo 🔗 正在添加新的遠端倉庫...
git remote add origin https://github.com/%username%/%reponame%.git

echo.
echo 📋 步驟 4: 設置主分支並推送
echo 🔄 設置主分支為 main...
git branch -M main

echo.
echo 🚀 現在推送到您的 GitHub 倉庫...
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ✅ 成功！您的代碼已推送到 GitHub
    echo 🌐 倉庫地址: https://github.com/%username%/%reponame%
    echo 💡 以後可以使用 'git push' 推送更新
) else (
    echo.
    echo ❌ 推送失敗，可能的原因：
    echo    1. GitHub 用戶名或倉庫名稱錯誤
    echo    2. 需要設置 Git 認證
    echo    3. 倉庫不存在或無權限
    echo.
    echo 💡 解決方案：
    echo    - 檢查 GitHub 倉庫是否正確建立
    echo    - 確認用戶名和倉庫名稱正確
    echo    - 可能需要使用 Personal Access Token 進行認證
)

:end
echo.
echo 🎯 完成！您現在有自己的 GitHub 倉庫了
pause
