@echo off
echo 正在安裝 OpenAI CLIP 套件...
echo 這是解決 k-diffusion 錯誤的必要步驟

pip install git+https://github.com/openai/CLIP.git

if %errorlevel% equ 0 (
    echo.
    echo ✓ CLIP 安裝成功！
    echo 現在可以重新啟動 WebUI
) else (
    echo.
    echo ✗ CLIP 安裝失敗
    echo 請檢查網路連線和 Git 是否安裝
)

echo.
echo 按任意鍵繼續...
pause > nul
