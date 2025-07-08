@echo off
echo ====================================
echo     部署 LoRA 模型到 WebUI
echo ====================================
echo.

cd /d "%~dp0"

python deploy_lora.py

echo.
pause
