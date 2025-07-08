@echo off
chcp 65001 >nul
echo 🔧 修復 PyTorch Lightning 版本相容性問題
echo ============================================================
echo 解決 pytorch_lightning.utilities.distributed 錯誤
echo ============================================================
echo.

cd /d "e:\Yung_Folder\Project\stable-diffusion-webui"

echo 📦 安裝相容的 PyTorch Lightning 版本...
pip uninstall pytorch_lightning -y
pip install pytorch_lightning==1.9.0

echo.
echo 📦 安裝其他可能需要的套件...
pip install lightning==1.9.0
pip install torchmetrics

echo.
echo 📦 檢查關鍵依賴版本...
python -c "
try:
    import pytorch_lightning as pl
    print(f'✅ PyTorch Lightning: {pl.__version__}')
    
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
    
    # 測試 utilities.distributed
    from pytorch_lightning.utilities.distributed import rank_zero_only
    print('✅ pytorch_lightning.utilities.distributed 可用')
    
except Exception as e:
    print(f'❌ 錯誤: {e}')
"

echo.
echo ✅ 版本修復完成！
echo.
echo 🚀 嘗試重新啟動 WebUI...
python webui.py --api --listen --skip-torch-cuda-test

pause
