# 🔧 PyTorch Lightning 版本相容性問題解決方案

## ❌ 問題
```
ModuleNotFoundError: No module named 'pytorch_lightning.utilities.distributed'
```

## 🎯 原因
Stable Diffusion WebUI 需要特定版本的 PyTorch Lightning (通常是 1.x 版本)，但系統可能安裝了更新的 2.x 版本，API 結構發生了變化。

## ✅ 解決方案

### 自動修復 (推薦)
```bash
.\FIX_PYTORCH_LIGHTNING.bat
```

### 手動修復
```bash
# 卸載當前版本
pip uninstall pytorch_lightning -y

# 安裝相容版本
pip install pytorch_lightning==1.9.0
pip install lightning==1.9.0
pip install torchmetrics

# 驗證安裝
python -c "from pytorch_lightning.utilities.distributed import rank_zero_only; print('✅ 修復成功')"
```

## 🔍 版本相容性對照表

| SD WebUI 版本 | PyTorch Lightning | Lightning | 狀態 |
|---------------|-------------------|-----------|------|
| 最新版        | 1.9.0            | 1.9.0     | ✅ 相容 |
| 較舊版        | 1.7.x - 1.8.x    | 1.7.x     | ✅ 相容 |
| 不相容        | 2.0+             | 2.0+      | ❌ 錯誤 |

## 🚀 啟動 WebUI

修復完成後啟動：
```bash
python webui.py --api --listen --skip-torch-cuda-test
```

## 🧪 驗證修復

成功的標誌：
```
✅ PyTorch Lightning: 1.9.0
✅ pytorch_lightning.utilities.distributed 可用
Running on local URL: http://127.0.0.1:7860
```

## 💡 其他可能的問題

### 如果仍有錯誤：
1. **xFormers 警告** (可忽略)
   ```
   WARNING[XFORMERS]: Need to compile C++ extensions
   ```
   這不會影響功能，只是性能優化。

2. **CUDA 相關錯誤**
   添加參數：`--skip-torch-cuda-test`

3. **模型載入錯誤**
   確保有 SD 模型在 `models/Stable-diffusion/` 目錄

### 快速診斷命令：
```bash
# 檢查版本
python -c "import pytorch_lightning; print(pytorch_lightning.__version__)"

# 測試導入
python -c "from pytorch_lightning.utilities.distributed import rank_zero_only"

# 檢查 CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🎯 完成後步驟

1. ✅ WebUI 啟動成功
2. ✅ 看到 `http://127.0.0.1:7860`
3. ✅ 執行 `python check_webui_for_clip.py` 驗證
4. ✅ 運行 `python day2_enhanced_test.py` 開始測試
