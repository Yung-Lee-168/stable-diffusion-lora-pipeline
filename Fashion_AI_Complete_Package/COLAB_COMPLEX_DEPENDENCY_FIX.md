# 🔧 Google Colab 複雜依賴衝突解決方案

## 🚨 問題描述

在 Google Colab 中可能遇到的複雜依賴衝突：

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
fastai 2.7.19 requires torch<2.7,>=1.10, but you have torch 2.7.1+cu118 which is incompatible.
sentence-transformers 4.1.0 requires transformers<5.0.0,>=4.41.0, but you have transformers 4.35.2 which is incompatible.
torchvision 0.21.0+cu124 requires torch==2.6.0, but you have torch 2.7.1+cu118 which is incompatible.
torchaudio 2.6.0+cu124 requires torch==2.6.0, but you have torch 2.7.1+cu118 which is incompatible.
```

## 🔄 完整解決方案

### 方法一：一鍵修復腳本
```python
# 在 Colab 中執行這個完整的修復腳本
import subprocess
import sys

def fix_all_dependencies():
    print("🔧 開始修復所有依賴衝突...")
    
    # 1. 完全清理環境
    print("🗑️ 清理所有相關套件...")
    packages_to_remove = [
        "torch", "torchvision", "torchaudio", 
        "transformers", "sentence-transformers", 
        "fastai", "diffusers", "accelerate", "peft"
    ]
    
    for pkg in packages_to_remove:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", pkg], 
                      capture_output=True)
    
    # 2. 安裝穩定的 PyTorch 組合
    print("📦 安裝穩定的 PyTorch...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch==2.1.0+cu118",
        "torchvision==0.16.0+cu118", 
        "torchaudio==2.1.0+cu118",
        "--index-url", "https://download.pytorch.org/whl/cu118",
        "--force-reinstall"
    ])
    
    # 3. 安裝兼容的其他套件
    print("📦 安裝其他套件...")
    other_packages = [
        "transformers>=4.41.0,<5.0.0",
        "diffusers[torch]",
        "accelerate", 
        "peft>=0.4.0",
        "sentence-transformers"
    ]
    
    for pkg in other_packages:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg])
    
    print("✅ 修復完成！請重新啟動運行時")

# 執行修復
fix_all_dependencies()
```

### 方法二：手動分步修復

#### 步驟 1：清理環境
```bash
!pip uninstall -y torch torchvision torchaudio transformers sentence-transformers fastai diffusers accelerate peft
```

#### 步驟 2：安裝兼容的 PyTorch
```bash
!pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

#### 步驟 3：安裝 AI 套件
```bash
!pip install transformers>=4.41.0 diffusers[torch] accelerate peft
```

#### 步驟 4：重新安裝 sentence-transformers
```bash
!pip install sentence-transformers
```

#### 步驟 5：重新啟動運行時
**重要**：執行 Runtime > Restart runtime

### 方法三：使用虛擬環境（進階）
```python
# 在 Colab 中創建乾淨的虛擬環境
!python -m venv /content/fashion_ai_env
!source /content/fashion_ai_env/bin/activate && pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
!source /content/fashion_ai_env/bin/activate && pip install transformers diffusers accelerate peft sentence-transformers

# 啟用虛擬環境
import sys
sys.path.insert(0, '/content/fashion_ai_env/lib/python3.10/site-packages')
```

## 📋 驗證安裝

修復後，執行以下代碼驗證：

```python
# 檢查版本
import torch
import transformers
import diffusers

print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"diffusers: {diffusers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# 測試導入
try:
    from diffusers import StableDiffusionPipeline
    from peft import LoraConfig
    from transformers import CLIPModel
    print("✅ 所有關鍵套件導入成功")
except ImportError as e:
    print(f"❌ 導入失敗: {e}")
```

## 🎯 推薦的穩定版本組合

### 最佳穩定組合（推薦）
- `torch==2.1.0+cu118`
- `torchvision==0.16.0+cu118`
- `torchaudio==2.1.0+cu118`
- `transformers>=4.41.0,<5.0.0`
- `diffusers>=0.21.0`
- `accelerate>=0.20.0`
- `peft>=0.4.0`

### 相容性矩陣

| PyTorch | transformers | diffusers | 相容性 |
|---------|-------------|-----------|--------|
| 2.1.0   | 4.41.x      | 0.21.x    | ✅ 優秀 |
| 2.0.0   | 4.35.x      | 0.18.x    | ⚠️ 一般 |
| 2.6.0   | 4.41.x      | 0.24.x    | ❌ 衝突 |

## 🔍 故障排除

### 常見錯誤 1：CUDA 版本不匹配
```python
# 檢查 CUDA 版本
!nvcc --version
!nvidia-smi

# 選擇對應的 PyTorch 版本
# CUDA 11.8: torch==2.1.0+cu118
# CUDA 12.1: torch==2.1.0+cu121
```

### 常見錯誤 2：記憶體不足
```python
# 清理記憶體
import gc
import torch
gc.collect()
torch.cuda.empty_cache()

# 減少批次大小
config["train_batch_size"] = 1
config["gradient_accumulation_steps"] = 8
```

### 常見錯誤 3：模型載入失敗
```python
# 檢查網路連接
!ping -c 4 huggingface.co

# 使用離線模式
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
```

## ⚡ 快速修復命令

如果遇到任何問題，執行這個一鍵修復：

```bash
# 一鍵修復命令
!pip uninstall -y torch torchvision torchaudio transformers sentence-transformers fastai && \
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
pip install transformers>=4.41.0 diffusers[torch] accelerate peft sentence-transformers
```

執行後務必重新啟動運行時！

## 📞 支援

如果問題持續存在：
1. 檢查 Colab 的 GPU 類型和 CUDA 版本
2. 嘗試使用全新的 Colab notebook
3. 確保網路連接穩定
4. 考慮使用本地環境或其他雲端平台
