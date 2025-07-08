# 安裝環境說明

## 🛠️ 系統需求

### 最低配置
- **操作系統**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8 - 3.11 (推薦 3.10)
- **GPU**: 4GB+ VRAM (推薦 8GB+)
- **磁碟空間**: 15GB+

### 推薦配置
- **GPU**: RTX 3070/4060 Ti 或更高
- **RAM**: 16GB+ 系統記憶體
- **VRAM**: 8GB+ GPU 記憶體

## 📦 安裝步驟

### 1. 安裝 Python 環境

#### Windows
```bash
# 下載並安裝 Python 3.10 from python.org
# 確保添加到 PATH

# 驗證安裝
python --version
pip --version
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.10 python3.10-pip python3.10-venv
sudo apt install git wget curl
```

#### macOS
```bash
# 使用 Homebrew
brew install python@3.10
```

### 2. 創建虛擬環境 (推薦)
```bash
# 創建虛擬環境
python -m venv fashion_ai_env

# 啟動虛擬環境
# Windows
fashion_ai_env\Scripts\activate
# Linux/macOS
source fashion_ai_env/bin/activate
```

### 3. 安裝 PyTorch

#### CUDA 版本 (有 NVIDIA GPU)
```bash
# CUDA 11.8 (推薦)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### CPU 版本
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. 安裝套件依賴
```bash
# 安裝主要依賴
pip install -r requirements.txt

# 驗證安裝
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 5. 安裝 Stable Diffusion WebUI

#### 方法 A: 全新安裝
```bash
# 克隆 WebUI 倉庫
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# 首次啟動 (會自動下載模型)
# Windows
./webui-user.bat
# Linux/macOS
./webui.sh
```

#### 方法 B: 使用現有 WebUI
```bash
# 如果您已有 WebUI，只需確保版本相容
cd your-existing-webui-folder
git pull  # 更新到最新版本
```

### 6. 設置 Fashion AI 套件
```bash
# 解壓 Fashion AI Training Package
unzip Fashion_AI_Training_Package.zip
cd Fashion_AI_Training_Package

# 安裝套件依賴
pip install -r requirements.txt

# 測試安裝
python launcher.py --test
```

## ⚙️ 配置設定

### 1. WebUI 配置
編輯 `webui-user.bat` (Windows) 或 `webui-user.sh` (Linux/macOS):

```bash
# 啟用 API 和外部訪問
set COMMANDLINE_ARGS=--api --listen --enable-insecure-extension-access
# Linux/macOS
export COMMANDLINE_ARGS="--api --listen --enable-insecure-extension-access"
```

### 2. 環境變數設定
```bash
# 設置快取目錄 (可選)
export HF_HOME="/path/to/huggingface/cache"
export TRANSFORMERS_CACHE="/path/to/transformers/cache"

# 設置 CUDA 記憶體分配 (如果有 GPU 記憶體問題)
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
```

### 3. Fashion AI 配置
編輯 `configs/default_config.json`:

```json
{
  "webui_api_url": "http://localhost:7860",
  "output_dir": "./outputs",
  "image_size": 512,
  "batch_size": 1,
  "use_gpu": true
}
```

## 🔧 驗證安裝

### 運行測試腳本
```bash
python launcher.py --test
```

預期輸出:
```
✅ Python 環境檢查通過
✅ PyTorch 安裝正確
✅ GPU 可用 (如果有 GPU)
✅ 必要套件已安裝
✅ WebUI API 連接成功
🎉 安裝驗證完成！
```

## 🐛 常見安裝問題

### 1. CUDA 版本不匹配
```bash
# 檢查 CUDA 版本
nvidia-smi

# 重新安裝對應版本的 PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 記憶體不足
```bash
# 設置環境變數限制記憶體使用
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"

# 或使用 CPU 模式
python launcher.py --device cpu
```

### 3. 網路連接問題
```bash
# 使用鏡像源 (中國用戶)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 設置 Hugging Face 鏡像
export HF_ENDPOINT=https://hf-mirror.com
```

### 4. 權限問題 (Linux/macOS)
```bash
# 確保有寫入權限
chmod +x webui.sh
chmod +x launcher.py

# 如果需要，使用 sudo (不推薦用於 pip)
sudo chown -R $USER:$USER ./Fashion_AI_Training_Package
```

## 📋 完整依賴清單

詳見 `requirements.txt`:
```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
diffusers>=0.21.0
accelerate>=0.24.0
pillow>=9.0.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
requests>=2.28.0
tqdm>=4.65.0
```

## 🆘 需要幫助？

如果安裝過程中遇到問題：

1. 📖 查看 `docs/troubleshooting.md`
2. 🔧 檢查系統配置是否符合需求
3. 🌐 確保網路連接正常
4. 💾 確保磁碟空間充足

## 🎯 下一步

安裝完成後，請參考：
- `SYSTEM_STARTUP_GUIDE.md` - 系統啟動說明
- `WEBUI_API_GUIDE.md` - API 使用指南
- `examples/` - 實際使用範例
