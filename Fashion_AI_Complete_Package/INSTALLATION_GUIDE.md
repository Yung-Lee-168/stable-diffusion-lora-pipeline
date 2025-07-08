# Fashion AI Complete Suite - 安裝指南

## 🚀 系統安裝

### 方式一: 自動安裝 (推薦)

1. **下載套件**
   ```bash
   # 下載並解壓 Fashion_AI_Complete_Package.zip
   # 或使用 git clone
   git clone https://github.com/你的用戶名/Fashion-AI-Complete-Suite.git
   cd Fashion-AI-Complete-Suite
   ```

2. **執行自動安裝**
   ```bash
   python setup_and_install.py
   ```
   
   自動安裝將會：
   - ✅ 檢查系統環境
   - ✅ 安裝 Python 依賴
   - ✅ 下載必要模型
   - ✅ 配置 WebUI API
   - ✅ 執行系統測試

### 方式二: 手動安裝

#### 1. 環境準備
```bash
# 建議使用虛擬環境
python -m venv fashion_ai_env

# Windows
fashion_ai_env\Scripts\activate

# Linux/macOS
source fashion_ai_env/bin/activate
```

#### 2. 安裝依賴
```bash
# 基本依賴
pip install -r requirements.txt

# GPU 支援 (NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 可選優化套件
pip install xformers  # 記憶體優化
```

#### 3. 模型下載
```bash
# 執行模型下載器
python utils/model_downloader.py

# 或手動下載
# FashionCLIP 模型會自動下載
# Stable Diffusion 模型請確保 WebUI 已安裝
```

#### 4. 配置設定
```bash
# 複製並編輯配置文件
cp config/default_config.yaml config/user_config.yaml
# 編輯 user_config.yaml 以符合您的設定
```

## 🔧 Stable Diffusion WebUI 設置

### 1. WebUI 安裝
如果尚未安裝 Stable Diffusion WebUI：

```bash
# 克隆 WebUI 倉庫
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git

# 進入目錄並安裝
cd stable-diffusion-webui

# Windows
webui-user.bat

# Linux/macOS
./webui.sh
```

### 2. API 啟用
編輯 WebUI 啟動腳本，添加 API 參數：

**Windows (webui-user.bat):**
```batch
set COMMANDLINE_ARGS=--api --listen --port 7860
```

**Linux/macOS (webui-user.sh):**
```bash
export COMMANDLINE_ARGS="--api --listen --port 7860"
```

### 3. 模型準備
確保以下模型已放置在 WebUI 的 models 目錄：
- `models/Stable-diffusion/`: SD 1.5 或 SDXL 模型
- `models/VAE/`: VAE 模型 (可選)
- `models/Lora/`: LoRA 模型 (可選)

### 4. 啟動 WebUI
```bash
cd stable-diffusion-webui

# Windows
webui-user.bat

# Linux/macOS
./webui.sh
```

驗證 API 可用：瀏覽器開啟 http://localhost:7860/docs

## 📋 系統配置

### 1. API 配置 (config/api_config.yaml)
```yaml
webui:
  host: "localhost"
  port: 7860
  timeout: 60
  
fashion_ai:
  host: "localhost"
  port: 8080
  debug: false

models:
  fashion_clip: "patrickjohncyh/fashion-clip"
  default_sd_model: "runwayml/stable-diffusion-v1-5"
```

### 2. 功能配置 (config/default_config.yaml)
```yaml
# 圖片處理設定
image_processing:
  max_size: 1024
  supported_formats: ["jpg", "jpeg", "png", "bmp"]
  auto_resize: true

# 分析設定
analysis:
  confidence_threshold: 0.3
  max_categories: 10
  enable_detailed_analysis: true

# 生成設定
generation:
  default_steps: 20
  default_cfg_scale: 7.5
  default_size: [512, 512]
  batch_size: 1

# 系統設定
system:
  cache_enabled: true
  log_level: "INFO"
  max_concurrent_requests: 4
```

## 🧪 系統測試

### 1. 執行系統檢查
```bash
python utils/system_check.py
```

檢查項目：
- ✅ Python 版本
- ✅ GPU 可用性
- ✅ 依賴套件
- ✅ WebUI API 連接
- ✅ 模型載入
- ✅ 範例功能測試

### 2. 基本功能測試
```bash
# 測試 FashionCLIP 分析
python examples/basic_usage.py

# 測試 WebUI API 連接
python examples/api_examples.py

# 測試批次處理
python examples/batch_processing.py
```

### 3. Web 界面測試
```bash
# 啟動 Web 界面
python fashion_web_ui.py

# 瀏覽器開啟 http://localhost:8080
# 上傳測試圖片並驗證功能
```

## 🔧 故障排除

### 常見問題

#### 1. GPU 記憶體不足
```bash
# 降低批次大小
export BATCH_SIZE=1

# 啟用 CPU 模式
export FORCE_CPU=true

# 使用低精度模式
export USE_HALF_PRECISION=true
```

#### 2. WebUI API 連接失敗
```bash
# 檢查 WebUI 是否運行
curl http://localhost:7860/sdapi/v1/options

# 檢查防火牆設定
# 確認 --api 參數已添加到 WebUI 啟動命令
```

#### 3. 模型下載失敗
```bash
# 設置 Hugging Face 鏡像
export HF_ENDPOINT=https://hf-mirror.com

# 手動下載模型
git lfs install
git clone https://huggingface.co/patrickjohncyh/fashion-clip

# 設置本地模型路徑
export FASHION_CLIP_PATH=/path/to/fashion-clip
```

#### 4. 依賴衝突
```bash
# 重新創建虛擬環境
deactivate
rm -rf fashion_ai_env
python -m venv fashion_ai_env
source fashion_ai_env/bin/activate
pip install -r requirements.txt
```

### 效能優化

#### 1. GPU 優化
```yaml
# config/user_config.yaml
gpu_optimization:
  enable_xformers: true
  enable_attention_slicing: true
  enable_cpu_offload: false
  mixed_precision: "fp16"
```

#### 2. 記憶體優化
```yaml
memory_optimization:
  enable_model_caching: true
  max_cache_size: "4GB"
  enable_garbage_collection: true
  gc_interval: 10
```

#### 3. 網路優化
```yaml
network_optimization:
  connection_pool_size: 10
  request_timeout: 30
  retry_attempts: 3
  enable_compression: true
```

## 📚 進階配置

### 1. 自定義模型
```python
# 添加自定義 SD 模型
# 將模型文件放置在 WebUI 的 models/Stable-diffusion/ 目錄
# 更新配置文件
custom_models:
  my_fashion_model:
    path: "models/my_fashion_model.safetensors"
    description: "專門的時尚模型"
    default_settings:
      steps: 25
      cfg_scale: 8.0
```

### 2. 自定義分析類別
```python
# config/custom_categories.yaml
custom_categories:
  seasons:
    - "spring_collection"
    - "summer_collection" 
    - "autumn_collection"
    - "winter_collection"
  
  price_ranges:
    - "budget_friendly"
    - "mid_range"
    - "luxury"
    - "haute_couture"
```

### 3. API 擴展
```python
# 添加自定義 API 端點
# web/api/custom_endpoints.py
from flask import Blueprint

custom_api = Blueprint('custom', __name__)

@custom_api.route('/custom/style_transfer', methods=['POST'])
def style_transfer():
    # 自定義風格轉換邏輯
    pass
```

## 🚀 部署到生產環境

### 1. Docker 部署
```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# 安裝依賴
COPY requirements.txt .
RUN pip install -r requirements.txt

# 複製應用
COPY . /app
WORKDIR /app

# 啟動命令
CMD ["python", "fashion_ai_main.py", "--production"]
```

### 2. 雲端部署
```bash
# AWS/Azure/GCP 部署腳本
# 確保 GPU 實例和足夠的記憶體
# 配置負載平衡和自動擴展
```

### 3. 監控和日誌
```yaml
# config/production_config.yaml
monitoring:
  enable_metrics: true
  metrics_port: 9090
  log_file: "/var/log/fashion_ai.log"
  error_tracking: true
```

---

**安裝完成後，請參考 USER_MANUAL.md 了解使用方法**
