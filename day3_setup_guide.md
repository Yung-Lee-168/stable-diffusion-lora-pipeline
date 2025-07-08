# Fashion AI 完整應用程式 - 安裝和使用指南

## 🎯 概述

Fashion AI 完整應用程式是一個整合了 FashionCLIP 特徵分析、Stable Diffusion v1.5 圖片生成、WebUI API 等功能的全功能時尚 AI 系統。

主要功能：
1. **智能時尚分析** - 使用 FashionCLIP 進行時尚圖片特徵分析
2. **AI 圖片生成** - 基於 Stable Diffusion WebUI API 的文本到圖片生成
3. **Web 介面操作** - 友好的 Web UI 界面，支持批次處理
4. **API 整合服務** - 完整的 REST API 支持，方便第三方整合

## 🏗️ 系統架構

```
Fashion AI Complete Package
├── 核心引擎 (Core Engine)
│   ├── FashionCLIP 特徵分析
│   ├── Stable Diffusion API 整合
│   └── 提示詞生成與優化
├── Web 介面 (Web Interface)
│   ├── 圖片上傳與預覽
│   ├── 即時生成結果展示
│   └── 批次處理管理
├── API 服務 (API Services)
│   ├── RESTful API 端點
│   ├── WebSocket 即時通信
│   └── 第三方整合接口
└── 工具與配置 (Tools & Config)
    ├── 模型管理
    ├── 系統監控
    └── 配置管理
```

## 📁 文件結構

```
Fashion_AI_Complete_Package/
├── fashion_ai_main.py              # 主啟動器
├── fashion_web_ui.py               # Web 介面主程式
├── core/                           # 核心模組
│   ├── fashion_analyzer.py         # FashionCLIP 分析器
│   ├── webui_connector.py          # WebUI API 連接器
│   ├── prompt_generator.py         # 提示詞生成器
│   └── batch_processor.py          # 批次處理器
├── web/                            # Web 介面資源
│   ├── templates/                  # HTML 模板
│   ├── static/                     # 靜態資源
│   └── api/                        # API 端點
├── utils/                          # 工具模組
│   ├── model_manager.py            # 模型管理
│   ├── system_check.py             # 系統檢查
│   └── logger.py                   # 日誌系統
├── config/                         # 配置文件
│   ├── default_config.yaml         # 默認配置
│   └── model_config.yaml           # 模型配置
├── examples/                       # 範例程式
│   ├── api_examples.py             # API 使用範例
│   └── batch_examples.py           # 批次處理範例
├── data/                           # 數據目錄
│   ├── input/                      # 輸入圖片
│   └── output/                     # 生成結果
├── README.md                       # 系統說明
├── INSTALLATION_GUIDE.md           # 安裝指南
├── WEBUI_API_GUIDE.md             # WebUI API 使用指南
├── USER_MANUAL.md                  # 使用手冊
└── requirements.txt                # 依賴列表
```

## 📋 系統需求

### 最低硬體需求
- **GPU**: NVIDIA RTX 3060 (8GB VRAM) 或以上
- **RAM**: 16GB 系統內存
- **存儲**: 50GB 可用空間（包含模型檔案）
- **網路**: 穩定的互聯網連接（首次下載模型時）

### 推薦硬體配置
- **GPU**: NVIDIA RTX 4070 (12GB VRAM) 或以上
- **RAM**: 32GB 系統內存
- **存儲**: 100GB SSD 存儲空間
- **CPU**: Intel i7-10700K 或 AMD Ryzen 7 3700X 以上

### 軟體需求
- **作業系統**: Windows 10/11 (64-bit)
- **Python**: 3.8-3.11
- **CUDA**: 11.7 或以上
- **Git**: 最新版本

## 🛠️ 安裝步驟

### 1. 準備環境
```bash
# 建立虛擬環境
python -m venv fashion_ai_env
fashion_ai_env\Scripts\activate

# 更新 pip
python -m pip install --upgrade pip
```

### 2. 安裝依賴
```bash
# 安裝核心依賴
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers diffusers accelerate
pip install pillow opencv-python scikit-learn
pip install matplotlib seaborn scipy
pip install requests tqdm flask

# 安裝可選依賴（推薦）
pip install xformers       # 優化注意力機制
pip install tensorboard    # TensorBoard 支持
```

### 3. 設置 Stable Diffusion WebUI
```bash
# 如果尚未安裝 WebUI
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# 首次啟動（下載模型）
webui.bat --api --listen

# 確認 API 可用
curl http://localhost:7860/sdapi/v1/options
```

### 4. 下載 Fashion AI 完整包
```bash
# 方法一：從 GitHub 下載（推薦）
git clone https://github.com/your-repo/Fashion_AI_Complete_Package.git
cd Fashion_AI_Complete_Package

# 方法二：手動創建目錄結構
mkdir Fashion_AI_Complete_Package
cd Fashion_AI_Complete_Package

# 從原始專案中複製需要的檔案
# 注意：不是複製整個 stable-diffusion-webui 目錄
# 只複製我們需要的核心功能檔案
```

**重要說明**：
- ❌ **不要** 複製整個 `stable-diffusion-webui` 目錄
- ✅ **只需要** 複製 Fashion AI 相關的核心檔案
- ✅ **保持** WebUI 在原本位置運行，通過 API 連接

## 🏗️ 部署架構說明

### 建議的目錄結構
```
工作目錄/
├── stable-diffusion-webui/          # 現有的 WebUI（保持不變）
│   ├── webui.bat
│   ├── models/
│   └── ...其他 WebUI 檔案
└── Fashion_AI_Complete_Package/      # 新的精簡應用包
    ├── fashion_ai_main.py            # 主啟動器
    ├── core/                         # 核心模組
    └── ...其他應用檔案
```

### 運行方式
1. **WebUI 服務**: 在 `stable-diffusion-webui/` 目錄運行，提供 API 服務
2. **Fashion AI 應用**: 在 `Fashion_AI_Complete_Package/` 目錄運行，通過 API 連接 WebUI
3. **獨立運行**: 兩個系統獨立運行，不互相干擾

### 需要從現有專案複製的檔案
```bash
# 從 stable-diffusion-webui 目錄複製以下檔案到 Fashion_AI_Complete_Package
day3_fashion_training.py        → core/fashion_analyzer.py
day3_colab_finetuning.py        → core/webui_connector.py
# 其他相關的分析和生成程式
```

## 🚀 快速啟動

### 方法一：一鍵啟動
```bash
# 啟動完整系統
python fashion_ai_main.py

# 帶參數啟動
python fashion_ai_main.py --port 8080 --webui-url http://localhost:7860
```

### 方法二：分步啟動
```bash
# 1. 啟動 Stable Diffusion WebUI
cd stable-diffusion-webui
webui.bat --api --listen

# 2. 啟動 Fashion AI 系統
cd Fashion_AI_Complete_Package
python fashion_web_ui.py
```

### 3. 驗證安裝
打開瀏覽器訪問：
- Fashion AI Web UI: http://localhost:8080
- WebUI API 文檔: http://localhost:7860/docs

## 🎯 主要功能

### 1. 智能時尚分析
- 上傳時尚圖片，自動分析服裝風格、顏色、材質等特徵
- 基於 FashionCLIP 的深度學習分析
- 生成詳細的特徵分析報告

### 2. AI 圖片生成
- 文本到圖片的 AI 生成功能
- 支持多種風格和參數調整
- 批次生成和即時預覽

### 3. Web 介面操作
- 直觀的 Web 用戶界面
- 拖拽上傳圖片功能
- 即時結果展示和下載

### 4. API 整合服務
- RESTful API 支持
- 第三方應用整合
- 程式化批次處理

## 💻 基本使用方法

### 透過 Web 介面使用
1. 啟動系統後，在瀏覽器中訪問 http://localhost:8080
2. 上傳您的時尚圖片
3. 選擇分析模式或生成模式
4. 查看結果並下載

### 透過 API 使用
```python
import requests

# 上傳圖片分析
response = requests.post('http://localhost:8080/api/analyze', 
                        files={'image': open('fashion_image.jpg', 'rb')})
result = response.json()

# 生成新圖片
response = requests.post('http://localhost:8080/api/generate', 
                        json={'prompt': 'elegant dress, runway style'})
image_url = response.json()['image_url']
```

## 🔧 API 端點說明

### 主要 API 端點
- `POST /api/analyze` - 分析時尚圖片
- `POST /api/generate` - 生成新圖片
- `GET /api/status` - 檢查系統狀態
- `GET /api/models` - 列出可用模型
- `POST /api/batch` - 批次處理請求

### WebUI API 整合
- `GET /sdapi/v1/options` - WebUI 配置選項
- `POST /sdapi/v1/txt2img` - 文本到圖片生成
- `POST /sdapi/v1/img2img` - 圖片到圖片轉換

### 微調訓練輸出

```
day3_finetuning_results/
├── checkpoints/                              # 訓練檢查點
├── fashion_lora_weights.pt                   # LoRA 權重
├── fashion_sd_model/                         # 完整模型
├── validation_images/                        # 驗證圖片
├── monitoring/                               # 監控圖表
└── finetuning_report_YYYYMMDD_HHMMSS.md     # 訓練報告
```

## 🔧 進階使用

### 自定義配置

```python
# 創建自定義微調配置
from day3_finetuning_config import FineTuningConfig

config_manager = FineTuningConfig()
custom_config = config_manager.create_custom_config(
    base_config="standard",
    learning_rate=2e-4,
    num_epochs=30,
## 📊 輸出結果

### 分析結果輸出
```
data/output/
├── analysis_results_YYYYMMDD_HHMMSS.json    # 詳細分析結果
├── analysis_summary_YYYYMMDD_HHMMSS.csv     # CSV 摘要
├── analysis_report_YYYYMMDD_HHMMSS.html     # HTML 報告
├── generated_images/                         # 生成的圖片
└── feature_analysis/                         # 特徵分析結果
```

### 生成圖片結果
```
data/output/generated_images/
├── single_generation/                        # 單次生成結果
├── batch_generation/                         # 批次生成結果
└── thumbnails/                               # 縮略圖
```

## 🔧 配置選項

### 系統配置
- **端口設置**: 修改 `config/default_config.yaml` 中的 `port` 設置
- **WebUI 連接**: 設置 `webui_url` 和 `webui_port`
- **模型路徑**: 配置本地模型存儲路徑
- **輸出設置**: 設置生成圖片的品質和格式

### 生成參數
- **圖片尺寸**: 512x512, 768x768, 1024x1024
- **推理步數**: 20-50 步（品質與速度平衡）
- **CFG Scale**: 7-15（提示詞遵循度）
- **採樣器**: DPM++ 2M Karras, Euler a, DDIM

## 🐛 常見問題和解決方案

### 1. WebUI API 連接失敗

**錯誤**: `Unable to connect to WebUI API`

**解決方案**:
```bash
# 確認 WebUI 已啟動並啟用 API
webui.bat --api --listen

# 檢查防火牆設置
netstat -an | findstr 7860

# 測試 API 連接
curl http://localhost:7860/sdapi/v1/options
```

### 2. 模型載入失敗

**錯誤**: `Model not found` 或載入錯誤

**解決方案**:
```bash
# 檢查模型檔案是否存在
ls models/Stable-diffusion/

# 下載基礎模型
wget https://huggingface.co/runwayml/stable-diffusion-v1-5

# 檢查 FashionCLIP 模型
python -c "from fashion_clip import FashionCLIP; print('FashionCLIP loaded successfully')"
```

### 3. 記憶體不足

**錯誤**: `CUDA out of memory` 或系統記憶體不足

**解決方案**:
```yaml
# 在 config/default_config.yaml 中調整
batch_size: 1
max_image_size: 512
enable_memory_optimization: true
use_half_precision: true
```

### 4. 圖片處理錯誤

**錯誤**: `Image processing failed`

**解決方案**:
```python
# 檢查圖片格式和尺寸
# 支援的格式: JPG, PNG, BMP, WEBP
# 建議尺寸: 512x512 或 768x768
# 最大檔案大小: 10MB
```

## ❓ 常見疑問解答

### Q: 是否需要複製整個 stable-diffusion-webui 目錄？
**A: 不需要！** 

- **WebUI 保持原位**：`stable-diffusion-webui` 目錄保持不變，繼續在原位置運行
- **Fashion AI 獨立運行**：創建新的 `Fashion_AI_Complete_Package` 目錄
- **通過 API 連接**：Fashion AI 通過 HTTP API 與 WebUI 通信

### Q: 兩個系統如何協作？
**A: API 連接方式**

1. **WebUI 提供 API 服務**：
   ```bash
   cd stable-diffusion-webui
   webui.bat --api --listen
   # 提供 API 服務在 http://localhost:7860
   ```

2. **Fashion AI 作為客戶端**：
   ```bash
   cd Fashion_AI_Complete_Package
   python fashion_ai_main.py --webui-url http://localhost:7860
   # 作為客戶端連接到 WebUI API
   ```

### Q: 需要佔用多少硬碟空間？
**A: 空間需求**

- **WebUI 原始目錄**：約 20-50GB（包含模型）
- **Fashion AI 應用包**：約 1-2GB（只包含程式碼）
- **總計**：約 21-52GB

### Q: 可以在不同電腦上運行嗎？
**A: 可以！**

- **WebUI 伺服器**：在一台有 GPU 的電腦上運行
- **Fashion AI 客戶端**：可以在任何電腦上運行，通過網路連接

## 📈 效能優化建議

### 1. GPU 優化
```bash
# 安裝 xformers 加速
pip install xformers

# 啟用混合精度
# 在配置文件中設置 use_half_precision: true
```

### 2. 記憶體優化
```yaml
# config/default_config.yaml
memory_optimization:
  enable_attention_slicing: true
  enable_sequential_cpu_offload: true
  max_batch_size: 1
  clear_cache_after_generation: true
```

### 3. 網路優化
```python
# 啟用批次處理
batch_processing:
  enabled: true
  max_batch_size: 4
  timeout: 300
## 📚 進一步學習資源

### 相關文檔
- [Stable Diffusion WebUI API 文檔](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API)
- [FashionCLIP 論文](https://arxiv.org/abs/2204.03972)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers)

### 範例項目
- [Fashion AI 應用範例](./examples/)
- [API 整合範例](./examples/api_examples.py)
- [批次處理範例](./examples/batch_examples.py)

## 🤝 技術支持

### 獲得幫助
1. 查看 [USER_MANUAL.md](./USER_MANUAL.md) 詳細使用說明
2. 閱讀 [WEBUI_API_GUIDE.md](./WEBUI_API_GUIDE.md) API 文檔
3. 檢查 [常見問題](#常見問題和解決方案) 部分

### 問題回報
如果遇到問題，請提供：
- 詳細的錯誤訊息
- 系統配置（GPU、記憶體、作業系統）
- 操作步驟
- 相關的日誌檔案

## 📝 版本資訊

### v1.0.0 (Current)
- ✅ 完整的 Fashion AI 分析和生成功能
- ✅ WebUI API 整合
- ✅ Web 介面支持
- ✅ 批次處理功能
- ✅ RESTful API 支持
- ✅ 詳細的安裝和使用文檔

### 未來計劃
- 🔄 ControlNet 整合
- 🔄 多模型支持
- 🔄 進階批次處理
- 🔄 雲端部署支持
- 🔄 更多 AI 模型整合

## � 總結

Fashion AI 完整應用程式提供了一個完整的時尚 AI 解決方案，包括：

1. **易於安裝** - 自動化安裝腳本和詳細文檔
2. **功能齊全** - 分析、生成、批次處理等完整功能
3. **API 優先** - 以 WebUI API 為核心的設計
4. **文檔完整** - 詳細的安裝、使用和 API 文檔
5. **實用性強** - 適合個人使用、研究和商業應用

立即開始您的 Fashion AI 之旅！

---

**📞 聯繫資訊**
- 技術問題：查看文檔或提交 Issue
- 功能建議：歡迎提交 Pull Request
- 商業合作：請聯繫項目維護者

**🔗 相關連結**
- [項目主頁](./README.md)
- [安裝指南](./INSTALLATION_GUIDE.md)
- [API 文檔](./WEBUI_API_GUIDE.md)
- [使用手冊](./USER_MANUAL.md)

### 🔧 Colab 專用配置

| GPU 類型 | VRAM | 自動配置 |
|----------|------|----------|
| T4 | 16GB | LoRA rank=4, batch_size=1 |
| V100 | 16GB | LoRA rank=8, batch_size=2 |
| A100 | 40GB | LoRA rank=16, batch_size=4 |

### 📦 Colab 輸出結果

```
fashion_ai_model_YYYYMMDD_HHMMSS.zip
├── model/                    # LoRA 權重檔案
├── validation/               # 訓練過程驗證圖片
├── test_generations/         # 最終測試圖片
├── training_progress.png     # 訓練損失曲線
└── README.md                # 使用說明
```

---
