# Fashion AI Training Suite - 精簡整合版

## 🎯 概述

這是一個專為 Stable Diffusion WebUI API 設計的時尚 AI 訓練套件，包含：
- 🎨 提示詞優化訓練
- 🔧 SD v1.5 微調 (LoRA)
- 🌐 Google Colab 支持
- 📡 WebUI API 完整整合

## 📁 打包文件結構

```
Fashion_AI_Training_Package/
├── README.md                           # 本文件
├── INSTALL.md                          # 安裝說明
├── WEBUI_API_GUIDE.md                 # WebUI API 使用指南
├── SYSTEM_STARTUP_GUIDE.md            # 系統啟動指南
├── requirements.txt                    # 依賴套件清單
├── launcher.py                         # 統一啟動器
├── core/                               # 核心程式
│   ├── __init__.py
│   ├── fashion_training.py             # 提示詞優化核心
│   ├── webui_api_client.py            # WebUI API 客戶端
│   ├── colab_finetuning.py            # Colab 微調
│   └── utils.py                        # 工具函數
├── examples/                           # 使用範例
│   ├── basic_text2img.py              # 基本文生圖範例
│   ├── fashion_optimization.py        # 時尚優化範例
│   └── batch_generation.py            # 批次生成範例
├── configs/                            # 配置文件
│   ├── default_config.json
│   └── prompt_templates.json
├── notebooks/                          # Colab Notebook
│   └── Fashion_AI_Colab.ipynb
└── docs/                               # 文檔
    ├── api_reference.md
    └── troubleshooting.md
```

## 🚀 快速開始

### 1. 安裝環境
```bash
# 詳見 INSTALL.md
pip install -r requirements.txt
```

### 2. 啟動 WebUI
```bash
# 詳見 SYSTEM_STARTUP_GUIDE.md
./webui.sh --api --listen
```

### 3. 運行訓練
```bash
python launcher.py --mode fashion_training
```

### 4. 使用 API
```python
# 詳見 WEBUI_API_GUIDE.md
from core.webui_api_client import WebUIClient
client = WebUIClient("http://localhost:7860")
image = client.text2img("elegant woman in fashion dress")
```

## 📚 文檔說明

- **INSTALL.md**: 完整安裝步驟和環境配置
- **WEBUI_API_GUIDE.md**: WebUI API 詳細使用說明和範例
- **SYSTEM_STARTUP_GUIDE.md**: 系統啟動和使用流程
- **docs/**: 詳細 API 參考和問題排除

## 🎁 主要特色

✅ **保留 WebUI 完整功能** - 完全相容現有 WebUI 生態系統
✅ **簡化核心程式** - 移除不必要的複雜功能
✅ **API 優先設計** - 專注於 text2img API 整合
✅ **詳細使用說明** - 包含安裝、啟動、API 使用完整指南
✅ **範例程式** - 提供實際可運行的使用案例

## 🆘 技術支持

如有問題請參考：
1. `docs/troubleshooting.md` - 常見問題解決
2. `WEBUI_API_GUIDE.md` - API 使用問題
3. `INSTALL.md` - 安裝問題
