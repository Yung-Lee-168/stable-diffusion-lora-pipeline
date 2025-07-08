# Fashion AI Complete Suite - 全功能時尚 AI 系統

## 🎯 系統概述

Fashion AI Complete Suite 是一個整合式時尚 AI 應用系統，結合 Stable Diffusion WebUI 和 FashionCLIP 技術，提供完整的時尚圖片分析、生成和處理功能。

### ✨ 核心功能
- 🎨 **智能圖片生成** - 基於 WebUI API 的 Text-to-Image 功能
- 🔍 **時尚特徵分析** - FashionCLIP 專業時尚圖片分析
- 📊 **風格分類與標註** - 自動識別服裝類型、風格、色彩
- 🎭 **提示詞優化** - 智能生成高品質 SD 提示詞
- 📈 **批次處理** - 大量圖片的自動化處理
- 🌐 **Web 界面** - 友好的用戶操作界面
- 📋 **報告生成** - 詳細的分析報告和統計

### 🏗️ 系統架構
```
Fashion AI Complete Suite
├── Core Engine (核心引擎)
│   ├── FashionCLIP 分析模組
│   ├── WebUI API 接口
│   └── 提示詞生成器
├── Web Interface (Web 界面)
│   ├── 圖片上傳界面
│   ├── 分析結果展示
│   └── 生成控制面板
├── Batch Processing (批次處理)
│   ├── 自動化分析
│   ├── 批次生成
│   └── 結果整理
└── Utilities (工具模組)
    ├── 配置管理
    ├── 系統監控
    └── 報告生成
```

## 📁 文件結構

```
Fashion_AI_Complete_Package/
├── 🚀 setup_and_install.py          # 自動安裝和配置
├── 🎯 fashion_ai_main.py            # 主程式啟動器
├── 🌐 fashion_web_ui.py             # Web 界面主程式
├── 📋 README.md                     # 本文件
├── 📖 INSTALLATION_GUIDE.md         # 安裝指南
├── 🔧 WEBUI_API_GUIDE.md           # WebUI API 使用指南
├── 🎮 USER_MANUAL.md               # 使用說明
├── core/                           # 核心模組
│   ├── fashion_analyzer.py         # 時尚分析引擎
│   ├── webui_connector.py          # WebUI API 連接器
│   ├── prompt_generator.py         # 提示詞生成器
│   ├── batch_processor.py          # 批次處理器
│   └── config_manager.py           # 配置管理器
├── web/                            # Web 界面
│   ├── app.py                      # Flask/FastAPI 應用
│   ├── templates/                  # HTML 模板
│   ├── static/                     # 靜態資源
│   └── api/                        # REST API
├── utils/                          # 工具模組
│   ├── system_check.py            # 系統檢查
│   ├── model_downloader.py        # 模型下載器
│   ├── report_generator.py        # 報告生成器
│   └── logger.py                  # 日誌系統
├── config/                         # 配置文件
│   ├── default_config.yaml        # 默認配置
│   ├── api_config.yaml            # API 配置
│   └── model_config.yaml          # 模型配置
├── examples/                       # 使用範例
│   ├── basic_usage.py             # 基本使用
│   ├── api_examples.py            # API 使用範例
│   ├── batch_processing.py        # 批次處理範例
│   └── sample_images/             # 範例圖片
├── data/                          # 數據目錄
│   ├── input/                     # 輸入圖片
│   ├── output/                    # 輸出結果
│   └── cache/                     # 緩存文件
└── requirements.txt                # 依賴列表
```

## 🛠️ 系統需求

### 硬體需求
- **GPU**: NVIDIA GTX 1060 6GB+ (推薦 RTX 3070+)
- **記憶體**: 16GB+ RAM
- **儲存**: 10GB+ 可用空間
- **網路**: 穩定的網路連接 (用於模型下載)

### 軟體需求
- **作業系統**: Windows 10/11, Linux, macOS
- **Python**: 3.8-3.11
- **CUDA**: 11.8+ (NVIDIA GPU)
- **Stable Diffusion WebUI**: 已安裝並運行

## ⚡ 快速開始

### 1. 自動安裝 (推薦)
```bash
# 下載並解壓套件
# 進入目錄
cd Fashion_AI_Complete_Package

# 執行自動安裝
python setup_and_install.py
```

### 2. 啟動系統
```bash
# 啟動 Stable Diffusion WebUI (在背景)
# 然後啟動 Fashion AI 系統
python fashion_ai_main.py

# 或啟動 Web 界面
python fashion_web_ui.py
```

### 3. 開始使用
- 瀏覽器開啟: http://localhost:8080
- 上傳圖片進行分析
- 使用分析結果生成新圖片
- 查看詳細報告

## 🎯 主要用途

### 1. 時尚電商
- 商品圖片自動標註
- 風格分類和推薦
- 產品描述生成

### 2. 設計創作
- 靈感圖片分析
- 風格參考生成
- 設計元素提取

### 3. 內容創作
- 時尚部落格配圖
- 社群媒體內容
- 廣告素材製作

### 4. 研究分析
- 時尚趨勢分析
- 風格演變研究
- 市場調研支援

## 🔗 相關連結

- [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [FashionCLIP](https://huggingface.co/patrickjohncyh/fashion-clip)
- [官方文檔](您的文檔連結)
- [GitHub 倉庫](您的GitHub連結)

## 📞 支援與聯繫

如有問題或建議，請：
1. 查看使用手冊 (USER_MANUAL.md)
2. 檢查常見問題 (FAQ)
3. 提交 Issue 或聯繫開發者

## 🔧 Google Colab 依賴衝突解決方案

如果在 Google Colab 中遇到依賴衝突錯誤：
```
ERROR: sentence-transformers 4.1.0 requires transformers<5.0.0,>=4.41.0, but you have transformers 4.35.2
```

### 解決方法：

1. **使用修復版本** (推薦):
   ```python
   # 上傳並運行 colab_training_fixed.py
   # 或使用 Fashion_AI_Colab_Fixed.ipynb
   ```

2. **重新執行腳本**:
   - 重新啟動運行時: `Runtime > Restart runtime`
   - 重新執行所有單元格: `Runtime > Run all`
   - 詳細步驟: 參考 [`COLAB_RERUN_GUIDE.md`](COLAB_RERUN_GUIDE.md)

3. **手動修復**:
   ```bash
   !pip uninstall -y sentence-transformers transformers
   !pip install transformers>=4.41.0 --force-reinstall
   !pip install diffusers[torch] accelerate peft
   ```

4. **詳細說明**: 
   - 基本修復: [`COLAB_DEPENDENCY_FIX.md`](COLAB_DEPENDENCY_FIX.md)
   - 複雜修復: [`COLAB_COMPLEX_DEPENDENCY_FIX.md`](COLAB_COMPLEX_DEPENDENCY_FIX.md)
   - 重新執行指南: [`COLAB_RERUN_GUIDE.md`](COLAB_RERUN_GUIDE.md)

---

**版本**: v1.0.0  
**更新日期**: 2024-07-04  
**授權**: MIT License
