# 🎨 SD v1.5 Colab Fine-tuning Package

這個資料夾包含所有在 Google Colab 上微調 Stable Diffusion v1.5 的必要檔案。

## 📁 資料夾結構

```
SD_v1.5_Colab_FineTuning/
├── README.md                              # 主要說明文件
├── main_training_script.py                # 主要訓練腳本
├── colab_environment_setup.py             # 環境設置腳本
├── dependency_fix.py                      # 依賴修復腳本
├── notebooks/                             # Jupyter Notebooks
│   ├── SD_v1.5_Complete_Training.ipynb    # 完整訓練 Notebook
│   ├── Quick_Test.ipynb                   # 快速測試 Notebook
│   └── Environment_Setup.ipynb            # 環境設置 Notebook
├── configs/                               # 配置檔案
│   ├── training_config.yaml               # 訓練配置
│   ├── gpu_optimization.yaml              # GPU 優化配置
│   └── model_settings.yaml                # 模型設定
├── utils/                                 # 工具函式
│   ├── data_loader.py                     # 資料載入器
│   ├── fashion_clip_analyzer.py           # FashionCLIP 分析器
│   ├── model_manager.py                   # 模型管理器
│   └── visualization.py                   # 視覺化工具
├── examples/                              # 範例檔案
│   ├── sample_images/                     # 範例圖片
│   ├── example_prompts.txt                # 範例提示詞
│   └── training_results/                  # 訓練結果範例
├── docs/                                  # 文檔
│   ├── INSTALLATION.md                    # 安裝指南
│   ├── USAGE.md                          # 使用說明
│   ├── TROUBLESHOOTING.md                # 故障排除
│   └── FAQ.md                            # 常見問題
└── requirements.txt                       # 依賴清單
```

## 🚀 快速開始

### 方法 1: 使用 Jupyter Notebook（推薦）
1. 上傳 `notebooks/SD_v1.5_Complete_Training.ipynb` 到 Google Colab
2. 執行所有單元格
3. 等待訓練完成並下載結果

### 方法 2: 使用 Python 腳本
```python
# 在 Colab 中執行
!git clone <your-repo-url>
%cd SD_v1.5_Colab_FineTuning
!python main_training_script.py
```

### 方法 3: 手動上傳
1. 將整個資料夾上傳到 Google Drive
2. 在 Colab 中掛載 Drive
3. 執行相關腳本

## 📋 使用步驟

1. **環境檢查**: 確保 GPU 已啟用
2. **依賴安裝**: 執行自動安裝腳本
3. **模型準備**: 下載並設置 SD v1.5 模型
4. **資料準備**: 上傳訓練圖片
5. **開始訓練**: 執行微調訓練
6. **結果下載**: 下載訓練好的模型

## 🔧 系統需求

- Google Colab (建議 Pro 版本)
- GPU 運行時 (T4/V100/A100)
- 至少 15GB 可用記憶體
- 穩定的網路連接

## 💡 特色功能

- ✅ 自動環境設置與依賴修復
- ✅ 多種 GPU 配置自動優化
- ✅ LoRA 高效微調支援
- ✅ FashionCLIP 特徵提取
- ✅ 實時訓練監控
- ✅ 自動保存與備份
- ✅ 完整的錯誤處理

## 🆘 遇到問題？

1. 查看 [故障排除指南](docs/TROUBLESHOOTING.md)
2. 參考 [常見問題解答](docs/FAQ.md)
3. 檢查 [使用說明](docs/USAGE.md)

## 📞 支援

如需協助，請參考相關文檔或聯繫開發者。

---

**版本**: v1.0.0  
**更新時間**: 2025-07-04  
**兼容性**: Google Colab, SD v1.5, PyTorch 2.0+
