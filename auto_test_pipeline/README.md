# LoRA 調優與分析完整系統

這是一個完整的 LoRA 模型調優、分析與優化系統，提供從訓練到評估的全自動化流程。

## 🎯 技術指標追蹤系統

### 💡 核心優勢
我們的系統能夠在 LoRA 調教過程中產出**完整的技術性指標**來追蹤模型進步：

- **🎯 三基準點評估**：總損失、視覺相似度、FashionCLIP 語意相似度、色彩相似度
- **🚀 LoRA 調優指標**：訓練效率、生成品質、特徵保持、整體分數
- **📈 即時監控**：趨勢分析、性能分布、預警系統、基準值比較
- **🔧 自動化建議**：參數調整建議、下一輪調優目標、改善策略

### 📊 快速查看指標
```bash
# 一鍵查看最新技術指標
查看技術指標.bat

# 或使用命令列
python quick_metrics_viewer.py --latest
```

### 🚀 智能訓練監控
```bash
# 使用智能訓練系統（推薦）
智能調教系統.bat

# 或手動執行監控訓練
python train_lora_monitored.py
```

我們的系統在 **訓練階段就開始監控技術指標**，能夠：
- 即時追蹤損失函數和學習率變化
- 檢測訓練異常（損失突增、學習率異常）
- 提供早停建議
- **基於訓練表現自動決定是否繼續推理**

## 🎯 訓練過程即時監控

### 💡 核心創新
我們的系統在 **train_network.py 訓練階段就開始監控技術指標**，而不是等到訓練完成後才分析：

#### 訓練中即時指標
- **損失函數追蹤**：即時監控 loss 變化趨勢
- **學習率監控**：追蹤學習率調整情況
- **收斂性分析**：評估訓練穩定性
- **異常檢測**：自動檢測損失突增、學習率異常
- **早停建議**：基於指標變化提供早停建議

#### 智能決策機制
```
訓練監控 → 性能評估 → 自動決策
    ↓           ↓          ↓
即時指標    LoRA 專業    是否繼續
追蹤      評估標準     推理階段
```

#### 決策邏輯
- **優秀 (Excellent)**：自動繼續推理
- **良好 (Good)**：建議繼續推理
- **一般 (Average)**：根據改善幅度決定
- **差 (Poor)**：建議調整參數重新訓練

### 🚀 使用智能監控訓練

#### 1. 啟動智能訓練
```bash
# 使用完整監控功能
python train_lora_monitored.py

# 或使用便捷啟動器
智能調教系統.bat
```

#### 2. 監控輸出範例
```
🚀 開始 LoRA 訓練...
📊 Step 50: Loss=0.8234, LR=0.000100
📊 Step 100: Loss=0.6789, LR=0.000095
🎯 損失改善: 0.1445, 新最佳: 0.6789
📊 Step 150: Loss=0.5432, LR=0.000090
⚠️ 收斂緩慢，建議檢查數據質量或調整優化器
📊 Step 200: Loss=0.4567, LR=0.000085
✅ 已達到目標損失，可以進行推理測試
```

#### 3. 自動決策
```
🎯 評估結果: GOOD
📊 決策: 繼續推理
💡 原因: 訓練表現良好 (損失改善: 0.3667)
```

### 📊 訓練記錄整合

訓練過程中的所有指標都會自動記錄到 `training_logs/` 目錄：
- `training_report_*.json` - 訓練詳細記錄
- `training_chart_*.png` - 訓練過程圖表
- `training_checkpoint_*.json` - 中間檢查點

這些記錄會在 `analyze_results.py` 中自動整合到最終報告中。

## 🎯 系統特色

### 核心功能
- **🚀 自動化訓練**：完整的 LoRA 訓練流程，支援自訂參數
- **🎨 智能推理**：基於訓練數據自動生成測試圖片
- **📊 深度分析**：整合 SSIM、FashionCLIP 等多種評估指標
- **🔧 智能優化**：基於分析結果自動調整訓練參數
- **📈 即時監控**：監控訓練進度和性能指標變化
- **🎯 多輪迭代**：支援多輪自動化調優直到達到目標

### 評估標準
- **三基準點評估**：參考 day3_fashion_training.py 的評估標準
- **LoRA 調優指標**：訓練效率、生成品質、特徵保持
- **FashionCLIP 語意分析**：專業的時尚特徵理解
- **視覺相似度評估**：基於 SSIM 的圖像品質評估

## 📁 檔案結構

```
auto_test_pipeline/
├── 🚀 核心腳本
│   ├── train_lora.py              # 基本 LoRA 訓練
│   ├── train_lora_monitored.py    # 智能監控訓練 (推薦)
│   ├── infer_lora.py              # 推理測試
│   ├── analyze_results.py         # 結果分析 (含訓練進度)
│   ├── generate_caption_fashionclip.py  # FashionCLIP 標籤生成
│   └── lora_optimization_pipeline.py    # 完整流程
├── 🔧 優化工具
│   ├── training_progress_monitor.py # 訓練過程監控器 ⭐
│   ├── lora_tuning_optimizer.py   # 參數優化器
│   ├── lora_tuning_monitor.py     # 監控儀表板
│   └── quick_metrics_viewer.py    # 快速指標查看器
├── 📊 啟動工具
│   ├── 智能調教系統.bat           # 智能訓練系統 (推薦) ⭐
│   ├── start_lora_optimization.bat # 基本啟動器
│   ├── 查看技術指標.bat           # 指標查看器
│   └── README.md                  # 使用說明
├── 📋 文檔
│   └── 技術指標追蹤指南.md        # 完整指標說明
├── 📂 訓練數據
│   └── lora_train_set/10_test/    # 訓練圖片目錄
├── 📂 訓練記錄
│   └── training_logs/             # 訓練過程記錄 ⭐
├── 📂 測試結果
│   └── test_results/              # 結果報告目錄
└── 📂 優化配置
    └── optimization_configs/      # 參數配置目錄
```

## 🚀 快速開始

### 方法1：使用智能訓練系統（推薦）
```batch
# 雙擊執行 - 自動監控訓練 + 智能決策
智能調教系統.bat
```

### 方法2：使用基本啟動器
```batch
# 雙擊執行 - 傳統流程
start_lora_optimization.bat
```

### 方法3：命令列執行
```bash
# 智能監控訓練（推薦）
python train_lora_monitored.py

# 完整自動化流程
python lora_optimization_pipeline.py --max_iterations 5 --target_overall 0.7

# 單次測試（基本模式）
python train_lora.py && python infer_lora.py && python analyze_results.py

# 分析現有結果
python analyze_results.py

# 監控現有訓練
python training_progress_monitor.py --training-command "python train_network.py"

# 生成優化配置
python lora_tuning_optimizer.py

# 監控儀表板
python lora_tuning_monitor.py --mode dashboard
```

## 📊 使用流程

### 1. 準備訓練數據
```
lora_train_set/10_test/
├── image1.jpg
├── image2.jpg
├── ...
└── image10.jpg
```

### 2. 執行調優流程
1. **訓練階段**：使用 LoRA 技術訓練模型
2. **推理階段**：生成測試圖片
3. **分析階段**：計算各種評估指標
4. **優化階段**：基於結果調整參數
5. **迭代階段**：重複流程直到達到目標

### 3. 查看結果
- **HTML 報告**：`test_results/training_report_*.html`
- **JSON 數據**：`test_results/training_report_*.json`
- **圖表分析**：`test_results/training_charts_*.png`
- **優化建議**：`optimization_configs/optimization_report_*.md`

## 🔧 高級配置

### 自訂訓練參數
```python
# 在 train_lora.py 中修改
TRAINING_CONFIG = {
    "learning_rate": 0.0005,      # 學習率
    "max_train_steps": 100,       # 訓練步數
    "resolution": "512x512",      # 解析度
    "batch_size": 1,              # 批次大小
    "network_dim": 32,            # 網路維度
    "network_alpha": 32,          # 網路 Alpha
}
```

### 調整目標指標
```python
# 在 lora_optimization_pipeline.py 中修改
target_metrics = {
    "total_loss": 0.4,                    # 總損失目標
    "visual_similarity": 0.5,             # 視覺相似度目標
    "fashion_clip_similarity": 0.6,       # FashionCLIP 相似度目標
    "overall_score": 0.7                  # 整體分數目標
}
```

### 自訂評估權重
```python
# 在 analyze_results.py 中修改
weights = {
    "visual": 0.2,        # 視覺權重
    "fashion_clip": 0.6,  # FashionCLIP 權重
    "color": 0.2          # 色彩權重
}
```

## 📊 技術指標追蹤

### 🎯 指標類型
我們的系統提供業界最完整的 LoRA 調教技術指標追蹤：

#### 1. 三基準點性能評估
| 指標 | 參考值 | 優秀 | 良好 | 一般 | 追蹤方式 |
|------|--------|------|------|------|----------|
| 總損失 | 0.709 | ≤0.3 | ≤0.5 | ≤0.7 | 越低越好 |
| 視覺相似度 (SSIM) | 0.326 | ≥0.7 | ≥0.5 | ≥0.3 | 越高越好 |
| FashionCLIP 語意相似度 | 0.523 | ≥0.7 | ≥0.5 | ≥0.3 | 越高越好 |
| 色彩相似度 | 0.012 | ≥0.8 | ≥0.6 | ≥0.4 | 越高越好 |

#### 2. LoRA 調優專業指標
- **訓練效率**：模型大小/訓練步數比率
- **生成品質**：基於 SSIM 的視覺品質評估
- **特徵保持**：基於 FashionCLIP 的語意保持能力
- **整體調優分數**：綜合所有指標的加權平均

#### 3. 即時監控指標
- **趨勢分析**：多輪調教的進步/退步趨勢
- **性能分布**：優秀/良好/一般/待改善的比例
- **預警系統**：指標異常時自動警告
- **基準值比較**：與參考標準的實時差異

### 📈 技術指標查看方式

#### 快速查看最新指標
```bash
# 使用 Windows 啟動器
查看技術指標.bat

# 或使用命令列
python quick_metrics_viewer.py --latest
```

#### 查看歷史趨勢
```bash
# 查看最近 5 次調教的趨勢
python quick_metrics_viewer.py --history

# 比較多輪結果
python quick_metrics_viewer.py --compare
```

#### 即時監控
```bash
# 開始即時監控（每 30 秒更新）
python quick_metrics_viewer.py --monitor

# 或使用專業監控工具
python lora_tuning_monitor.py --mode dashboard
```

### 🎯 指標追蹤輸出

#### 控制台輸出範例
```
🎯 三基準點性能評估...
  📊 image_1.jpg: EXCELLENT
     總損失: 0.245, 視覺: 0.723, FashionCLIP: 0.645
  📊 image_2.jpg: GOOD
     總損失: 0.387, 視覺: 0.567, FashionCLIP: 0.589

🎯 LoRA 調優指標...
  📊 訓練效率: EXCELLENT (比率: 0.142)
  🎨 生成品質: GOOD (SSIM: 0.623)
  🎯 特徵保持: GOOD (FashionCLIP: 0.578)
🎯 整體調優分數: 0.870 (EXCELLENT)
```

#### 報告檔案
- **JSON 數據**：`test_results/training_report_*.json`
- **HTML 報告**：`test_results/training_report_*.html`
- **監控圖表**：`test_results/training_charts_*.png`
- **即時儀表板**：`test_results/tuning_dashboard_*.png`

## 📈 監控與分析

### 即時監控
```bash
# 啟動監控
python lora_tuning_monitor.py --mode monitor --interval 30

# 生成儀表板
python lora_tuning_monitor.py --mode dashboard --output dashboard.png
```

### 分析報告
系統自動生成多種格式的分析報告：

1. **HTML 報告**：包含圖表和視覺化分析
2. **JSON 數據**：結構化的分析結果
3. **Markdown 總結**：人類易讀的總結報告
4. **PNG 圖表**：性能指標趨勢圖

## 🎯 評估指標說明

### 基礎指標
- **總損失**：綜合訓練損失（越低越好）
- **視覺相似度**：SSIM 計算的圖像相似度
- **FashionCLIP 相似度**：語意特徵相似度
- **色彩相似度**：色彩分布相似度

### 進階指標
- **訓練效率**：模型大小與訓練步數的比值
- **生成品質**：基於多個指標的綜合評估
- **特徵保持**：特徵一致性評估
- **整體分數**：加權平均的綜合評分

### 三基準點評估
參考 day3_fashion_training.py 的評估標準：
- **優秀**：總損失 < 0.3，各相似度 > 0.7
- **良好**：總損失 < 0.5，各相似度 > 0.5
- **一般**：總損失 < 0.7，各相似度 > 0.3
- **待改善**：低於一般標準

## 🔧 故障排除

### 常見問題

1. **訓練失敗**
   - 檢查訓練數據是否存在
   - 確認 GPU 記憶體足夠
   - 降低批次大小或解析度

2. **推理失敗**
   - 確認模型文件已生成
   - 檢查 WebUI 是否運行
   - 驗證 API 連接

3. **分析失敗**
   - 確認測試圖片已生成
   - 檢查 FashionCLIP 模型載入
   - 驗證相依套件安裝

4. **優化建議不準確**
   - 增加訓練數據數量
   - 調整評估權重
   - 檢查目標指標設定

### 依賴套件
```bash
# 安裝必要套件
pip install torch torchvision transformers
pip install opencv-python scikit-image matplotlib
pip install numpy pillow requests
```

## 💡 最佳實踐

### 訓練數據準備
1. **圖片數量**：建議 10-50 張高品質圖片
2. **圖片解析度**：512x512 或 768x768
3. **圖片品質**：清晰、光線充足、主體明確
4. **標籤描述**：使用 FashionCLIP 生成精確標籤

### 參數調整策略
1. **學習率**：從 0.0005 開始，根據收斂情況調整
2. **訓練步數**：100-200 步通常足夠
3. **網路維度**：32-64，過大可能過擬合
4. **解析度**：平衡品質與效率

### 多輪調優
1. **第一輪**：使用預設參數建立基線
2. **第二輪**：根據分析結果調整主要參數
3. **第三輪**：微調權重和細節參數
4. **後續輪**：逐步優化直到收斂

## 🎉 進階功能

### 自動化腳本
系統提供完整的自動化功能，可以：
- 自動執行多輪調優
- 自動選擇最佳參數
- 自動生成優化建議
- 自動保存最佳模型

### 監控預警
- 性能指標異常自動預警
- 訓練進度即時追蹤
- 歷史趨勢分析
- 自動建議調整方向

### 報告系統
- 多格式報告輸出
- 詳細的性能分析
- 視覺化圖表
- 改善建議與行動計劃

## 📞 支援與回饋

如果您在使用過程中遇到問題或有改善建議，歡迎提出回饋。系統會持續優化以提供更好的使用體驗。

---

**🎯 祝您調優成功！**
