# Google Colab 快速部署指南

## 🚀 一鍵部署到 Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/你的用戶名/你的倉庫/blob/main/Day3_Fashion_AI_Colab.ipynb)

## 📋 部署步驟

### 1. 準備工作
1. 確保有 Google 帳號
2. 開啟 Google Colab (colab.research.google.com)
3. 準備 10-50 張時尚圖片 (JPG/PNG 格式)

### 2. 上傳 Notebook
有三種方式上傳我們的 Notebook：

#### 方式 A: 直接上傳檔案
1. 下載 `Day3_Fashion_AI_Colab.ipynb`
2. 在 Colab 中選擇「檔案」→「上傳筆記本」
3. 選擇下載的 `.ipynb` 檔案

#### 方式 B: 從 GitHub 載入 (推薦)
1. 將代碼上傳到您的 GitHub 倉庫
2. 在 Colab 中選擇「檔案」→「在 GitHub 中開啟」
3. 輸入倉庫 URL

#### 方式 C: 從 Google Drive 載入
1. 將 `.ipynb` 檔案上傳到 Google Drive
2. 在 Colab 中選擇「檔案」→「在 Drive 中開啟」

### 3. 設置 GPU 運行時
1. 在 Colab 中點擊「執行階段」→「變更執行階段類型」
2. 「硬體加速器」選擇「GPU」
3. 建議選擇「高 RAM」(如果有 Colab Pro)

### 4. 執行訓練
按照 Notebook 中的步驟順序執行：
1. 安裝依賴套件
2. 檢查 GPU 狀態
3. 掛載 Google Drive
4. 上傳訓練圖片
5. 開始自動訓練
6. 下載結果

## ⚙️ 配置選項

### GPU 類型優化
| GPU 類型 | VRAM | 建議配置 |
|----------|------|----------|
| T4 | 16GB | LoRA rank=4, batch_size=1 |
| V100 | 16GB | LoRA rank=8, batch_size=2 |
| A100 | 40GB | LoRA rank=16, batch_size=4 |

### 訓練參數調整
```python
# 在 Notebook 中可以調整這些參數
config = {
    "num_epochs": 20,        # 訓練輪數 (10-50)
    "learning_rate": 1e-4,   # 學習率
    "lora_rank": 8,          # LoRA 複雜度 (4-16)
    "batch_size": 1,         # 批次大小
    "save_steps": 50         # 保存頻率
}
```

## 📊 預期結果

### 訓練時間
- **T4 GPU**: 約 30-60 分鐘 (20 epochs, 20 張圖片)
- **V100 GPU**: 約 20-40 分鐘
- **A100 GPU**: 約 10-20 分鐘

### 輸出檔案
訓練完成後會自動打包下載：
```
fashion_ai_model_YYYYMMDD_HHMMSS.zip
├── model/                    # LoRA 權重檔案
├── validation/               # 驗證圖片
├── test_generations/         # 測試生成圖片
├── training_progress.png     # 訓練曲線
└── README.md                # 使用說明
```

## 🔧 故障排除

### 常見問題

#### 1. 記憶體不足 (CUDA OOM)
```python
# 解決方案：減少批次大小和 LoRA rank
config = {
    "train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "lora_rank": 4
}
```

#### 2. 套件安裝失敗
```bash
# 在 Colab 中執行
!pip install --upgrade pip
!pip install -q diffusers==0.21.4 transformers==4.35.2
```

#### 3. 模型下載失敗
```python
# 設置離線模式或使用鏡像
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

#### 4. Google Drive 掛載失敗
```python
# 重新執行掛載
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### 效能優化建議

#### 1. 圖片預處理
- 建議圖片尺寸：512x512 或 768x768
- 檔案格式：JPG (較小) 或 PNG (高品質)
- 數量：20-50 張 (品質比數量重要)

#### 2. 訓練優化
```python
# 啟用混合精度和 xformers (如果可用)
config = {
    "mixed_precision": "fp16",
    "use_xformers": True,
    "gradient_checkpointing": True
}
```

#### 3. 記憶體管理
```python
# 定期清理記憶體
import gc
gc.collect()
torch.cuda.empty_cache()
```

## 💡 進階使用

### 1. 多樣本測試
```python
# 在訓練完成後測試不同提示詞
test_prompts = [
    "elegant woman in evening dress",
    "casual man in street fashion",
    "business professional outfit",
    "vintage style clothing"
]
```

### 2. 與其他 LoRA 合併
```python
# 可以將訓練好的 LoRA 與其他 LoRA 合併使用
from peft import PeftModel

# 載入多個 LoRA
model = base_model
model = PeftModel.from_pretrained(model, "fashion_lora")
model = PeftModel.from_pretrained(model, "style_lora")
```

### 3. 批次生成
```python
# 批次生成多張圖片進行比較
for i in range(5):
    image = pipeline(prompt, seed=i).images[0]
    image.save(f"generated_{i}.png")
```

## 📚 額外資源

### 學習資料
- [Diffusers 文檔](https://huggingface.co/docs/diffusers)
- [LoRA 論文](https://arxiv.org/abs/2106.09685)
- [Stable Diffusion 原理](https://arxiv.org/abs/2112.10752)

### 社群和支援
- [Hugging Face 論壇](https://discuss.huggingface.co/)
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [Discord AI 社群](https://discord.gg/hugging-face)

## 🎯 成功案例

### 典型訓練效果
經過 20-30 epochs 的訓練，您可以期望：
- 🎨 生成符合您訓練數據風格的時尚圖片
- 👔 更好的服裝細節和質感
- 🎭 保持人物姿態的自然性
- 🌈 改善色彩搭配和整體美感

### 最佳實踐
1. **圖片準備**: 選擇高品質、多樣化的時尚圖片
2. **參數調整**: 根據 GPU 類型調整配置
3. **監控訓練**: 觀察損失曲線，適時調整
4. **測試驗證**: 使用多樣化的提示詞測試效果

---

**注意**: 這個 Colab 版本專為解決本地 GPU 記憶體不足的問題而設計。即使您的筆電只有 4GB VRAM，也可以在 Colab 的 T4 GPU (16GB) 上順利完成訓練！
