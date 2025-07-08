# LoRA 訓練流程完整指南

## 📋 概述

本指南說明了 LoRA 訓練流程的所有改進，包括靈活步數設定、loss 日誌記錄和精確的訓練停止控制。

## 🎯 主要改進

### 1. 靈活的步數設定

#### 方法一：命令行參數
```bash
# 新訓練 500 步
python train_lora.py --new --steps 500

# 繼續訓練 200 步
python train_lora.py --continue --steps 200
```

#### 方法二：交互式設定
```bash
# 執行後會詢問步數
python train_lora.py --new
python train_lora.py --continue
```

### 2. 完整的 Loss 日誌記錄

#### 新增功能
- ✅ **TensorBoard 日誌**: 每步 loss 自動記錄到 `lora_output/logs/`
- ✅ **多種 loss 指標**: current loss, average loss, learning rate
- ✅ **視覺化支援**: 可用 TensorBoard 查看訓練曲線

#### 查看日誌方法
```bash
# 方法一：使用批處理檔案
查看訓練日誌.bat

# 方法二：手動啟動 TensorBoard
cd lora_output/logs
tensorboard --logdir .
# 然後在瀏覽器開啟 http://localhost:6006

# 方法三：使用Python腳本
python check_training_logs.py
```

### 3. 精確的訓練步數控制

#### 改進內容
- ✅ **明確的累積步數**: 設定 `gradient_accumulation_steps=1`
- ✅ **雙重檢查機制**: 在步數和epoch層級都檢查停止條件
- ✅ **立即保存**: 達到目標步數時立即保存模型並退出

#### 步數計算公式
```
每張圖片被訓練次數 = 總步數 ÷ 圖片數量
```

例如：100 張圖片，訓練 500 步
- 每張圖片會被訓練 500 ÷ 100 = 5 次

## 📁 輸出文件詳細說明

### 主要文件結構
```
lora_output/
├── last.safetensors          # 🎯 主要 LoRA 模型文件
├── logs/                     # 📊 訓練日誌目錄
│   ├── events.out.tfevents.* # TensorBoard 事件文件
│   └── ...
├── 狀態目錄_YYYYMMDD_HHMMSS/ # 🔄 訓練狀態備份
│   ├── optimizer.pt          # 優化器狀態
│   ├── train_state.json      # 訓練狀態資訊
│   ├── random_states.pkl     # 隨機數狀態
│   └── ...
└── ...

lora_output_backup/           # 🗄️ 舊模型備份
├── last_backup_YYYYMMDD_HHMMSS.safetensors
└── ...
```

### 文件說明

#### 1. 主要 LoRA 文件 (`last.safetensors`)
- **用途**: 這是最終的 LoRA 權重文件
- **大小**: 通常 10-50 MB（取決於 network_dim 設定）
- **使用方法**: 
  - 複製到 WebUI 的 `models/Lora/` 目錄
  - 在 WebUI 中直接選擇使用

#### 2. 訓練狀態目錄
- **用途**: 包含完整的訓練狀態，可用於繼續訓練
- **內容**:
  - `optimizer.pt`: 優化器狀態（Adam/AdaFactor等）
  - `train_state.json`: 當前步數、epoch等資訊
  - `random_states.pkl`: 隨機數種子狀態
  - `lr_scheduler.pt`: 學習率調度器狀態
- **使用方法**: `python train_lora.py --continue`

#### 3. TensorBoard 日誌 (`logs/`)
- **用途**: 記錄詳細的訓練過程數據
- **內容**:
  - 每步的 loss 值
  - 學習率變化
  - 其他訓練指標
- **查看方法**: 
  - `tensorboard --logdir lora_output/logs`
  - 瀏覽器開啟 `http://localhost:6006`

#### 4. 備份文件 (`lora_output_backup/`)
- **用途**: 防止意外覆蓋現有模型
- **內容**: 訓練前的舊 LoRA 文件
- **命名**: `原檔名_backup_時間戳.safetensors`

## 🔧 技術細節

### 為什麼之前沒有 loss 日誌？
**原因**: 缺少 `--logging_dir` 參數
- train_network.py 的日誌記錄功能需要明確指定日誌目錄
- 現在已在 train_lora.py 中自動添加此參數

### 為什麼訓練可能超出設定步數？
**可能原因**:
1. **累積梯度**: 之前未明確設定 `gradient_accumulation_steps`
2. **Epoch 結束處理**: 可能在 epoch 結束時進行額外操作
3. **觀察誤差**: 看到的可能是最終保存步數而非訓練步數

**解決方案**:
- 明確設定 `gradient_accumulation_steps=1`
- 雙重檢查：在步數和epoch層級都檢查停止條件
- 達到目標步數時立即保存並退出

### 步數計算邏輯
```python
# 在 train_network.py 中的控制邏輯
if global_step >= args.max_train_steps:
    logger.info(f"Training completed: reached max_train_steps {args.max_train_steps}")
    # 立即保存並退出
    break
```

## 📝 使用範例

### 範例 1: 新訓練 300 步
```bash
python train_lora.py --new --steps 300
```

### 範例 2: 繼續訓練 100 步
```bash
python train_lora.py --continue --steps 100
```

### 範例 3: 查看訓練結果
```bash
# 檢查所有輸出文件
python check_training_logs.py

# 啟動 TensorBoard
查看訓練日誌.bat
```

## 🎯 最佳實踐

1. **步數設定**:
   - 小數據集（<50張）: 200-500 步
   - 中數據集（50-200張）: 500-1000 步
   - 大數據集（>200張）: 1000+ 步

2. **訓練監控**:
   - 每次訓練後檢查 TensorBoard 日誌
   - 觀察 loss 是否正常下降
   - 避免過度訓練（loss 過低或不再下降）

3. **檔案管理**:
   - 定期清理舊的狀態目錄
   - 保留重要的備份文件
   - 為不同實驗使用不同的輸出目錄

## ❓ 常見問題

**Q: 為什麼沒看到 loss 日誌文件？**
A: 確保使用更新後的 train_lora.py，它會自動設定 `--logging_dir` 參數。

**Q: 訓練似乎沒有在指定步數停止？**
A: 檢查是否在看最終保存的檔案名稱。實際訓練會在指定步數精確停止。

**Q: 如何查看詳細的訓練曲線？**
A: 使用 `查看訓練日誌.bat` 或手動啟動 TensorBoard。

**Q: 繼續訓練時從哪個步數開始？**
A: 系統會自動從上次的檢查點繼續，並顯示當前已完成的步數。
