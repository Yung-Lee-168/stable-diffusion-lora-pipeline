# LoRA 訓練與自動化流程 - 完整指南

## 📋 目錄結構
```
auto_test_pipeline/
├── train_lora.py                    # 主要訓練腳本
├── train_lora_monitor.py            # 最小監控版本
├── train_lora_monitored_new.py      # 完整監控版本
├── train_lora_silent.py             # 靜默包裝器
├── infer_lora_direct.py             # 推理腳本
├── test_training_flow.py            # 測試腳本
├── test_training.bat                # 快速測試批次檔
├── run_train_lora_silent.bat        # 靜默訓練批次檔
├── lora_train_set/10_test/          # 訓練資料集
└── lora_output/                     # 輸出目錄
```

## 🎯 主要功能

### 1. train_lora.py (主要訓練腳本)
- ✅ 完整的 LoRA 訓練流程
- ✅ 自動備份現有模型
- ✅ 智能繼續訓練 (支援狀態目錄和 LoRA 檔案)
- ✅ 自動清理舊狀態目錄
- ✅ 完整的參數配置
- ✅ 錯誤處理和日誌記錄

### 2. 繼續訓練邏輯
**優先級順序：**
1. 🔄 **狀態目錄** (state_dir) - 完整斷點續訓
2. 🔄 **LoRA 檔案** - 從現有模型繼續訓練
3. 🆕 **全新訓練** - 沒有任何現有資料

### 3. 自動化特性
- 自動檢查並轉換 .JPG → .jpg
- 自動檢查圖片尺寸和資料完整性
- 自動備份現有模型為 backup_*.safetensors
- 自動清理舊的狀態目錄（保留最新的）
- 自動生成訓練報告

## 🚀 使用方法

### 基本使用
```bash
# 全新訓練
python train_lora.py --epochs 10 --no-continue

# 繼續訓練
python train_lora.py --epochs 10 --continue

# 使用監控版本
python train_lora_monitor.py --epochs 10 --continue
```

### 批次檔使用
```bash
# 快速測試 (1 epoch)
test_training.bat

# 靜默訓練 (完全無輸出)
run_train_lora_silent.bat
```

### 參數說明
```bash
python train_lora.py [選項]

必要參數：
  --epochs        訓練輪數 (預設: 10)
  
可選參數：
  --continue      從現有檢查點繼續訓練
  --no-continue   強制全新訓練
  --lr            學習率 (預設: 1e-4)
  --batch-size    批次大小 (預設: 1)
  --dim           LoRA 維度 (預設: 32)
  --alpha         LoRA alpha (預設: 16)
```

## 📊 訓練監控

### 監控功能
- 📈 **訓練進度追蹤** - 即時顯示訓練狀態
- 📊 **損失值監控** - 記錄訓練損失變化
- ⏱️ **時間統計** - 訓練時間和預估完成時間
- 📁 **檔案管理** - 自動備份和清理
- 📝 **詳細日誌** - 完整的訓練記錄

### 監控版本選擇
1. **train_lora_monitor.py** - 最小監控，基本日誌
2. **train_lora_monitored_new.py** - 完整監控，詳細分析

## 🔧 故障排除

### 常見問題

#### 1. 訓練早期退出
**原因：** subprocess 參數設定問題
**解決：** 使用與 train_lora.py 相同的 subprocess.run 設定

#### 2. resume 失敗
**原因：** 狀態目錄路徑或檔案格式問題
**解決：** 
- 檢查 lora_output 目錄權限
- 確認狀態目錄格式正確
- 清理損壞的狀態目錄

#### 3. 圖片尺寸不一致
**原因：** 訓練資料集中圖片尺寸不統一
**解決：** 自動檢查會提醒，可手動調整或使用自動縮放

#### 4. xFormers 警告
**原因：** xFormers 版本相容性問題
**解決：** 使用 train_lora_silent.py 或設定環境變數

### 環境檢查
```bash
# 檢查訓練資料
python check_training_data.py

# 檢查基礎模型
python check_base_model.py

# 快速測試
python test_training_flow.py simple
```

## 📈 效能最佳化

### 訓練參數調整
- **學習率：** 1e-4 (穩定) 到 5e-4 (快速)
- **批次大小：** 根據 GPU 記憶體調整
- **LoRA 維度：** 32 (標準) 到 128 (高品質)
- **LoRA alpha：** 通常設為 dim/2

### 資料集建議
- 圖片數量：50-200 張
- 圖片解析度：512x512 或 768x768
- 標籤品質：清晰、準確的描述
- 資料多樣性：不同角度、光線、背景

## 🎨 推理使用

### 基本推理
```bash
python infer_lora_direct.py
```

### 推理參數
- 自動載入最新的 LoRA 模型
- 支援批次生成
- 可調整 LoRA 強度 (0.0-1.0)

## 📝 總結

此 LoRA 訓練系統提供：
1. **穩定可靠** - 經過多次測試和修正
2. **自動化** - 最小化人工干預
3. **智能化** - 自動處理各種情況
4. **可監控** - 完整的訓練追蹤
5. **易使用** - 簡單的命令行介面

無論是新手還是專業用戶，都能快速上手並獲得良好的訓練效果。

---
*更新日期：2025-01-06*
*版本：v2.0 - 統一優化版*
