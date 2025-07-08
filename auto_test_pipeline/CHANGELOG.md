# 專案變更日誌 (CHANGELOG)

## [2.1.0] - 2025-07-08 - 重大機制修正

### 🚨 重大變更 (BREAKING CHANGES)
- **移除虛假性能指標計算**: LoRA 訓練期間不再計算 Visual/FashionCLIP/Color Loss
- **修正數據記錄格式**: 無法計算的指標記錄為 "N/A"
- **澄清 LoRA 訓練機制**: 正確理解噪聲預測損失的含義

### ✨ 新增功能 (Added)
- 精確的訓練步數控制機制
- 超時保護機制 (防止無限等待)
- 多重訓練完成檢測
- 詳細的機制說明文檔
- 誠實的數據記錄系統

### 🔧 修復 (Fixed)
- 修復訓練步數超出設定值不停止的問題
- 修復報告無法生成的問題
- 修復概念混淆導致的錯誤數據記錄
- 修復 Total Loss 計算公式錯誤

### 📚 文檔更新 (Documentation)
- `LORA_TRAINING_MECHANISM_EXPLAINED.md`: LoRA 訓練機制詳解
- `LORA_LOSS_DATA_CLARIFICATION.md`: Loss 數據含義澄清
- `TRAIN_LORA_BUG_FIX_DETAILS.md`: Bug 修復詳情
- `PROJECT_MAJOR_CHANGES_REPORT.md`: 重大變更報告

### ⚠️ 棄用警告 (Deprecated)
- 訓練期間的虛假性能指標 (將在訓練完成後單獨計算)

### 🗑️ 移除 (Removed)
- 訓練期間的虛假 Visual/FashionCLIP/Color Loss 計算
- 誤導性的性能指標顯示

---

## [2.0.3] - 2025-07-07 - Bug 修復

### 🔧 修復 (Fixed)
- 修復報告生成失敗問題
- 強化日誌解析邏輯
- 移除不存在的函數調用

---

## [2.0.2] - 2025-07-06 - 性能指標整合

### ✨ 新增功能 (Added)
- 整合三大性能指標計算
- 確保與 analyze_results.py 公式一致
- 增強 loss 監控 regex

---

## [2.0.1] - 2025-07-05 - 環境相容性

### 🔧 修復 (Fixed)
- 修正 train_network.py 不支援的參數
- 增強 Conda 環境自動適配
- 抑制 xformers 等警告訊息

---

## [2.0.0] - 2025-07-04 - 四種 Loss 記錄

### 🚨 重大變更 (BREAKING CHANGES)
- 支援詳細記錄四種 Loss 類型
- 每 10 步自動計算性能指標

### ✨ 新增功能 (Added)
- Total/Visual/FashionCLIP/Color Loss 追蹤
- 詳細 JSON/PNG 報告生成
- 性能指標即時計算

---

## [1.5.0] - 2025-07-03 - 自動化強化

### ✨ 新增功能 (Added)
- 完整的訓練、日誌、報告、備份輔助腳本
- Git 自動備份功能
- 環境檢查和快速啟動腳本

---

## [1.0.0] - 2025-07-01 - 初始版本

### ✨ 新增功能 (Added)
- 基礎 LoRA 訓練功能
- 訓練狀態監控
- 基本報告生成
- 檢查點續訓功能

---

## 版本編號說明

- **主版本號 (Major)**: 重大變更或不向後相容的修改
- **次版本號 (Minor)**: 新功能添加，向後相容
- **修訂版本號 (Patch)**: Bug 修復和小幅改進

## 變更類型說明

- 🚨 **重大變更 (BREAKING CHANGES)**: 可能影響現有使用方式
- ✨ **新增功能 (Added)**: 新的功能特性
- 🔧 **修復 (Fixed)**: Bug 修復
- 📚 **文檔更新 (Documentation)**: 文檔相關變更
- ⚠️ **棄用警告 (Deprecated)**: 將來會移除的功能
- 🗑️ **移除 (Removed)**: 已移除的功能
