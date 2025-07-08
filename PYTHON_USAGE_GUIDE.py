#!/usr/bin/env python3
"""
LoRA訓練正確使用方法說明
強調在Python環境中運行的重要性
"""

def print_usage_guide():
    """打印正確的使用方法"""
    
    guide = """
🎯 LoRA訓練正確使用方法
===============================================

## ✅ 正確的運行方式

### 在Python環境中運行：
```bash
# 1. 新訓練
python auto_test_pipeline/train_lora.py --new

# 2. 繼續訓練
python auto_test_pipeline/train_lora.py --continue

# 3. 交互式選擇
python auto_test_pipeline/train_lora.py
```

## 🔧 關鍵改進

### 1. Python解釋器檢測
腳本現在會自動使用當前Python環境：
- 使用 `sys.executable` 獲取當前Python路徑
- 確保子進程使用相同的Python環境
- 避免環境不一致問題

### 2. 智能步數管理
- 自動檢測當前訓練步數
- 智能計算max_train_steps
- 避免步數衝突錯誤

### 3. 訓練停止修復
- 修復了訓練不在指定步數停止的問題
- 添加雙重break邏輯
- 確保精確控制訓練長度

## 📊 使用示例

### 場景1：第一次訓練
```bash
python auto_test_pipeline/train_lora.py --new
```
輸出示例：
```
🆕 模式：開始新的獨立訓練
🐍 使用Python解釋器: /path/to/your/python
📊 新的最大步數: 100
🚀 開始 LoRA 微調 ...
```

### 場景2：繼續現有訓練
```bash
python auto_test_pipeline/train_lora.py --continue
```
輸出示例：
```
🔄 模式：從檢查點繼續訓練
🔄 找到訓練狀態目錄: 20250708_123456
📊 當前已完成步數: 50
📊 計劃增加步數: 100
📊 新的最大步數: 150
```

## 🎯 為什麼要在Python中運行？

1. **環境一致性** - 確保主腳本和子進程使用相同Python
2. **依賴管理** - 正確加載已安裝的包
3. **路徑解析** - 避免Windows命令行的路徑問題
4. **錯誤處理** - 更好的異常捕獲和處理

## ⚠️ 避免的問題

### ❌ 不要直接從Windows運行
```cmd
# 避免這樣做
train_lora.py --new
```

### ❌ 不要混用不同Python環境
```cmd
# 避免路徑不一致
C:\\Python39\\python.exe train_lora.py  # 如果你的環境是Python 3.10
```

### ✅ 正確方式
```bash
# 在你的工作環境中
python auto_test_pipeline/train_lora.py --new
```

## 📈 最佳實踐

1. **激活正確環境** - 確保在正確的Python環境中
2. **檢查依賴** - 確認PyTorch、Diffusers等已安裝
3. **監控日誌** - 關注訓練過程和停止時機
4. **備份管理** - 腳本會自動備份現有模型

## 🔍 故障排除

### 問題：找不到train_network.py
**解決**：確保在auto_test_pipeline目錄中運行

### 問題：Python路徑錯誤
**解決**：檢查 `sys.executable` 輸出是否正確

### 問題：訓練不停止
**解決**：已修復，現在會精確在max_train_steps停止

---
現在您的LoRA訓練腳本已經優化為在Python環境中正確運行！
"""
    
    print(guide)

if __name__ == "__main__":
    print_usage_guide()
