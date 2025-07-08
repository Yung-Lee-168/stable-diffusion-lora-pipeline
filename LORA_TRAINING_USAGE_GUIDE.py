#!/usr/bin/env python3
"""
LoRA訓練完整流水線使用指南
包含修復後的訓練停止邏輯和統一的性能指標
"""

def print_usage_guide():
    """打印完整的使用指南"""
    
    guide = """
🎯 LoRA訓練完整流水線使用指南
==================================================

## 📋 功能完整性確認

✅ 已完成的修復和統一：
   
   1. 性能指標統一
      - SSIM計算公式統一
      - FashionCLIP標籤匹配邏輯統一
      - 顏色直方圖相似性計算統一
      - 圖像尺寸處理統一（≤512x512）

   2. 訓練停止修復
      - 修復了max_train_steps不停止的問題
      - 添加雙重break邏輯
      - 訓練現在精確在指定步數停止

   3. 智能步數管理
      - 自動檢測當前訓練步數
      - 智能計算max_train_steps
      - 避免"步數衝突"錯誤

## 🚀 推薦使用流程

### 1. 準備數據
```bash
# 確保數據目錄結構正確
auto_test_pipeline/fashion_dataset/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── captions/
    ├── image1.txt
    ├── image2.txt
    └── ...
```

### 2. 執行完整訓練
```bash
# 使用統一的訓練腳本
python day3_fashion_training.py

# 或使用詳細參數的訓練
python auto_test_pipeline/train_lora.py \\
    --base_model runwayml/stable-diffusion-v1-5 \\
    --data_dir auto_test_pipeline/fashion_dataset \\
    --output_dir auto_test_pipeline/lora_output \\
    --max_train_steps 100 \\
    --learning_rate 1e-4
```

### 3. 監控訓練過程
訓練現在會正確顯示：
```
Training completed: reached max_train_steps 100 at global_step 100
Breaking out of epoch loop: max_train_steps 100 reached
```

### 4. 評估結果
```bash
# 使用統一的評估腳本
python auto_test_pipeline/analyze_results.py
```

## 📊 性能指標說明

所有指標現在在訓練和評估中完全一致：

### SSIM (結構相似性)
- 範圍：0-1，越高越好
- 用於：圖像整體質量評估
- 圖像尺寸：min(height, width)用於SSIM計算

### FashionCLIP標籤匹配
- 範圍：0-1，越高越好  
- 用於：時尚語義準確性評估
- 權重：在總分中佔重要比例

### 顏色直方圖相似性
- 範圍：0-1，越高越好
- 用於：顏色保真度評估
- 方法：巴塔查理雅距離

## 🔧 故障排除

### 如果訓練不停止：
```bash
# 運行驗證腳本
python verify_training_stop_fix.py
```

### 如果步數衝突：
```bash
# 運行修復腳本
python auto_test_pipeline/fix_training_steps.py
```

### 如果指標不一致：
```bash
# 運行一致性檢查
python auto_test_pipeline/performance_metrics_final_confirmation.py
```

## 📁 關鍵文件位置

### 訓練相關
- `day3_fashion_training.py` - 主要訓練腳本
- `auto_test_pipeline/train_lora.py` - 詳細訓練腳本
- `auto_test_pipeline/train_network.py` - 核心訓練邏輯（已修復）

### 評估相關
- `auto_test_pipeline/analyze_results.py` - 結果分析
- `auto_test_pipeline/performance_metrics_final_confirmation.py` - 指標確認

### 工具腳本
- `auto_test_pipeline/fix_training_steps.py` - 步數修復
- `verify_training_stop_fix.py` - 停止邏輯驗證
- `training_stop_fix_summary.py` - 修復總結

## ⚠️ 重要注意事項

1. **訓練步數**：現在會精確停止，無需擔心超步問題
2. **性能指標**：訓練和評估使用相同公式，結果可靠
3. **圖像尺寸**：統一處理為≤512x512，確保一致性
4. **檢查點恢復**：智能步數計算避免衝突

## 🎉 成功標誌

訓練成功完成後，你應該看到：
- 精確在max_train_steps停止
- 生成的LoRA權重文件
- 詳細的性能評估報告
- 一致的評估指標

---
現在你的LoRA訓練流水線已經完全準備就緒！
所有問題都已修復，可以放心進行生產級訓練。
"""
    
    print(guide)

def create_quick_start_script():
    """創建快速開始腳本"""
    
    script_content = """@echo off
echo ===============================================
echo LoRA訓練快速開始
echo ===============================================

echo 1. 檢查修復狀態...
python verify_training_stop_fix.py

echo.
echo 2. 確認性能指標一致性...
python auto_test_pipeline\\performance_metrics_final_confirmation.py

echo.
echo 3. 開始LoRA訓練...
python day3_fashion_training.py

echo.
echo 4. 分析訓練結果...
python auto_test_pipeline\\analyze_results.py

echo.
echo ===============================================
echo 訓練完成！檢查結果輸出。
echo ===============================================
pause
"""
    
    with open("快速開始LoRA訓練.bat", 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("✅ 創建了快速開始腳本: 快速開始LoRA訓練.bat")

if __name__ == "__main__":
    print_usage_guide()
    create_quick_start_script()
    
    print("\n" + "="*50)
    print("🎯 使用指南總結")
    print("="*50)
    print("1. 📖 閱讀上述完整指南")
    print("2. 🚀 使用 '快速開始LoRA訓練.bat' 開始")
    print("3. 📊 監控訓練日誌確認正確停止")
    print("4. 📈 檢查評估結果的指標一致性")
    print("="*50)
