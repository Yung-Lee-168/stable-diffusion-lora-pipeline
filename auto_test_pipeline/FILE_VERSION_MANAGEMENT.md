# 文件版本管理說明

**創建日期:** 2025年7月8日  
**分支:** feature/realtime-generation-evaluation

## 📁 文件版本說明

### 核心訓練腳本
| 文件名 | 版本 | 功能描述 | 使用場景 |
|--------|------|----------|----------|
| `train_lora.py` | **實時評估版** | 含實時生圖評估功能 | 新功能開發和測試 |
| `train_lora_stable.py` | **穩定版備份** | 原始穩定功能 | 生產環境備用 |

### 功能差異對比

#### train_lora_stable.py (穩定版)
```python
✅ 基礎 LoRA 訓練功能
✅ 準確的 loss 記錄 (修復虛假數據問題)
✅ 精確的步數控制
✅ 完整的錯誤處理
❌ 無實時生圖評估
❌ 無真實性能指標計算
```

#### train_lora.py (實時評估版 - 開發中)
```python
✅ 包含所有穩定版功能
✨ 實時生圖評估 (新功能)
✨ 真實性能指標計算 (新功能)
✨ 訓練過程可視化 (新功能)
⚠️ 可能包含未測試的新代碼
⚠️ 需要更多計算資源
```

## 🔄 版本切換指南

### 使用穩定版本
```bash
# 如果需要可靠的訓練，使用穩定版
python train_lora_stable.py --new --steps 100
```

### 使用實時評估版本
```bash
# 如果要體驗新功能，使用開發版
python train_lora.py --new --steps 100 --enable-realtime-eval
```

### 回滾到穩定版本
```bash
# 如果新版本有問題，可以立即回滾
cp train_lora_stable.py train_lora.py
```

## 🚀 開發計劃

### 階段1: 基礎實現 (當前)
- [ ] 在 train_lora.py 中添加推理管道設置
- [ ] 實現基本的實時生圖功能
- [ ] 添加配置選項控制

### 階段2: 功能完善
- [ ] 性能優化和記憶體管理
- [ ] 錯誤處理和異常恢復
- [ ] 詳細的進度報告

### 階段3: 測試驗證
- [ ] 小規模功能測試
- [ ] 性能基準測試
- [ ] 文檔完善

## 📊 風險管理

### 數據安全
- ✅ 原始穩定版本已備份
- ✅ Git 分支隔離開發
- ✅ 定期提交和推送

### 功能回滾
```bash
# 緊急回滾方案
git checkout main  # 切回穩定分支
# 或
cp train_lora_stable.py train_lora.py  # 文件級回滾
```

### 並行維護
- 主分支 (main): 只進行緊急 Bug 修復
- 功能分支: 專注於新功能開發
- 定期同步兩個分支的 Bug 修復

## 📝 文檔策略

### 各版本對應文檔
```
穩定版文檔:
├── 現有的所有修復和機制說明文檔
├── 已驗證的使用指南
└── 問題排解指南

實時評估版文檔:
├── 新功能使用指南 (待建立)
├── 性能調優建議 (待建立)
├── 故障排除指南 (待建立)
└── 功能對比說明 (當前文檔)
```

## 🎯 提交策略

### 開發期間提交規範
```bash
# 功能開發
git commit -m "✨ feat(realtime): Add basic inference pipeline setup"

# Bug 修復
git commit -m "🔧 fix(realtime): Fix memory leak in generation loop"

# 性能優化
git commit -m "⚡ perf(realtime): Optimize GPU memory usage"

# 文檔更新
git commit -m "📚 docs(realtime): Add realtime evaluation usage guide"
```

### 里程碑標記
```bash
# 重要版本標記
git tag v2.2.0-dev-start -m "Start realtime evaluation development"
git tag v2.2.0-alpha -m "Basic realtime evaluation working"
git tag v2.2.0-beta -m "Optimized realtime evaluation"
```

**準備就緒！現在可以安全地開始開發實時生圖評估功能，同時保持穩定版本可用。** ✅
