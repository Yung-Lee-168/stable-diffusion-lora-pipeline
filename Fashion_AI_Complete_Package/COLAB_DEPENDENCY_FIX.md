# Google Colab 使用指南

## 🔧 依賴衝突解決方案

### 問題描述
在 Google Colab 中安裝 Fashion AI 套件時，可能會遇到以下錯誤：
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
sentence-transformers 4.1.0 requires transformers<5.0.0,>=4.41.0, but you have transformers 4.35.2 which is incompatible.
```

### 解決方法

#### 方法一：使用修復版本腳本（推薦）
1. 使用 `colab_training_fixed.py` 腳本，它會自動處理依賴衝突
2. 在 Colab 中運行此腳本，會自動：
   - 卸載衝突的套件
   - 安裝兼容版本
   - 重新安裝必要套件

#### 方法二：手動修復
在 Colab 中依序執行以下命令：

```python
# 1. 卸載衝突套件
!pip uninstall -y sentence-transformers transformers

# 2. 安裝兼容版本
!pip install transformers>=4.41.0 --force-reinstall

# 3. 安裝其他套件
!pip install diffusers[torch] accelerate peft

# 4. 重新安裝 sentence-transformers
!pip install sentence-transformers

# 5. 可選：安裝 xformers
!pip install xformers --index-url https://download.pytorch.org/whl/cu118
```

#### 方法三：建立新的 Colab 環境
1. 使用全新的 Colab notebook
2. 在第一個 cell 中運行：
```python
!pip install transformers>=4.41.0 diffusers[torch] accelerate peft
```
3. 重新啟動運行時 (Runtime > Restart runtime)
4. 然後運行 Fashion AI 腳本

### 重要提示

1. **重新啟動運行時**：安裝套件後，務必重新啟動運行時
2. **按順序執行**：不要同時執行多個安裝命令
3. **檢查版本**：確保 transformers 版本 >= 4.41.0

### 檢查安裝是否成功

```python
import transformers
import diffusers
print(f"transformers: {transformers.__version__}")
print(f"diffusers: {diffusers.__version__}")
```

### 常見問題

#### Q: 為什麼會有依賴衝突？
A: Colab 預裝的 transformers 版本較舊，而 sentence-transformers 需要較新版本。

#### Q: 是否需要每次都修復？
A: 通常只需要在第一次使用時修復，之後的 session 會記住設置。

#### Q: 修復後還是有問題怎麼辦？
A: 嘗試使用全新的 Colab notebook，或者重新啟動運行時。

## 🚀 快速開始

使用修復版本腳本：
1. 上傳 `colab_training_fixed.py` 到 Colab
2. 運行腳本，它會自動處理所有依賴問題
3. 按照提示進行訓練

## 📝 技術細節

### 套件版本需求
- `transformers >= 4.41.0`
- `diffusers[torch]`
- `accelerate`
- `peft`
- `torch >= 2.0.0`

### GPU 記憶體需求
- T4 (16GB): 基本訓練
- V100 (16GB): 標準訓練
- A100 (40GB): 高品質訓練

### 自動優化配置
腳本會根據檢測到的 GPU 自動調整：
- 批次大小
- LoRA 參數
- 記憶體使用策略
