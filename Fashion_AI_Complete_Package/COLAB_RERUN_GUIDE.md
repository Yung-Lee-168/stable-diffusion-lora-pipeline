# 🚀 Colab 快速重新執行指南

## 📋 重新執行步驟

### 方法 1: 完全重新執行（推薦）

1. **重新啟動運行時**
   ```
   Runtime > Restart runtime
   ```

2. **重新執行所有單元格**
   ```
   Runtime > Run all
   ```

3. **或者逐步執行**
   - 按 `Shift + Enter` 逐個執行單元格
   - 確保每個步驟都成功完成

### 方法 2: 使用更新的腳本

1. **上傳新的腳本文件**
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```

2. **直接執行腳本**
   ```python
   exec(open('day3_colab_finetuning.py').read())
   ```

### 方法 3: 直接複製程式碼

將 `day3_colab_finetuning.py` 的內容直接複製到 Colab 單元格中執行。

## 🔧 執行前檢查清單

- [ ] 確認 GPU 已啟用
- [ ] 檢查 Python 版本 (建議 3.8+)
- [ ] 確認網路連接正常
- [ ] 有足夠的 Google Drive 空間（如需保存模型）

## 🎯 執行模式選擇

### 快速測試模式
```python
# 在腳本中選擇模式 1
mode = "1"  # 快速測試
```

### 完整訓練模式
```python
# 在腳本中選擇模式 2
mode = "2"  # 完整訓練
```

### 範例數據模式
```python
# 在腳本中選擇模式 3
mode = "3"  # 使用範例數據
```

## 📦 依賴管理

### 自動修復（推薦）
腳本會自動檢查並修復依賴衝突：

```python
# 自動執行依賴修復
setup_success = setup_colab_environment()
```

### 手動修復
如果自動修復失敗，可以手動執行：

```bash
# 重新安裝 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 重新安裝 Transformers
pip install transformers==4.21.0

# 重新安裝其他依賴
pip install diffusers accelerate peft
```

## 🚨 常見問題解決

### 問題 1: 依賴衝突
**解決方案:**
1. 重新啟動運行時
2. 重新執行安裝步驟
3. 使用固定版本的依賴

### 問題 2: GPU 記憶體不足
**解決方案:**
```python
# 減少 batch size
config = {
    "train_batch_size": 1,  # 降低到 1
    "gradient_accumulation_steps": 4  # 增加累積步驟
}
```

### 問題 3: 模型下載失敗
**解決方案:**
1. 檢查網路連接
2. 重新執行下載步驟
3. 使用鏡像源

### 問題 4: 腳本執行中斷
**解決方案:**
1. 檢查 Colab 會話是否過期
2. 重新連接並繼續執行
3. 保存中間結果

## 💡 最佳實踐

### 1. 定期保存
```python
# 定期保存到 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 保存模型
torch.save(model.state_dict(), '/content/drive/MyDrive/fashion_ai_model.pth')
```

### 2. 監控資源使用
```python
# 檢查記憶體使用
import psutil
print(f"記憶體使用: {psutil.virtual_memory().percent}%")

# 檢查 GPU 使用
import torch
if torch.cuda.is_available():
    print(f"GPU 記憶體: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
```

### 3. 錯誤日誌
```python
# 啟用詳細日誌
import logging
logging.basicConfig(level=logging.INFO)
```

## 📞 取得幫助

如果遇到問題：
1. 查看 [完整說明文檔](README.md)
2. 參考 [依賴修復指南](COLAB_DEPENDENCY_FIX.md)
3. 檢查 [常見問題解答](FAQ.md)

## 🔄 版本更新

定期檢查是否有新版本：
```python
# 檢查最新版本
import requests
response = requests.get('https://api.github.com/repos/your-repo/releases/latest')
print(f"最新版本: {response.json()['tag_name']}")
```

---

**💡 提示**: 建議在重新執行前先備份重要數據，並確保有足夠的時間完成整個訓練過程。
