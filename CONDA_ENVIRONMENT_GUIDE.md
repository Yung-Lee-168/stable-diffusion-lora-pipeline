# Conda環境中的LoRA訓練指南

## 🐻 為什麼需要Conda環境？

在使用LoRA訓練時，Conda環境提供了以下優勢：

1. **依賴隔離** - 避免不同項目間的包衝突
2. **版本控制** - 確保使用正確的PyTorch、CUDA版本
3. **環境重現** - 確保在不同機器上的一致性
4. **簡化管理** - 統一的包管理方式

## 🚀 快速開始

### 方法1：使用自動化腳本

雙擊運行以下任一腳本：
- `Conda環境LoRA訓練.bat` - 完整的環境管理
- `快速開始LoRA訓練.bat` - 包含環境檢查的訓練流程

### 方法2：命令行操作

```bash
# 1. 檢查環境
python conda_environment_checker.py

# 2. 直接訓練
python auto_test_pipeline/train_lora.py --new
```

## 🔧 環境設置

### 1. 創建專用環境

```bash
# 創建Python 3.10環境
conda create -n lora_training python=3.10 -y

# 激活環境
conda activate lora_training

# 安裝PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安裝其他依賴
pip install diffusers transformers accelerate pillow numpy tqdm
```

### 2. 使用現有環境

```bash
# 激活現有環境
conda activate your_existing_env

# 檢查依賴
python conda_environment_checker.py
```

## 🎯 修復的關鍵改進

### 1. Python解釋器檢測

修改後的`train_lora.py`現在會：
- 自動檢測當前Conda環境
- 使用`sys.executable`確保使用正確的Python
- 顯示詳細的環境信息

```python
# 自動使用當前Python解釋器
python_executable = sys.executable
print(f"🐍 使用Python解釋器: {python_executable}")

cmd_parts = [
    f'"{python_executable}" train_network.py',  # 使用當前環境的Python
    # ... 其他參數
]
```

### 2. 環境檢查功能

新增的`check_conda_environment()`函數會檢查：
- Python版本和路徑
- Conda環境名稱
- 關鍵依賴版本
- CUDA可用性和GPU信息

### 3. 自動化工具

提供了多個工具確保環境正確：
- `conda_environment_checker.py` - 全面環境檢查
- `Conda環境LoRA訓練.bat` - 一鍵環境管理
- 修改後的訓練腳本 - 自動環境適配

## 📊 環境驗證

運行訓練前，系統會自動顯示：

```
🔍 檢查Python環境...
🐍 Python解釋器: C:\Users\...\anaconda3\envs\lora\python.exe
🐻 Conda環境: lora_training
📊 Python版本: 3.10.12
🔥 PyTorch版本: 2.1.0+cu118
🎮 CUDA可用: True
📱 GPU數量: 1
🎨 Diffusers版本: 0.21.4
```

## 🛠️ 故障排除

### 問題1：找不到CUDA
```bash
# 重新安裝PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 問題2：環境未激活
```bash
# 確認當前環境
conda info --envs

# 激活正確環境
conda activate your_env_name
```

### 問題3：依賴衝突
```bash
# 創建全新環境
conda create -n lora_clean python=3.10 -y
conda activate lora_clean
# 重新安裝所有依賴
```

## 📈 性能優化建議

1. **使用專用環境** - 為LoRA訓練創建專門的Conda環境
2. **固定版本** - 記錄working的依賴版本
3. **CUDA優化** - 確保PyTorch版本與CUDA驅動兼容
4. **內存管理** - 在環境中設置適當的內存限制

## ⭐ 最佳實踐

1. **環境文件** - 導出環境配置用於重現
   ```bash
   conda env export > lora_environment.yml
   conda env create -f lora_environment.yml
   ```

2. **版本鎖定** - 使用requirements.txt鎖定版本
   ```bash
   pip freeze > requirements.txt
   pip install -r requirements.txt
   ```

3. **定期清理** - 清理不需要的環境
   ```bash
   conda env list
   conda env remove -n unused_env
   ```

---

通過這些改進，您的LoRA訓練現在完全兼容Conda環境，確保在任何設置下都能穩定運行！
