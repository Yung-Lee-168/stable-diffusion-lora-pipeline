# 🔧 WebUI 依賴安裝指南

## ❌ 問題
```
ModuleNotFoundError: No module named 'pytorch_lightning'
```

## ✅ 解決方案

### 方法 1: 執行安裝腳本
```bash
.\INSTALL_WEBUI_DEPS.bat
```

### 方法 2: 手動安裝
在 VS Code 終端逐一執行：

```bash
# 進入 WebUI 目錄
cd "e:\Yung_Folder\Project\stable-diffusion-webui"

# 安裝關鍵依賴
pip install pytorch_lightning
pip install gradio==3.41.2
pip install fastapi>=0.90.1
pip install transformers==4.30.2
pip install safetensors

# 安裝完整依賴
pip install -r requirements.txt
```

### 方法 3: 使用 WebUI 內建啟動腳本
```bash
# WebUI 有自己的啟動腳本，會自動安裝依賴
.\webui.bat --api
```

## 🚀 啟動 WebUI
依賴安裝完成後：
```bash
python webui.py --api --listen
```

## 🎯 成功標誌
看到這個訊息表示成功：
```
Running on local URL: http://127.0.0.1:7860
```

## 🔧 故障排除

### 如果仍有模組缺失：
1. 檢查 Python 版本 (建議 3.10.x)
2. 檢查 pip 版本：`pip --version`
3. 升級 pip：`pip install --upgrade pip`
4. 清除快取：`pip cache purge`

### 如果網路問題：
```bash
# 使用國內鏡像
pip install pytorch_lightning -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 虛擬環境問題：
確認是否在正確的虛擬環境中：
```bash
# 檢查環境
where python
pip list | findstr pytorch
```
