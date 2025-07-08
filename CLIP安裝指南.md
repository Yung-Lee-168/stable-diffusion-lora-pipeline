# CLIP 安裝完整指南

## 問題描述
Stable Diffusion WebUI 啟動時出現錯誤：
```
ModuleNotFoundError: No module named 'clip'
```

這是因為 k-diffusion 模組需要 OpenAI 的 CLIP 套件。

## 解決步驟

### 第 1 步：安裝 OpenAI CLIP
執行以下任一方法：

**方法 A：使用批次檔（推薦）**
```cmd
INSTALL_CLIP.bat
```

**方法 B：手動安裝**
```cmd
pip install git+https://github.com/openai/CLIP.git
```

### 第 2 步：檢查安裝狀態
```cmd
python check_clip_status.py
```

### 第 3 步：重新啟動 WebUI
```cmd
START_WEBUI.bat
```

### 第 4 步：測試 API 功能
```cmd
python check_webui_for_clip.py
```

### 第 5 步：執行完整測試
```cmd
python day2_enhanced_test.py
```

## 常見問題

### Q: Git 不可用怎麼辦？
A: 下載並安裝 Git for Windows：https://git-scm.com/download/win

### Q: 網路連線問題？
A: 嘗試使用鏡像源：
```cmd
pip install -i https://pypi.douban.com/simple/ git+https://github.com/openai/CLIP.git
```

### Q: 還是有錯誤？
A: 檢查 Python 環境：
```cmd
python --version
pip --version
```

## 驗證清單
- [ ] OpenAI CLIP 安裝成功
- [ ] WebUI 啟動無錯誤
- [ ] API 端點可訪問
- [ ] 圖像生成功能正常
- [ ] CLIP vs FashionCLIP 測試完成

## 後續步驟
安裝完成後，你可以：
1. 執行完整的 3 天測試計劃
2. 比較標準 CLIP 與 FashionCLIP 的效果
3. 開發時尚相關的 AI 應用
