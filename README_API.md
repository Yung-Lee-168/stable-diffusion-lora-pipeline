# Stable Diffusion WebUI API 完整解決方案

## 🎯 功能概述

這是一個完整的文字轉圖片 API 解決方案，讓您可以：
- **輸入文字描述** → **自動生成對應圖片**
- 透過 Python 函數呼叫或 HTTP API 使用
- 支援自定義參數和批次處理
- 自動保存生成的圖片

## 📁 檔案結構

```
├── webui-user.bat                 # Stable Diffusion WebUI 啟動檔 (已設定 API 模式)
├── text_to_image_service.py       # 核心服務 - 文字轉圖片主要功能
├── web_api_server.py              # Web API 服務器 (HTTP 接口)
├── api_usage_examples.py          # 使用範例和測試程式
├── start_api_service.bat          # 一鍵啟動和管理腳本
└── README.md                      # 說明文件 (本檔案)
```

## 🚀 快速開始

### 1. 啟動服務

```bash
# 執行管理腳本
start_api_service.bat

# 或手動啟動
webui-user.bat
```

### 2. 基本使用

```python
from text_to_image_service import text_to_image_service

# 簡單使用
result = text_to_image_service("a beautiful sunset over the ocean")

if result["success"]:
    print(f"圖片已保存: {result['saved_files'][0]}")
else:
    print(f"生成失敗: {result['error']}")
```

### 3. 進階使用

```python
result = text_to_image_service(
    prompt="a cyberpunk city at night, neon lights, highly detailed",
    negative_prompt="blurry, low quality, watermark",
    width=768,
    height=768,
    steps=30,
    cfg_scale=8
)
```

## 🌐 Web API 使用

### 啟動 Web API 服務器

```bash
python web_api_server.py
# 服務地址: http://localhost:8000
```

### HTTP API 呼叫

```bash
# 使用 curl
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful landscape"}'

# 使用 Python requests
import requests

response = requests.post('http://localhost:8000/generate', json={
    'prompt': 'a cute cat sitting on a table',
    'negative_prompt': 'blurry, low quality',
    'width': 512,
    'height': 512,
    'steps': 20
})

result = response.json()
```

## 📊 API 參數說明

### 必需參數
- **prompt** (string): 圖片描述文字

### 可選參數
- **negative_prompt** (string): 負向描述，排除不想要的元素
- **width** (int): 圖片寬度，預設 512
- **height** (int): 圖片高度，預設 512  
- **steps** (int): 生成步數，預設 20 (範圍 1-150)
- **cfg_scale** (float): 提示詞遵循度，預設 7 (範圍 1-30)
- **sampler_name** (string): 採樣器名稱，預設 "Euler"
- **seed** (int): 隨機種子，預設 -1 (隨機)

### 回應格式

```json
{
    "success": true,
    "images": ["base64_encoded_image_data"],
    "saved_files": ["generated_images/generated_20240703_143052_1.png"],
    "generation_time": 15.23,
    "parameters": {...}
}
```

## 🔧 完整使用流程

### 第一步：環境準備
1. 確保已安裝 Python 3.8+
2. 執行 `start_api_service.bat` → 選項 5 安裝必要套件

### 第二步：啟動服務器
1. 執行 `start_api_service.bat` → 選項 1
2. 等待看到 "Running on local URL: http://127.0.0.1:7860"

### 第三步：測試功能
1. 執行 `start_api_service.bat` → 選項 2 (測試連接)
2. 執行 `start_api_service.bat` → 選項 3 (運行範例)

### 第四步：整合到您的程式

```python
# 方式 1: 直接呼叫函數
from text_to_image_service import text_to_image_service

def generate_image_from_text(user_input):
    result = text_to_image_service(user_input)
    if result["success"]:
        return result["saved_files"][0]  # 回傳圖片路徑
    else:
        return None

# 方式 2: HTTP API 呼叫
import requests

def generate_via_api(prompt):
    response = requests.post('http://localhost:8000/generate', 
                           json={'prompt': prompt})
    return response.json()
```

## 💡 使用技巧

### 提示詞建議
- 使用英文描述
- 描述越詳細越好
- 加入風格關鍵字：`photorealistic`, `anime style`, `oil painting`
- 使用負向提示排除不想要的元素

### 範例提示詞

**風景類：**
```
"a serene mountain landscape at dawn, mist rising from valleys, highly detailed, 4k"
```

**人物類：**
```
"portrait of a wise old wizard, long beard, detailed eyes, fantasy art"
```

**科幻類：**
```
"futuristic cyberpunk city, neon lights, flying cars, rain, cinematic"
```

**動物類：**
```
"a cute fluffy cat sitting by a window, soft lighting, adorable"
```

### 參數調整建議

**高品質設定：**
- steps: 30-50
- cfg_scale: 7-12
- 尺寸: 768x768 或更高

**快速預覽設定：**
- steps: 15-20
- cfg_scale: 7
- 尺寸: 512x512

## 🛠️ 故障排除

### 常見問題

**Q: 顯示 "Server not ready" 錯誤**
- A: 請先啟動 `webui-user.bat` 並等待完全載入

**Q: 生成圖片品質不好**
- A: 增加 steps 參數，使用更詳細的提示詞

**Q: 生成速度很慢**
- A: 降低圖片尺寸和 steps 參數，確保使用 GPU

**Q: 記憶體不足錯誤**
- A: 降低圖片尺寸，關閉其他程式釋放記憶體

### 性能優化

1. **使用 GPU 加速** (NVIDIA 顯卡)
2. **適當的圖片尺寸** (512x512 到 1024x1024)
3. **合理的 steps 設定** (20-30 通常足夠)
4. **批次處理** 多張圖片時使用批次功能

## 📝 更多範例

查看 `api_usage_examples.py` 檔案獲得更多詳細範例：
- 簡單文字轉圖片
- 自定義參數使用
- 批次生成多張圖片
- 程式整合範例

## 🔗 相關連結

- [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [API 文檔](http://localhost:7860/docs) (服務器啟動後可存取)

## 📞 支援

如果遇到問題：
1. 檢查 `start_api_service.bat` 中的故障排除選項
2. 確認所有依賴套件已正確安裝
3. 查看控制台錯誤訊息進行診斷
