# 🔧 故障排除指南

## 🚨 常見問題與解決方案

### 1. 環境和依賴問題

#### 問題：依賴衝突錯誤
```
ERROR: sentence-transformers 4.1.0 requires transformers<5.0.0,>=4.41.0, but you have transformers 4.35.2
```

**解決方案：**
1. **重新啟動運行時**
   ```
   Runtime > Restart runtime
   ```

2. **重新執行環境設置**
   ```
   執行 Environment_Setup.ipynb
   ```

3. **手動修復**
   ```bash
   !pip uninstall -y sentence-transformers transformers torch torchvision torchaudio
   !pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
   !pip install "transformers>=4.41.0,<5.0.0" diffusers[torch] accelerate peft
   !pip install sentence-transformers
   ```

#### 問題：CUDA 不可用
```
RuntimeError: No CUDA GPUs are available
```

**解決方案：**
1. **啟用 GPU 運行時**
   ```
   Runtime > Change runtime type > Hardware accelerator > GPU
   ```

2. **重新連接**
   ```
   Runtime > Reconnect
   ```

3. **檢查 GPU 配額**
   - 確認 Colab 使用額度
   - 考慮升級到 Colab Pro

#### 問題：套件導入失敗
```
ImportError: cannot import name 'StableDiffusionPipeline'
```

**解決方案：**
1. **確認 diffusers 版本**
   ```python
   import diffusers
   print(diffusers.__version__)
   ```

2. **重新安裝**
   ```bash
   !pip install --upgrade diffusers[torch]
   ```

3. **清除緩存**
   ```bash
   !pip cache purge
   ```

### 2. 記憶體問題

#### 問題：GPU 記憶體不足
```
RuntimeError: CUDA out of memory
```

**解決方案：**
1. **調整批次大小**
   ```python
   config = {
       "train_batch_size": 1,  # 降到最小
       "gradient_accumulation_steps": 8  # 增加累積步驟
   }
   ```

2. **啟用記憶體優化**
   ```python
   config.update({
       "enable_xformers": True,
       "attention_slicing": True,
       "vae_slicing": True
   })
   ```

3. **使用較小的圖片**
   ```python
   config["image_size"] = 256  # 從 512 降到 256
   ```

4. **清理記憶體**
   ```python
   import gc
   import torch
   gc.collect()
   torch.cuda.empty_cache()
   ```

#### 問題：系統記憶體不足
```
MemoryError: Unable to allocate array
```

**解決方案：**
1. **重新啟動運行時**
2. **減少同時載入的模型**
3. **使用 CPU 卸載**
   ```python
   config["enable_cpu_offload"] = True
   ```

### 3. 模型載入問題

#### 問題：模型下載失敗
```
OSError: Can't load tokenizer for 'runwayml/stable-diffusion-v1-5'
```

**解決方案：**
1. **檢查網路連接**
2. **重新下載**
   ```python
   from diffusers import StableDiffusionPipeline
   pipe = StableDiffusionPipeline.from_pretrained(
       "runwayml/stable-diffusion-v1-5", 
       force_download=True
   )
   ```

3. **使用本地緩存**
   ```python
   import os
   os.environ["HF_HUB_OFFLINE"] = "1"  # 使用離線模式
   ```

#### 問題：模型權重載入錯誤
```
RuntimeError: Error(s) in loading state_dict
```

**解決方案：**
1. **檢查模型兼容性**
2. **重新下載模型**
3. **使用安全張量格式**
   ```python
   pipe = StableDiffusionPipeline.from_pretrained(
       "runwayml/stable-diffusion-v1-5",
       use_safetensors=True
   )
   ```

### 4. 訓練問題

#### 問題：訓練損失不下降
```
Training loss remains high or increases
```

**解決方案：**
1. **調整學習率**
   ```python
   config["learning_rate"] = 5e-5  # 降低學習率
   ```

2. **檢查數據品質**
   - 確保圖片清晰
   - 檢查標籤正確性
   - 避免過於複雜的圖片

3. **增加訓練時間**
   ```python
   config["num_epochs"] = 30  # 增加輪數
   ```

#### 問題：訓練過慢
```
Training takes too long
```

**解決方案：**
1. **啟用 xformers**
   ```bash
   !pip install xformers --index-url https://download.pytorch.org/whl/cu118
   ```

2. **使用混合精度**
   ```python
   config["mixed_precision"] = "fp16"
   ```

3. **優化數據載入**
   ```python
   config["dataloader_num_workers"] = 2  # 但 Colab 建議為 0
   ```

#### 問題：過擬合
```
Validation images look worse than training
```

**解決方案：**
1. **增加訓練數據**
2. **使用正則化**
   ```python
   config["weight_decay"] = 0.01
   ```

3. **降低 LoRA 等級**
   ```python
   config["lora_rank"] = 2
   ```

### 5. 生成問題

#### 問題：生成圖片品質差
```
Generated images are blurry or distorted
```

**解決方案：**
1. **調整推理參數**
   ```python
   image = pipe(
       prompt,
       num_inference_steps=50,  # 增加步驟
       guidance_scale=7.5,      # 調整指導強度
       width=512, height=512
   ).images[0]
   ```

2. **使用更好的調度器**
   ```python
   from diffusers import DDIMScheduler
   pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
   ```

3. **檢查訓練結果**
   - 確認訓練完成
   - 檢查最終損失值
   - 驗證模型保存

#### 問題：生成速度慢
```
Image generation takes too long
```

**解決方案：**
1. **使用 xformers**
2. **降低推理步驟**
   ```python
   num_inference_steps=20  # 從 50 降到 20
   ```

3. **使用較小的圖片**
   ```python
   width=256, height=256
   ```

### 6. 文件和存儲問題

#### 問題：Google Drive 空間不足
```
No space left on device
```

**解決方案：**
1. **清理舊文件**
2. **使用外部存儲**
3. **壓縮模型**
   ```python
   # 只保存 LoRA 權重，不保存完整模型
   config["save_full_model"] = False
   ```

#### 問題：文件上傳失敗
```
Upload timeout or connection error
```

**解決方案：**
1. **檢查網路連接**
2. **分批上傳**
3. **壓縮檔案**
   ```python
   from PIL import Image
   
   # 壓縮圖片
   def compress_image(image_path, quality=85):
       img = Image.open(image_path)
       img.save(image_path, "JPEG", quality=quality, optimize=True)
   ```

### 7. Colab 特定問題

#### 問題：會話過期
```
Session expired or disconnected
```

**解決方案：**
1. **使用 Colab Pro**
2. **定期保存檢查點**
3. **保持瀏覽器活躍**
   ```javascript
   // 在瀏覽器控制台執行
   setInterval(() => {
       document.querySelector('#connect')?.click();
   }, 60000);
   ```

#### 問題：運行時重置
```
Runtime restarted unexpectedly
```

**解決方案：**
1. **從檢查點恢復**
   ```python
   config["resume_from_checkpoint"] = "/path/to/checkpoint"
   ```

2. **檢查資源使用**
3. **分段執行**

### 8. 進階故障排除

#### 調試模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 啟用詳細錯誤信息
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
```

#### 記憶體監控
```python
import psutil
import torch

def print_memory_usage():
    # 系統記憶體
    ram = psutil.virtual_memory()
    print(f"RAM: {ram.used/1024**3:.1f}/{ram.total/1024**3:.1f} GB ({ram.percent:.1f}%)")
    
    # GPU 記憶體
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_mem:.1f}/{gpu_total:.1f} GB")

# 定期調用
print_memory_usage()
```

#### 性能分析
```python
import time
import torch

def profile_training_step():
    start_time = time.time()
    
    # 訓練步驟代碼
    
    end_time = time.time()
    step_time = end_time - start_time
    
    print(f"Training step time: {step_time:.2f}s")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
```

## 🆘 緊急恢復步驟

### 完全重置
如果遇到無法解決的問題：

1. **保存重要數據**
2. **重新啟動運行時**
   ```
   Runtime > Factory reset runtime
   ```
3. **重新執行所有設置步驟**

### 最小可行配置
如果記憶體嚴重不足：

```python
minimal_config = {
    "train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "mixed_precision": "fp16",
    "image_size": 256,
    "lora_rank": 2,
    "num_epochs": 5,
    "learning_rate": 5e-5
}
```

## 📞 取得幫助

### 檢查清單
在尋求幫助前，請確認：

- [ ] 已嘗試重新啟動運行時
- [ ] 已執行 Environment_Setup.ipynb
- [ ] 已檢查 GPU 是否啟用
- [ ] 已查看完整錯誤信息
- [ ] 已嘗試相關解決方案

### 報告問題
報告問題時請提供：

1. **錯誤信息**（完整的 traceback）
2. **系統信息**（GPU 類型、記憶體等）
3. **執行步驟**（重現問題的步驟）
4. **配置參數**（使用的設置）

---

**更新日期**：2025-07-04  
**版本**：v1.0.0
