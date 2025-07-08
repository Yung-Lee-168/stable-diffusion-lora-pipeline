import os
import sys
import subprocess
import json
import warnings
from datetime import datetime

# 減少警告訊息
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*xformers.*")
warnings.filterwarnings("ignore", message=".*triton.*")

# 確保在腳本所在目錄執行
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"📁 切換到腳本目錄: {script_dir}")

def find_latest_lora():
    """找到最新的 LoRA 模型檔案"""
    lora_path = "lora_output"
    if not os.path.exists(lora_path):
        print("Cannot find LoRA output folder")
        return None
    
    lora_files = [f for f in os.listdir(lora_path) if f.endswith('.safetensors')]
    if not lora_files:
        print("No LoRA model files found")
        return None
    
    # 找最新的檔案
    latest_lora = max(lora_files, key=lambda x: os.path.getmtime(os.path.join(lora_path, x)))
    lora_full_path = os.path.join(lora_path, latest_lora)
    
    print(f"Found LoRA model: {latest_lora}")
    print(f"File size: {os.path.getsize(lora_full_path) / (1024*1024):.2f} MB")
    
    return lora_full_path

def generate_test_images():
    """使用 LoRA 產生測試圖片"""
    
    print("Starting test image generation...")
    
    # 檢查 LoRA 模型
    lora_path = find_latest_lora()
    if not lora_path:
        return False
    
    # 讀取原始訓練資料的提示詞
    train_data_dir = "lora_train_set/10_test"
    
    if not os.path.exists(train_data_dir):
        print(f"Cannot find training data directory: {train_data_dir}")
        return False
    
    # 獲取所有文字檔案
    txt_files = [f for f in os.listdir(train_data_dir) if f.endswith('.txt')]
    txt_files.sort()  # 確保順序一致
    
    test_prompts = []
    for txt_file in txt_files:
        txt_path = os.path.join(train_data_dir, txt_file)
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # 加上 LoRA 觸發詞
            prompt = f"test, {content}"
            test_prompts.append(prompt)
    
    print(f"Found {len(test_prompts)} training prompts to test")
    
    # 建立輸出資料夾
    output_dir = "test_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # 建立測試腳本 - 正確載入 LoRA 權重
    test_script = """# -*- coding: utf-8 -*-
import torch
from diffusers import StableDiffusionPipeline
from diffusers.loaders import LoraLoaderMixin
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# 設定編碼環境
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# 禁用 xFormers 相關警告
os.environ['DISABLE_XFORMERS'] = '1'
os.environ['XFORMERS_MORE_DETAILS'] = '0'

def generate_with_lora(prompt, lora_path, output_path):
    try:
        print("Loading model...")
        
        # 設定設備
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # 使用線上模型
        model_path = "runwayml/stable-diffusion-v1-5"
        print(f"Loading base model: {model_path}")
        
        # 載入基礎管線
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True,
            variant="fp16" if device == "cuda" else None
        )
        
        pipe = pipe.to(device)
        
        # 啟用記憶體優化
        try:
            pipe.enable_attention_slicing()
            if device == "cuda":
                pipe.enable_model_cpu_offload()
        except Exception as e:
            print(f"Memory optimization warning: {e}")
            pass
        
        # 載入 LoRA 權重
        print(f"Loading LoRA weights: {os.path.basename(lora_path)}")
        try:
            # 使用 diffusers 的 LoRA 載入功能
            pipe.load_lora_weights(".", weight_name=os.path.basename(lora_path))
            print("LoRA weights loaded successfully")
        except Exception as e:
            print(f"Failed to load LoRA weights: {e}")
            print("Continuing with base model only...")
        
        print("Generating image...")
        
        # 生成圖片 - 使用 LoRA 觸發詞
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                num_inference_steps=30,  # 使用更多步數展示 LoRA 效果
                guidance_scale=7.5,
                width=512,
                height=512,
                generator=torch.Generator(device=device).manual_seed(42)
            ).images[0]
        
        # 儲存圖片
        image.save(output_path)
        print(f"Image saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <prompt> <lora_path> <output_path>")
        sys.exit(1)
        
    prompt = sys.argv[1]
    lora_path = sys.argv[2]
    output_path = sys.argv[3]
    
    success = generate_with_lora(prompt, lora_path, output_path)
    sys.exit(0 if success else 1)
"""
    
    # 儲存測試腳本
    script_path = "temp_generate.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    # 產生測試圖片
    successful_images = 0
    test_info = {
        "lora_model": os.path.basename(lora_path),
        "test_time": datetime.now().isoformat(),
        "prompts": [],
        "success_count": 0,
        "total_count": len(test_prompts)
    }
    
    for i, prompt in enumerate(test_prompts):
        output_path = os.path.join(output_dir, f"test_{i+1:02d}.png")
        print(f"  生成圖片 {i+1}/{len(test_prompts)}: {prompt[:50]}...")
        
        # 嘗試生成圖片
        try:
            # 設定環境變數避免編碼問題和 xFormers 警告
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            env['DISABLE_XFORMERS'] = '1'
            env['XFORMERS_MORE_DETAILS'] = '0'
            
            result = subprocess.run([
                sys.executable, script_path, prompt, lora_path, output_path
            ], capture_output=True, text=True, timeout=180, env=env, encoding='utf-8', errors='ignore')
            
            if result.returncode == 0 and os.path.exists(output_path):
                successful_images += 1
                status = "SUCCESS"
                print(f"    SUCCESS")
            else:
                status = "FAILED"
                print(f"    FAILED")
                if result.stderr:
                    print(f"    Error: {result.stderr.strip()}")
                if result.stdout:
                    print(f"    Output: {result.stdout.strip()}")
                
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
            print(f"    TIMEOUT")
        except Exception as e:
            status = f"ERROR: {str(e)}"
            print(f"    ERROR: {str(e)}")
        
        test_info["prompts"].append({
            "prompt": prompt,
            "output_file": f"test_{i+1:02d}.png",
            "status": status
        })
        
        print(f"    {status}")
    
    test_info["success_count"] = successful_images
    
    # 儲存測試資訊
    with open(os.path.join(output_dir, "test_info.json"), 'w', encoding='utf-8') as f:
        json.dump(test_info, f, indent=2, ensure_ascii=False)
    
    # 清理臨時檔案
    if os.path.exists(script_path):
        os.remove(script_path)
    
    print(f"Test image generation completed: {successful_images}/{len(test_prompts)} successful")
    return successful_images > 0

if __name__ == "__main__":
    generate_test_images()