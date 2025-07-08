#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接 LoRA 推理 - 不使用子進程，避免編碼問題
"""
import os
import sys
import json
import warnings
from datetime import datetime

# 設定編碼
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['DISABLE_XFORMERS'] = '1'
os.environ['XFORMERS_MORE_DETAILS'] = '0'

warnings.filterwarnings("ignore")

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

def generate_with_lora_direct():
    """直接生成 LoRA 圖片"""
    
    print("Starting direct LoRA inference...")
    
    # 檢查 LoRA 模型
    lora_path = find_latest_lora()
    if not lora_path:
        return False
    
    try:
        import torch
        from diffusers import StableDiffusionPipeline
        
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
            requires_safety_checker=False
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
            lora_dir = os.path.dirname(lora_path)
            lora_filename = os.path.basename(lora_path)
            pipe.load_lora_weights(lora_dir, weight_name=lora_filename)
            print("LoRA weights loaded successfully")
        except Exception as e:
            print(f"Failed to load LoRA weights: {e}")
            print("Continuing with base model only...")
        
        # 建立輸出資料夾
        output_dir = "test_images"
        os.makedirs(output_dir, exist_ok=True)
        
        # 測試提示詞
        test_prompts = [
            "test, 1girl, portrait, high quality, detailed face, beautiful eyes",
            "test, 1boy, full body, anime style, standing pose",
            "test, character design, colorful, beautiful, fantasy style",
            "test, artwork, professional illustration, detailed shading",
            "test, detailed character, masterpiece, best quality, ultra detailed"
        ]
        
        successful_images = 0
        test_info = {
            "lora_model": os.path.basename(lora_path),
            "test_time": datetime.now().isoformat(),
            "prompts": [],
            "success_count": 0,
            "total_count": len(test_prompts)
        }
        
        # 生成圖片
        for i, prompt in enumerate(test_prompts):
            output_path = os.path.join(output_dir, f"test_{i+1:02d}.png")
            print(f"  Generating image {i+1}/{len(test_prompts)}: {prompt[:50]}...")
            
            try:
                print("    Generating...")
                
                # 生成圖片
                with torch.no_grad():
                    image = pipe(
                        prompt=prompt,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        width=512,
                        height=512,
                        generator=torch.Generator(device=device).manual_seed(42)
                    ).images[0]
                
                # 儲存圖片
                image.save(output_path)
                successful_images += 1
                status = "SUCCESS"
                print(f"    SUCCESS - Image saved to: {output_path}")
                
            except Exception as e:
                status = "FAILED"
                print(f"    FAILED: {str(e)}")
            
            test_info["prompts"].append({
                "prompt": prompt,
                "output_file": f"test_{i+1:02d}.png",
                "status": status
            })
        
        test_info["success_count"] = successful_images
        
        # 儲存測試資訊
        with open(os.path.join(output_dir, "test_info.json"), 'w', encoding='utf-8') as f:
            json.dump(test_info, f, indent=2, ensure_ascii=False)
        
        print(f"Test completed: {successful_images}/{len(test_prompts)} successful")
        return successful_images > 0
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    generate_with_lora_direct()
