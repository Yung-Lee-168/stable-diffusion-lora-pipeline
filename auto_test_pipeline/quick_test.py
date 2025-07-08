#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡化版圖片生成測試 - 不使用 LoRA，只測試基礎生成功能
"""
import torch
from diffusers import StableDiffusionPipeline
import os
import warnings
warnings.filterwarnings("ignore")

# 禁用 xFormers 相關警告
os.environ['DISABLE_XFORMERS'] = '1'
os.environ['XFORMERS_MORE_DETAILS'] = '0'

def quick_test():
    print("Starting quick generation test...")
    
    # 設定設備
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 使用線上模型
    model_path = "runwayml/stable-diffusion-v1-5"
    
    try:
        # 載入管線
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
        except:
            pass
        
        print("Generating test image...")
        
        # 生成一張測試圖片
        with torch.no_grad():
            image = pipe(
                prompt="a beautiful girl, portrait, high quality",
                num_inference_steps=20,
                guidance_scale=7.5,
                width=512,
                height=512
            ).images[0]
        
        # 儲存圖片
        output_path = "quick_test.png"
        image.save(output_path)
        print(f"SUCCESS! Image saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    quick_test()
