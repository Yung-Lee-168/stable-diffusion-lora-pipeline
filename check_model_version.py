#!/usr/bin/env python3
"""
檢查 Stable Diffusion 模型版本和信息
"""

import requests
import os
import json

def check_current_model():
    """檢查當前使用的模型"""
    try:
        print("🔍 正在檢查當前模型...")
        response = requests.get('http://localhost:7860/sdapi/v1/options', timeout=10)
        
        if response.status_code == 200:
            options = response.json()
            current_model = options.get('sd_model_checkpoint', 'Unknown')
            print(f"🎯 當前使用的模型: {current_model}")
            return current_model
        else:
            print(f"❌ 無法獲取選項: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ 連接錯誤: {e}")
        return None

def check_available_models():
    """檢查所有可用模型"""
    try:
        print("\n📋 正在獲取可用模型列表...")
        response = requests.get('http://localhost:7860/sdapi/v1/sd-models', timeout=10)
        
        if response.status_code == 200:
            models = response.json()
            print(f"找到 {len(models)} 個可用模型:")
            
            for i, model in enumerate(models, 1):
                model_name = model.get('model_name', 'Unknown')
                title = model.get('title', 'Unknown')
                filename = model.get('filename', 'Unknown')
                hash_value = model.get('hash', 'Unknown')
                
                print(f"\n   模型 {i}:")
                print(f"     名稱: {model_name}")
                if title != model_name:
                    print(f"     標題: {title}")
                print(f"     檔案: {filename}")
                print(f"     Hash: {hash_value}")
                
                # 判斷模型版本
                model_version = identify_model_version(filename, hash_value)
                print(f"     版本: {model_version}")
                
            return models
        else:
            print(f"❌ 無法獲取模型列表: HTTP {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ 獲取模型列表錯誤: {e}")
        return []

def identify_model_version(filename, hash_value):
    """根據檔名和hash識別模型版本"""
    
    # 常見的 Stable Diffusion 模型識別
    known_models = {
        'v1-5-pruned-emaonly.safetensors': 'Stable Diffusion v1.5 (EMA only)',
        'v1-5-pruned.safetensors': 'Stable Diffusion v1.5',
        'v1-4.safetensors': 'Stable Diffusion v1.4',
        'v2-1_768-ema-pruned.safetensors': 'Stable Diffusion v2.1 (768px)',
        'v2-1_512-ema-pruned.safetensors': 'Stable Diffusion v2.1 (512px)',
        'sd_xl_base_1.0.safetensors': 'Stable Diffusion XL 1.0 Base',
        'sd_xl_refiner_1.0.safetensors': 'Stable Diffusion XL 1.0 Refiner',
    }
    
    # 檢查確切的檔名
    if filename in known_models:
        return known_models[filename]
    
    # 檢查檔名包含的關鍵字
    filename_lower = filename.lower()
    
    if 'xl' in filename_lower:
        return 'Stable Diffusion XL (SDXL)'
    elif 'v2' in filename_lower or '2.1' in filename_lower:
        return 'Stable Diffusion v2.x'
    elif 'v1-5' in filename_lower or '1.5' in filename_lower:
        return 'Stable Diffusion v1.5'
    elif 'v1-4' in filename_lower or '1.4' in filename_lower:
        return 'Stable Diffusion v1.4'
    elif any(x in filename_lower for x in ['anime', 'waifu', 'nai']):
        return 'Anime/Illustration Model (基於 SD v1.x)'
    elif any(x in filename_lower for x in ['realistic', 'photo']):
        return 'Realistic Model (基於 SD v1.x)'
    else:
        return 'Custom/Unknown Model'

def check_local_models():
    """檢查本地模型檔案"""
    print("\n📁 檢查本地模型檔案...")
    
    models_dir = "models/Stable-diffusion"
    if os.path.exists(models_dir):
        files = [f for f in os.listdir(models_dir) if f.endswith(('.safetensors', '.ckpt'))]
        
        if files:
            print(f"在 {models_dir} 中找到 {len(files)} 個模型檔案:")
            
            for i, filename in enumerate(files, 1):
                filepath = os.path.join(models_dir, filename)
                file_size = os.path.getsize(filepath) / (1024**3)  # GB
                model_version = identify_model_version(filename, '')
                
                print(f"\n   檔案 {i}:")
                print(f"     檔名: {filename}")
                print(f"     大小: {file_size:.2f} GB")
                print(f"     版本: {model_version}")
        else:
            print("❌ 沒有找到模型檔案")
    else:
        print(f"❌ 模型目錄不存在: {models_dir}")

def main():
    """主函數"""
    print("🔍 Stable Diffusion 模型版本檢查器")
    print("=" * 60)
    
    # 檢查本地模型檔案
    check_local_models()
    
    print("\n" + "=" * 60)
    
    # 檢查 WebUI API
    current_model = check_current_model()
    
    if current_model:
        models = check_available_models()
        
        print("\n" + "=" * 60)
        print("📊 摘要:")
        
        if current_model != 'Unknown':
            current_version = identify_model_version(current_model, '')
            print(f"   目前使用: {current_model}")
            print(f"   模型版本: {current_version}")
        
        if models:
            print(f"   可用模型數量: {len(models)}")
    else:
        print("\n⚠️ 無法連接到 WebUI API")
        print("請確認 Stable Diffusion WebUI 是否正在運行")
    
    print("\n💡 模型版本說明:")
    print("   • SD v1.4: 原始版本，512x512")
    print("   • SD v1.5: 改進版本，512x512，最常用")
    print("   • SD v2.x: 新架構，支援 512x512 和 768x768")
    print("   • SD XL: 最新版本，1024x1024，更高品質")

if __name__ == "__main__":
    main()
