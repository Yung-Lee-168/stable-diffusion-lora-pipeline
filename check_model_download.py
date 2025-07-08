#!/usr/bin/env python3
"""
檢查 CLIP 和 FashionCLIP 模型下載狀態
"""

import os
import torch
from transformers import CLIPModel, CLIPProcessor

def check_model_download_status():
    print("🔍 檢查模型下載狀態")
    print("=" * 60)
    
    # 檢查 HuggingFace cache 目錄
    cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
    print(f"📁 HuggingFace 快取目錄: {cache_dir}")
    print(f"📁 目錄存在: {os.path.exists(cache_dir)}")
    
    if os.path.exists(cache_dir):
        cache_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(cache_dir)
            for filename in filenames
        ) / (1024 * 1024 * 1024)  # 轉換為 GB
        print(f"📦 快取大小: {cache_size:.2f} GB")
    
    print("\n" + "=" * 60)
    print("🧪 實際測試模型載入")
    print("=" * 60)
    
    # 測試標準 CLIP
    print("1️⃣ 測試標準 CLIP (openai/clip-vit-base-patch32)")
    try:
        print("   📥 嘗試載入...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 檢查模型參數
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        model_size = param_count * 4 / 1024  # 估算 GB
        
        print(f"   ✅ 標準 CLIP 載入成功")
        print(f"   📊 參數數量: {param_count:.1f}M")
        print(f"   💾 模型大小: ~{model_size:.2f} GB")
        print(f"   🎮 推薦設備: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        # 快速功能測試
        from PIL import Image
        test_image = Image.new('RGB', (224, 224), color='red')
        test_texts = ["a red image", "a blue image"]
        
        inputs = processor(text=test_texts, images=test_image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"   🧪 功能測試: 通過")
        standard_clip_ok = True
        
    except Exception as e:
        print(f"   ❌ 標準 CLIP 載入失敗: {e}")
        standard_clip_ok = False
    
    print()
    
    # 測試 FashionCLIP
    print("2️⃣ 測試 FashionCLIP (patrickjohncyh/fashion-clip)")
    try:
        print("   📥 嘗試載入...")
        fashion_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        fashion_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        
        # 檢查模型參數
        fashion_param_count = sum(p.numel() for p in fashion_model.parameters()) / 1e6
        fashion_model_size = fashion_param_count * 4 / 1024  # 估算 GB
        
        print(f"   ✅ FashionCLIP 載入成功")
        print(f"   📊 參數數量: {fashion_param_count:.1f}M")
        print(f"   💾 模型大小: ~{fashion_model_size:.2f} GB")
        print(f"   🎮 推薦設備: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"   👗 專業優勢: 時尚圖片分析")
        
        # 快速功能測試
        test_image = Image.new('RGB', (224, 224), color='blue')
        fashion_texts = ["elegant dress", "casual outfit", "formal wear"]
        
        inputs = fashion_processor(text=fashion_texts, images=test_image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = fashion_model(**inputs)
        
        print(f"   🧪 功能測試: 通過")
        fashion_clip_ok = True
        
    except Exception as e:
        print(f"   ❌ FashionCLIP 載入失敗: {e}")
        if "connection" in str(e).lower() or "network" in str(e).lower():
            print(f"   💡 可能是網路問題，模型尚未下載")
        fashion_clip_ok = False
    
    print("\n" + "=" * 60)
    print("📋 下載狀態總結")
    print("=" * 60)
    
    print(f"標準 CLIP: {'✅ 已下載並可用' if standard_clip_ok else '❌ 未下載或有問題'}")
    print(f"FashionCLIP: {'✅ 已下載並可用' if fashion_clip_ok else '❌ 未下載或有問題'}")
    
    if standard_clip_ok and fashion_clip_ok:
        print("\n🎉 兩個模型都已準備就緒！")
        print("✅ 可以直接執行: python day2_enhanced_test.py")
    elif standard_clip_ok:
        print("\n⚠️ 只有標準 CLIP 可用")
        print("💡 程式會自動使用標準 CLIP 進行測試")
    else:
        print("\n❌ 模型載入問題")
        print("💡 請檢查網路連線並重新嘗試")
    
    print("\n" + "=" * 60)
    print("🚀 執行建議")
    print("=" * 60)
    
    if standard_clip_ok or fashion_clip_ok:
        print("1. 模型已準備就緒，可以開始測試")
        print("2. 執行命令: python day2_enhanced_test.py")
        print("3. 程式會自動選擇可用的模型進行比較")
        
        if torch.cuda.is_available():
            print("4. 🎮 GPU 加速已啟用，測試速度會更快")
        else:
            print("4. 💻 使用 CPU 模式，可能需要較長時間")
    else:
        print("1. 請確保網路連線正常")
        print("2. 重新執行此檢查腳本")
        print("3. 如果問題持續，可以先使用 transformers 的預設快取")

if __name__ == "__main__":
    check_model_download_status()
