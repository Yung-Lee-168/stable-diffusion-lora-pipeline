#!/usr/bin/env python3
"""
預下載 CLIP 和 FashionCLIP 模型
確保模型在執行測試前已準備就緒
"""

import os
import time
from transformers import CLIPModel, CLIPProcessor

def download_models():
    print("🚀 開始下載 CLIP 模型")
    print("=" * 60)
    print("⏳ 這可能需要幾分鐘時間，請耐心等待...")
    print()
    
    # 下載標準 CLIP
    print("1️⃣ 下載標準 CLIP (openai/clip-vit-base-patch32)")
    try:
        start_time = time.time()
        print("   📥 正在下載模型檔案...")
        
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        download_time = time.time() - start_time
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        
        print(f"   ✅ 標準 CLIP 下載成功！")
        print(f"   ⏱️ 下載時間: {download_time:.1f} 秒")
        print(f"   📊 參數數量: {param_count:.1f}M")
        print(f"   💾 大小: ~{param_count * 4 / 1024:.2f} GB")
        
        # 清理記憶體
        del model, processor
        
    except Exception as e:
        print(f"   ❌ 標準 CLIP 下載失敗: {e}")
        return False
    
    print()
    
    # 下載 FashionCLIP
    print("2️⃣ 下載 FashionCLIP (patrickjohncyh/fashion-clip)")
    try:
        start_time = time.time()
        print("   📥 正在下載專業時尚模型...")
        
        fashion_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        fashion_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        
        download_time = time.time() - start_time
        fashion_param_count = sum(p.numel() for p in fashion_model.parameters()) / 1e6
        
        print(f"   ✅ FashionCLIP 下載成功！")
        print(f"   ⏱️ 下載時間: {download_time:.1f} 秒")
        print(f"   📊 參數數量: {fashion_param_count:.1f}M")
        print(f"   💾 大小: ~{fashion_param_count * 4 / 1024:.2f} GB")
        print(f"   👗 專業領域: 時尚圖片分析")
        
        # 清理記憶體
        del fashion_model, fashion_processor
        
    except Exception as e:
        print(f"   ❌ FashionCLIP 下載失敗: {e}")
        print(f"   💡 如果網路問題，程式會自動回退到標準 CLIP")
    
    print()
    print("=" * 60)
    print("🎉 模型下載完成！")
    print("=" * 60)
    
    # 檢查快取目錄
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(cache_dir):
        cache_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(cache_dir)
            for filename in filenames
        ) / (1024 * 1024 * 1024)
        print(f"📁 模型快取位置: {cache_dir}")
        print(f"📦 總快取大小: {cache_size:.2f} GB")
    
    print()
    print("✅ 現在可以執行測試:")
    print("   python day2_enhanced_test.py")
    
    return True

if __name__ == "__main__":
    print("🔍 CLIP 模型預下載工具")
    print("確保模型在測試前已準備就緒")
    print()
    
    try:
        import torch
        import transformers
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ Transformers: {transformers.__version__}")
        print(f"✅ 網路連線: 準備下載")
        print()
        
        download_models()
        
    except ImportError as e:
        print(f"❌ 缺少必要套件: {e}")
        print("請先安裝: pip install torch transformers")
    except Exception as e:
        print(f"❌ 下載過程發生錯誤: {e}")
        print("請檢查網路連線並重試")
