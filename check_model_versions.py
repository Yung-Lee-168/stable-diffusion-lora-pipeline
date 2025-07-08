#!/usr/bin/env python3
"""
檢查 day2_enhanced_test.py 使用的模型版本
"""

def check_models():
    print("🔍 檢查 day2_enhanced_test.py 使用的模型版本")
    print("=" * 60)
    
    # 檢查基礎環境
    try:
        import torch
        import transformers
        print(f"✅ PyTorch 版本: {torch.__version__}")
        print(f"✅ Transformers 版本: {transformers.__version__}")
        print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")
        
    except ImportError as e:
        print(f"❌ 基礎套件缺失: {e}")
        return
    
    print("\n" + "=" * 60)
    print("📋 day2_enhanced_test.py 配置的模型版本")
    print("=" * 60)
    
    # 標準 CLIP 版本
    print("🔍 標準 CLIP:")
    print("   模型: openai/clip-vit-base-patch32")
    print("   來源: HuggingFace Transformers")
    print("   用途: 通用圖片-文字理解")
    
    # FashionCLIP 版本
    print("\n👗 FashionCLIP:")
    print("   主要模型: patrickjohncyh/fashion-clip")
    print("   備用模型: openai/clip-vit-base-patch32")
    print("   來源: HuggingFace Transformers")
    print("   用途: 專業時尚圖片分析")
    
    print("\n" + "=" * 60)
    print("⚡ 系統優化配置")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print("🎮 GPU 模式:")
        print("   設備: CUDA (RTX 3050 Ti)")
        print("   精度: float16 (節省記憶體)")
        print("   記憶體管理: device_map='auto'")
    else:
        print("💻 CPU 模式:")
        print("   設備: CPU")
        print("   精度: float32 (標準精度)")
    
    print("\n" + "=" * 60)
    print("🧪 測試實際載入")
    print("=" * 60)
    
    # 測試標準 CLIP
    try:
        print("📥 載入標準 CLIP...")
        from transformers import CLIPModel, CLIPProcessor
        
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ 標準 CLIP 載入成功")
        print(f"   參數數量: {param_count:.1f}M")
        print(f"   模型大小: ~{param_count * 4 / 1024:.1f} GB (float32)")
        
    except Exception as e:
        print(f"❌ 標準 CLIP 載入失敗: {e}")
    
    # 測試 FashionCLIP
    try:
        print("\n📥 載入 FashionCLIP...")
        fashion_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        fashion_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        
        fashion_param_count = sum(p.numel() for p in fashion_model.parameters()) / 1e6
        print(f"✅ FashionCLIP 載入成功")
        print(f"   參數數量: {fashion_param_count:.1f}M")
        print(f"   模型大小: ~{fashion_param_count * 4 / 1024:.1f} GB (float32)")
        print(f"   專業優勢: 時尚圖片分析準確度更高")
        
    except Exception as e:
        print(f"❌ FashionCLIP 載入失敗: {e}")
        print("   將回退到標準 CLIP")
    
    print("\n" + "=" * 60)
    print("🎯 執行建議")
    print("=" * 60)
    print("1. 直接執行: python day2_enhanced_test.py")
    print("2. 程式會自動選擇最適合的模型版本")
    print("3. 如果 FashionCLIP 不可用，會自動使用標準 CLIP")
    print("4. GPU 加速會自動啟用 (如果可用)")

if __name__ == "__main__":
    check_models()
