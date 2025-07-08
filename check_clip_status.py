#!/usr/bin/env python3
"""檢查 CLIP 套件安裝狀態"""

import sys

def check_clip_installation():
    """檢查各種 CLIP 套件的安裝狀態"""
    print("=== CLIP 套件安裝狀態檢查 ===\n")
    
    # 檢查 OpenAI CLIP
    try:
        import clip
        print("✓ OpenAI CLIP 已安裝")
        print(f"  版本: {getattr(clip, '__version__', '未知')}")
        
        # 測試基本功能
        try:
            model, preprocess = clip.load("ViT-B/32", device="cpu")
            print("  ✓ 模型載入測試成功")
        except Exception as e:
            print(f"  ⚠️ 模型載入測試失敗: {e}")
            
    except ImportError as e:
        print("✗ OpenAI CLIP 未安裝")
        print(f"  錯誤: {e}")
        print("  解決方案: 執行 INSTALL_CLIP.bat")
    
    # 檢查 transformers CLIP
    try:
        from transformers import CLIPModel, CLIPProcessor
        print("✓ Transformers CLIP 已安裝")
    except ImportError:
        print("✗ Transformers CLIP 未安裝")
    
    # 檢查 FashionCLIP
    try:
        from fashion_clip.fashion_clip import FashionCLIP
        print("✓ FashionCLIP 已安裝")
    except ImportError:
        print("✗ FashionCLIP 未安裝")
        print("  安裝方法: pip install fashion-clip")
    
    # 檢查 torch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA {torch.version.cuda} 可用")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠️ CUDA 不可用")
    except ImportError:
        print("✗ PyTorch 未安裝")

if __name__ == "__main__":
    check_clip_installation()
    
    print("\n=== 建議操作 ===")
    print("1. 執行 INSTALL_CLIP.bat 安裝 OpenAI CLIP")
    print("2. 重新啟動 WebUI")
    print("3. 執行 check_webui_for_clip.py 測試 API")
    print("4. 執行 day2_enhanced_test.py 進行完整測試")
