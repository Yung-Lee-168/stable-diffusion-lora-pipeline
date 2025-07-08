#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3: 快速相容性測試
快速檢查系統狀態和可用功能
"""

def quick_test():
    """快速測試主要功能"""
    print("🧪 Day 3 快速相容性測試")
    print("=" * 40)
    
    test_results = {}
    
    # 1. 測試基本 Python 環境
    try:
        import sys
        print(f"🐍 Python 版本: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        test_results['python'] = True
    except:
        test_results['python'] = False
    
    # 2. 測試 PyTorch
    try:
        import torch
        print(f"🔥 PyTorch: {torch.__version__}")
        print(f"📱 CUDA 可用: {'是' if torch.cuda.is_available() else '否'}")
        test_results['torch'] = True
    except ImportError:
        print("❌ PyTorch 未安裝")
        test_results['torch'] = False
    
    # 3. 測試 Transformers
    try:
        import transformers
        print(f"🤗 Transformers: {transformers.__version__}")
        
        # 測試 CLIP 導入
        from transformers import CLIPModel, CLIPProcessor
        print("✅ CLIP 模型可用")
        test_results['transformers'] = True
    except ImportError as e:
        print(f"❌ Transformers 問題: {e}")
        test_results['transformers'] = False
    
    # 4. 測試 Diffusers
    try:
        import diffusers
        print(f"🎨 Diffusers: {diffusers.__version__}")
        
        # 測試 SD Pipeline 導入
        from diffusers import StableDiffusionPipeline
        print("✅ Stable Diffusion Pipeline 可用")
        test_results['diffusers'] = True
    except ImportError as e:
        print(f"❌ Diffusers 問題: {e}")
        test_results['diffusers'] = False
    
    # 5. 測試其他必要套件
    packages = {
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn'
    }
    
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✅ {name} 可用")
            test_results[module] = True
        except ImportError:
            print(f"❌ {name} 未安裝")
            test_results[module] = False
    
    # 6. 測試我們的模組
    try:
        import day3_fashion_training
        print("✅ day3_fashion_training 可用")
        test_results['day3_fashion_training'] = True
    except ImportError as e:
        print(f"❌ day3_fashion_training 問題: {e}")
        test_results['day3_fashion_training'] = False
    
    # 7. 測試目錄結構
    import os
    if os.path.exists("day1_results"):
        print("✅ day1_results 目錄存在")
        test_results['day1_results'] = True
    else:
        print("❌ day1_results 目錄不存在")
        test_results['day1_results'] = False
    
    # 總結
    print("\n" + "=" * 40)
    print("📊 測試結果總結")
    print("=" * 40)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"總測試: {total_tests}")
    print(f"通過: {passed_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    
    # 建議
    print("\n💡 建議:")
    
    if not test_results.get('transformers', False):
        print("🔧 修復 Transformers:")
        print("   pip install --upgrade transformers>=4.37.0")
    
    if not test_results.get('diffusers', False):
        print("🔧 修復 Diffusers:")
        print("   pip install --upgrade diffusers>=0.27.0")
    
    if not test_results.get('day1_results', False):
        print("📁 創建測試目錄:")
        print("   mkdir day1_results")
        print("   # 然後放入一些測試圖片")
    
    # 確定可用的功能
    print("\n🎯 可用功能:")
    
    if test_results.get('day3_fashion_training', False):
        print("✅ 提示詞優化訓練 (day3_fashion_training.py)")
    
    if test_results.get('transformers', False):
        print("✅ FashionCLIP 特徵提取")
    
    if test_results.get('diffusers', False):
        print("✅ Stable Diffusion 生成")
        print("✅ 真正的模型微調")
    else:
        print("⚠️  Stable Diffusion 功能受限")
        print("   可以使用相容性模式: python day3_compatible_finetuning.py")
    
    return test_results

def suggest_next_steps(test_results):
    """建議下一步操作"""
    print("\n🚀 建議的下一步:")
    
    if test_results.get('day3_fashion_training', False):
        print("1. 🎯 運行提示詞優化訓練:")
        print("   python day3_integrated_launcher.py --mode prompt")
    
    if test_results.get('transformers', False) and test_results.get('diffusers', False):
        print("2. 🔧 運行完整微調訓練:")
        print("   python day3_integrated_launcher.py --mode finetune")
    else:
        print("2. 🔧 運行相容性微調:")
        print("   python day3_compatible_finetuning.py")
    
    print("3. 🔍 查看訓練監控:")
    print("   python day3_integrated_launcher.py --mode monitor")
    
    print("4. ⚙️  管理配置:")
    print("   python day3_integrated_launcher.py --mode config")

if __name__ == "__main__":
    test_results = quick_test()
    suggest_next_steps(test_results)
