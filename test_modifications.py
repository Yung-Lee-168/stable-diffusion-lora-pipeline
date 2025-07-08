#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速測試修改後的提示詞優化功能
"""

import os
import sys

def test_imports():
    """測試導入功能"""
    print("📦 測試程式導入...")
    
    try:
        from day3_fashion_training import FashionTrainingPipeline
        print("✅ day3_fashion_training 導入成功")
        
        # 測試實例化
        pipeline = FashionTrainingPipeline()
        print("✅ FashionTrainingPipeline 實例化成功")
        
        # 測試新增的方法
        if hasattr(pipeline, 'analyze_prompt_composition'):
            print("✅ analyze_prompt_composition 方法存在")
        else:
            print("❌ analyze_prompt_composition 方法缺失")
            
        if hasattr(pipeline, 'analyze_loss_performance'):
            print("✅ analyze_loss_performance 方法存在")
        else:
            print("❌ analyze_loss_performance 方法缺失")
            
        return True
        
    except Exception as e:
        print(f"❌ 導入失敗: {e}")
        return False

def test_configuration():
    """測試配置功能"""
    print("\n⚙️ 測試配置功能...")
    
    try:
        from day3_fashion_training import FashionTrainingPipeline
        pipeline = FashionTrainingPipeline()
        
        # 測試提示詞配置
        configs = ["minimal_prompt", "high_confidence_only", "detailed_focused"]
        for config in configs:
            result = pipeline.set_prompt_config(config)
            if result:
                print(f"✅ {config} 配置設定成功")
            else:
                print(f"❌ {config} 配置設定失敗")
        
        # 測試權重配置
        weights = ["balanced", "fashion_focused", "visual_enhanced"]
        for weight in weights:
            result = pipeline.set_loss_weights(weight)
            if result:
                print(f"✅ {weight} 權重設定成功")
            else:
                print(f"❌ {weight} 權重設定失敗")
                
        return True
        
    except Exception as e:
        print(f"❌ 配置測試失敗: {e}")
        return False

def check_files():
    """檢查必要檔案"""
    print("\n📁 檢查檔案結構...")
    
    required_files = [
        "day3_fashion_training.py",
        "demo_prompt_optimization.py", 
        "prompt_optimization_config.json",
        "check_optimization_status.py"
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (缺失)")
            all_exist = False
    
    # 檢查來源目錄
    if os.path.exists("day1_results"):
        image_files = [f for f in os.listdir("day1_results") 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"📷 day1_results: {len(image_files)} 張圖片")
    else:
        print("📷 day1_results: 目錄不存在")
        
    return all_exist

def main():
    """主測試函數"""
    print("🧪 提示詞優化程式修改驗證測試")
    print("=" * 50)
    
    # 執行各項測試
    import_ok = test_imports()
    config_ok = test_configuration() if import_ok else False
    files_ok = check_files()
    
    print("\n📊 測試結果摘要")
    print("=" * 30)
    print(f"程式導入: {'✅ 通過' if import_ok else '❌ 失敗'}")
    print(f"配置功能: {'✅ 通過' if config_ok else '❌ 失敗'}")
    print(f"檔案結構: {'✅ 完整' if files_ok else '❌ 缺失'}")
    
    overall_status = import_ok and config_ok and files_ok
    print(f"\n🎯 整體狀態: {'✅ 就緒，可以執行' if overall_status else '❌ 需要修復'}")
    
    if overall_status:
        print("\n🚀 建議執行步驟:")
        print("1. python demo_prompt_optimization.py  # 互動式演示")
        print("2. python day3_fashion_training.py     # 完整訓練")
    else:
        print("\n⚠️ 需要先修復上述問題")

if __name__ == "__main__":
    main()
