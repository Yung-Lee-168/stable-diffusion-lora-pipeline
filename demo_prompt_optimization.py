#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提示詞優化演示腳本
展示如何使用不同配置進行提示詞優化訓練
"""

import os
import sys
from day3_fashion_training import FashionTrainingPipeline

def demo_basic_training():
    """基礎提示詞優化訓練演示"""
    print("🎯 基礎提示詞優化訓練演示")
    print("=" * 50)
    
    # 初始化訓練管道
    pipeline = FashionTrainingPipeline()
    
    # 檢查來源圖片
    if not os.path.exists("day1_results"):
        print("❌ day1_results 目錄不存在，請先準備來源圖片")
        return
    
    image_files = [f for f in os.listdir("day1_results") 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("❌ day1_results 中沒有圖片檔案")
        return
    
    print(f"📁 找到 {len(image_files)} 張圖片，開始處理前3張...")
    
    # 處理前3張圖片作為演示
    for i, image_file in enumerate(image_files[:3], 1):
        print(f"\n📷 處理第 {i} 張圖片: {image_file}")
        
        image_path = os.path.join("day1_results", image_file)
        result = pipeline.process_single_image(image_path)
        
        if result:
            print(f"✅ 成功處理: {image_file}")
        else:
            print(f"❌ 處理失敗: {image_file}")

def demo_prompt_config_comparison():
    """提示詞配置比較演示"""
    print("\n🧪 提示詞配置比較演示")
    print("=" * 50)
    
    # 初始化訓練管道
    pipeline = FashionTrainingPipeline()
    
    # 檢查測試圖片
    test_image = None
    if os.path.exists("day1_results"):
        image_files = [f for f in os.listdir("day1_results") 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            test_image = os.path.join("day1_results", image_files[0])
    
    if not test_image:
        print("❌ 找不到測試圖片")
        return
    
    print(f"🎯 使用測試圖片: {os.path.basename(test_image)}")
    
    # 比較不同提示詞配置
    configs = ["default", "minimal_prompt", "high_confidence_only", "detailed_focused"]
    comparison_results = pipeline.compare_prompt_configs(test_image, configs)
    
    if comparison_results:
        print("\n📊 配置比較結果摘要:")
        for config, data in comparison_results.items():
            print(f"   {config}: {data['prompt_length']} 字符")

def demo_weight_optimization():
    """權重優化演示"""
    print("\n⚖️ 權重優化演示")
    print("=" * 50)
    
    # 初始化訓練管道
    pipeline = FashionTrainingPipeline()
    
    # 檢查測試圖片
    test_image = None
    if os.path.exists("day1_results"):
        image_files = [f for f in os.listdir("day1_results") 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            test_image = os.path.join("day1_results", image_files[0])
    
    if not test_image:
        print("❌ 找不到測試圖片")
        return
    
    print(f"🎯 使用測試圖片: {os.path.basename(test_image)}")
    
    # 比較不同權重方案
    schemes = ["default", "balanced", "fashion_focused", "visual_enhanced"]
    comparison_results = pipeline.compare_weight_schemes(test_image, schemes)
    
    if comparison_results:
        print("\n📊 權重方案比較結果摘要:")
        for scheme, data in comparison_results.items():
            print(f"   {scheme}: 總損失 {data['total_loss']:.3f}")

def main():
    """主要演示函數"""
    print("🎨 Day 3 提示詞優化訓練演示")
    print("基於 FashionCLIP 的智能提示詞優化系統")
    print("=" * 60)
    
    while True:
        print("\n🔧 選擇演示模式:")
        print("1. 基礎提示詞優化訓練")
        print("2. 提示詞配置比較")
        print("3. 權重優化比較")
        print("4. 執行完整訓練流程")
        print("0. 退出")
        
        try:
            choice = input("\n請選擇 (0-4): ").strip()
            
            if choice == "0":
                print("👋 演示結束")
                break
            elif choice == "1":
                demo_basic_training()
            elif choice == "2":
                demo_prompt_config_comparison()
            elif choice == "3":
                demo_weight_optimization()
            elif choice == "4":
                print("🚀 執行完整訓練流程...")
                pipeline = FashionTrainingPipeline()
                pipeline.run_training_pipeline()
            else:
                print("❌ 無效選擇，請重新輸入")
                
        except KeyboardInterrupt:
            print("\n👋 演示中斷")
            break
        except Exception as e:
            print(f"❌ 執行出錯: {e}")

if __name__ == "__main__":
    main()
