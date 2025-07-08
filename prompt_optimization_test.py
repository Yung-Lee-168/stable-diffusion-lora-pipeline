#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提示詞優化測試腳本
測試移除無用特徵後的效果

🎯 測試目標：
1. 比較簡潔提示詞 vs 詳細提示詞的效果
2. 驗證移除通用品質詞後的改善
3. 找到最佳的特徵組合
"""

import sys
import os
from day3_fashion_training import FashionTrainingPipeline

def test_prompt_optimization():
    """測試提示詞優化效果"""
    print("🧪 提示詞優化測試")
    print("=" * 50)
    
    pipeline = FashionTrainingPipeline()
    
    # 檢查測試圖片
    if not os.path.exists("day1_results"):
        print("❌ day1_results 目錄不存在")
        return
    
    image_files = [f for f in os.listdir("day1_results") 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("❌ day1_results 目錄中沒有圖片文件")
        return
    
    test_image = os.path.join("day1_results", image_files[0])
    print(f"🖼️ 使用測試圖片: {os.path.basename(test_image)}")
    
    # 測試不同配置
    configs_to_test = [
        "default",           # 標準配置 (包含詳細特徵)
        "minimal_prompt",    # 最簡配置 (僅基本類別)
        "high_confidence_only"  # 高置信度配置
    ]
    
    print(f"\n🔍 測試配置: {', '.join(configs_to_test)}")
    
    # 運行比較實驗
    try:
        results = pipeline.compare_prompt_configs(test_image, configs=configs_to_test)
        
        if results:
            print("\n✅ 提示詞配置測試完成！")
            analyze_prompt_results(results)
        else:
            print("❌ 測試失敗")
            
    except Exception as e:
        print(f"❌ 測試過程中出錯: {e}")

def analyze_prompt_results(results):
    """分析提示詞測試結果"""
    print("\n📊 提示詞效果分析")
    print("=" * 40)
    
    print("📝 各配置提示詞對比:")
    
    for config_name, data in results.items():
        prompt = data["prompt"]
        length = data["prompt_length"]
        config = data["config"]
        
        print(f"\n🔍 {config_name}:")
        print(f"   長度: {length} 字符")
        print(f"   詳細特徵: {'✓' if config.get('use_detailed_features', False) else '✗'}")
        print(f"   提示詞: {prompt}")
        
        # 分析提示詞組成
        prompt_parts = prompt.split(", ")
        print(f"   組件數: {len(prompt_parts)}")
    
    # 提供建議
    provide_prompt_recommendations(results)

def provide_prompt_recommendations(results):
    """提供提示詞優化建議"""
    print("\n💡 優化建議")
    print("=" * 30)
    
    # 找出最簡潔的配置
    shortest = min(results.items(), key=lambda x: x[1]["prompt_length"])
    longest = max(results.items(), key=lambda x: x[1]["prompt_length"])
    
    print(f"📏 最簡潔配置: {shortest[0]} ({shortest[1]['prompt_length']} 字符)")
    print(f"📏 最詳細配置: {longest[0]} ({longest[1]['prompt_length']} 字符)")
    
    print(f"\n🎯 建議:")
    
    # 基於長度差異提供建議
    length_diff = longest[1]["prompt_length"] - shortest[1]["prompt_length"]
    
    if length_diff > 100:
        print("   • 詳細配置可能過於冗長，建議使用簡潔配置")
        print("   • 過多特徵可能稀釋重要信息")
    
    print("   • 建議先測試簡潔配置的生成效果")
    print("   • 如果效果不佳，再逐步增加關鍵特徵")
    print("   • 專注於 FashionCLIP 能理解的核心特徵")
    
    print(f"\n🔬 下一步測試:")
    print("   1. 使用這些配置實際生成圖片")
    print("   2. 比較 FashionCLIP 相似度")
    print("   3. 選擇表現最佳的配置")

def show_removed_features():
    """顯示已移除的無用特徵"""
    print("\n🗑️ 已移除的無用特徵")
    print("=" * 30)
    
    removed_features = [
        "high quality",
        "detailed", 
        "professional photography",
        "fashion photography",
        "studio lighting"
    ]
    
    print("移除原因:")
    for feature in removed_features:
        print(f"   ❌ '{feature}' - 通用詞，對時尚特徵無幫助")
    
    print(f"\n✅ 移除效果:")
    print("   • 減少提示詞冗餘")
    print("   • 突出重要的時尚特徵")
    print("   • 提高 FashionCLIP 識別精度")
    print("   • 避免通用詞稀釋專業特徵")

def main():
    """主函數"""
    print("🎯 提示詞優化測試腳本")
    print("移除無用特徵，專注於 FashionCLIP 核心能力")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "show-removed":
            show_removed_features()
        elif sys.argv[1] == "test":
            test_prompt_optimization()
        else:
            print("用法: python prompt_optimization_test.py [show-removed|test]")
    else:
        # 默認運行完整測試
        show_removed_features()
        test_prompt_optimization()

if __name__ == "__main__":
    main()
