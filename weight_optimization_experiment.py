#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3: 權重優化實驗腳本
測試不同損失函數權重配置的效果

🎯 目標：找到最佳的損失函數權重組合
- 視覺相似度權重
- FashionCLIP 語意相似度權重
- 色彩分布相似度權重
"""

import sys
import os
from day3_fashion_training import FashionTrainingPipeline

def analyze_current_weights():
    """分析當前權重配置的問題"""
    print("📊 當前權重配置分析")
    print("=" * 50)
    
    pipeline = FashionTrainingPipeline()
    
    print("🔍 當前權重配置:")
    current_weights = pipeline.training_config["loss_weights"]
    for key, value in current_weights.items():
        print(f"   {key}: {value}")
    
    print("\n🎯 備選權重方案:")
    alt_weights = pipeline.training_config["alternative_weights"]
    for scheme, weights in alt_weights.items():
        print(f"   {scheme}: {weights}")
    
    return pipeline

def run_weight_comparison_experiment():
    """運行權重比較實驗"""
    print("\n🧪 開始權重比較實驗")
    print("=" * 50)
    
    pipeline = analyze_current_weights()
    
    # 檢查測試圖片
    if not os.path.exists("day1_results"):
        print("❌ day1_results 目錄不存在")
        return
    
    image_files = [f for f in os.listdir("day1_results") 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("❌ day1_results 目錄中沒有圖片文件")
        return
    
    print(f"📁 找到 {len(image_files)} 張測試圖片")
    
    # 選擇第一張圖片進行測試
    test_image = os.path.join("day1_results", image_files[0])
    print(f"🖼️ 使用測試圖片: {os.path.basename(test_image)}")
    
    # 定義要測試的權重方案
    test_schemes = ["default", "balanced", "fashion_focused", "visual_enhanced", "color_enhanced"]
    
    # 運行比較實驗
    try:
        results = pipeline.compare_weight_schemes(test_image, schemes=test_schemes)
        
        if results:
            print("\n✅ 權重比較實驗完成！")
            analyze_results(results)
        else:
            print("❌ 實驗失敗")
            
    except Exception as e:
        print(f"❌ 實驗過程中出錯: {e}")

def analyze_results(results):
    """分析實驗結果"""
    print("\n📊 實驗結果分析")
    print("=" * 50)
    
    # 按總損失排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]["total_loss"])
    
    print("🏆 權重方案性能排名:")
    for i, (scheme, data) in enumerate(sorted_results, 1):
        total_loss = data["total_loss"]
        weights = data["weights"]
        similarities = data["similarities"]
        
        print(f"\n{i}. 方案: {scheme}")
        print(f"   總損失: {total_loss:.4f}")
        print(f"   權重: V={weights.get('visual', 0):.2f}, "
              f"F={weights.get('fashion_clip', 0):.2f}, "
              f"C={weights.get('color', 0):.2f}")
        print(f"   相似度: 視覺={similarities.get('visual_ssim', 0):.3f}, "
              f"FashionCLIP={similarities.get('fashion_clip', 0):.3f}, "
              f"色彩={similarities.get('color_distribution', 0):.3f}")
        
        if i == 1:
            print("   🎯 **推薦方案**")
    
    # 提供優化建議
    provide_optimization_suggestions(sorted_results)

def provide_optimization_suggestions(sorted_results):
    """提供優化建議"""
    print("\n💡 優化建議")
    print("=" * 30)
    
    best_scheme, best_data = sorted_results[0]
    best_weights = best_data["weights"]
    best_similarities = best_data["similarities"]
    
    print(f"🎯 最佳方案: {best_scheme}")
    print(f"   建議權重配置: {best_weights}")
    
    # 基於相似度分析提供建議
    visual_sim = best_similarities.get('visual_ssim', 0)
    fashion_sim = best_similarities.get('fashion_clip', 0)
    color_sim = best_similarities.get('color_distribution', 0)
    
    print("\n📈 進一步優化建議:")
    
    if visual_sim < 0.4:
        print("   • 視覺相似度偏低，考慮增加視覺權重或改進視覺相似度算法")
    
    if fashion_sim < 0.5:
        print("   • FashionCLIP 相似度中等，考慮調整提示詞生成策略")
    
    if color_sim < 0.3:
        print("   • 色彩相似度偏低，可能需要在提示詞中加強色彩描述")
    
    print(f"\n🔄 建議將權重配置更新為: {best_weights}")

def main():
    """主函數"""
    print("🎯 Day 3: 損失函數權重優化實驗")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "analyze":
            analyze_current_weights()
        elif sys.argv[1] == "test":
            run_weight_comparison_experiment()
        else:
            print("用法: python weight_optimization.py [analyze|test]")
    else:
        # 默認運行完整實驗
        run_weight_comparison_experiment()

if __name__ == "__main__":
    main()
