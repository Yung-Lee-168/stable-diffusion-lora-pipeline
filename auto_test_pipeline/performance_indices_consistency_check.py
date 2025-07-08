#!/usr/bin/env python3
"""
三個核心性能指標一致性檢查腳本
確認 analyze_results.py 和 day3_fashion_training.py 使用完全相同的計算方法和公式
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import torch
from sklearn.metrics.pairwise import cosine_similarity

class PerformanceIndicesChecker:
    """性能指標一致性檢查器"""
    
    def __init__(self):
        print("🔍 三個核心性能指標一致性檢查")
        print("=" * 80)
    
    def check_ssim_implementation(self):
        """檢查 SSIM (結構相似度) 實現一致性"""
        print("\n1️⃣ 結構相似度 (SSIM) 實現檢查")
        print("-" * 50)
        
        # analyze_results.py 實現
        print("📁 analyze_results.py 實現:")
        print("   函數名: calculate_image_similarity()")
        print("   行數: 第70行")
        print("   實現步驟:")
        print("   1. cv2.imread() 讀取圖片")
        print("   2. cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 轉灰階")
        print("   3. 尺寸對齊: 使用較小尺寸 min(shape)")
        print("   4. cv2.resize() 調整尺寸")
        print("   5. ssim(gray1, gray2) from skimage.metrics")
        print("   6. 返回 SSIM 相似度值")
        
        # day3_fashion_training.py 實現  
        print("\n📁 day3_fashion_training.py 實現:")
        print("   函數名: calculate_image_similarity() > visual_ssim")
        print("   行數: 第365行")
        print("   實現步驟:")
        print("   1. np.array(img.resize((256, 256))) 預處理")
        print("   2. cv2.cvtColor(array, cv2.COLOR_RGB2GRAY) 轉灰階")
        print("   3. cv2.matchTemplate(gen_gray, src_gray, cv2.TM_CCOEFF_NORMED)")
        print("   ❌ 注意: 使用 matchTemplate 而非 SSIM!")
        
        print("\n⚠️  發現差異:")
        print("   • analyze_results.py: 使用 skimage.metrics.ssim")
        print("   • day3_fashion_training.py: 使用 cv2.matchTemplate")
        print("   • 這兩個算法會產生不同的結果!")
        
        return {
            "analyze_results": "skimage.metrics.ssim",
            "day3_fashion_training": "cv2.matchTemplate",
            "consistent": False
        }
    
    def check_color_similarity_implementation(self):
        """檢查色彩相似度實現一致性"""
        print("\n2️⃣ 色彩分布相似度實現檢查")
        print("-" * 50)
        
        # analyze_results.py 實現
        print("📁 analyze_results.py 實現:")
        print("   函數名: calculate_color_similarity()")
        print("   行數: 第100行")
        print("   實現步驟:")
        print("   1. cv2.imread() 讀取圖片")
        print("   2. cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 轉RGB")
        print("   3. cv2.calcHist([img_rgb], [0,1,2], None, [32,32,32], [0,256,0,256,0,256])")
        print("   4. cv2.normalize(hist, hist).flatten()")
        print("   5. cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)")
        print("   6. 返回相關係數")
        
        # day3_fashion_training.py 實現
        print("\n📁 day3_fashion_training.py 實現:")
        print("   函數名: calculate_image_similarity() > color_distribution")
        print("   行數: 第365行內")
        print("   實現步驟:")
        print("   1. np.array(img.resize((256, 256))) 預處理")
        print("   2. cv2.calcHist([gen_array], [0,1,2], None, [32,32,32], [0,256,0,256,0,256])")
        print("   3. cv2.compareHist(gen_hist, src_hist, cv2.HISTCMP_CORREL)")
        print("   4. max(0, color_similarity) 保證非負")
        print("   ⚠️  注意: 缺少 normalize 步驟!")
        
        print("\n⚠️  發現差異:")
        print("   • analyze_results.py: BGR→RGB + normalize")
        print("   • day3_fashion_training.py: 直接RGB + 無normalize")
        print("   • normalize 步驟會影響計算結果!")
        
        return {
            "analyze_results": "BGR→RGB + normalize + HISTCMP_CORREL",
            "day3_fashion_training": "RGB + no_normalize + HISTCMP_CORREL",
            "consistent": False
        }
    
    def check_fashionclip_implementation(self):
        """檢查 FashionCLIP 相似度實現一致性"""
        print("\n3️⃣ FashionCLIP 語義相似度實現檢查")
        print("-" * 50)
        
        # analyze_results.py 實現
        print("📁 analyze_results.py 實現:")
        print("   函數名: compare_fashion_features()")
        print("   行數: 第727行")
        print("   實現步驟:")
        print("   1. 比較每個類別的標籤匹配")
        print("   2. label_match = 1.0 if orig_top == gen_top else 0.0")
        print("   3. conf_similarity = 1.0 - abs(orig_conf - gen_conf)")
        print("   4. category_similarity = 0.7 * label_match + 0.3 * conf_similarity")
        print("   5. average_similarity = sum(similarities) / len(similarities)")
        print("   6. 返回平均相似度")
        
        # day3_fashion_training.py 實現
        print("\n📁 day3_fashion_training.py 實現:")
        print("   函數名: calculate_image_similarity() > fashion_clip")
        print("   行數: 第365行內")
        print("   實現步驟:")
        print("   1. fashion_clip_processor(images=[gen_img, src_img])")
        print("   2. fashion_clip_model.get_image_features(**inputs)")
        print("   3. cosine_similarity(features[0:1], features[1:2])")
        print("   4. 返回餘弦相似度")
        print("   ❌ 完全不同的實現方法!")
        
        print("\n❌ 嚴重差異:")
        print("   • analyze_results.py: 特徵標籤比較 (離散)")
        print("   • day3_fashion_training.py: 特徵向量餘弦相似度 (連續)")
        print("   • 這是完全不同的算法!")
        
        return {
            "analyze_results": "label_matching + confidence_similarity",
            "day3_fashion_training": "cosine_similarity of feature_vectors", 
            "consistent": False
        }
    
    def check_loss_combination_formula(self):
        """檢查組合損失公式一致性"""
        print("\n4️⃣ 組合損失公式檢查")
        print("-" * 50)
        
        # analyze_results.py 實現
        print("📁 analyze_results.py 組合公式:")
        print("   行數: 第314行")
        print("   公式: total_loss = 0.2 * visual_loss + 0.6 * fashion_clip_loss + 0.2 * color_loss")
        print("   權重: visual=0.2, fashion_clip=0.6, color=0.2")
        print("   損失轉換: loss = 1.0 - similarity")
        
        # day3_fashion_training.py 實現
        print("\n📁 day3_fashion_training.py 組合公式:")
        print("   行數: 第445行 + 配置第49行")
        print("   公式: total_loss = weights['visual'] * visual_loss + weights['fashion_clip'] * fashion_clip_loss + weights['color'] * color_loss")
        print("   權重配置: visual=0.2, fashion_clip=0.6, color=0.2")
        print("   損失轉換: loss = 1.0 - similarity")
        
        print("\n✅ 權重配置一致:")
        print("   • 兩個腳本使用相同的權重: 0.2:0.6:0.2")
        print("   • 損失轉換公式相同: 1.0 - similarity")
        print("   • 組合公式結構相同")
        
        return {
            "analyze_results": "0.2 * visual + 0.6 * fashion + 0.2 * color",
            "day3_fashion_training": "0.2 * visual + 0.6 * fashion + 0.2 * color",
            "consistent": True
        }
    
    def generate_consistency_report(self):
        """生成一致性檢查報告"""
        print("\n" + "=" * 80)
        print("📊 一致性檢查總結報告")
        print("=" * 80)
        
        # 執行所有檢查
        ssim_check = self.check_ssim_implementation()
        color_check = self.check_color_similarity_implementation() 
        fashion_check = self.check_fashionclip_implementation()
        loss_check = self.check_loss_combination_formula()
        
        # 統計結果
        total_checks = 4
        consistent_checks = sum([
            ssim_check["consistent"],
            color_check["consistent"], 
            fashion_check["consistent"],
            loss_check["consistent"]
        ])
        
        print(f"\n📈 一致性統計:")
        print(f"   總檢查項目: {total_checks}")
        print(f"   ✅ 一致項目: {consistent_checks}")
        print(f"   ❌ 不一致項目: {total_checks - consistent_checks}")
        print(f"   🎯 一致性百分比: {(consistent_checks/total_checks)*100:.1f}%")
        
        # 詳細問題列表
        print(f"\n🚨 發現的不一致問題:")
        
        if not ssim_check["consistent"]:
            print("   1. SSIM計算: skimage.ssim vs cv2.matchTemplate")
            
        if not color_check["consistent"]:
            print("   2. 色彩相似度: 缺少normalize步驟")
            
        if not fashion_check["consistent"]:
            print("   3. FashionCLIP: 標籤比較 vs 特徵向量相似度")
        
        print(f"\n✅ 確認一致的項目:")
        if loss_check["consistent"]:
            print("   1. 組合損失權重: 0.2:0.6:0.2")
        
        # 修復建議
        print(f"\n🔧 修復建議:")
        print("   1. 統一SSIM實現: 都使用 skimage.metrics.ssim")
        print("   2. 統一色彩處理: 添加 normalize 步驟") 
        print("   3. 統一FashionCLIP: 選擇一種實現方法")
        print("   4. 建立共用函數庫: 避免重複實現")
        
        return {
            "total_checks": total_checks,
            "consistent_checks": consistent_checks,
            "consistency_percentage": (consistent_checks/total_checks)*100,
            "issues": {
                "ssim": not ssim_check["consistent"],
                "color": not color_check["consistent"], 
                "fashionclip": not fashion_check["consistent"]
            },
            "consistent_items": {
                "loss_weights": loss_check["consistent"]
            }
        }

def main():
    """主函數"""
    checker = PerformanceIndicesChecker()
    report = checker.generate_consistency_report()
    
    print(f"\n🎯 檢查完成!")
    print(f"請根據報告修復不一致的實現，確保訓練和評估使用相同的計算方法。")

if __name__ == "__main__":
    main()
