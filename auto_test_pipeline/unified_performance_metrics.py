#!/usr/bin/env python3
"""
統一三個性能指標實現 - 修復腳本
確保 analyze_results.py 和 day3_fashion_training.py 使用完全相同的函數和公式
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import torch
from sklearn.metrics.pairwise import cosine_similarity

class UnifiedPerformanceMetrics:
    """統一的性能指標實現"""
    
    @staticmethod
    def calculate_ssim_similarity(img1_path_or_array, img2_path_or_array):
        """
        統一的 SSIM 結構相似度計算
        兩個腳本都應該使用這個實現
        """
        try:
            # 處理輸入 - 支持文件路徑或圖片數組
            if isinstance(img1_path_or_array, str):
                img1 = cv2.imread(img1_path_or_array)
            else:
                img1 = np.array(img1_path_or_array)
                
            if isinstance(img2_path_or_array, str):
                img2 = cv2.imread(img2_path_or_array)  
            else:
                img2 = np.array(img2_path_or_array)
            
            if img1 is None or img2 is None:
                return None
            
            # 確保是 BGR 格式 (OpenCV 標準)
            if len(img1.shape) == 3 and img1.shape[2] == 3:
                if isinstance(img1_path_or_array, str):
                    # 從文件讀取，已經是 BGR
                    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                else:
                    # 從數組轉換，假設是 RGB
                    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            else:
                gray1 = img1
                
            if len(img2.shape) == 3 and img2.shape[2] == 3:
                if isinstance(img2_path_or_array, str):
                    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                else:
                    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            else:
                gray2 = img2
            
            # 尺寸對齊 - 使用較小尺寸
            if gray1.shape != gray2.shape:
                target_shape = (min(gray1.shape[0], gray2.shape[0]), 
                              min(gray1.shape[1], gray2.shape[1]))
                gray1 = cv2.resize(gray1, (target_shape[1], target_shape[0]))
                gray2 = cv2.resize(gray2, (target_shape[1], target_shape[0]))
            
            # 🎯 統一使用 skimage.metrics.ssim
            similarity = ssim(gray1, gray2)
            return similarity
            
        except Exception as e:
            print(f"❌ SSIM計算失敗: {e}")
            return None
    
    @staticmethod 
    def calculate_color_similarity(img1_path_or_array, img2_path_or_array):
        """
        統一的色彩分布相似度計算
        兩個腳本都應該使用這個實現
        """
        try:
            # 處理輸入
            if isinstance(img1_path_or_array, str):
                img1 = cv2.imread(img1_path_or_array)
                img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            else:
                img1_rgb = np.array(img1_path_or_array)
                
            if isinstance(img2_path_or_array, str):
                img2 = cv2.imread(img2_path_or_array)
                img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            else:
                img2_rgb = np.array(img2_path_or_array)
            
            if img1_rgb is None or img2_rgb is None:
                return None
            
            # 🎯 統一的直方圖計算 - 32x32x32 RGB bins
            hist1 = cv2.calcHist([img1_rgb], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([img2_rgb], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
            
            # 🎯 統一正規化步驟
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            
            # 🎯 統一相關係數計算
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            return correlation
            
        except Exception as e:
            print(f"❌ 色彩相似度計算失敗: {e}")
            return None
    
    @staticmethod
    def calculate_fashionclip_feature_similarity(fashion_clip_model, fashion_clip_processor, img1, img2):
        """
        統一的 FashionCLIP 特徵向量相似度計算
        使用特徵向量餘弦相似度 (更準確的語義比較)
        """
        try:
            if not fashion_clip_model or not fashion_clip_processor:
                return None
                
            device = next(fashion_clip_model.parameters()).device
            model_dtype = next(fashion_clip_model.parameters()).device
            
            # 處理圖片輸入
            inputs = fashion_clip_processor(
                images=[img1, img2], 
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 確保數據類型一致
            if model_dtype == torch.float16:
                for key in inputs:
                    if inputs[key].dtype == torch.float32:
                        inputs[key] = inputs[key].half()
            
            with torch.no_grad():
                image_features = fashion_clip_model.get_image_features(**inputs)
                # 計算餘弦相似度
                fashion_similarity = cosine_similarity(
                    image_features[0:1].cpu().numpy(), 
                    image_features[1:2].cpu().numpy()
                )[0][0]
                
            return float(fashion_similarity)
            
        except Exception as e:
            print(f"❌ FashionCLIP相似度計算失敗: {e}")
            return None
    
    @staticmethod
    def calculate_fashionclip_label_similarity(orig_analysis, gen_analysis):
        """
        統一的 FashionCLIP 標籤比較相似度計算
        使用標籤匹配和信心度比較 (適用於已分析的特徵)
        """
        try:
            if not orig_analysis or not gen_analysis:
                return None
            
            similarities = []
            
            # 比較每個類別
            for category in orig_analysis.keys():
                if category in gen_analysis:
                    orig_top = orig_analysis[category]["top_label"]
                    gen_top = gen_analysis[category]["top_label"]
                    orig_conf = orig_analysis[category]["confidence"]
                    gen_conf = gen_analysis[category]["confidence"]
                    
                    # 標籤匹配度
                    label_match = 1.0 if orig_top == gen_top else 0.0
                    
                    # 信心度相似性
                    conf_similarity = 1.0 - abs(orig_conf - gen_conf)
                    
                    # 🎯 統一的類別相似度公式
                    category_similarity = 0.7 * label_match + 0.3 * conf_similarity
                    similarities.append(category_similarity)
            
            # 計算平均相似度
            average_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            
            return average_similarity
            
        except Exception as e:
            print(f"❌ FashionCLIP標籤相似度計算失敗: {e}")
            return None
    
    @staticmethod
    def calculate_combined_loss(visual_similarity, fashion_similarity, color_similarity, 
                              weights=None):
        """
        統一的組合損失函數
        兩個腳本都應該使用這個實現
        """
        if weights is None:
            # 🎯 統一的默認權重配置
            weights = {
                "visual": 0.2,      # SSIM 結構相似度
                "fashion_clip": 0.6, # FashionCLIP 語義相似度 (主要指標)
                "color": 0.2        # 色彩分布相似度
            }
        
        # 將相似度轉換為損失 (1 - similarity)
        visual_loss = 1.0 - (visual_similarity if visual_similarity is not None else 0.0)
        fashion_clip_loss = 1.0 - (fashion_similarity if fashion_similarity is not None else 0.0)
        color_loss = 1.0 - (color_similarity if color_similarity is not None else 0.0)
        
        # 🎯 統一的組合損失公式
        total_loss = (
            weights["visual"] * visual_loss +
            weights["fashion_clip"] * fashion_clip_loss +
            weights["color"] * color_loss
        )
        
        return {
            "total_loss": total_loss,
            "visual_loss": visual_loss,
            "fashion_clip_loss": fashion_clip_loss,
            "color_loss": color_loss,
            "weights": weights,
            "similarities": {
                "visual": visual_similarity,
                "fashion_clip": fashion_similarity,
                "color": color_similarity
            }
        }

def demonstrate_unified_implementation():
    """演示統一實現的使用方法"""
    print("🎯 統一性能指標實現演示")
    print("=" * 60)
    
    print("\n📊 三個核心指標的統一實現:")
    print("1. SSIM 結構相似度:")
    print("   函數: UnifiedPerformanceMetrics.calculate_ssim_similarity()")
    print("   實現: skimage.metrics.ssim + 尺寸對齊")
    print("   範圍: [-1, 1], 1=完全相同")
    
    print("\n2. 色彩分布相似度:")
    print("   函數: UnifiedPerformanceMetrics.calculate_color_similarity()")
    print("   實現: 32×32×32 RGB直方圖 + normalize + 相關係數")
    print("   範圍: [-1, 1], 1=完全相關")
    
    print("\n3. FashionCLIP 語義相似度:")
    print("   特徵向量版本: calculate_fashionclip_feature_similarity()")
    print("   標籤比較版本: calculate_fashionclip_label_similarity()")
    print("   範圍: [0, 1], 1=完全相似")
    
    print("\n4. 組合損失函數:")
    print("   函數: UnifiedPerformanceMetrics.calculate_combined_loss()")
    print("   公式: 0.2×visual_loss + 0.6×fashion_loss + 0.2×color_loss")
    print("   範圍: [0, 1], 0=完全相同")
    
    print("\n✅ 使用建議:")
    print("1. 將此類導入到 analyze_results.py 和 day3_fashion_training.py")
    print("2. 替換現有的分散實現")
    print("3. 確保兩個腳本調用完全相同的函數")
    print("4. 統一處理圖片輸入格式 (路徑或數組)")

def generate_implementation_guide():
    """生成實現指南"""
    print("\n" + "=" * 60)
    print("🔧 修復指南 - 如何統一三個性能指標")
    print("=" * 60)
    
    print("\n步驟 1: 在 analyze_results.py 中替換函數")
    print("━" * 40)
    print("替換 calculate_image_similarity() 為:")
    print("  similarity = UnifiedPerformanceMetrics.calculate_ssim_similarity(img1_path, img2_path)")
    
    print("\n替換 calculate_color_similarity() 為:")
    print("  correlation = UnifiedPerformanceMetrics.calculate_color_similarity(img1_path, img2_path)")
    
    print("\n保留 compare_fashion_features() 或使用:")
    print("  similarity = UnifiedPerformanceMetrics.calculate_fashionclip_label_similarity(orig, gen)")
    
    print("\n步驟 2: 在 day3_fashion_training.py 中替換函數")
    print("━" * 40)
    print("在 calculate_image_similarity() 中替換:")
    print("  similarities['visual_ssim'] = UnifiedPerformanceMetrics.calculate_ssim_similarity(gen_img, src_img)")
    print("  similarities['color_distribution'] = UnifiedPerformanceMetrics.calculate_color_similarity(gen_img, src_img)")
    print("  similarities['fashion_clip'] = UnifiedPerformanceMetrics.calculate_fashionclip_feature_similarity(...)")
    
    print("\n步驟 3: 統一組合損失計算")
    print("━" * 40)
    print("兩個腳本都使用:")
    print("  loss_result = UnifiedPerformanceMetrics.calculate_combined_loss(visual_sim, fashion_sim, color_sim)")
    
    print("\n步驟 4: 驗證一致性")
    print("━" * 40)
    print("1. 使用相同測試圖片對比兩個腳本的輸出")
    print("2. 確保三個指標值完全相同")
    print("3. 確保組合損失值完全相同")
    print("4. 運行完整測試驗證")

if __name__ == "__main__":
    demonstrate_unified_implementation()
    generate_implementation_guide()
    
    print(f"\n🎯 總結:")
    print(f"此統一實現確保了三個性能指標在所有腳本中使用完全相同的:")
    print(f"✅ 算法 (SSIM, 色彩直方圖, FashionCLIP)")
    print(f"✅ 預處理步驟 (尺寸對齊, 色彩空間轉換)")
    print(f"✅ 權重配置 (0.2:0.6:0.2)")
    print(f"✅ 損失轉換公式 (1.0 - similarity)")
    print(f"\n請按照修復指南更新代碼以確保完全一致性!")
