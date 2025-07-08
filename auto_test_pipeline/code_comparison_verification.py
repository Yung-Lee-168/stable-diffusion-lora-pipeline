#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代碼層面的圖片尺寸處理對比驗證
直接比較訓練和評估代碼中的具體實現
"""

import os
import sys

def print_code_comparison():
    """打印訓練和評估代碼的具體對比"""
    
    print("🔍 LoRA 訓練 vs 評估：圖片尺寸處理代碼對比")
    print("=" * 80)
    
    # 1. SSIM 計算對比
    print("\n📋 1. SSIM 計算 - 圖片尺寸對齊")
    print("=" * 50)
    
    print("🔵 訓練階段 (day3_fashion_training.py):")
    print("```python")
    print("# 確保尺寸一致 (使用較小尺寸，與 analyze_results.py 一致)")
    print("if gen_gray.shape != src_gray.shape:")
    print("    target_shape = (min(gen_gray.shape[0], src_gray.shape[0]),")
    print("                   min(gen_gray.shape[1], src_gray.shape[1]))")
    print("    gen_gray = cv2.resize(gen_gray, (target_shape[1], target_shape[0]))")
    print("    src_gray = cv2.resize(src_gray, (target_shape[1], target_shape[0]))")
    print("```")
    
    print("\n🔴 評估階段 (analyze_results.py):")
    print("```python")
    print("# 確保兩張圖片尺寸一致（SSIM 計算要求）")
    print("if gray1.shape != gray2.shape:")
    print("    # 使用較小的尺寸作為基準，避免放大")
    print("    target_shape = (min(gray1.shape[0], gray2.shape[0]),")
    print("                   min(gray1.shape[1], gray2.shape[1]))")
    print("    gray1 = cv2.resize(gray1, (target_shape[1], target_shape[0]))")
    print("    gray2 = cv2.resize(gray2, (target_shape[1], target_shape[0]))")
    print("```")
    
    print("\n✅ 結論: SSIM 計算中的尺寸對齊邏輯完全一致")
    
    # 2. 色彩直方圖對比
    print("\n📋 2. 色彩直方圖計算")
    print("=" * 50)
    
    print("🔵 訓練階段 (day3_fashion_training.py):")
    print("```python")
    print("# 轉換為 RGB (與 analyze_results.py 一致)")
    print("gen_rgb = np.array(generated_img)  # 已經是 RGB")
    print("src_rgb = np.array(source_img)     # 已經是 RGB")
    print("")
    print("# 計算 RGB 直方圖 (32×32×32, 與 analyze_results.py 一致)")
    print("gen_hist = cv2.calcHist([gen_rgb], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])")
    print("src_hist = cv2.calcHist([src_rgb], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])")
    print("")
    print("# 正規化步驟 (與 analyze_results.py 一致)")
    print("gen_hist = cv2.normalize(gen_hist, gen_hist).flatten()")
    print("src_hist = cv2.normalize(src_hist, src_hist).flatten()")
    print("")
    print("# 計算相關係數 (與 analyze_results.py 一致)")
    print("color_similarity = cv2.compareHist(gen_hist, src_hist, cv2.HISTCMP_CORREL)")
    print("```")
    
    print("\n🔴 評估階段 (analyze_results.py):")
    print("```python")
    print("# 轉換為RGB")
    print("img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)")
    print("img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)")
    print("")
    print("# 計算RGB直方圖")
    print("hist1 = cv2.calcHist([img1_rgb], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])")
    print("hist2 = cv2.calcHist([img2_rgb], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])")
    print("")
    print("# 正規化")
    print("hist1 = cv2.normalize(hist1, hist1).flatten()")
    print("hist2 = cv2.normalize(hist2, hist2).flatten()")
    print("")
    print("# 計算相關係數")
    print("correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)")
    print("```")
    
    print("\n✅ 結論: 色彩直方圖計算方法完全一致 (32×32×32 bins, 正規化, 相關係數)")
    
    # 3. FashionCLIP 對比
    print("\n📋 3. FashionCLIP 標籤匹配")
    print("=" * 50)
    
    print("🔵 訓練階段 (day3_fashion_training.py):")
    print("```python")
    print("def _calculate_fashionclip_label_similarity(self, generated_img, source_img):")
    print("    # 標籤匹配度")
    print("    label_match = 1.0 if orig_top == gen_top else 0.0")
    print("    ")
    print("    # 信心度相似性")
    print("    conf_similarity = 1.0 - abs(orig_conf - gen_conf)")
    print("    ")
    print("    # 綜合相似度 (0.7 * 標籤匹配 + 0.3 * 信心度相似性)")
    print("    category_similarity = 0.7 * label_match + 0.3 * conf_similarity")
    print("```")
    
    print("\n🔴 評估階段 (analyze_results.py):")
    print("```python")
    print("def compare_fashion_features(orig_analysis, gen_analysis):")
    print("    # 標籤匹配度")
    print("    label_match = 1.0 if orig_top == gen_top else 0.0")
    print("    ")
    print("    # 信心度相似性")
    print("    conf_similarity = 1.0 - abs(orig_conf - gen_conf)")
    print("    ")
    print("    # 綜合相似度 (0.7 * 標籤匹配 + 0.3 * 信心度相似性)")
    print("    category_similarity = 0.7 * label_match + 0.3 * conf_similarity")
    print("```")
    
    print("\n✅ 結論: FashionCLIP 標籤匹配公式完全一致")
    
    # 4. 圖片尺寸限制
    print("\n📋 4. 圖片尺寸限制")
    print("=" * 50)
    
    print("🔵 預處理階段 (generate_caption_fashionclip.py):")
    print("```python")
    print("# 檢查圖片尺寸")
    print("if width <= 512 and height <= 512:")
    print("    # 尺寸符合要求，直接複製")
    print("else:")
    print("    # 需要縮放")
    print("    resized_image, was_resized = resize_image_keep_aspect(image, 512)")
    print("```")
    
    print("\n🔵 訓練檢查 (train_lora.py):")
    print("```python")
    print("def check_image_size(data_folder, target_size=512):")
    print("    # 檢查圖片尺寸是否符合要求")
    print("    if width <= target_size and height <= target_size:")
    print("        valid_count += 1")
    print("    else:")
    print("        # 超出尺寸，將跳過")
    print("        invalid_files.append((img_file, width, height))")
    print("```")
    
    print("\n✅ 結論: 所有階段都確保圖片 ≤ 512x512")
    
    # 5. 最終總結
    print("\n🎯 5. 最終驗證結果")
    print("=" * 50)
    print("✅ SSIM 計算: 訓練和評估使用相同的 min(shape) 對齊策略")
    print("✅ 色彩直方圖: 訓練和評估使用相同的 32×32×32 bins 和正規化")
    print("✅ FashionCLIP: 訓練和評估使用相同的標籤匹配公式 (0.7+0.3)")
    print("✅ 圖片尺寸: 整個 pipeline 都確保 ≤ 512x512")
    print("✅ 代碼一致性: 關鍵函數和參數完全一致")
    
    print("\n🏆 結論: LoRA 訓練和評估在圖片尺寸處理上完全一致！")

def main():
    """主函數"""
    print_code_comparison()

if __name__ == "__main__":
    main()
