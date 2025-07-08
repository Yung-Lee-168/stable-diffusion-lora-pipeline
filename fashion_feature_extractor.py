#!/usr/bin/env python3
"""
Fashion Feature Extraction Pipeline
使用 FashionCLIP 分析服裝雜誌圖片並提取特徵
"""

import os
import json
import torch
import clip
from PIL import Image
import pandas as pd
from datetime import datetime
import numpy as np

class FashionFeatureExtractor:
    """時尚特徵提取器"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 使用設備: {self.device}")
        
        # 載入 CLIP 模型 (可以替換為 FashionCLIP)
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # 定義時尚相關的特徵類別
        self.feature_categories = {
            "gender": ["male", "female", "unisex"],
            "age_group": ["children", "teenager", "young adult", "middle aged", "elderly"],
            "top_clothing": [
                "t-shirt", "shirt", "blouse", "sweater", "jacket", "coat", 
                "hoodie", "tank top", "cardigan", "blazer", "vest"
            ],
            "bottom_clothing": [
                "jeans", "trousers", "shorts", "skirt", "dress", "leggings",
                "pants", "chinos", "joggers", "overalls"
            ],
            "style": [
                "casual", "formal", "business", "sporty", "elegant", "vintage",
                "streetwear", "bohemian", "minimalist", "punk", "gothic"
            ],
            "color_scheme": [
                "monochrome", "colorful", "pastel", "bright", "dark", "neutral",
                "warm tones", "cool tones", "earth tones"
            ],
            "season": ["spring", "summer", "autumn", "winter"],
            "occasion": [
                "everyday", "work", "party", "wedding", "beach", "gym",
                "date", "travel", "home", "outdoor"
            ]
        }
    
    def extract_features_from_image(self, image_path):
        """從單張圖片提取特徵"""
        try:
            # 載入和預處理圖片
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            features = {}
            
            # 對每個特徵類別進行分析
            for category, options in self.feature_categories.items():
                print(f"   分析 {category}...")
                
                # 創建文字提示
                text_prompts = [f"a photo of {option} clothing" for option in options]
                text_inputs = clip.tokenize(text_prompts).to(self.device)
                
                # 計算相似度
                with torch.no_grad():
                    image_features = self.model.encode_image(image_input)
                    text_features = self.model.encode_text(text_inputs)
                    
                    # 計算相似度分數
                    similarities = torch.cosine_similarity(
                        image_features, text_features, dim=1
                    )
                    
                    # 轉換為機率分布
                    probs = torch.softmax(similarities, dim=0)
                    
                    # 儲存結果
                    category_features = {}
                    for i, option in enumerate(options):
                        category_features[option] = float(probs[i])
                    
                    features[category] = category_features
            
            return features
            
        except Exception as e:
            print(f"❌ 處理圖片 {image_path} 時發生錯誤: {e}")
            return None
    
    def process_fashion_magazine_dataset(self, dataset_dir, output_file="fashion_features.json"):
        """處理整個時尚雜誌資料集"""
        
        print(f"🔍 掃描資料集目錄: {dataset_dir}")
        
        # 支援的圖片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # 收集所有圖片檔案
        image_files = []
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        print(f"📋 找到 {len(image_files)} 張圖片")
        
        # 處理每張圖片
        all_features = []
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n🎨 處理圖片 {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            features = self.extract_features_from_image(image_path)
            
            if features:
                # 添加元數據
                image_data = {
                    "image_path": image_path,
                    "image_name": os.path.basename(image_path),
                    "processing_time": datetime.now().isoformat(),
                    "features": features
                }
                
                all_features.append(image_data)
                print(f"✅ 特徵提取完成")
            else:
                print(f"❌ 特徵提取失敗")
        
        # 儲存結果
        print(f"\n💾 儲存特徵資料到 {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "dataset_info": {
                    "total_images": len(image_files),
                    "processed_images": len(all_features),
                    "processing_date": datetime.now().isoformat(),
                    "dataset_directory": dataset_dir
                },
                "features": all_features
            }, f, indent=2, ensure_ascii=False)
        
        print(f"🎉 處理完成! 成功處理 {len(all_features)}/{len(image_files)} 張圖片")
        return all_features
    
    def generate_sd_prompts(self, features_data, output_file="sd_prompts.json"):
        """將特徵轉換為 Stable Diffusion 提示詞"""
        
        print("🔄 轉換特徵為 SD 提示詞...")
        
        sd_prompts = []
        
        for item in features_data:
            features = item["features"]
            
            # 構建提示詞
            prompt_parts = []
            negative_parts = []
            
            # 性別
            gender_scores = features.get("gender", {})
            top_gender = max(gender_scores.items(), key=lambda x: x[1])
            if top_gender[1] > 0.3:
                prompt_parts.append(top_gender[0])
            
            # 年齡層
            age_scores = features.get("age_group", {})
            top_age = max(age_scores.items(), key=lambda x: x[1])
            if top_age[1] > 0.3:
                prompt_parts.append(f"{top_age[0]} person")
            
            # 服裝類型
            top_scores = features.get("top_clothing", {})
            top_top = max(top_scores.items(), key=lambda x: x[1])
            if top_top[1] > 0.3:
                prompt_parts.append(top_top[0])
            
            bottom_scores = features.get("bottom_clothing", {})
            top_bottom = max(bottom_scores.items(), key=lambda x: x[1])
            if top_bottom[1] > 0.3:
                prompt_parts.append(top_bottom[0])
            
            # 風格
            style_scores = features.get("style", {})
            top_style = max(style_scores.items(), key=lambda x: x[1])
            if top_style[1] > 0.3:
                prompt_parts.append(f"{top_style[0]} style")
            
            # 場合
            occasion_scores = features.get("occasion", {})
            top_occasion = max(occasion_scores.items(), key=lambda x: x[1])
            if top_occasion[1] > 0.3:
                prompt_parts.append(f"{top_occasion[0]} outfit")
            
            # 組合提示詞
            main_prompt = ", ".join(prompt_parts)
            main_prompt += ", fashion photography, high quality, detailed"
            
            negative_prompt = "blurry, low quality, distorted, deformed, ugly"
            
            sd_prompt_data = {
                "original_image": item["image_path"],
                "prompt": main_prompt,
                "negative_prompt": negative_prompt,
                "features_scores": features
            }
            
            sd_prompts.append(sd_prompt_data)
        
        # 儲存 SD 提示詞
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "generation_info": {
                    "total_prompts": len(sd_prompts),
                    "generation_date": datetime.now().isoformat()
                },
                "prompts": sd_prompts
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 已生成 {len(sd_prompts)} 個 SD 提示詞")
        return sd_prompts

def main():
    """主函數 - 示範使用"""
    
    print("👗 Fashion Feature Extraction Pipeline")
    print("=" * 60)
    
    # 創建特徵提取器
    extractor = FashionFeatureExtractor()
    
    # 設定資料集路徑 (請修改為您的資料集路徑)
    dataset_directory = "fashion_magazine_images"
    
    # 檢查資料集是否存在
    if not os.path.exists(dataset_directory):
        print(f"⚠️ 資料集目錄不存在: {dataset_directory}")
        print("請創建目錄並放入時尚雜誌圖片")
        
        # 創建示例目錄結構
        os.makedirs(dataset_directory, exist_ok=True)
        print(f"📁 已創建目錄: {dataset_directory}")
        print("請將時尚雜誌圖片放入此目錄中")
        return
    
    # 處理資料集
    features_data = extractor.process_fashion_magazine_dataset(
        dataset_directory, 
        "fashion_features.json"
    )
    
    if features_data:
        # 生成 SD 提示詞
        sd_prompts = extractor.generate_sd_prompts(
            features_data, 
            "sd_prompts.json"
        )
        
        print(f"\n🎉 Pipeline 完成!")
        print(f"📊 處理了 {len(features_data)} 張圖片")
        print(f"🔤 生成了 {len(sd_prompts)} 個提示詞")
        print(f"📁 特徵檔案: fashion_features.json")
        print(f"📝 提示詞檔案: sd_prompts.json")
    
    print("\n下一步: 使用這些提示詞訓練 SD 模型")

if __name__ == "__main__":
    main()
