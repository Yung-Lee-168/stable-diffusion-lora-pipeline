#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 資料集生成器 - 基於 day2 分析結果
快速生成 Standard CLIP 和 FashionCLIP 的 CSV 資料集
"""

import requests
import json
import base64
import os
import sys
import csv
from datetime import datetime
from PIL import Image
import numpy as np
import torch

# Windows 編碼設定
if sys.platform.startswith('win'):
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'zh_TW.UTF-8')
        except:
            pass

class CSVGenerator:
    def __init__(self):
        self.output_dir = "day2_advanced_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 定義分類
        self.categories = {
            "Gender": ["male", "female"],
            "Age": ["child", "teenager", "young adult", "adult", "senior"],
            "Season": ["spring", "summer", "autumn", "winter"],
            "Occasion": ["casual", "formal", "business", "sport", "party", "beach", "wedding", "date", "travel", "home"],
            "Upper Body": ["t-shirt", "shirt", "jacket", "coat", "sweater", "blazer", "hoodie", "tank top", "blouse", "dress"],
            "Lower Body": ["jeans", "trousers", "shorts", "skirt", "leggings", "cargo pants", "sweatpants", "culottes", "capris", "dress"]
        }
        
        self.detailed_clothing_features = {
            "Dress Style": ["A-line dress", "sheath dress", "wrap dress", "maxi dress", "midi dress", "mini dress", "bodycon dress", "shift dress", "empire waist dress", "fit and flare dress", "slip dress", "shirt dress", "sweater dress"],
            "Shirt Features": ["button-down shirt", "polo shirt", "henley shirt", "flannel shirt", "dress shirt", "peasant blouse", "crop top", "off-shoulder top", "turtleneck", "v-neck shirt", "crew neck", "collared shirt"],
            "Jacket Types": ["denim jacket", "leather jacket", "bomber jacket", "trench coat", "peacoat", "blazer jacket", "cardigan", "windbreaker", "puffer jacket", "motorcycle jacket", "varsity jacket"],
            "Pants Details": ["skinny jeans", "straight leg jeans", "bootcut jeans", "wide leg pants", "high-waisted pants", "low-rise pants", "cropped pants", "palazzo pants", "joggers", "dress pants", "cargo pants with pockets"],
            "Skirt Varieties": ["pencil skirt", "A-line skirt", "pleated skirt", "wrap skirt", "mini skirt", "maxi skirt", "denim skirt", "leather skirt", "tulle skirt", "asymmetrical skirt"],
            "Fabric Texture": ["cotton fabric", "silk material", "denim texture", "leather finish", "wool texture", "linen fabric", "chiffon material", "velvet texture", "knit fabric", "lace material", "satin finish", "corduroy texture"],
            "Pattern Details": ["solid color", "striped pattern", "floral print", "polka dots", "geometric pattern", "animal print", "plaid pattern", "paisley design", "abstract print", "tie-dye pattern", "checkered pattern"],
            "Color Scheme": ["monochrome outfit", "pastel colors", "bright colors", "earth tones", "neutral colors", "bold colors", "metallic accents", "neon colors", "vintage colors", "gradient colors"],
            "Fit Description": ["loose fit", "tight fit", "oversized", "fitted", "relaxed fit", "tailored fit", "slim fit", "regular fit", "cropped length", "flowing silhouette", "structured shape"],
            "Style Details": ["minimalist style", "vintage style", "bohemian style", "gothic style", "preppy style", "streetwear style", "romantic style", "edgy style", "classic style", "trendy style", "elegant style"]
        }

    def load_clip_models(self):
        """載入 CLIP 模型"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            models = {}
            
            print("📥 載入 Standard CLIP...")
            standard_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16 if device == "cuda" else torch.float32)
            standard_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            standard_model.to(device)
            models["standard_clip"] = (standard_model, standard_processor)
            print("✅ Standard CLIP 載入成功")
            
            try:
                print("📥 載入 FashionCLIP...")
                fashion_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip", torch_dtype=torch.float16 if device == "cuda" else torch.float32)
                fashion_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
                fashion_model.to(device)
                models["fashion_clip"] = (fashion_model, fashion_processor)
                print("✅ FashionCLIP 載入成功")
            except Exception as e:
                print(f"⚠️ FashionCLIP 載入失敗: {e}")
            
            return models
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            return {}

    def analyze_image(self, image_path, models):
        """分析單張圖片"""
        try:
            image = Image.open(image_path).convert("RGB")
            all_categories = {**self.categories, **self.detailed_clothing_features}
            results = {}
            
            for model_name, (model, processor) in models.items():
                print(f"   🔍 {model_name} 分析中...")
                model_results = {}
                device = next(model.parameters()).device
                model_dtype = next(model.parameters()).dtype
                
                for category_name, labels in all_categories.items():
                    try:
                        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        if model_dtype == torch.float16:
                            for key in inputs:
                                if inputs[key].dtype == torch.float32:
                                    inputs[key] = inputs[key].half()
                        
                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits_per_image = outputs.logits_per_image
                            probs = logits_per_image.softmax(dim=1)
                        
                        top_indices = probs[0].topk(min(3, len(labels))).indices
                        top_labels = [labels[i] for i in top_indices]
                        top_scores = [probs[0][i].item() for i in top_indices]
                        
                        model_results[category_name] = {
                            "top_labels": top_labels,
                            "scores": top_scores,
                            "confidence": max(top_scores)
                        }
                    except Exception as e:
                        model_results[category_name] = {"error": str(e), "confidence": 0.0}
                
                results[model_name] = model_results
            
            return results
        except Exception as e:
            print(f"❌ 圖片分析失敗: {e}")
            return {}

    def generate_csv_datasets(self):
        """生成 CSV 資料集"""
        print("🔍 尋找圖片...")
        
        # 尋找圖片
        image_files = []
        search_dirs = ["day1_results", "outputs", "day2_enhanced_results", "test_images"]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                print(f"   📂 搜索: {search_dir}")
                for file in os.listdir(search_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(os.path.join(search_dir, file))
                if image_files and search_dir == "day1_results":
                    break
        
        if not image_files:
            print("❌ 找不到圖片")
            return False
        
        print(f"✅ 找到 {len(image_files)} 張圖片")
        
        # 載入模型
        models = self.load_clip_models()
        if not models:
            print("❌ 模型載入失敗")
            return False
        
        # 分析所有圖片
        all_results = []
        for i, image_path in enumerate(image_files):
            print(f"\n--- 分析圖片 {i+1}/{len(image_files)}: {os.path.basename(image_path)} ---")
            analysis = self.analyze_image(image_path, models)
            if analysis:
                all_results.append({
                    "filename": os.path.basename(image_path),
                    "analysis": analysis
                })
        
        # 生成 CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.create_csv_files(all_results, timestamp)
        return True

    def create_csv_files(self, results, timestamp):
        """創建 CSV 檔案"""
        print("\n📄 生成 CSV 檔案...")
        
        all_categories = {**self.categories, **self.detailed_clothing_features}
        headers = ["filename"] + list(all_categories.keys())
        
        # 為每個模型生成 CSV
        for model_name in ["standard_clip", "fashion_clip"]:
            csv_filename = f"dataset_{model_name}_{timestamp}.csv"
            csv_path = os.path.join(self.output_dir, csv_filename)
            
            print(f"   📝 生成 {model_name} CSV: {csv_filename}")
            
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                
                for result in results:
                    if model_name not in result["analysis"]:
                        continue
                    
                    row = [result["filename"]]
                    analysis = result["analysis"][model_name]
                    
                    for category_name in all_categories.keys():
                        if category_name in analysis and "top_labels" in analysis[category_name]:
                            best_label = analysis[category_name]["top_labels"][0]
                            cell_value = best_label  # 移除機率部分，只保留標籤
                        else:
                            cell_value = "N/A"
                        row.append(cell_value)
                    
                    writer.writerow(row)
        
        # 生成簡潔提示詞 CSV (移除攝影後綴)
        self.create_clean_prompts_csv(results, timestamp)
        
        print("✅ CSV 檔案生成完成!")

    def create_clean_prompts_csv(self, results, timestamp):
        """生成簡潔提示詞 CSV (移除攝影相關後綴)"""
        csv_filename = f"clean_prompts_{timestamp}.csv"
        csv_path = os.path.join(self.output_dir, csv_filename)
        
        print(f"   📝 生成簡潔提示詞 CSV: {csv_filename}")
        
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "standard_clip_prompt", "fashion_clip_prompt"])
            
            for result in results:
                filename = result["filename"]
                prompts = {}
                
                # 為每個模型生成簡潔提示詞
                for model_name, analysis in result["analysis"].items():
                    prompt_parts = []
                    for category_name, category_data in analysis.items():
                        if "top_labels" in category_data and category_data["confidence"] > 0.3:
                            prompt_parts.append(category_data["top_labels"][0])
                    
                    # 簡潔提示詞 (移除攝影後綴)
                    prompts[model_name] = ", ".join(prompt_parts) if prompt_parts else "elegant fashion"
                
                writer.writerow([
                    filename,
                    prompts.get("standard_clip", "N/A"),
                    prompts.get("fashion_clip", "N/A")
                ])

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 快速 CSV 資料集生成器")
    print("=" * 60)
    
    generator = CSVGenerator()
    success = generator.generate_csv_datasets()
    
    if success:
        print(f"\n🎉 成功! 檔案保存在: {generator.output_dir}")
    else:
        print("\n❌ 生成失敗")
