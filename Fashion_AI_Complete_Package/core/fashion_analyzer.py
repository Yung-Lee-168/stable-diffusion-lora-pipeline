#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3: Fashion Training Pipeline
基於 FashionCLIP 特徵提取與 SD v1.5 的自監督學習系統

🎯 重要說明：
- 🚫 完全禁用標準 CLIP 模型
- 🎯 僅使用 FashionCLIP 進行圖片特徵提取和相似度計算
- 所有類別定義參考 day2_csv_generator.py

處理流程:
來源圖輸入 → 🎯FashionCLIP特徵提取 → 結構化輸入 → SD圖生成 → 🎯FashionCLIP相似度比對 → 微調 → 風格化生成
"""

import os
import json
import requests
import base64
import io
from PIL import Image
import numpy as np
from datetime import datetime
import csv
import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
from sklearn.metrics.pairwise import cosine_similarity

class FashionTrainingPipeline:
    def __init__(self):
        print("🚀 初始化 Fashion Training Pipeline...")
        
        # 基本設定
        self.api_url = "http://localhost:7860"
        self.source_dir = "day1_results"  # 來源圖片目錄
        self.output_dir = "day3_training_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化模型
        self.init_models()
        
        # 訓練配置
        self.training_config = {
            "epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 1,
            # 🎯 優化後的損失權重配置
            "loss_weights": {
                "visual": 0.2,      # 視覺相似度 (降低權重，因為生成圖片風格可能不同)
                "fashion_clip": 0.6, # FashionCLIP 語意相似度 (主要指標，提高權重)
                "color": 0.2        # 色彩分布相似度 (適中權重，關注色彩匹配)
            },
            # 🔄 備選權重方案 (可供測試)
            "alternative_weights": {
                "balanced": {"visual": 0.33, "fashion_clip": 0.34, "color": 0.33},
                "fashion_focused": {"visual": 0.15, "fashion_clip": 0.7, "color": 0.15},
                "visual_enhanced": {"visual": 0.5, "fashion_clip": 0.4, "color": 0.1},
                "color_enhanced": {"visual": 0.3, "fashion_clip": 0.4, "color": 0.3}
            },
            # 📝 提示詞生成配置
            "prompt_config": {
                "use_detailed_features": True,      # 是否使用詳細特徵
                "detailed_confidence_threshold": 0.3,  # 詳細特徵置信度閾值
                "max_detailed_features": 5,        # 最大詳細特徵數量
                "use_basic_categories_only": False  # 僅使用基本類別
            },
            # 🧪 實驗性配置
            "experimental_configs": {
                "minimal_prompt": {
                    "use_detailed_features": False,
                    "use_basic_categories_only": True,
                    "description": "僅使用核心特徵，測試簡潔提示詞效果"
                },
                "high_confidence_only": {
                    "use_detailed_features": True,
                    "detailed_confidence_threshold": 0.5,
                    "max_detailed_features": 3,
                    "description": "僅使用高置信度特徵"
                },
                "detailed_focused": {
                    "use_detailed_features": True,
                    "detailed_confidence_threshold": 0.2,
                    "max_detailed_features": 8,
                    "description": "包含更多詳細特徵"
                }
            }
        }
        
        # 記錄訓練歷史
        self.training_history = {
            "epochs": [],
            "losses": [],
            "similarities": [],
            "generated_images": []
        }
        
    def init_models(self):
        """初始化所需的模型"""
        print("📦 載入模型中...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"🚀 使用設備: {device}")
        print(f"🔧 使用精度: {torch_dtype}")
        
        # 載入 FashionCLIP 模型 (唯一使用的模型)
        try:
            print("📥 載入 FashionCLIP (專業時尚模型)...")
            self.fashion_clip_model = CLIPModel.from_pretrained(
                "patrickjohncyh/fashion-clip",
                torch_dtype=torch_dtype
            ).to(device)
            self.fashion_clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
            print("✅ FashionCLIP 模型載入成功")
        except Exception as e:
            print(f"❌ FashionCLIP 模型載入失敗: {e}")
            self.fashion_clip_model = None
            self.fashion_clip_processor = None
            
        # 🚫 完全禁用標準 CLIP
        print("� 標準 CLIP 已禁用 - 僅使用 FashionCLIP")
        self.clip_model = None
        self.clip_processor = None
            
        # 使用 day2_csv_generator.py 中的完整分類結構
        self.categories = {
            "Gender": ["male", "female"],
            "Age": ["child", "teenager", "young adult", "adult", "senior"],
            "Season": ["spring", "summer", "autumn", "winter"],
            "Occasion": ["casual", "formal", "business", "sport", "party", "beach", "wedding", "date", "travel", "home"],
            "Upper Body": ["t-shirt", "shirt", "jacket", "coat", "sweater", "blazer", "hoodie", "tank top", "blouse", "dress"],
            "Lower Body": ["jeans", "trousers", "shorts", "skirt", "leggings", "cargo pants", "sweatpants", "culottes", "capris", "dress"]
        }
        
        # 詳細服裝特徵分析 - 來自 day2_csv_generator.py
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
        
    def extract_fashion_features(self, image_path):
        """使用 FashionCLIP 提取圖片特徵 - 參考 day2_csv_generator.py 的分析方法"""
        print(f"🔍 使用 FashionCLIP 分析圖片: {os.path.basename(image_path)}")
        
        if not self.fashion_clip_model or not self.fashion_clip_processor:
            print("❌ FashionCLIP 模型未載入，無法進行特徵提取")
            return {}
        
        try:
            # 載入圖片
            image = Image.open(image_path).convert("RGB")
            device = next(self.fashion_clip_model.parameters()).device
            model_dtype = next(self.fashion_clip_model.parameters()).dtype
            
            features = {}
            all_categories = {**self.categories, **self.detailed_clothing_features}
            
            # 對每個類別進行分析
            for category_name, labels in all_categories.items():
                try:
                    # 準備輸入
                    inputs = self.fashion_clip_processor(
                        text=labels, 
                        images=image, 
                        return_tensors="pt", 
                        padding=True
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # 處理數據類型
                    if model_dtype == torch.float16:
                        for key in inputs:
                            if inputs[key].dtype == torch.float32:
                                inputs[key] = inputs[key].half()
                    
                    # FashionCLIP 模型推理
                    with torch.no_grad():
                        outputs = self.fashion_clip_model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        probs = logits_per_image.softmax(dim=1)
                    
                    # 獲取前3名結果
                    top_indices = probs[0].topk(min(3, len(labels))).indices
                    top_labels = [labels[i] for i in top_indices]
                    top_scores = [probs[0][i].item() for i in top_indices]
                    
                    features[category_name] = {
                        "top_labels": top_labels,
                        "scores": top_scores,
                        "confidence": max(top_scores)
                    }
                    
                    print(f"   {category_name}: {top_labels[0]} (置信度: {top_scores[0]:.3f})")
                    
                except Exception as e:
                    print(f"⚠️ 分析類別 {category_name} 時出錯: {e}")
                    features[category_name] = {
                        "error": str(e), 
                        "confidence": 0.0
                    }
            
            print(f"✅ FashionCLIP 特徵提取完成，共分析 {len(features)} 個類別")
            return features
            
        except Exception as e:
            print(f"❌ 圖片載入或處理失敗: {e}")
            return {}
    
    def structure_features(self, features):
        """將特徵結構化為 SD 可用的格式 - 參考 day2_csv_generator.py 的格式"""
        print("🔧 結構化特徵數據...")
        
        structured = {
            "prompt_components": [],
            "style_tags": [],
            "technical_params": {}
        }
        
        # 提取基本特徵
        gender = self._get_best_feature(features, "Gender", "person")
        age = self._get_best_feature(features, "Age", "adult")
        upper_body = self._get_best_feature(features, "Upper Body", "clothing")
        lower_body = self._get_best_feature(features, "Lower Body", "")
        occasion = self._get_best_feature(features, "Occasion", "casual")
        season = self._get_best_feature(features, "Season", "")
        
        # 基本人物描述
        if gender != "person":
            person_desc = f"{age} {gender}"
        else:
            person_desc = age
            
        structured["prompt_components"].append(person_desc)
        
        # 服裝描述
        clothing_desc = f"wearing {upper_body}"
        if lower_body and lower_body != upper_body and lower_body != "dress":
            clothing_desc += f" and {lower_body}"
        structured["prompt_components"].append(clothing_desc)
        
        # 風格標籤
        if occasion:
            structured["style_tags"].append(occasion)
        if season:
            structured["style_tags"].append(f"{season} fashion")
        
        # 詳細特徵 - 根據配置決定是否使用
        prompt_config = self.training_config.get("prompt_config", {})
        use_detailed = prompt_config.get("use_detailed_features", True)
        confidence_threshold = prompt_config.get("detailed_confidence_threshold", 0.3)
        max_features = prompt_config.get("max_detailed_features", 5)
        
        if use_detailed and not prompt_config.get("use_basic_categories_only", False):
            detailed_features = []
            for category_name in self.detailed_clothing_features.keys():
                feature_value = self._get_best_feature(features, category_name, "")
                confidence = features.get(category_name, {}).get("confidence", 0)
                
                # 只添加高置信度且有意義的特徵
                if feature_value and confidence > confidence_threshold:
                    detailed_features.append((feature_value, confidence))
            
            # 按置信度排序，取前 N 個
            detailed_features.sort(key=lambda x: x[1], reverse=True)
            selected_features = [feat[0] for feat in detailed_features[:max_features]]
            
            # 添加到風格標籤
            structured["style_tags"].extend(selected_features)
            
            print(f"🔍 選擇了 {len(selected_features)} 個詳細特徵 (閾值: {confidence_threshold})")
        else:
            print("📋 僅使用基本類別，跳過詳細特徵")
        
        # 技術參數
        structured["technical_params"] = {
            "steps": 25,
            "cfg_scale": 7.5,
            "width": 512,
            "height": 512,
            "sampler": "DPM++ 2M Karras"
        }
        
        print(f"✅ 結構化完成，生成 {len(structured['prompt_components'])} 個主要組件和 {len(structured['style_tags'])} 個風格標籤")
        return structured
    
    def _get_best_feature(self, features, category_name, default=""):
        """獲取最佳特徵值"""
        if category_name in features and "top_labels" in features[category_name]:
            return features[category_name]["top_labels"][0]
        return default
    
    def features_to_prompt(self, structured_features):
        """將結構化特徵轉換為 SD 提示詞"""
        prompt_parts = []
        
        # 主要描述
        main_desc = ", ".join(structured_features["prompt_components"])
        prompt_parts.append(main_desc)
        
        # 風格標籤
        if structured_features["style_tags"]:
            style_desc = ", ".join(structured_features["style_tags"])
            prompt_parts.append(style_desc)
        
        # 🚫 移除無用的通用品質詞，專注於 FashionCLIP 特徵
        # 這些詞對時尚特徵訓練沒有幫助，反而稀釋重要特徵
        
        final_prompt = ", ".join(prompt_parts)
        
        # 負面提示詞 - 保持簡潔，專注於避免變形
        negative_prompt = "deformed, bad anatomy, blurry"
        
        return {
            "prompt": final_prompt,
            "negative_prompt": negative_prompt
        }
    
    def generate_image_with_sd(self, prompt_data, structured_features):
        """使用 SD v1.5 生成圖片"""
        print("🎨 使用 Stable Diffusion 生成圖片...")
        
        payload = {
            "prompt": prompt_data["prompt"],
            "negative_prompt": prompt_data["negative_prompt"],
            "steps": structured_features["technical_params"]["steps"],
            "cfg_scale": structured_features["technical_params"]["cfg_scale"],
            "width": structured_features["technical_params"]["width"],
            "height": structured_features["technical_params"]["height"],
            "sampler_name": structured_features["technical_params"]["sampler"],
            "batch_size": 1,
            "n_iter": 1
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/sdapi/v1/txt2img",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("images"):
                    # 解碼生成的圖片
                    img_data = base64.b64decode(result["images"][0])
                    generated_image = Image.open(io.BytesIO(img_data))
                    print("✅ 圖片生成成功")
                    return generated_image
                    
        except Exception as e:
            print(f"❌ 圖片生成失敗: {e}")
            
        return None
    
    def calculate_image_similarity(self, generated_img, source_img):
        """計算生成圖片與原圖的相似度 - 主要使用 FashionCLIP"""
        print("📊 計算圖片相似度...")
        
        similarities = {}
        
        try:
            # 1. 基本視覺相似度 (使用結構相似性)
            gen_array = np.array(generated_img.resize((256, 256)))
            src_array = np.array(source_img.resize((256, 256)))
            
            # 轉換為灰度圖計算 SSIM
            gen_gray = cv2.cvtColor(gen_array, cv2.COLOR_RGB2GRAY)
            src_gray = cv2.cvtColor(src_array, cv2.COLOR_RGB2GRAY)
            
            ssim_score = cv2.matchTemplate(gen_gray, src_gray, cv2.TM_CCOEFF_NORMED)[0][0]
            similarities["visual_ssim"] = float(max(0, ssim_score))
            
            # 2. FashionCLIP 語意相似度 (唯一使用的語意模型)
            if self.fashion_clip_model and self.fashion_clip_processor:
                device = next(self.fashion_clip_model.parameters()).device
                model_dtype = next(self.fashion_clip_model.parameters()).dtype
                
                # 處理圖片
                inputs = self.fashion_clip_processor(
                    images=[generated_img, source_img], 
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 確保輸入數據類型與模型一致
                if model_dtype == torch.float16:
                    for key in inputs:
                        if inputs[key].dtype == torch.float32:
                            inputs[key] = inputs[key].half()
                
                with torch.no_grad():
                    image_features = self.fashion_clip_model.get_image_features(**inputs)
                    # 計算餘弦相似度
                    from sklearn.metrics.pairwise import cosine_similarity
                    fashion_similarity = cosine_similarity(
                        image_features[0:1].cpu().numpy(), 
                        image_features[1:2].cpu().numpy()
                    )[0][0]
                    
                similarities["fashion_clip"] = float(fashion_similarity)
                print(f"   ✅ FashionCLIP 相似度: {fashion_similarity:.3f}")
            else:
                print("   ❌ FashionCLIP 模型未載入")
                similarities["fashion_clip"] = 0.0
            
            # � 標準 CLIP 已完全禁用
            similarities["standard_clip"] = 0.0
            
            # 4. 色彩分布相似度
            gen_hist = cv2.calcHist([gen_array], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
            src_hist = cv2.calcHist([src_array], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
            
            color_similarity = cv2.compareHist(gen_hist, src_hist, cv2.HISTCMP_CORREL)
            similarities["color_distribution"] = float(max(0, color_similarity))
            
            print(f"相似度分數: 視覺={similarities.get('visual_ssim', 0):.3f}, "
                  f"🎯FashionCLIP={similarities.get('fashion_clip', 0):.3f}, "
                  f"色彩={similarities.get('color_distribution', 0):.3f}")
            print(f"🚫 標準 CLIP 已禁用")
            
        except Exception as e:
            print(f"⚠️ 計算相似度時出錯: {e}")
            # 設定預設值
            similarities = {
                "visual_ssim": 0.5,
                "fashion_clip": 0.5,
                "standard_clip": 0.5,
                "color_distribution": 0.5
            }
        
        return similarities
    
    def calculate_combined_loss(self, similarities):
        """計算組合損失函數 - 優化權重分配，主要基於 FashionCLIP"""
        weights = self.training_config["loss_weights"]
        
        # 將相似度轉換為損失 (1 - similarity)
        visual_loss = 1.0 - similarities.get("visual_ssim", 0)
        fashion_clip_loss = 1.0 - similarities.get("fashion_clip", 0)  # 🎯 主要指標
        color_loss = 1.0 - similarities.get("color_distribution", 0)
        
        # 🚫 標準 CLIP 已禁用，設為 0
        standard_clip_loss = 0.0
        
        # 🎯 優化後的加權組合
        total_loss = (
            weights["visual"] * visual_loss +           # 視覺結構相似度
            weights["fashion_clip"] * fashion_clip_loss +  # FashionCLIP 語意相似度 (主要)
            weights["color"] * color_loss               # 色彩分布相似度
        )
        
        # 📊 詳細損失分析
        print(f"📊 損失分析:")
        print(f"   總損失: {total_loss:.4f}")
        print(f"   🎯 FashionCLIP損失: {fashion_clip_loss:.4f} (權重: {weights['fashion_clip']})")
        print(f"   �️ 視覺損失: {visual_loss:.4f} (權重: {weights['visual']})")
        print(f"   🎨 色彩損失: {color_loss:.4f} (權重: {weights['color']})")
        print(f"   �🚫 標準 CLIP: 已禁用")
        
        return {
            "total_loss": total_loss,
            "visual_loss": visual_loss,
            "fashion_clip_loss": fashion_clip_loss,  # 主要且唯一語意指標
            "standard_clip_loss": standard_clip_loss,  # 已禁用
            "color_loss": color_loss,
            # 💡 權重信息
            "weight_distribution": weights,
            "loss_breakdown": {
                "visual_weighted": weights["visual"] * visual_loss,
                "fashion_clip_weighted": weights["fashion_clip"] * fashion_clip_loss,
                "color_weighted": weights["color"] * color_loss
            }
        }
    
    def process_single_image(self, image_path):
        """處理單一圖片的完整流程"""
        print(f"\n🎯 處理圖片: {os.path.basename(image_path)}")
        print("=" * 50)
        
        try:
            # 載入原始圖片
            source_image = Image.open(image_path).convert("RGB")
            
            # 1. 特徵提取
            features = self.extract_fashion_features(image_path)
            
            # 2. 結構化特徵
            structured_features = self.structure_features(features)
            
            # 3. 生成提示詞
            prompt_data = self.features_to_prompt(structured_features)
            
            # 📝 分析提示詞組成 (新增)
            prompt_analysis = self.analyze_prompt_composition(structured_features, features)
            
            print(f"📝 生成的提示詞: {prompt_data['prompt']}")
            
            # 4. 生成圖片
            generated_image = self.generate_image_with_sd(prompt_data, structured_features)
            
            if generated_image is None:
                print("❌ 圖片生成失敗，跳過此圖片")
                return None
            
            # 5. 計算相似度
            similarities = self.calculate_image_similarity(generated_image, source_image)
            
            # 6. 計算損失
            losses = self.calculate_combined_loss(similarities)
            
            # 📊 詳細損失分析 (新增)
            self.analyze_loss_performance(losses, similarities)
            
            # 7. 保存結果
            result_data = {
                "source_image": os.path.basename(image_path),
                "features": features,
                "structured_features": structured_features,
                "prompt": prompt_data,
                "similarities": similarities,
                "losses": losses,
                "timestamp": datetime.now().isoformat()
            }
            
            # 保存生成的圖片
            output_filename = f"generated_{os.path.splitext(os.path.basename(image_path))[0]}.png"
            output_path = os.path.join(self.output_dir, output_filename)
            generated_image.save(output_path)
            
            print(f"✅ 結果已保存: {output_filename}")
            print(f"📊 總損失: {losses['total_loss']:.4f}")
            
            return result_data
            
        except Exception as e:
            print(f"❌ 處理圖片時出錯: {e}")
            return None
    
    def run_training_pipeline(self):
        """執行完整的訓練流程"""
        print("🚀 開始 Fashion Training Pipeline")
        print("=" * 60)
        
        # 檢查來源目錄
        if not os.path.exists(self.source_dir):
            print(f"❌ 來源目錄不存在: {self.source_dir}")
            return
        
        # 搜尋圖片檔案
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend([
                os.path.join(self.source_dir, f) 
                for f in os.listdir(self.source_dir) 
                if f.lower().endswith(ext)
            ])
        
        if not image_files:
            print(f"❌ 在 {self.source_dir} 中找不到圖片檔案")
            return
        
        print(f"📁 找到 {len(image_files)} 張圖片")
        
        # 處理每張圖片
        all_results = []
        total_loss = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n📷 處理第 {i}/{len(image_files)} 張圖片")
            
            result = self.process_single_image(image_path)
            if result:
                all_results.append(result)
                total_loss += result["losses"]["total_loss"]
        
        # 計算平均損失
        if all_results:
            avg_loss = total_loss / len(all_results)
            print(f"\n📊 平均損失: {avg_loss:.4f}")
        
        # 保存完整結果
        self.save_training_results(all_results)
        
        # 生成報告
        self.generate_training_report(all_results)
        
        print(f"\n🎉 訓練流程完成！結果保存在: {self.output_dir}")
    
    def save_training_results(self, results):
        """保存訓練結果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存 JSON 格式結果
        json_path = os.path.join(self.output_dir, f"training_results_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存 CSV 格式結果
        csv_path = os.path.join(self.output_dir, f"training_summary_{timestamp}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 寫入標題
            writer.writerow([
                "Source Image", "Generated Prompt", "Total Loss",
                "Visual Loss", "FashionCLIP Loss", "Color Loss",
                "Visual Similarity", "FashionCLIP Similarity", "Color Similarity",
                "Visual Weight", "FashionCLIP Weight", "Color Weight"
            ])
            
            # 寫入數據
            for result in results:
                weights = result["losses"].get("weight_distribution", {})
                writer.writerow([
                    result["source_image"],
                    result["prompt"]["prompt"][:100] + "...",  # 截斷長提示詞
                    f"{result['losses']['total_loss']:.4f}",
                    f"{result['losses']['visual_loss']:.4f}",
                    f"{result['losses']['fashion_clip_loss']:.4f}",
                    f"{result['losses']['color_loss']:.4f}",
                    f"{result['similarities'].get('visual_ssim', 0):.4f}",
                    f"{result['similarities'].get('fashion_clip', 0):.4f}",
                    f"{result['similarities'].get('color_distribution', 0):.4f}",
                    f"{weights.get('visual', 0):.2f}",
                    f"{weights.get('fashion_clip', 0):.2f}",
                    f"{weights.get('color', 0):.2f}"
                ])
        
        print(f"📄 結果已保存: {json_path}")
        print(f"📊 CSV 摘要已保存: {csv_path}")
    
    def generate_training_report(self, results):
        """生成訓練報告"""
        if not results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"training_report_{timestamp}.html")
        
        # 計算統計數據
        total_images = len(results)
        avg_total_loss = sum(r["losses"]["total_loss"] for r in results) / total_images
        avg_visual_sim = sum(r["similarities"].get("visual_ssim", 0) for r in results) / total_images
        avg_fashion_sim = sum(r["similarities"].get("fashion_clip", 0) for r in results) / total_images
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Fashion Training Pipeline Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 10px; }}
        .summary {{ background: #ecf0f1; padding: 15px; margin: 20px 0; border-radius: 8px; }}
        .result-item {{ border: 1px solid #bdc3c7; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .prompt {{ background: #e8f6f3; padding: 10px; border-radius: 5px; font-family: monospace; }}
        .metrics {{ display: flex; justify-content: space-around; margin: 10px 0; }}
        .metric {{ text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        .generated-image {{ max-width: 300px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 Fashion Training Pipeline Report</h1>
        <p>訓練時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="summary">
        <h2>📊 訓練摘要</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{total_images}</div>
                <div>處理圖片數</div>
            </div>
            <div class="metric">
                <div class="metric-value">{avg_total_loss:.3f}</div>
                <div>平均總損失</div>
            </div>
            <div class="metric">
                <div class="metric-value">{avg_visual_sim:.3f}</div>
                <div>平均視覺相似度</div>
            </div>
            <div class="metric">
                <div class="metric-value">{avg_fashion_sim:.3f}</div>
                <div>平均FashionCLIP相似度</div>
            </div>
        </div>
    </div>
    
    <h2>🖼️ 詳細結果</h2>
"""
        
        # 添加每個結果的詳細信息
        for i, result in enumerate(results, 1):
            generated_img_name = f"generated_{os.path.splitext(result['source_image'])[0]}.png"
            
            html_content += f"""
    <div class="result-item">
        <h3>圖片 {i}: {result['source_image']}</h3>
        
        <div class="prompt">
            <strong>生成提示詞:</strong><br>
            {result['prompt']['prompt']}
        </div>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{result['losses']['total_loss']:.3f}</div>
                <div>總損失</div>
            </div>
            <div class="metric">
                <div class="metric-value">{result['similarities'].get('visual_ssim', 0):.3f}</div>
                <div>視覺相似度</div>
            </div>
            <div class="metric">
                <div class="metric-value">{result['similarities'].get('fashion_clip', 0):.3f}</div>
                <div>FashionCLIP相似度</div>
            </div>
            <div class="metric">
                <div class="metric-value">{result['similarities'].get('color_distribution', 0):.3f}</div>
                <div>色彩相似度</div>
            </div>
        </div>
        
        <p><strong>生成圖片:</strong> {generated_img_name}</p>
        
        <details>
            <summary>詳細特徵分析</summary>
            <pre>{json.dumps(result['features'], ensure_ascii=False, indent=2)}</pre>
        </details>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📋 訓練報告已生成: {report_path}")

    def set_loss_weights(self, weight_name="default"):
        """動態設置損失權重配置"""
        if weight_name == "default":
            # 使用優化後的默認權重
            return self.training_config["loss_weights"]
        elif weight_name in self.training_config["alternative_weights"]:
            # 使用備選權重方案
            self.training_config["loss_weights"] = self.training_config["alternative_weights"][weight_name]
            print(f"🔄 切換到權重方案: {weight_name}")
            print(f"   新權重: {self.training_config['loss_weights']}")
            return self.training_config["loss_weights"]
        else:
            print(f"❌ 未知的權重方案: {weight_name}")
            return self.training_config["loss_weights"]
    
    def compare_weight_schemes(self, image_path, schemes=["default", "balanced", "fashion_focused"]):
        """比較不同權重方案的效果"""
        print(f"\n🧪 權重方案比較實驗: {os.path.basename(image_path)}")
        print("=" * 60)
        
        # 載入原始圖片並提取特徵（只做一次）
        source_image = Image.open(image_path).convert("RGB")
        features = self.extract_fashion_features(image_path)
        structured_features = self.structure_features(features)
        prompt_data = self.features_to_prompt(structured_features)
        
        # 生成圖片（只做一次）
        generated_image = self.generate_image_with_sd(prompt_data, structured_features)
        if generated_image is None:
            print("❌ 圖片生成失敗，無法進行比較")
            return None
        
        # 計算相似度（只做一次）
        similarities = self.calculate_image_similarity(generated_image, source_image)
        
        # 比較不同權重方案
        comparison_results = {}
        original_weights = self.training_config["loss_weights"].copy()
        
        for scheme in schemes:
            print(f"\n🎯 測試權重方案: {scheme}")
            print("-" * 30)
            
            # 設置權重
            self.set_loss_weights(scheme)
            
            # 計算損失
            losses = self.calculate_combined_loss(similarities)
            
            comparison_results[scheme] = {
                "weights": self.training_config["loss_weights"].copy(),
                "total_loss": losses["total_loss"],
                "losses": losses,
                "similarities": similarities
            }
        
        # 恢復原始權重
        self.training_config["loss_weights"] = original_weights
        
        # 生成比較報告
        self._generate_weight_comparison_report(comparison_results, os.path.basename(image_path))
        
        return comparison_results
    
    def _generate_weight_comparison_report(self, results, image_name):
        """生成權重比較報告"""
        print(f"\n📊 權重方案比較報告: {image_name}")
        print("=" * 60)
        
        # 按總損失排序
        sorted_results = sorted(results.items(), key=lambda x: x[1]["total_loss"])
        
        print("🏆 權重方案排名 (按總損失從低到高):")
        for i, (scheme, data) in enumerate(sorted_results, 1):
            weights = data["weights"]
            total_loss = data["total_loss"]
            
            print(f"\n{i}. 📋 方案: {scheme}")
            print(f"   總損失: {total_loss:.4f}")
            print(f"   權重配置: 視覺={weights.get('visual', 0):.2f}, "
                  f"FashionCLIP={weights.get('fashion_clip', 0):.2f}, "
                  f"色彩={weights.get('color', 0):.2f}")
            
            if i == 1:
                print("   🎯 **最佳方案**")
        
        # 保存詳細比較結果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = os.path.join(
            self.output_dir, 
            f"weight_comparison_{image_name}_{timestamp}.json"
        )
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 詳細比較結果已保存: {comparison_file}")

    def set_prompt_config(self, config_name="default"):
        """設置提示詞生成配置"""
        if config_name == "default":
            # 保持當前配置
            return self.training_config["prompt_config"]
            
        elif config_name in self.training_config["experimental_configs"]:
            # 使用實驗性配置
            exp_config = self.training_config["experimental_configs"][config_name]
            
            # 更新提示詞配置
            for key, value in exp_config.items():
                if key != "description":
                    self.training_config["prompt_config"][key] = value
            
            print(f"🔄 切換到提示詞配置: {config_name}")
            print(f"   描述: {exp_config.get('description', '')}")
            print(f"   配置: {self.training_config['prompt_config']}")
            
            return self.training_config["prompt_config"]
        else:
            print(f"❌ 未知的配置: {config_name}")
            return self.training_config["prompt_config"]
    
    def compare_prompt_configs(self, image_path, configs=["default", "minimal_prompt", "high_confidence_only"]):
        """比較不同提示詞配置的效果"""
        print(f"\n🧪 提示詞配置比較實驗: {os.path.basename(image_path)}")
        print("=" * 60)
        
        # 載入原始圖片並提取特徵（只做一次）
        source_image = Image.open(image_path).convert("RGB")
        features = self.extract_fashion_features(image_path)
        
        comparison_results = {}
        original_config = self.training_config["prompt_config"].copy()
        
        for config_name in configs:
            print(f"\n🎯 測試配置: {config_name}")
            print("-" * 30)
            
            # 設置配置
            self.set_prompt_config(config_name)
            
            # 生成提示詞
            structured_features = self.structure_features(features)
            prompt_data = self.features_to_prompt(structured_features)
            
            print(f"📝 生成的提示詞: {prompt_data['prompt']}")
            print(f"📏 提示詞長度: {len(prompt_data['prompt'])} 字符")
            
            # 記錄結果
            comparison_results[config_name] = {
                "config": self.training_config["prompt_config"].copy(),
                "prompt": prompt_data["prompt"],
                "prompt_length": len(prompt_data["prompt"]),
                "structured_features": structured_features
            }
        
        # 恢復原始配置
        self.training_config["prompt_config"] = original_config
        
        # 生成比較報告
        self._generate_prompt_comparison_report(comparison_results, os.path.basename(image_path))
        
        return comparison_results
    
    def _generate_prompt_comparison_report(self, results, image_name):
        """生成提示詞比較報告"""
        print(f"\n📊 提示詞配置比較報告: {image_name}")
        print("=" * 60)
        
        # 按提示詞長度排序
        sorted_results = sorted(results.items(), key=lambda x: x[1]["prompt_length"])
        
        print("📏 配置比較 (按提示詞長度排序):")
        for i, (config_name, data) in enumerate(sorted_results, 1):
            prompt_length = data["prompt_length"]
            config = data["config"]
            
            print(f"\n{i}. 📋 配置: {config_name}")
            print(f"   提示詞長度: {prompt_length} 字符")
            print(f"   使用詳細特徵: {config.get('use_detailed_features', False)}")
            print(f"   置信度閾值: {config.get('detailed_confidence_threshold', 0.3)}")
            print(f"   最大特徵數: {config.get('max_detailed_features', 5)}")
            print(f"   提示詞: {data['prompt'][:100]}...")
            
            if i == 1:
                print("   🎯 **最簡潔**")
        
        # 保存比較結果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = os.path.join(
            self.output_dir, 
            f"prompt_comparison_{image_name}_{timestamp}.json"
        )
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 詳細比較結果已保存: {comparison_file}")

    def analyze_prompt_composition(self, structured_features, features):
        """分析提示詞組成與特徵分布"""
        print("🔍 分析提示詞組成...")
        
        analysis = {
            "component_count": len(structured_features["prompt_components"]),
            "style_tag_count": len(structured_features["style_tags"]),
            "feature_distribution": {},
            "confidence_analysis": {},
            "optimization_suggestions": []
        }
        
        # 分析特徵分布
        for category, feature_data in features.items():
            if isinstance(feature_data, dict) and "confidence" in feature_data:
                confidence = feature_data["confidence"]
                analysis["feature_distribution"][category] = confidence
                
                # 信心度分析
                if confidence > 0.7:
                    analysis["confidence_analysis"][category] = "高信心"
                elif confidence > 0.4:
                    analysis["confidence_analysis"][category] = "中信心"
                else:
                    analysis["confidence_analysis"][category] = "低信心"
        
        # 優化建議
        low_confidence_features = [k for k, v in analysis["feature_distribution"].items() if v < 0.3]
        if len(low_confidence_features) > 5:
            analysis["optimization_suggestions"].append("建議提高置信度閾值，減少低信心特徵")
        
        if analysis["style_tag_count"] > 8:
            analysis["optimization_suggestions"].append("風格標籤過多，建議限制最大特徵數量")
        
        print(f"   📊 主要組件: {analysis['component_count']}, 風格標籤: {analysis['style_tag_count']}")
        print(f"   💡 優化建議: {len(analysis['optimization_suggestions'])} 項")
        
        return analysis
    
    def analyze_loss_performance(self, losses, similarities):
        """分析損失性能與建議"""
        print("📈 損失性能分析...")
        
        total_loss = losses["total_loss"]
        fashion_clip_loss = losses["fashion_clip_loss"]
        
        # 性能評估
        if total_loss < 0.3:
            performance = "優秀"
            emoji = "🎯"
        elif total_loss < 0.5:
            performance = "良好"
            emoji = "✅"
        elif total_loss < 0.7:
            performance = "一般"
            emoji = "⚠️"
        else:
            performance = "需改善"
            emoji = "❌"
        
        print(f"   {emoji} 整體性能: {performance} (總損失: {total_loss:.3f})")
        
        # FashionCLIP 主要指標分析
        if fashion_clip_loss < 0.2:
            print("   🎯 FashionCLIP 表現優異")
        elif fashion_clip_loss < 0.4:
            print("   ✅ FashionCLIP 表現良好")
        else:
            print("   ⚠️ FashionCLIP 需要優化提示詞策略")
        
        # 權重建議
        if losses["visual_loss"] > 0.8 and losses["fashion_clip_loss"] < 0.3:
            print("   💡 建議: 降低視覺權重，專注語意相似度")
        
        return {
            "performance": performance,
            "total_loss": total_loss,
            "main_metric_loss": fashion_clip_loss
        }

    # ...existing code...
def main():
    """主函數"""
    print("🎯 Day 3: Fashion Training Pipeline")
    print("基於 FashionCLIP 特徵提取與 SD v1.5 的自監督學習系統")
    print("=" * 60)
    
    # 初始化訓練管道
    pipeline = FashionTrainingPipeline()
    
    # 執行訓練流程
    pipeline.run_training_pipeline()

if __name__ == "__main__":
    main()

def test_weight_optimization():
    """測試權重優化功能"""
    print("🧪 權重優化測試")
    print("=" * 40)
    
    # 初始化訓練管道
    pipeline = FashionTrainingPipeline()
    
    # 檢查是否有測試圖片
    test_image = None
    if os.path.exists("day1_results"):
        image_files = [f for f in os.listdir("day1_results") 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            test_image = os.path.join("day1_results", image_files[0])
    
    if test_image:
        # 進行權重方案比較
        results = pipeline.compare_weight_schemes(
            test_image, 
            schemes=["default", "balanced", "fashion_focused", "visual_enhanced"]
        )
        
        if results:
            print("\n✅ 權重優化測試完成！")
            print("💡 建議查看生成的比較報告以選擇最佳權重方案")
        else:
            print("❌ 權重測試失敗")
    else:
        print("❌ 找不到測試圖片")
