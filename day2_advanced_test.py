#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2天：進階測試 - 圖片分析和提示詞生成
目標：測試圖片特徵提取和自動提示詞生成，比較標準 CLIP 和 FashionCLIP
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

# Windows 編碼設定 - 確保 emoji 正確顯示
if sys.platform.startswith('win'):
    import locale
    # 嘗試設定 UTF-8 編碼
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'zh_TW.UTF-8')
        except:
            pass

class Day2Tester:
    def __init__(self):
        self.api_url = "http://localhost:7860"
        self.output_dir = "day2_advanced_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 定義詳細的時尚分類
        self.categories = {
            "Gender": ["male", "female"],
            "Age": ["child", "teenager", "young adult", "adult", "senior"],
            "Season": ["spring", "summer", "autumn", "winter"],
            "Occasion": [
                "casual", "formal", "business", "sport", "party",
                "beach", "wedding", "date", "travel", "home"
            ],
            "Upper Body": [
                "t-shirt", "shirt", "jacket", "coat", "sweater",
                "blazer", "hoodie", "tank top", "blouse", "dress"
            ],
            "Lower Body": [
                "jeans", "trousers", "shorts", "skirt", "leggings",
                "cargo pants", "sweatpants", "culottes", "capris", "dress"
            ]
        }
        
        # 詳細服裝特徵分析 - 深入描述
        self.detailed_clothing_features = {
            "Dress Style": [
                "A-line dress", "sheath dress", "wrap dress", "maxi dress", "midi dress",
                "mini dress", "bodycon dress", "shift dress", "empire waist dress",
                "fit and flare dress", "slip dress", "shirt dress", "sweater dress"
            ],
            "Shirt Features": [
                "button-down shirt", "polo shirt", "henley shirt", "flannel shirt",
                "dress shirt", "peasant blouse", "crop top", "off-shoulder top",
                "turtleneck", "v-neck shirt", "crew neck", "collared shirt"
            ],
            "Jacket Types": [
                "denim jacket", "leather jacket", "bomber jacket", "trench coat",
                "peacoat", "blazer jacket", "cardigan", "windbreaker",
                "puffer jacket", "motorcycle jacket", "varsity jacket"
            ],
            "Pants Details": [
                "skinny jeans", "straight leg jeans", "bootcut jeans", "wide leg pants",
                "high-waisted pants", "low-rise pants", "cropped pants", "palazzo pants",
                "joggers", "dress pants", "cargo pants with pockets"
            ],
            "Skirt Varieties": [
                "pencil skirt", "A-line skirt", "pleated skirt", "wrap skirt",
                "mini skirt", "maxi skirt", "denim skirt", "leather skirt",
                "tulle skirt", "asymmetrical skirt"
            ],
            "Fabric Texture": [
                "cotton fabric", "silk material", "denim texture", "leather finish",
                "wool texture", "linen fabric", "chiffon material", "velvet texture",
                "knit fabric", "lace material", "satin finish", "corduroy texture"
            ],
            "Pattern Details": [
                "solid color", "striped pattern", "floral print", "polka dots",
                "geometric pattern", "animal print", "plaid pattern", "paisley design",
                "abstract print", "tie-dye pattern", "checkered pattern"
            ],
            "Color Scheme": [
                "monochrome outfit", "pastel colors", "bright colors", "earth tones",
                "neutral colors", "bold colors", "metallic accents", "neon colors",
                "vintage colors", "gradient colors"
            ],
            "Fit Description": [
                "loose fit", "tight fit", "oversized", "fitted", "relaxed fit",
                "tailored fit", "slim fit", "regular fit", "cropped length",
                "flowing silhouette", "structured shape"
            ],
            "Style Details": [
                "minimalist style", "vintage style", "bohemian style", "gothic style",
                "preppy style", "streetwear style", "romantic style", "edgy style",
                "classic style", "trendy style", "elegant style"
            ]
        }
        
    def install_requirements(self):
        """檢查並安裝必要的套件"""
        required_packages = {
            'torch': 'torch',
            'transformers': 'transformers', 
            'PIL': 'pillow',
            'numpy': 'numpy'
        }
        
        missing_packages = []
        
        for import_name, package_name in required_packages.items():
            try:
                __import__(import_name)
                print(f"✅ {package_name} 已安裝")
            except ImportError:
                missing_packages.append(package_name)
                print(f"❌ 缺少套件: {package_name}")
        
        if missing_packages:
            print(f"\n🔧 請安裝缺少的套件:")
            print(f"pip install {' '.join(missing_packages)}")
            print("\n或者運行完整安裝命令:")
            print("pip install torch transformers pillow numpy")
            return False
        
        print("✅ 所有必要套件已安裝")
        return True
    
    def load_clip_models(self):
        """載入標準 CLIP 和 FashionCLIP 模型用於比較"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            models = {}
            
            # 載入標準 CLIP
            print("📥 載入標準 CLIP...")
            standard_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            standard_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            standard_model.to(device)
            models["standard_clip"] = (standard_model, standard_processor)
            print("✅ 標準 CLIP 載入成功")
            
            # 載入 FashionCLIP
            try:
                print("📥 載入 FashionCLIP...")
                fashion_model = CLIPModel.from_pretrained(
                    "patrickjohncyh/fashion-clip",
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                fashion_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
                fashion_model.to(device)
                models["fashion_clip"] = (fashion_model, fashion_processor)
                print("✅ FashionCLIP 載入成功")
            except Exception as e:
                print(f"⚠️ FashionCLIP 載入失敗，僅使用標準 CLIP: {e}")
            
            return models
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            return {}
    
    def analyze_image_features(self, image_path, models):
        """使用多個 CLIP 模型分析圖片特徵並比較結果 - 包含詳細服裝特徵"""
        try:
            from PIL import Image
            import torch
            
            image = Image.open(image_path).convert("RGB")
            
            # 合併基本分類和詳細特徵分析
            all_categories = {**self.categories, **self.detailed_clothing_features}
            
            results = {}
            
            for model_name, (model, processor) in models.items():
                print(f"   🔍 使用 {model_name} 進行詳細分析...")
                
                model_results = {}
                device = next(model.parameters()).device
                model_dtype = next(model.parameters()).dtype
                
                # 對每個類別進行分析
                for category_name, labels in all_categories.items():
                    try:
                        # 使用 CLIP 分析圖片
                        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        # 確保輸入精度與模型匹配
                        if model_dtype == torch.float16:
                            for key in inputs:
                                if inputs[key].dtype == torch.float32:
                                    inputs[key] = inputs[key].half()
                        
                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits_per_image = outputs.logits_per_image
                            probs = logits_per_image.softmax(dim=1)
                        
                        # 獲取最相關的標籤（前3個）
                        top_indices = probs[0].topk(min(3, len(labels))).indices
                        top_labels = [labels[i] for i in top_indices]
                        top_scores = [probs[0][i].item() for i in top_indices]
                        
                        model_results[category_name] = {
                            "top_labels": top_labels,
                            "scores": top_scores,
                            "confidence": max(top_scores),
                            "category_type": "detailed" if category_name in self.detailed_clothing_features else "basic"
                        }
                        
                    except Exception as e:
                        model_results[category_name] = {
                            "error": str(e),
                            "confidence": 0.0,
                            "category_type": "detailed" if category_name in self.detailed_clothing_features else "basic"
                        }
                
                results[model_name] = {
                    "categories": model_results,
                    "analysis_success": True
                }
            
            return results
            
        except Exception as e:
            print(f"❌ 圖片分析失敗: {e}")
            return {"analysis_success": False, "error": str(e)}
    
    def generate_prompt_from_analysis(self, analysis_results):
        """根據多模型分析結果生成 SD 提示詞並比較差異"""
        prompts = {}
        
        for model_name, model_result in analysis_results.items():
            if not model_result.get("analysis_success", False):
                prompts[model_name] = "elegant fashion, high quality, professional photography"
                continue
            
            categories = model_result["categories"]
            prompt_parts = []
            
            # 從每個類別提取最佳標籤
            for category_name, category_result in categories.items():
                if "top_labels" in category_result and category_result["top_labels"]:
                    # 選擇置信度最高的標籤
                    best_label = category_result["top_labels"][0]
                    if category_result["confidence"] > 0.3:  # 只有置信度足夠高才加入
                        prompt_parts.append(best_label)
            
            # 構建提示詞
            if prompt_parts:
                base_prompt = ", ".join(prompt_parts)
                enhanced_prompt = f"{base_prompt}, high fashion, professional photography, detailed, high quality, studio lighting"
            else:
                enhanced_prompt = "elegant fashion, high quality, professional photography"
            
            prompts[model_name] = enhanced_prompt
        
        return prompts
    
    def compare_model_performance(self, all_results):
        """詳細比較不同模型的表現 - 特別關注服裝細節分析能力"""
        print("\n" + "=" * 80)
        print("📊 Standard CLIP vs FashionCLIP 詳細比較分析")
        print("=" * 80)
        
        model_stats = {}
        category_comparisons = {}
        
        # 收集統計數據
        for result in all_results:
            if result["success"] and "analysis" in result:
                for model_name, model_result in result["analysis"].items():
                    if model_name not in model_stats:
                        model_stats[model_name] = {
                            "total_analyses": 0,
                            "basic_categories": {},
                            "detailed_categories": {},
                            "overall_confidences": []
                        }
                    
                    model_stats[model_name]["total_analyses"] += 1
                    
                    if model_result.get("analysis_success", False):
                        categories = model_result["categories"]
                        for category_name, category_result in categories.items():
                            if "confidence" in category_result:
                                confidence = category_result["confidence"]
                                model_stats[model_name]["overall_confidences"].append(confidence)
                                
                                # 區分基本類別和詳細類別
                                category_type = category_result.get("category_type", "basic")
                                target_dict = model_stats[model_name]["detailed_categories" if category_type == "detailed" else "basic_categories"]
                                
                                if category_name not in target_dict:
                                    target_dict[category_name] = []
                                target_dict[category_name].append(confidence)
                                
                                # 收集兩個模型在同一類別的比較數據
                                if category_name not in category_comparisons:
                                    category_comparisons[category_name] = {}
                                if model_name not in category_comparisons[category_name]:
                                    category_comparisons[category_name][model_name] = []
                                category_comparisons[category_name][model_name].append({
                                    "confidence": confidence,
                                    "prediction": category_result.get("top_labels", ["unknown"])[0],
                                    "image": result["image_name"]
                                })
        
        # 顯示整體統計
        print("\n🏆 整體表現比較:")
        print("-" * 60)
        
        for model_name, stats in model_stats.items():
            overall_avg = sum(stats["overall_confidences"]) / len(stats["overall_confidences"]) if stats["overall_confidences"] else 0
            basic_confidences = []
            detailed_confidences = []
            
            for confidences in stats["basic_categories"].values():
                basic_confidences.extend(confidences)
            for confidences in stats["detailed_categories"].values():
                detailed_confidences.extend(confidences)
            
            basic_avg = sum(basic_confidences) / len(basic_confidences) if basic_confidences else 0
            detailed_avg = sum(detailed_confidences) / len(detailed_confidences) if detailed_confidences else 0
            
            print(f"\n🤖 {model_name.upper().replace('_', ' ')}:")
            print(f"   整體平均置信度: {overall_avg:.3f}")
            print(f"   基本分類平均: {basic_avg:.3f} (共 {len(basic_confidences)} 個預測)")
            print(f"   詳細特徵平均: {detailed_avg:.3f} (共 {len(detailed_confidences)} 個預測)")
        
        # 詳細類別比較
        print(f"\n🔍 詳細服裝特徵分析比較:")
        print("-" * 60)
        
        detailed_categories = [cat for cat in category_comparisons.keys() if cat in self.detailed_clothing_features]
        
        for category_name in detailed_categories[:5]:  # 顯示前5個詳細類別
            if len(category_comparisons[category_name]) >= 2:  # 確保兩個模型都有數據
                print(f"\n📋 {category_name}:")
                
                for model_name, predictions in category_comparisons[category_name].items():
                    avg_conf = sum(p["confidence"] for p in predictions) / len(predictions)
                    best_prediction = max(predictions, key=lambda x: x["confidence"])
                    print(f"   {model_name.replace('_', ' ').title()}:")
                    print(f"     平均置信度: {avg_conf:.3f}")
                    print(f"     最佳預測: {best_prediction['prediction']} (置信度: {best_prediction['confidence']:.3f})")
        
        # 模型優勢分析
        print(f"\n🎯 模型優勢分析:")
        print("-" * 60)
        
        standard_wins = 0
        fashion_wins = 0
        
        for category_name, models_data in category_comparisons.items():
            if len(models_data) >= 2:
                model_avgs = {}
                for model_name, predictions in models_data.items():
                    model_avgs[model_name] = sum(p["confidence"] for p in predictions) / len(predictions)
                
                best_model = max(model_avgs, key=model_avgs.get)
                if "standard" in best_model.lower():
                    standard_wins += 1
                elif "fashion" in best_model.lower():
                    fashion_wins += 1
        
        print(f"Standard CLIP 領先類別: {standard_wins}")
        print(f"FashionCLIP 領先類別: {fashion_wins}")
        
        if fashion_wins > standard_wins:
            print("🏆 FashionCLIP 在大多數類別中表現更佳")
        elif standard_wins > fashion_wins:
            print("🏆 Standard CLIP 在大多數類別中表現更佳")
        else:
            print("🤝 兩個模型表現相當")
        
        return {
            "model_stats": model_stats,
            "category_comparisons": category_comparisons,
            "winner_summary": {
                "standard_clip_wins": standard_wins,
                "fashion_clip_wins": fashion_wins,
                "total_categories": len(category_comparisons)
            }
        }
    
    def test_image_to_prompt_generation(self):
        """測試圖片分析到提示詞生成的完整流程 - 分析所有找到的圖片"""
        print("🔍 尋找所有可分析的圖片...")
        
        # 尋找可用的圖片文件 - 移除數量限制
        image_files = []
        search_dirs = ["day1_results", "outputs", "day2_enhanced_results", "test_images"]
        
        for search_dir in search_dirs:
                print(f"   📂 搜索資料夾: {search_dir}")
                dir_files = []
                for file in os.listdir(search_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        dir_files.append(os.path.join(search_dir, file))
                
                if dir_files:
                    print(f"      找到 {len(dir_files)} 張圖片")
                    image_files.extend(dir_files)
                    # 如果是 day1_results，優先分析這些圖片
                    if search_dir == "day1_results":
                        break
        
        if not image_files:
            print("❌ 找不到可分析的圖片文件")
            print("請確保以下資料夾中有圖片文件:")
            for dir_name in search_dirs:
                print(f"  - {dir_name}")
            return []
        
        print(f"✅ 總共找到 {len(image_files)} 張圖片進行深度分析")
        
        # 載入模型
        models = self.load_clip_models()
        if not models:
            print("❌ 無法載入 CLIP 模型，跳過圖片分析測試")
            return []
        
        results = []
        
        # 分析每張找到的圖片
        for i, image_path in enumerate(image_files):
            print(f"\n--- 深度分析圖片 {i+1}/{len(image_files)}: {os.path.basename(image_path)} ---")
            
            try:
                # 分析圖片特徵
                analysis = self.analyze_image_features(image_path, models)
                
                # 根據分析結果生成提示詞
                generated_prompts = self.generate_prompt_from_analysis(analysis)
                
                results.append({
                    "image_path": image_path,
                    "image_name": os.path.basename(image_path),
                    "analysis": analysis,
                    "generated_prompts": generated_prompts,
                    "success": True
                })
                print(f"✅ 分析完成")
                
            except Exception as e:
                results.append({
                    "image_path": image_path,
                    "image_name": os.path.basename(image_path),
                    "error": str(e),
                    "success": False
                })
                print(f"❌ 分析失敗: {e}")
        
        print(f"\n🏁 完成所有圖片分析：{len([r for r in results if r['success']])}/{len(results)} 成功")
        return results
    
    def run_day2_tests(self):
        """運行第2天的所有測試"""
        print("=" * 60)
        print("第2天進階測試：CLIP 模型比較分析")
        print("=" * 60)
        
        # 檢查環境
        if not self.install_requirements():
            print("❌ 環境檢查失敗，請安裝必要套件")
            return False
        
        # 運行圖片分析測試
        results = self.test_image_to_prompt_generation()
        if not results:
            print("❌ 沒有可分析的結果")
            return False
        
        successful = sum(1 for r in results if r["success"])
        print(f"\n第2天測試完成：{successful}/{len(results)} 個分析成功")
        
        # 顯示詳細模型比較分析
        comparison_results = None
        if successful > 0:
            comparison_results = self.compare_model_performance(results)
        
        # 生成詳細報告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            "test_name": "Day 2 Deep Fashion Analysis: CLIP vs FashionCLIP",
            "timestamp": datetime.now().isoformat(),
            "tests_run": len(results),
            "tests_successful": successful,
            "success_rate": successful / len(results) if results else 0,
            "results": results,
            "categories_analyzed": list(self.categories.keys()),
            "detailed_features_analyzed": list(self.detailed_clothing_features.keys()),
            "comparison_summary": comparison_results
        }
        
        # 保存 JSON 報告
        json_path = os.path.join(self.output_dir, f"day2_advanced_report_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 生成 HTML 報告
        html_path = os.path.join(self.output_dir, f"day2_advanced_report_{timestamp}.html")
        self.generate_html_report(report, html_path)
        
        # 生成 Markdown 報告
        md_path = os.path.join(self.output_dir, f"day2_advanced_report_{timestamp}.md")
        self.generate_markdown_report(report, md_path)
        
        print(f"\n📊 報告已生成:")
        print(f"   JSON: {json_path}")
        print(f"   HTML: {html_path}")
        print(f"   Markdown: {md_path}")
        
        return True
    
    def generate_html_report(self, report, html_path):
        """生成詳細的 HTML 格式報告 - 展示服裝特徵深度分析"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>深度時尚分析：CLIP 模型比較報告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 10px; }}
        h3 {{ color: #7f8c8d; }}
        .model-comparison {{ display: flex; gap: 20px; margin: 20px 0; }}
        .model-card {{ flex: 1; border: 1px solid #ddd; border-radius: 8px; padding: 15px; background-color: #f9f9f9; }}
        .confidence-high {{ color: #27ae60; font-weight: bold; }}
        .confidence-medium {{ color: #f39c12; font-weight: bold; }}
        .confidence-low {{ color: #e74c3c; font-weight: bold; }}
        .basic-category {{ background-color: #ecf0f1; border-left: 3px solid #3498db; }}
        .detailed-category {{ background-color: #fef9e7; border-left: 3px solid #f39c12; }}
        .category-result {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
        .image-analysis {{ border: 1px solid #ddd; margin: 20px 0; padding: 15px; border-radius: 8px; }}
        .prompt-box {{ background-color: #e8f5e8; padding: 10px; border-radius: 5px; margin: 5px 0; font-family: monospace; font-size: 0.9em; }}
        .feature-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
        .feature-section {{ border: 1px solid #ddd; padding: 15px; border-radius: 8px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .winner {{ background-color: #d5f4e6; }}
        .legend {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>� 深度時尚分析：CLIP 模型比較報告</h1>
        <p><strong>生成時間:</strong> {report["timestamp"]}</p>
        <p><strong>測試成功率:</strong> {report["success_rate"]:.1%} ({report["tests_successful"]}/{report["tests_run"]})</p>
        
        <div class="legend">
            <strong>🔍 分析說明:</strong><br>
            <span style="background-color: #ecf0f1; padding: 2px 5px; border-radius: 3px;">藍色區塊</span> = 基本分類 (性別、年齡、場合等)<br>
            <span style="background-color: #fef9e7; padding: 2px 5px; border-radius: 3px;">黃色區塊</span> = 詳細特徵 (款式、材質、剪裁等)
        </div>
        
        <h2>🔍 詳細分析結果</h2>
"""
        
        # 顯示所有成功分析的結果
        successful_results = [r for r in report["results"] if r["success"]]
        for i, result in enumerate(successful_results, 1):
            if not result["success"]:
                continue
            
            html_content += f"""
        <div class="image-analysis">
            <h3>📸 圖片 {i}: {result["image_name"]}</h3>
            
            <div class="feature-grid">
"""
            
            if "analysis" in result:
                for model_name, analysis in result["analysis"].items():
                    if analysis.get("analysis_success", False):
                        html_content += f"""
                <div class="feature-section">
                    <h4>🤖 {model_name.replace('_', ' ').title()} 分析結果</h4>
"""
                        
                        categories = analysis.get("categories", {})
                        
                        # 分別顯示基本分類和詳細特徵
                        basic_categories = []
                        detailed_categories = []
                        
                        for category_name, category_result in categories.items():
                            if "top_labels" in category_result:
                                category_type = category_result.get("category_type", "basic")
                                if category_type == "basic":
                                    basic_categories.append((category_name, category_result))
                                else:
                                    detailed_categories.append((category_name, category_result))
                        
                        # 顯示基本分類
                        if basic_categories:
                            html_content += "<h5>📋 基本分類</h5>"
                            for category_name, category_result in basic_categories[:3]:
                                confidence = category_result["confidence"]
                                confidence_class = "confidence-high" if confidence >= 0.7 else "confidence-medium" if confidence >= 0.5 else "confidence-low"
                                html_content += f"""
                    <div class="category-result basic-category">
                        <strong>{category_name}:</strong> {category_result["top_labels"][0]} 
                        <span class="{confidence_class}">(置信度: {confidence:.3f})</span>
                    </div>
"""
                        
                        # 顯示詳細特徵
                        if detailed_categories:
                            html_content += "<h5>🎨 詳細特徵</h5>"
                            for category_name, category_result in detailed_categories[:4]:
                                confidence = category_result["confidence"]
                                confidence_class = "confidence-high" if confidence >= 0.7 else "confidence-medium" if confidence >= 0.5 else "confidence-low"
                                html_content += f"""
                    <div class="category-result detailed-category">
                        <strong>{category_name}:</strong> {category_result["top_labels"][0]} 
                        <span class="{confidence_class}">(置信度: {confidence:.3f})</span>
                    </div>
"""
                        
                        html_content += """
                </div>
"""
                
                html_content += """
            </div>
"""
                
                # 顯示生成的提示詞比較
                if "generated_prompts" in result:
                    html_content += """
            <h4>💭 生成的提示詞比較</h4>
            <div class="model-comparison">
"""
                    for model_name, prompt in result["generated_prompts"].items():
                        html_content += f"""
                <div class="model-card">
                    <h5>{model_name.replace('_', ' ').title()}</h5>
                    <div class="prompt-box">{prompt}</div>
                </div>
"""
                    html_content += """
            </div>
"""
            
            html_content += "</div>"
        
        # 添加模型比較總結
        html_content += """
        <h2>🏆 模型表現總結</h2>
        <div class="model-comparison">
            <div class="model-card">
                <h3>Standard CLIP</h3>
                <p><strong>優勢:</strong> 通用性強，對整體場景理解好</p>
                <p><strong>適用:</strong> 一般圖像分析、多領域應用</p>
            </div>
            <div class="model-card">
                <h3>FashionCLIP</h3>
                <p><strong>優勢:</strong> 時尚專業性，服飾細節識別精準</p>
                <p><strong>適用:</strong> 時尚電商、服裝設計、風格分析</p>
            </div>
        </div>
        
        <h2>💡 深度分析洞察</h2>
        <ul>
            <li><strong>詳細特徵識別:</strong> 本測試分析了 {len(report.get("categories_analyzed", []))} 個基本類別和多個詳細服裝特徵</li>
            <li><strong>模型專業性:</strong> FashionCLIP 在識別具體服裝款式（如 A-line dress、skinny jeans）上通常更準確</li>
            <li><strong>應用建議:</strong> 根據應用場景選擇合適的模型，時尚相關應用推薦 FashionCLIP</li>
            <li><strong>置信度參考:</strong> 高於 0.7 為優秀，0.5-0.7 為良好，低於 0.5 需要謹慎參考</li>
        </ul>
    </div>
</body>
</html>
"""
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    
    def generate_markdown_report(self, report, md_path):
        """生成 Markdown 格式報告"""
        md_content = f"""# 📊 Day 2 進階 CLIP 比較報告

**生成時間:** {report["timestamp"]}  
**測試成功率:** {report["success_rate"]:.1%} ({report["tests_successful"]}/{report["tests_run"]})

## 🔍 分析結果

"""
        
        # 顯示所有成功分析的結果
        successful_results = [r for r in report["results"] if r["success"]]
        for i, result in enumerate(successful_results, 1):
            if not result["success"]:
                continue
            
            md_content += f"### 圖片 {i}: {result['image_name']}\n\n"
            
            if "analysis" in result:
                for model_name, analysis in result["analysis"].items():
                    if analysis.get("analysis_success", False):
                        md_content += f"#### {model_name.replace('_', ' ').title()}\n\n"
                        
                        categories = analysis.get("categories", {})
                        for category_name, category_result in list(categories.items())[:4]:
                            if "top_labels" in category_result:
                                confidence = category_result["confidence"]
                                md_content += f"- **{category_name}:** {category_result['top_labels'][0]} (置信度: {confidence:.3f})\n"
                        
                        md_content += "\n"
                
                # 顯示生成的提示詞
                if "generated_prompts" in result:
                    md_content += "**生成的提示詞:**\n\n"
                    for model_name, prompt in result["generated_prompts"].items():
                        md_content += f"- **{model_name.replace('_', ' ').title()}:** `{prompt}`\n"
                    md_content += "\n"
        
        md_content += """
## 💡 總結

這個測試比較了標準 CLIP 和 FashionCLIP 在時尚圖片分析上的表現差異。您可以參考置信度分數來判斷哪個模型在特定類別上表現更好。

## 🔧 技術說明

- 使用 PyTorch 和 Transformers 庫
- 支援 CUDA 加速和混合精度推理
- 分析了 {len(report.get("categories_analyzed", []))} 個時尚類別
- 自動化的提示詞生成流程
"""
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

if __name__ == "__main__":
    tester = Day2Tester()
    tester.run_day2_tests()
