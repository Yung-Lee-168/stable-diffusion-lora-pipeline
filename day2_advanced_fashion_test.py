#!/usr/bin/env python3
"""
第2天：進階時尚 CLIP 比較測試
目標：深度比較標準 CLIP 與 FashionCLIP 在時尚圖片分析上的差異
"""

import json
import os
from datetime import datetime
from PIL import Image
import numpy as np
import torch

class FashionCLIPComparison:
    def __init__(self):
        self.output_dir = "day2_fashion_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 詳細時尚分類結構
        self.fashion_categories = {
            "Gender": {
                "labels": ["male", "female", "unisex"],
                "description": "性別定位"
            },
            "Age Group": {
                "labels": ["child", "teenager", "young adult", "adult", "senior"],
                "description": "年齡群體"
            },
            "Season": {
                "labels": ["spring", "summer", "autumn", "winter", "all season"],
                "description": "季節適用性"
            },
            "Occasion": {
                "labels": [
                    "casual", "formal", "business", "sport", "party",
                    "beach", "wedding", "date", "travel", "home", "work"
                ],
                "description": "場合穿搭"
            },
            "Style": {
                "labels": [
                    "minimalist", "vintage", "bohemian", "street style", "classic",
                    "punk", "romantic", "sporty", "elegant", "trendy"
                ],
                "description": "風格類型"
            },
            "Upper Body": {
                "labels": [
                    "t-shirt", "shirt", "blouse", "jacket", "coat", "sweater",
                    "hoodie", "tank top", "blazer", "cardigan", "vest"
                ],
                "description": "上身服飾"
            },
            "Lower Body": {
                "labels": [
                    "jeans", "trousers", "shorts", "skirt", "dress", "leggings",
                    "cargo pants", "sweatpants", "culottes", "palazzo pants"
                ],
                "description": "下身服飾"
            },
            "Color Palette": {
                "labels": [
                    "monochrome", "bright colors", "pastel", "earth tones",
                    "neon", "metallic", "neutral", "bold colors"
                ],
                "description": "色彩風格"
            },
            "Pattern": {
                "labels": [
                    "solid", "stripes", "floral", "geometric", "polka dots",
                    "animal print", "plaid", "paisley", "abstract"
                ],
                "description": "圖案類型"
            },
            "Fabric Feel": {
                "labels": [
                    "cotton", "silk", "denim", "leather", "wool", "knit",
                    "chiffon", "lace", "velvet", "linen"
                ],
                "description": "材質感覺"
            }
        }
        
    def install_requirements(self):
        """檢查並安裝必要套件"""
        required_packages = [
            ("torch", "torch"),
            ("transformers", "transformers"), 
            ("PIL", "pillow"),
            ("numpy", "numpy")
        ]
        
        missing_packages = []
        installed_packages = []
        
        for import_name, package_name in required_packages:
            try:
                __import__(import_name)
                installed_packages.append(package_name)
            except ImportError:
                missing_packages.append(package_name)
        
        if installed_packages:
            print(f"✅ 已安裝的套件: {', '.join(installed_packages)}")
        
        if missing_packages:
            print(f"❌ 缺少套件: {', '.join(missing_packages)}")
            print(f"\n🔧 請運行以下命令安裝缺少的套件:")
            print(f"pip install {' '.join(missing_packages)}")
            print(f"\n或運行完整安裝命令:")
            print(f"pip install torch transformers pillow numpy")
            print(f"\n您也可以運行: INSTALL_DEPENDENCIES.bat")
            return False
        
        print("✅ 所有必要套件已安裝")
        return True
    
    def load_clip_models(self):
        """載入標準 CLIP 和 FashionCLIP 模型"""
        from transformers import CLIPProcessor, CLIPModel
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"🚀 使用設備: {device}")
        print(f"🔧 使用精度: {torch_dtype}")
        
        models = {}
        
        # 載入標準 CLIP
        try:
            print("📥 載入標準 CLIP (OpenAI ViT-B/32)...")
            standard_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                torch_dtype=torch_dtype
            ).to(device)
            standard_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            models["Standard CLIP"] = {
                "model": standard_model,
                "processor": standard_processor,
                "description": "通用圖像理解模型"
            }
            print("✅ 標準 CLIP 載入成功")
        except Exception as e:
            print(f"❌ 標準 CLIP 載入失敗: {e}")
        
        # 載入 FashionCLIP
        try:
            print("📥 載入 FashionCLIP (專業時尚模型)...")
            fashion_model = CLIPModel.from_pretrained(
                "patrickjohncyh/fashion-clip",
                torch_dtype=torch_dtype
            ).to(device)
            fashion_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
            
            models["FashionCLIP"] = {
                "model": fashion_model,
                "processor": fashion_processor,
                "description": "時尚專業圖像理解模型"
            }
            print("✅ FashionCLIP 載入成功")
        except Exception as e:
            print(f"❌ FashionCLIP 載入失敗: {e}")
        
        if not models:
            print("❌ 無法載入任何模型")
            return None
        
        print(f"🎯 成功載入 {len(models)} 個模型")
        return models
    
    def analyze_single_image(self, image_path, models):
        """使用多個模型分析單張圖片"""
        try:
            image = Image.open(image_path).convert("RGB")
            print(f"🔍 分析圖片: {os.path.basename(image_path)}")
            
            results = {}
            
            for model_name, model_info in models.items():
                print(f"   📊 {model_name} 分析中...")
                
                model = model_info["model"]
                processor = model_info["processor"]
                device = next(model.parameters()).device
                
                model_result = {
                    "categories": {},
                    "overall_confidence": 0.0,
                    "description": model_info["description"]
                }
                
                total_confidence = 0
                valid_categories = 0
                
                # 分析每個時尚類別
                for category_name, category_info in self.fashion_categories.items():
                    labels = category_info["labels"]
                    
                    try:
                        # 準備輸入
                        inputs = processor(
                            text=labels, 
                            images=image, 
                            return_tensors="pt", 
                            padding=True
                        )
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        # 模型推理
                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits_per_image = outputs.logits_per_image
                            probs = logits_per_image.softmax(dim=1)
                        
                        # 獲取最佳匹配
                        top_3_indices = probs[0].topk(min(3, len(labels))).indices
                        top_3_labels = [labels[i] for i in top_3_indices]
                        top_3_scores = [probs[0][i].item() for i in top_3_indices]
                        
                        model_result["categories"][category_name] = {
                            "top_predictions": list(zip(top_3_labels, top_3_scores)),
                            "best_match": top_3_labels[0],
                            "confidence": top_3_scores[0],
                            "description": category_info["description"]
                        }
                        
                        total_confidence += top_3_scores[0]
                        valid_categories += 1
                        
                    except Exception as e:
                        model_result["categories"][category_name] = {
                            "error": str(e),
                            "confidence": 0.0,
                            "description": category_info["description"]
                        }
                
                # 計算整體置信度
                if valid_categories > 0:
                    model_result["overall_confidence"] = total_confidence / valid_categories
                
                results[model_name] = model_result
            
            return results
            
        except Exception as e:
            print(f"❌ 圖片分析失敗: {e}")
            return {}
    
    def generate_fashion_prompt(self, analysis_result, model_name):
        """根據分析結果生成時尚描述提示詞"""
        if not analysis_result or "categories" not in analysis_result:
            return "elegant fashion photography, high quality"
        
        categories = analysis_result["categories"]
        prompt_parts = []
        
        # 提取高置信度的特徵
        for category_name, category_result in categories.items():
            if "best_match" in category_result and category_result["confidence"] > 0.3:
                prompt_parts.append(category_result["best_match"])
        
        # 構建提示詞
        if prompt_parts:
            base_prompt = ", ".join(prompt_parts[:6])  # 限制長度
            enhanced_prompt = f"{base_prompt}, professional fashion photography, high quality, detailed, studio lighting"
        else:
            enhanced_prompt = "elegant fashion style, professional photography, high quality"
        
        return enhanced_prompt
    
    def compare_models_on_images(self, image_folder="day1_results"):
        """比較兩個模型在所有圖片上的表現"""
        # 載入模型
        models = self.load_clip_models()
        if not models:
            return None
        
        # 尋找圖片
        image_files = []
        if os.path.exists(image_folder):
            for file in os.listdir(image_folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(image_folder, file))
        
        if not image_files:
            print(f"❌ 在 {image_folder} 中找不到圖片文件")
            return None
        
        print(f"🖼️ 找到 {len(image_files)} 張圖片進行分析")
        
        all_results = []
        
        # 分析每張圖片
        for i, image_path in enumerate(image_files, 1):
            print(f"\n--- 分析第 {i}/{len(image_files)} 張圖片 ---")
            
            analysis = self.analyze_single_image(image_path, models)
            
            if analysis:
                # 為每個模型生成提示詞
                prompts = {}
                for model_name, model_result in analysis.items():
                    prompts[model_name] = self.generate_fashion_prompt(model_result, model_name)
                
                result = {
                    "image_path": image_path,
                    "image_name": os.path.basename(image_path),
                    "analysis": analysis,
                    "generated_prompts": prompts,
                    "success": True
                }
            else:
                result = {
                    "image_path": image_path,
                    "image_name": os.path.basename(image_path),
                    "success": False,
                    "error": "分析失敗"
                }
            
            all_results.append(result)
        
        return all_results
    
    def generate_comparison_report(self, results):
        """生成詳細的模型比較報告"""
        if not results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 生成 JSON 報告
        report = {
            "test_name": "Fashion CLIP Comparison Test",
            "timestamp": datetime.now().isoformat(),
            "model_comparison": {
                "Standard CLIP": "通用圖像理解模型，適用於廣泛的圖像分析任務",
                "FashionCLIP": "專門針對時尚領域訓練的模型，對服飾理解更精準"
            },
            "fashion_categories": self.fashion_categories,
            "results": results,
            "summary": self._calculate_summary(results)
        }
        
        json_path = os.path.join(self.output_dir, f"fashion_comparison_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 生成 HTML 報告
        html_path = os.path.join(self.output_dir, f"fashion_comparison_{timestamp}.html")
        self._generate_html_report(report, html_path)
        
        # 生成 Markdown 報告
        md_path = os.path.join(self.output_dir, f"fashion_comparison_{timestamp}.md")
        self._generate_markdown_report(report, md_path)
        
        print(f"\n📊 報告已生成:")
        print(f"   JSON: {json_path}")
        print(f"   HTML: {html_path}")
        print(f"   Markdown: {md_path}")
        
        return report
    
    def _calculate_summary(self, results):
        """計算統計摘要"""
        if not results:
            return {}
        
        successful_results = [r for r in results if r["success"]]
        
        if not successful_results:
            return {"error": "沒有成功的分析結果"}
        
        model_names = list(successful_results[0]["analysis"].keys())
        summary = {}
        
        for model_name in model_names:
            model_confidences = []
            category_stats = {}
            
            for result in successful_results:
                analysis = result["analysis"].get(model_name, {})
                categories = analysis.get("categories", {})
                
                for category_name, category_result in categories.items():
                    if "confidence" in category_result:
                        if category_name not in category_stats:
                            category_stats[category_name] = []
                        category_stats[category_name].append(category_result["confidence"])
                        model_confidences.append(category_result["confidence"])
            
            # 計算每個類別的平均置信度
            category_averages = {}
            for category_name, confidences in category_stats.items():
                if confidences:
                    category_averages[category_name] = {
                        "average_confidence": sum(confidences) / len(confidences),
                        "sample_count": len(confidences),
                        "max_confidence": max(confidences),
                        "min_confidence": min(confidences)
                    }
            
            summary[model_name] = {
                "overall_average_confidence": sum(model_confidences) / len(model_confidences) if model_confidences else 0,
                "total_predictions": len(model_confidences),
                "category_performance": category_averages
            }
        
        return summary
    
    def _generate_html_report(self, report, html_path):
        """生成 HTML 格式報告"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fashion CLIP 模型比較報告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 10px; }}
        h3 {{ color: #7f8c8d; }}
        .model-comparison {{ display: flex; gap: 20px; margin: 20px 0; }}
        .model-card {{ flex: 1; border: 1px solid #ddd; border-radius: 8px; padding: 15px; background-color: #f9f9f9; }}
        .confidence-high {{ color: #27ae60; font-weight: bold; }}
        .confidence-medium {{ color: #f39c12; font-weight: bold; }}
        .confidence-low {{ color: #e74c3c; font-weight: bold; }}
        .category-result {{ margin: 10px 0; padding: 10px; border-left: 3px solid #3498db; background-color: #ecf0f1; }}
        .image-analysis {{ border: 1px solid #ddd; margin: 20px 0; padding: 15px; border-radius: 8px; }}
        .prompt-box {{ background-color: #e8f5e8; padding: 10px; border-radius: 5px; margin: 5px 0; font-family: monospace; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Fashion CLIP 模型比較報告</h1>
        <p><strong>生成時間:</strong> {report["timestamp"]}</p>
        
        <h2>📋 模型簡介</h2>
        <div class="model-comparison">
"""
        
        for model_name, description in report["model_comparison"].items():
            html_content += f"""
            <div class="model-card">
                <h3>{model_name}</h3>
                <p>{description}</p>
            </div>
"""
        
        # 添加統計摘要
        if "summary" in report:
            html_content += """
        </div>
        
        <h2>📊 整體表現統計</h2>
        <table>
            <tr>
                <th>模型</th>
                <th>整體平均置信度</th>
                <th>總預測次數</th>
                <th>表現評級</th>
            </tr>
"""
            
            for model_name, stats in report["summary"].items():
                avg_conf = stats.get("overall_average_confidence", 0)
                total_pred = stats.get("total_predictions", 0)
                
                if avg_conf >= 0.7:
                    grade = "優秀 ⭐⭐⭐"
                    grade_class = "confidence-high"
                elif avg_conf >= 0.5:
                    grade = "良好 ⭐⭐"
                    grade_class = "confidence-medium"
                else:
                    grade = "一般 ⭐"
                    grade_class = "confidence-low"
                
                html_content += f"""
            <tr>
                <td><strong>{model_name}</strong></td>
                <td>{avg_conf:.3f}</td>
                <td>{total_pred}</td>
                <td class="{grade_class}">{grade}</td>
            </tr>
"""
            
            html_content += "</table>"
        
        # 添加詳細分析結果
        html_content += """
        <h2>🔍 詳細分析結果</h2>
"""
        
        for i, result in enumerate(report["results"][:5], 1):  # 只顯示前5個結果
            if not result["success"]:
                continue
            
            html_content += f"""
        <div class="image-analysis">
            <h3>圖片 {i}: {result["image_name"]}</h3>
"""
            
            for model_name, analysis in result["analysis"].items():
                html_content += f"""
            <h4>{model_name} 分析結果</h4>
            <p><strong>整體置信度:</strong> 
                <span class="{'confidence-high' if analysis.get('overall_confidence', 0) >= 0.7 else 'confidence-medium' if analysis.get('overall_confidence', 0) >= 0.5 else 'confidence-low'}">
                    {analysis.get('overall_confidence', 0):.3f}
                </span>
            </p>
"""
                
                # 顯示類別結果
                categories = analysis.get("categories", {})
                for category_name, category_result in list(categories.items())[:3]:  # 只顯示前3個類別
                    if "best_match" in category_result:
                        html_content += f"""
            <div class="category-result">
                <strong>{category_name}:</strong> {category_result["best_match"]} 
                (置信度: {category_result["confidence"]:.3f})
            </div>
"""
                
                # 顯示生成的提示詞
                if "generated_prompts" in result:
                    prompt = result["generated_prompts"].get(model_name, "")
                    html_content += f"""
            <div class="prompt-box">
                <strong>生成的提示詞:</strong><br>{prompt}
            </div>
"""
            
            html_content += "</div>"
        
        html_content += """
    </div>
</body>
</html>
"""
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    
    def _generate_markdown_report(self, report, md_path):
        """生成 Markdown 格式報告"""
        md_content = f"""# 🎨 Fashion CLIP 模型比較報告

**生成時間:** {report["timestamp"]}

## 📋 模型簡介

"""
        
        for model_name, description in report["model_comparison"].items():
            md_content += f"### {model_name}\n{description}\n\n"
        
        # 添加統計摘要
        if "summary" in report:
            md_content += """## 📊 整體表現統計

| 模型 | 整體平均置信度 | 總預測次數 | 表現評級 |
|------|----------------|------------|----------|
"""
            
            for model_name, stats in report["summary"].items():
                avg_conf = stats.get("overall_average_confidence", 0)
                total_pred = stats.get("total_predictions", 0)
                
                if avg_conf >= 0.7:
                    grade = "優秀 ⭐⭐⭐"
                elif avg_conf >= 0.5:
                    grade = "良好 ⭐⭐"
                else:
                    grade = "一般 ⭐"
                
                md_content += f"| **{model_name}** | {avg_conf:.3f} | {total_pred} | {grade} |\n"
        
        # 添加類別表現比較
        if "summary" in report:
            md_content += "\n## 📈 各類別表現比較\n\n"
            
            for model_name, stats in report["summary"].items():
                md_content += f"### {model_name}\n\n"
                
                category_performance = stats.get("category_performance", {})
                if category_performance:
                    md_content += "| 類別 | 平均置信度 | 樣本數 | 最高置信度 |\n"
                    md_content += "|------|------------|--------|------------|\n"
                    
                    for category_name, perf in category_performance.items():
                        avg_conf = perf["average_confidence"]
                        sample_count = perf["sample_count"]
                        max_conf = perf["max_confidence"]
                        md_content += f"| {category_name} | {avg_conf:.3f} | {sample_count} | {max_conf:.3f} |\n"
                    
                    md_content += "\n"
        
        # 添加詳細結果示例
        md_content += "## 🔍 分析結果示例\n\n"
        
        for i, result in enumerate(report["results"][:3], 1):  # 只顯示前3個結果
            if not result["success"]:
                continue
            
            md_content += f"### 圖片 {i}: {result['image_name']}\n\n"
            
            for model_name, analysis in result["analysis"].items():
                md_content += f"#### {model_name}\n\n"
                md_content += f"**整體置信度:** {analysis.get('overall_confidence', 0):.3f}\n\n"
                
                # 顯示前幾個類別結果
                categories = analysis.get("categories", {})
                for category_name, category_result in list(categories.items())[:3]:
                    if "best_match" in category_result:
                        md_content += f"- **{category_name}:** {category_result['best_match']} (置信度: {category_result['confidence']:.3f})\n"
                
                # 顯示生成的提示詞
                if "generated_prompts" in result:
                    prompt = result["generated_prompts"].get(model_name, "")
                    md_content += f"\n**生成的提示詞:**\n```\n{prompt}\n```\n\n"
        
        md_content += """
## 💡 結論與建議

1. **FashionCLIP** 專門針對時尚領域訓練，在服飾識別和風格分析上通常表現更佳
2. **Standard CLIP** 具有更廣泛的理解能力，適合一般性的圖像分析
3. 建議根據具體應用場景選擇合適的模型
4. 可以考慮結合兩個模型的結果來獲得更全面的分析

## 🔧 技術說明

- 使用 PyTorch 和 Transformers 庫
- 支援 CUDA 加速和混合精度推理
- 詳細的時尚類別分析框架
- 自動化的提示詞生成流程
"""
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
    
    def run_comparison_test(self):
        """運行完整的比較測試"""
        print("🚀 開始 Fashion CLIP 模型比較測試")
        print("=" * 60)
        
        # 檢查環境
        if not self.install_requirements():
            return False
        
        # 運行比較分析
        results = self.compare_models_on_images()
        
        if not results:
            print("❌ 測試失敗，無法獲得分析結果")
            return False
        
        # 生成報告
        report = self.generate_comparison_report(results)
        
        # 顯示簡要結果
        successful = sum(1 for r in results if r["success"])
        print(f"\n✅ 測試完成:")
        print(f"   分析圖片數: {len(results)}")
        print(f"   成功分析: {successful}")
        print(f"   成功率: {successful/len(results)*100:.1f}%")
        
        if "summary" in report:
            print(f"\n📊 模型表現摘要:")
            for model_name, stats in report["summary"].items():
                avg_conf = stats.get("overall_average_confidence", 0)
                print(f"   {model_name}: 平均置信度 {avg_conf:.3f}")
        
        return True

if __name__ == "__main__":
    tester = FashionCLIPComparison()
    tester.run_comparison_test()
